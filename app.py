from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.db import DB_PATH, bootstrap_database
from src.document_store import CHROMA_DIR, ingest_documents
from src.query_engine import execute_query_plan, generate_fallback_query_plan, generate_query_plan, generate_response, get_llm_config


load_dotenv()
st.set_page_config(page_title="Risk Explain Copilot", page_icon="Risk", layout="wide")


@st.cache_resource(show_spinner="Indexing knowledge base...")
def _ensure_documents_indexed() -> bool:
    return ingest_documents()


def main() -> None:
    bootstrap_database()
    _ensure_documents_indexed()

    st.title("Risk Explain Copilot")
    st.caption("Ask market risk questions. VaR is aggregated from actual trade-level P&L; drivers are explained via risk-factor attribution on the VaR date.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for idx, message in enumerate(st.session_state["chat_history"]):
        _render_chat_message(message, idx)

    question = st.chat_input("Ask about VaR, PNL, desks, drivers, market moves, or trends")
    if question:
        conversation_context = _conversation_context(st.session_state["chat_history"])
        user_message = {"role": "user", "content": question}
        st.session_state["chat_history"].append(user_message)
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Generating SQL, querying SQLite, and forming the response..."):
                assistant_message = _answer_question(question, conversation_context)
            _render_assistant_message(assistant_message, len(st.session_state["chat_history"]))
        st.session_state["chat_history"].append(assistant_message)
        st.rerun()


def _answer_question(question: str, conversation_context: str = "") -> dict[str, object]:
    try:
        plan = generate_query_plan(question, conversation_context=conversation_context)
        try:
            result = execute_query_plan(plan)
        except Exception as exc:
            if not plan.llm_used:
                raise
            fallback_plan = generate_fallback_query_plan(question, conversation_context=conversation_context)
            result = execute_query_plan(fallback_plan)
            plan = fallback_plan
            plan = type(plan)(
                question=plan.question,
                sql=plan.sql,
                intent=plan.intent,
                metric=plan.metric,
                visualization=plan.visualization,
                llm_used=plan.llm_used,
                notes=f"{plan.notes} LLM SQL failed at execution and deterministic fallback was used: {exc}",
                retrieved_context=plan.retrieved_context,
                conversation_context=plan.conversation_context,
            )
        if plan.llm_used and result.empty:
            fallback_plan = generate_fallback_query_plan(question, conversation_context=conversation_context)
            fallback_result = execute_query_plan(fallback_plan)
            if not fallback_result.empty:
                result = fallback_result
                plan = type(fallback_plan)(
                    question=fallback_plan.question,
                    sql=fallback_plan.sql,
                    intent=fallback_plan.intent,
                    metric=fallback_plan.metric,
                    visualization=fallback_plan.visualization,
                    llm_used=fallback_plan.llm_used,
                    notes=f"{fallback_plan.notes} LLM SQL returned no rows and deterministic fallback was used.",
                    retrieved_context=fallback_plan.retrieved_context,
                    conversation_context=fallback_plan.conversation_context,
                )
        answer = generate_response(question, plan, result)
        llm_config = get_llm_config()
        answer_source = llm_config.provider if llm_config else "deterministic fallback"
        trace = {
            "intent": plan.intent,
            "metric": plan.metric,
            "visualization": plan.visualization,
            "sql_source": "LLM" if plan.llm_used else "rule-based fallback",
            "answer_source": answer_source,
            "rows_returned": len(result),
            "database": str(DB_PATH),
            "vector_store": str(CHROMA_DIR),
            "notes": plan.notes,
        }
        return {
            "role": "assistant",
            "question": question,
            "answer": answer,
            "sql": plan.sql.strip(),
            "trace": trace,
            "retrieved_context": list(plan.retrieved_context),
            "conversation_context": conversation_context,
            "result_records": result.to_dict(orient="records"),
            "result_columns": list(result.columns),
        }
    except Exception as exc:
        return {
            "role": "assistant",
            "question": question,
            "answer": f"Could not answer the query: {exc}",
            "sql": "",
            "trace": {"error": str(exc)},
            "retrieved_context": [],
            "conversation_context": conversation_context,
            "result_records": [],
            "result_columns": [],
        }


def _render_chat_message(message: dict[str, object], idx: int) -> None:
    role = str(message["role"])
    with st.chat_message(role):
        if role == "user":
            st.markdown(str(message["content"]))
        else:
            _render_assistant_message(message, idx)


def _render_assistant_message(message: dict[str, object], idx: int) -> None:
    answer = str(message.get("answer", ""))
    result = _message_result_frame(message)
    trace = dict(message.get("trace", {}))

    st.markdown(answer)

    chart = _chart_for_result(result, str(trace.get("visualization", "")))
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True, key=f"chart_{idx}")

    with st.expander("Explanation details", expanded=False):
        sql = str(message.get("sql", ""))
        st.markdown("**Generated SQL**")
        if sql:
            st.code(sql, language="sql")
        else:
            st.caption("No SQL was generated for this response.")

        st.markdown("**Calculation trace**")
        st.write(trace)


def _chart_for_result(result: pd.DataFrame, visualization: str):
    if result.empty:
        return None
    if {"date", "value"}.issubset(result.columns):
        return px.line(result, x="date", y="value", markers=True, title="Returned Trend")
    if "driver_pnl" in result.columns:
        label = _label_column(result)
        frame = result.sort_values("driver_pnl", key=lambda s: s.abs())
        return px.bar(frame, x="driver_pnl", y=label, orientation="h", title="VaR Scenario Drivers")
    if "pnl" in result.columns and "trade_id" in result.columns:
        frame = result.sort_values("pnl", key=lambda s: s.abs())
        return px.bar(frame, x="pnl", y="trade_id", orientation="h", title="Trade-Level P&L")
    if "delta" in result.columns:
        label = _label_column(result)
        frame = result.sort_values("delta", key=lambda s: s.abs())
        return px.bar(frame, x="delta", y=label, orientation="h", title="Returned Driver Deltas")
    if "bar" in visualization.lower():
        numeric_cols = [col for col in result.columns if pd.api.types.is_numeric_dtype(result[col])]
        if numeric_cols:
            label = _label_column(result)
            return px.bar(result, x=numeric_cols[0], y=label, orientation="h", title="Returned Result")
    return None


def _label_column(result: pd.DataFrame) -> str:
    for col in ("risk_factor", "driver", "trade_id", "scenario", "desk", "book", "date", "metric"):
        if col in result.columns:
            return col
    return result.columns[0]


def _message_result_frame(message: dict[str, object]) -> pd.DataFrame:
    records = list(message.get("result_records", []))
    columns = list(message.get("result_columns", []))
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(records, columns=columns)


def _conversation_context(history: list[dict[str, object]], max_turns: int = 6) -> str:
    if not history:
        return ""
    recent = history[-max_turns:]
    lines: list[str] = []
    for message in recent:
        role = str(message.get("role", ""))
        if role == "user":
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"User: {content}")
        elif role == "assistant":
            question = str(message.get("question", "")).strip()
            trace = dict(message.get("trace", {}))
            sql = str(message.get("sql", "")).strip()
            parts = []
            if question:
                parts.append(f"answered_question={question}")
            if trace:
                for key in ("intent", "metric", "visualization", "rows_returned"):
                    if key in trace:
                        parts.append(f"{key}={trace[key]}")
            if sql:
                parts.append(f"sql={sql}")
            if parts:
                lines.append("Assistant: " + "; ".join(parts))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
