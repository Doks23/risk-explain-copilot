from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.db import DATA_DIR, DB_PATH, bootstrap_database, load_csvs_to_sqlite, save_uploaded_csv
from src.query_engine import execute_query_plan, generate_fallback_query_plan, generate_query_plan, generate_response, get_llm_config
from src.vector_store import VECTOR_DB_PATH, initialize_vector_store


load_dotenv()
st.set_page_config(page_title="Risk Explain Copilot", page_icon="Risk", layout="wide")


def main() -> None:
    bootstrap_database()
    initialize_vector_store()
    _sidebar()

    st.title("Risk Explain Copilot")
    st.caption("Ask market risk questions in chat. Each answer keeps its supporting SQL, bounded result, and trace in collapsed details.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if not st.session_state["chat_history"]:
        st.info("Ask a question below. The app will generate safe SQL, execute only that bounded query, then answer from the returned rows.")

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
            "vector_store": str(VECTOR_DB_PATH),
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

        st.markdown("**Data used for this response**")
        st.caption("This is the bounded SQL result returned to the answer layer, not the full database.")
        if result.empty:
            st.caption("No rows returned.")
        else:
            st.dataframe(_display_frame(result), use_container_width=True, hide_index=True)

        st.markdown("**Calculation trace**")
        st.write(trace)
        context = list(message.get("retrieved_context", []))
        if context:
            st.markdown("**Retrieved context**")
            for item in context:
                st.markdown(f"- **{item['title']}** `{item['score']}`: {item['text']}")
        conversation_context = str(message.get("conversation_context", "")).strip()
        if conversation_context:
            st.markdown("**Conversation context used**")
            st.text(conversation_context)


def _sidebar() -> None:
    with st.sidebar:
        st.header("Data Admin")
        if st.button("Clear chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

        st.divider()
        uploaded_files = st.file_uploader(
            "Upload replacement CSVs",
            type=["csv"],
            accept_multiple_files=True,
            help="Accepted names: hierarchy.csv, pnl_results.csv, var_results.csv, sensitivities.csv, market_data.csv, scenario_data.csv",
        )
        if uploaded_files:
            for uploaded in uploaded_files:
                save_uploaded_csv(uploaded.name, uploaded.getvalue())
            load_csvs_to_sqlite(reset=True)
            st.success(f"Loaded {len(uploaded_files)} uploaded file(s).")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reload CSVs", use_container_width=True):
                load_csvs_to_sqlite(reset=True)
                st.success("Reloaded SQLite database.")
        with col2:
            if st.button("Reset data", use_container_width=True):
                from src.data_generator import generate_sample_data

                generate_sample_data(DATA_DIR)
                load_csvs_to_sqlite(reset=True)
                st.success("Regenerated sample data.")

        st.divider()
        llm_config = get_llm_config()
        if llm_config:
            st.success(f"{llm_config.provider} mode enabled")
            st.caption(f"Model: {llm_config.model}")
        else:
            st.warning("Mock mode")
            st.caption("No OpenAI or Gemini key found. Using deterministic SQL and response fallback.")
        st.caption(f"Vector context store: {VECTOR_DB_PATH.name}")
        st.caption("The UI does not expose raw database tables.")


def _chart_for_result(result: pd.DataFrame, visualization: str):
    if result.empty:
        return None
    if {"date", "value"}.issubset(result.columns):
        return px.line(result, x="date", y="value", markers=True, title="Returned Trend")
    if "estimated_pnl_impact" in result.columns and "risk_factor" in result.columns:
        frame = result.sort_values("estimated_pnl_impact", key=lambda s: s.abs())
        return px.bar(frame, x="estimated_pnl_impact", y="risk_factor", orientation="h", title="Estimated PNL Impact")
    if "delta" in result.columns:
        label = _label_column(result)
        frame = result.sort_values("delta", key=lambda s: s.abs())
        return px.bar(frame, x="delta", y=label, orientation="h", title="Returned Driver Deltas")
    if visualization == "bar":
        numeric_cols = [col for col in result.columns if pd.api.types.is_numeric_dtype(result[col])]
        if numeric_cols:
            label = _label_column(result)
            return px.bar(result, x=numeric_cols[0], y=label, orientation="h", title="Returned Result")
    return None


def _label_column(result: pd.DataFrame) -> str:
    if {"desk", "book", "portfolio", "scenario", "risk_factor"}.issubset(result.columns):
        return "risk_factor"
    for col in ("scenario", "risk_factor", "desk", "book", "date", "metric"):
        if col in result.columns:
            return col
    return result.columns[0]


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for col in display.select_dtypes(include=["float", "int"]).columns:
        display[col] = display[col].map(lambda value: round(float(value), 4))
    return display


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
