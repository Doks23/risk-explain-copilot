# Desk-Level Risk Policy

## Scope

This policy applies to all trading desks booking trades into the firm's risk system,
covering IR Swaps, FX Forwards and Swaps, Equity Options and Total Return Swaps,
Credit Index Swaps, Rates Options, and Commodity Forwards.

## Daily Risk Review

Each desk head is responsible for reviewing their desk's VaR and largest scenario P&L
drivers every trading day before market open. Any day where a desk's VaR increases by
more than 20% versus the prior day must be flagged to the desk's assigned Market Risk
officer with a one-line explanation of the driver, sourced from the risk-factor
attribution (not from memory or intuition).

## Escalation Thresholds

- A desk VaR breach above its assigned limit for one day does not require escalation
  beyond the desk head and Market Risk, provided the breach is explained by an
  identifiable risk-factor driver.
- A desk VaR breach above its assigned limit for two consecutive days must be escalated
  to the Head of Market Risk.
- Any unexplained residual (the gap between actual P&L and the linear risk-factor
  attribution on the VaR scenario date) exceeding 10% of the VaR figure itself must be
  investigated before the desk's VaR is reported as final, since a large unexplained
  residual usually indicates a mismodeled or missing sensitivity rather than genuine
  non-linear risk.

## Desk VaR Is Not a Limit Allocation Tool

Desk-level VaR figures must not be summed to derive or validate the entity-level VaR
limit, and the entity-level VaR limit must not be allocated to desks by simple
subtraction or proportional splitting of desk VaRs. Limit allocation across desks is a
separate governance exercise performed quarterly by the Risk Committee and is not
derived from the additive properties of VaR, because VaR does not have additive
properties across desks.

## New Product Approval

Any new product type introduced to a desk (e.g. a new derivative structure) requires a
sign-off from Market Risk confirming that appropriate risk factors and sensitivity
types exist in the sensitivity model before the desk may book live trades of that type.
Booking a trade whose risk factors are not represented in the sensitivity model results
in an unattributable driver gap that will appear as unexplained residual in VaR explain
and will trigger the residual-investigation threshold above.
