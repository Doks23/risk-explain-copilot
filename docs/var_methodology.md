# Value-at-Risk Methodology

## Purpose

This document describes how the firm calculates and explains historical-simulation
Value-at-Risk (VaR) for trading desks, products, and the overall entity. It is the
reference used by Market Risk when responding to audit, regulatory, or desk queries
about how a VaR number was produced.

## Historical Simulation Approach

VaR is calculated using full historical simulation over a rolling window of historical
scenario dates. Each historical date represents one full set of risk-factor shocks
observed on that date (rates, FX, equity, credit spreads, commodities). For every
scenario date, the firm revalues every trade in scope and sums the resulting P&L to
produce one portfolio P&L observation for that date. Across all scenario dates this
produces a P&L distribution for the scope being measured.

## Percentile Convention

The firm uses the nearest-rank method for percentile selection, not linear
interpolation. Given N historical scenario observations and a confidence level c
(typically 95%), the VaR is the ceil(N x c)-th smallest loss in the ranked loss
distribution. For a 50-day window at 95% confidence, this selects the 3rd-worst
historical day (only 2 of 50 days are permitted to breach VaR, which is more
conservative than the raw 2.5-day tail implied by 5%).

Linear-interpolation percentile methods (as used by default in many statistics
packages) are NOT used for regulatory VaR reporting at this firm, because they
can select a loss level that does not correspond to any actual historical
scenario, which complicates explainability.

## Non-Additivity of VaR

VaR is explicitly NOT additive across scopes. The entity-level VaR is not the sum of
desk-level VaRs, and a desk's VaR is not the sum of its trades' VaRs. This is because
VaR is a percentile of an aggregated distribution, and the worst-case historical date
for a broad scope (e.g. the entity) is frequently a different date than the worst-case
date for a narrower scope (e.g. a single desk), since offsetting positions across desks
change which historical date is worst for the combined book. Only P&L is additive;
VaR is not. Any request to "add up desk VaRs to check the entity VaR" reflects a
misunderstanding of the methodology and should be corrected, not accommodated.

## Explaining a VaR Number

When asked to explain what "drove" a VaR figure, Market Risk does not use full
revaluation P&L. Instead, it performs a linear risk-factor attribution on the single
historical date that was selected as the VaR scenario: for each risk factor, multiply
the trade-level sensitivity (e.g. IR Delta, FX Delta, Vega, CS01) by that risk factor's
shock on the VaR date. This attributed P&L is summed by risk factor and ranked by
absolute contribution to identify the largest drivers.

Because sensitivities are a linear (first-order) approximation of a trade's true
P&L response, the sum of attributed driver P&L will generally not exactly equal the
actual full-revaluation P&L on that date. The difference is convexity, cross-gamma, and
other non-linear effects not captured by static point-in-time sensitivities. This gap
must always be disclosed explicitly as an "unexplained residual" line alongside the
attributed drivers, rather than silently ignored — a VaR explain that does not
reconcile to the actual P&L is considered incomplete for review purposes.

## Confidence Level

95% is the standard confidence level for daily desk-level and entity-level VaR
reporting. Higher confidence levels (e.g. 99%) are used for regulatory capital
calculations under a separate methodology and are out of scope for this document.
