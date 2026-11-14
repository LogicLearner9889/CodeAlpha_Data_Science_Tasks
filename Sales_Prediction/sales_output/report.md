# Sales prediction and advertising impact

## Forecast model

- Best model by cross-validated error: **Linear: TV+Radio+TV^2+TVxRadio** (CV R2 0.982, RMSE 0.62).
- Held-out test performance: R2 **0.989**, MAE 0.50, RMSE 0.60 thousand units (average sales are 14.0).
- Typical 90% prediction band: about -0.8 to +0.8 thousand units around a forecast.

## How advertising affects sales

- **TV shows diminishing returns** (squared-spend term -0.00011, p = 0.0000). Extra units per extra $1,000 on TV: 71 at low TV spend, 43 at median, 19 at high spend.
- **Radio pays off more when TV spend is high:** its return per $1,000 rises from 62 units at low TV spend to 317 at high TV spend.
- **No significant saturation was detected for Radio** within the observed range (data are sparse at the highest Radio spend, so this is not proof it never saturates).
- **TV:** about **44 extra units per extra $1,000** at average spend (95% CI 43 to 45).
- **Radio:** about **193 extra units per extra $1,000** at average spend (95% CI 188 to 199).
- **Newspaper adds no reliable effect** once TV and Radio are included: the full-model estimate is 3 units per $1,000 with a 95% CI of -1 to 7, which includes zero (p = 0.15). Its raw correlation with sales (0.23) mostly reflects that Newspaper spend moves together with Radio (correlation 0.35).

Selected regression terms and statistics:

```
               coef  std_err         t  p_value   ci_low  ci_high
Intercept   5.13706  0.19267  26.66290      0.0  4.75708  5.51704
TV          0.05092  0.00223  22.80977      0.0  0.04652  0.05532
Radio       0.03516  0.00590   5.95909      0.0  0.02353  0.04680
TV_sq      -0.00011  0.00001 -15.92010      0.0 -0.00012 -0.00010
TV_x_Radio  0.00108  0.00003  31.06104      0.0  0.00101  0.00114
```

## What-if scenarios (from the average spend mix)

```
                              Sales  Change  Change_%
Baseline (average spend)      14.75    0.00      0.00
TV +10%                       15.37    0.62      4.19
Radio +10%                    15.20    0.45      3.05
TV +10% and Radio +10%        15.86    1.11      7.49
Move $10k Newspaper -> Radio  16.69    1.93     13.11
```

## Budget optimisation

- For a $201k budget (the average total today is $201k): current average mix predicts **14.8** thousand units; the model-optimal mix (TV 150k, Radio 50k, Newspaper 1k) predicts **20.2** (+37%).
- The optimum sits at the edge of observed spend for: Radio, Newspaper. The data cannot show what happens beyond that edge, so treat this as a direction to test, not a guaranteed gain.
- A smaller, safer step: **move $10k newspaper -> radio** is predicted to change sales by +1.9 thousand units (+13%).

```
           Current mix ($k)  Optimal mix ($k)
TV                   147.04            150.20
Radio                 23.26             50.00
Newspaper             30.55              0.66
```

## Recommendations

- Radio currently returns about 4.4x more extra units per dollar than TV at average spend (193 vs 44 per $1,000), but TV is still essential: the two work together, and Radio's gain is only proven up to the observed maximum ($50k).
- Do not increase Newspaper spend on the strength of this data; test reductions and redirect that money to Radio and TV.
- Move toward the suggested mix gradually and measure results (A/B or regional tests) before committing the full budget.
- Re-fit the model each quarter as new campaign data arrives.

## Limitations

- This is **observational data from 200 markets**, not a controlled experiment, so effects show association; other factors (price, seasonality, competitors) are not in the data.
- There is **no date column**, so time-series forecasting and seasonality cannot be analysed, and **no customer-segment column**, so segment effects are not estimated.
- Forecasts outside the observed spend ranges are extrapolations and should be treated with caution.
- Units: spend in thousands of dollars, sales in thousands of units; revenue or profit per unit is needed to turn unit gains into financial ROI.
