-- installments_payments: one row per payment made (or missed) on previous Home Credit loans.
-- days_late = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT (> 0 means paid after the due date).
-- Both dates go through past(): a positive value would be after the application date.
WITH p AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        past(DAYS_INSTALMENT)                                       AS due_day,
        past(DAYS_ENTRY_PAYMENT) - past(DAYS_INSTALMENT)            AS days_late,
        AMT_INSTALMENT,
        AMT_PAYMENT
    FROM installments_payments
)
SELECT
    SK_ID_CURR,
    COUNT(*)                                                            AS inst_count,
    COUNT(DISTINCT SK_ID_PREV)                                          AS inst_loans_nunique,
    -- days late
    AVG(GREATEST(days_late, 0))                                         AS inst_days_late_mean,
    MAX(days_late)                                                      AS inst_days_late_max,
    AVG(GREATEST(-days_late, 0))                                        AS inst_days_early_mean,
    AVG(CASE WHEN days_late > 0 THEN 1.0 ELSE 0.0 END)                  AS inst_late_rate,
    AVG(CASE WHEN days_late > 30 THEN 1.0 ELSE 0.0 END)                 AS inst_late30_rate,
    -- amounts and payment / instalment ratios
    SUM(AMT_INSTALMENT)                                                 AS inst_amt_instalment_sum,
    SUM(AMT_PAYMENT)                                                    AS inst_amt_payment_sum,
    MAX(AMT_INSTALMENT)                                                 AS inst_amt_instalment_max,
    AVG(AMT_PAYMENT / NULLIF(AMT_INSTALMENT, 0))                        AS inst_payment_ratio_mean,
    MIN(AMT_PAYMENT / NULLIF(AMT_INSTALMENT, 0))                        AS inst_payment_ratio_min,
    SUM(AMT_PAYMENT) / NULLIF(SUM(AMT_INSTALMENT), 0)                   AS inst_paid_to_due,
    AVG(CASE WHEN AMT_PAYMENT < AMT_INSTALMENT THEN 1.0 ELSE 0.0 END)   AS inst_underpaid_rate,
    SUM(GREATEST(AMT_INSTALMENT - AMT_PAYMENT, 0))                      AS inst_amt_shortfall_sum,
    -- recency window: instalments due in the last 12 months
    COUNT(*) FILTER (WHERE due_day > -365)                              AS inst_count_12m,
    AVG(CASE WHEN days_late > 0 THEN 1.0 ELSE 0.0 END)
        FILTER (WHERE due_day > -365)                                   AS inst_late_rate_12m,
    AVG(AMT_PAYMENT / NULLIF(AMT_INSTALMENT, 0))
        FILTER (WHERE due_day > -365)                                   AS inst_payment_ratio_mean_12m,
    MAX(due_day)                                                        AS inst_days_last_due
FROM p
GROUP BY SK_ID_CURR
