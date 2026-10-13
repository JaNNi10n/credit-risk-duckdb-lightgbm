-- credit_card_balance: monthly snapshots of previous Home Credit credit cards.
-- Utilisation = balance / limit, a classic revolving-credit risk signal.
SELECT
    SK_ID_CURR,
    COUNT(*)                                                                AS cc_months_count,
    COUNT(DISTINCT SK_ID_PREV)                                              AS cc_cards_nunique,
    AVG(AMT_BALANCE)                                                        AS cc_balance_mean,
    MAX(AMT_BALANCE)                                                        AS cc_balance_max,
    AVG(AMT_CREDIT_LIMIT_ACTUAL)                                            AS cc_limit_mean,
    AVG(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))                   AS cc_utilisation_mean,
    MAX(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))                   AS cc_utilisation_max,
    SUM(AMT_DRAWINGS_CURRENT)                                               AS cc_drawings_sum,
    SUM(AMT_DRAWINGS_ATM_CURRENT)                                           AS cc_drawings_atm_sum,
    SUM(AMT_PAYMENT_TOTAL_CURRENT)                                          AS cc_payment_sum,
    AVG(AMT_PAYMENT_CURRENT / NULLIF(AMT_INST_MIN_REGULARITY, 0))           AS cc_payment_to_min_mean,
    SUM(AMT_PAYMENT_TOTAL_CURRENT) / NULLIF(SUM(AMT_DRAWINGS_CURRENT), 0)   AS cc_payment_to_drawings,
    MAX(SK_DPD)                                                             AS cc_dpd_max,
    AVG(CASE WHEN SK_DPD > 0 THEN 1.0 ELSE 0.0 END)                         AS cc_dpd_rate,
    AVG(CASE WHEN SK_DPD > 30 THEN 1.0 ELSE 0.0 END)                        AS cc_dpd30_rate,
    AVG(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))
        FILTER (WHERE MONTHS_BALANCE >= -12)                                AS cc_utilisation_mean_12m,
    AVG(CASE WHEN SK_DPD > 0 THEN 1.0 ELSE 0.0 END)
        FILTER (WHERE MONTHS_BALANCE >= -12)                                AS cc_dpd_rate_12m
FROM credit_card_balance
WHERE MONTHS_BALANCE <= 0
GROUP BY SK_ID_CURR
