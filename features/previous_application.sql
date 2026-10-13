-- previous_application: earlier Home Credit applications by the same client -> SK_ID_CURR.
-- NAME_CONTRACT_STATUS in {Approved, Refused, Canceled, Unused offer}.
-- DAYS_DECISION is always <= 0; DAYS_LAST_DUE / DAYS_TERMINATION contain the 365243 sentinel and
-- future dates, so they only enter through past().
SELECT
    SK_ID_CURR,
    COUNT(*)                                                                    AS prev_count,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END)          AS prev_approved_count,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END)           AS prev_refused_count,
    AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1.0 ELSE 0.0 END)       AS prev_refused_share,
    AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Canceled' THEN 1.0 ELSE 0.0 END)      AS prev_canceled_share,
    AVG(CASE WHEN NAME_CONTRACT_TYPE = 'Revolving loans' THEN 1.0 ELSE 0.0 END) AS prev_revolving_share,
    AVG(CASE WHEN NAME_YIELD_GROUP = 'high' THEN 1.0 ELSE 0.0 END)              AS prev_yield_high_share,
    -- amounts
    AVG(AMT_APPLICATION)                                                        AS prev_amt_application_mean,
    MAX(AMT_APPLICATION)                                                        AS prev_amt_application_max,
    SUM(AMT_CREDIT)                                                             AS prev_amt_credit_sum,
    AVG(AMT_CREDIT)                                                             AS prev_amt_credit_mean,
    MAX(AMT_CREDIT)                                                             AS prev_amt_credit_max,
    AVG(AMT_ANNUITY)                                                            AS prev_amt_annuity_mean,
    AVG(AMT_DOWN_PAYMENT)                                                       AS prev_amt_down_payment_mean,
    AVG(RATE_DOWN_PAYMENT)                                                      AS prev_rate_down_payment_mean,
    AVG(CNT_PAYMENT)                                                            AS prev_cnt_payment_mean,
    -- ratios: how much of what was asked for was granted
    AVG(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0))                                AS prev_credit_to_application_mean,
    MIN(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0))                                AS prev_credit_to_application_min,
    AVG(AMT_CREDIT / NULLIF(AMT_ANNUITY, 0))                                    AS prev_credit_to_annuity_mean,
    -- recency
    MAX(past(DAYS_DECISION))                                                    AS prev_days_decision_max,
    MIN(past(DAYS_DECISION))                                                    AS prev_days_decision_min,
    AVG(past(DAYS_DECISION))                                                    AS prev_days_decision_mean,
    MAX(past(DAYS_LAST_DUE))                                                    AS prev_days_last_due_max,
    MAX(past(DAYS_TERMINATION))                                                 AS prev_days_termination_max,
    SUM(CASE WHEN past(DAYS_DECISION) > -365 THEN 1 ELSE 0 END)                 AS prev_count_12m,
    SUM(CASE WHEN past(DAYS_DECISION) > -365
              AND NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END)           AS prev_refused_count_12m,
    AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1.0 ELSE 0.0 END)
        FILTER (WHERE past(DAYS_DECISION) > -365)                               AS prev_refused_share_12m
FROM previous_application
GROUP BY SK_ID_CURR
