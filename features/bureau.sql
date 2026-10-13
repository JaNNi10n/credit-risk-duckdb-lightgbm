-- bureau (other lenders' credits reported to the Credit Bureau) + bureau_balance (monthly status).
-- Two-level aggregation: bureau_balance -> SK_ID_BUREAU, joined to bureau, then -> SK_ID_CURR.
-- past(x) (macro defined in pipeline.py) keeps x only if x <= 0: a positive DAYS_* value is a date
-- after the application (planned end date, or the 365243 sentinel), so it is never used.
-- bureau_balance STATUS: C closed, X unknown, 0 no DPD, 1 = 1-30 DPD, 2 = 31-60, ..., 5 = 120+ / written off.
WITH bb AS (
    SELECT
        SK_ID_BUREAU,
        COUNT(*)                                                                    AS bb_months,
        AVG(CASE WHEN CAST(STATUS AS VARCHAR) IN ('1','2','3','4','5') THEN 1.0 ELSE 0.0 END) AS bb_dpd_rate,
        AVG(CASE WHEN CAST(STATUS AS VARCHAR) IN ('2','3','4','5') THEN 1.0 ELSE 0.0 END)     AS bb_dpd30_rate,
        MAX(CASE WHEN CAST(STATUS AS VARCHAR) IN ('1','2','3','4','5')
                 THEN CAST(CAST(STATUS AS VARCHAR) AS INTEGER) ELSE 0 END)          AS bb_worst_status,
        SUM(CASE WHEN MONTHS_BALANCE >= -12
                  AND CAST(STATUS AS VARCHAR) IN ('1','2','3','4','5') THEN 1 ELSE 0 END) AS bb_dpd_months_12m
    FROM bureau_balance
    WHERE MONTHS_BALANCE <= 0
    GROUP BY SK_ID_BUREAU
),
b AS (
    SELECT bureau.*, bb.* EXCLUDE (SK_ID_BUREAU)
    FROM bureau
    LEFT JOIN bb USING (SK_ID_BUREAU)
)
SELECT
    SK_ID_CURR,
    -- counts
    COUNT(*)                                                                AS bur_count,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END)               AS bur_active_count,
    AVG(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1.0 ELSE 0.0 END)           AS bur_active_share,
    AVG(CASE WHEN CREDIT_ACTIVE = 'Bad debt' THEN 1.0 ELSE 0.0 END)         AS bur_bad_debt_share,
    COUNT(DISTINCT CREDIT_TYPE)                                             AS bur_credit_type_nunique,
    -- recency (DAYS_CREDIT = days before application the credit was opened)
    MIN(past(DAYS_CREDIT))                                                  AS bur_days_credit_min,
    MAX(past(DAYS_CREDIT))                                                  AS bur_days_credit_max,
    AVG(past(DAYS_CREDIT))                                                  AS bur_days_credit_mean,
    MAX(past(DAYS_ENDDATE_FACT))                                            AS bur_days_enddate_fact_max,
    MAX(past(DAYS_CREDIT_UPDATE))                                           AS bur_days_update_max,
    SUM(CASE WHEN past(DAYS_CREDIT) > -365 THEN 1 ELSE 0 END)               AS bur_count_12m,
    -- amounts
    SUM(AMT_CREDIT_SUM)                                                     AS bur_amt_credit_sum,
    AVG(AMT_CREDIT_SUM)                                                     AS bur_amt_credit_mean,
    MAX(AMT_CREDIT_SUM)                                                     AS bur_amt_credit_max,
    SUM(AMT_CREDIT_SUM_DEBT)                                                AS bur_amt_debt_sum,
    MAX(AMT_CREDIT_SUM_DEBT)                                                AS bur_amt_debt_max,
    SUM(AMT_CREDIT_SUM_OVERDUE)                                             AS bur_amt_overdue_sum,
    MAX(AMT_CREDIT_MAX_OVERDUE)                                             AS bur_amt_max_overdue_max,
    SUM(AMT_CREDIT_SUM_LIMIT)                                               AS bur_amt_limit_sum,
    SUM(CASE WHEN past(DAYS_CREDIT) > -365 THEN AMT_CREDIT_SUM END)         AS bur_amt_credit_sum_12m,
    -- ratios
    SUM(AMT_CREDIT_SUM_DEBT) / NULLIF(SUM(AMT_CREDIT_SUM), 0)               AS bur_debt_to_credit,
    SUM(AMT_CREDIT_SUM_OVERDUE) / NULLIF(SUM(AMT_CREDIT_SUM_DEBT), 0)       AS bur_overdue_to_debt,
    -- days past due
    MAX(CREDIT_DAY_OVERDUE)                                                 AS bur_day_overdue_max,
    AVG(CASE WHEN CREDIT_DAY_OVERDUE > 0 THEN 1.0 ELSE 0.0 END)             AS bur_dpd_rate,
    AVG(CASE WHEN CREDIT_DAY_OVERDUE > 30 THEN 1.0 ELSE 0.0 END)            AS bur_dpd30_rate,
    SUM(CNT_CREDIT_PROLONG)                                                 AS bur_prolong_sum,
    -- bureau_balance, already aggregated per bureau credit
    AVG(bb_months)                                                          AS bb_months_mean,
    AVG(bb_dpd_rate)                                                        AS bb_dpd_rate_mean,
    MAX(bb_dpd_rate)                                                        AS bb_dpd_rate_max,
    AVG(bb_dpd30_rate)                                                      AS bb_dpd30_rate_mean,
    MAX(bb_worst_status)                                                    AS bb_worst_status_max,
    SUM(bb_dpd_months_12m)                                                  AS bb_dpd_months_12m_sum
FROM b
GROUP BY SK_ID_CURR
