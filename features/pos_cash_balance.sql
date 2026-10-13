-- POS_CASH_balance: monthly snapshots of previous point-of-sale and cash loans.
-- MONTHS_BALANCE is relative to the application (-1 = the month before). Only <= 0 is used.
-- SK_DPD = days past due that month; recency window = last 12 months.
SELECT
    SK_ID_CURR,
    COUNT(*)                                                            AS pos_months_count,
    COUNT(DISTINCT SK_ID_PREV)                                          AS pos_loans_nunique,
    MIN(MONTHS_BALANCE)                                                 AS pos_months_balance_min,
    MAX(SK_DPD)                                                         AS pos_dpd_max,
    AVG(SK_DPD)                                                         AS pos_dpd_mean,
    MAX(SK_DPD_DEF)                                                     AS pos_dpd_def_max,
    AVG(CASE WHEN SK_DPD > 0 THEN 1.0 ELSE 0.0 END)                     AS pos_dpd_rate,
    AVG(CASE WHEN SK_DPD > 30 THEN 1.0 ELSE 0.0 END)                    AS pos_dpd30_rate,
    AVG(CNT_INSTALMENT)                                                 AS pos_cnt_instalment_mean,
    AVG(CNT_INSTALMENT_FUTURE / NULLIF(CNT_INSTALMENT, 0))              AS pos_remaining_share_mean,
    AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Completed' THEN 1.0 ELSE 0.0 END) AS pos_completed_share,
    COUNT(*) FILTER (WHERE MONTHS_BALANCE >= -12)                       AS pos_months_count_12m,
    AVG(CASE WHEN SK_DPD > 0 THEN 1.0 ELSE 0.0 END)
        FILTER (WHERE MONTHS_BALANCE >= -12)                            AS pos_dpd_rate_12m,
    MAX(SK_DPD) FILTER (WHERE MONTHS_BALANCE >= -12)                    AS pos_dpd_max_12m
FROM POS_CASH_balance
WHERE MONTHS_BALANCE <= 0
GROUP BY SK_ID_CURR
