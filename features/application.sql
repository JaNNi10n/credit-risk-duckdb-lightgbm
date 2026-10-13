-- Main table: application_train -> one row per SK_ID_CURR (already unique).
-- * DAYS_EMPLOYED = 365243 is a sentinel (pensioners / unemployed), not "1000 years in the future".
--   It becomes NULL and a 0/1 flag keeps the information.
-- * Ratio features: NULLIF guards every denominator so we never produce inf.
SELECT
    * REPLACE (
        CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL ELSE DAYS_EMPLOYED END AS DAYS_EMPLOYED
    ),
    CAST(DAYS_EMPLOYED = 365243 AS INTEGER)                                  AS DAYS_EMPLOYED_ANOM,
    AMT_CREDIT / NULLIF(AMT_INCOME_TOTAL, 0)                                 AS APP_CREDIT_TO_INCOME,
    AMT_ANNUITY / NULLIF(AMT_INCOME_TOTAL, 0)                                AS APP_ANNUITY_TO_INCOME,
    AMT_CREDIT / NULLIF(AMT_ANNUITY, 0)                                      AS APP_CREDIT_TO_ANNUITY,
    AMT_GOODS_PRICE / NULLIF(AMT_CREDIT, 0)                                  AS APP_GOODS_TO_CREDIT,
    (CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL ELSE DAYS_EMPLOYED END)
        / NULLIF(DAYS_BIRTH, 0)                                              AS APP_EMPLOYED_TO_AGE,
    AMT_INCOME_TOTAL / NULLIF(CNT_FAM_MEMBERS, 0)                            AS APP_INCOME_PER_PERSON,
    (COALESCE(EXT_SOURCE_1, 0) + COALESCE(EXT_SOURCE_2, 0) + COALESCE(EXT_SOURCE_3, 0))
        / NULLIF((EXT_SOURCE_1 IS NOT NULL)::INT + (EXT_SOURCE_2 IS NOT NULL)::INT
                 + (EXT_SOURCE_3 IS NOT NULL)::INT, 0)                       AS APP_EXT_SOURCE_MEAN
FROM application
