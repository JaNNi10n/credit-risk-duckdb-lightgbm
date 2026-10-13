"""Synthetic Home-Credit-shaped CSVs for tests and CI ONLY.

The real Kaggle data can't be committed or downloaded in CI, so tests run on small generated tables
that use the real column names, dtypes and quirks: the DAYS_EMPLOYED=365243 sentinel, positive
(future) DAYS_* values in the secondary tables, applicants with no history, and bureau rows for IDs
that are not in application_train (as in the real data, where they belong to application_test).
A hidden risk score drives TARGET and the secondary tables, so the models have some signal to learn.
Numbers produced from this data say nothing about real credit risk and are never reported as results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _repeat(ids, z, lam, rng):
    n = rng.poisson(lam, size=len(ids))
    return np.repeat(ids, n), np.repeat(z, n)


def make_tables(n_apps: int = 300, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.arange(100001, 100001 + n_apps)
    z = rng.normal(size=n_apps)  # hidden risk score
    target = (rng.random(n_apps) < _sig(-2.7 + 1.1 * z)).astype(int)

    income = np.round(rng.lognormal(11.9, 0.5, n_apps), -2)
    credit = np.round(income * rng.uniform(1, 6, n_apps), -2)
    annuity = np.round(credit / rng.uniform(10, 40, n_apps), 1)
    employed = -rng.integers(30, 12000, n_apps).astype(float)
    employed[rng.random(n_apps) < 0.15] = 365243  # pensioner / unemployed sentinel

    def ext(scale):
        v = np.clip(_sig(-0.9 * z + rng.normal(0, scale, n_apps)), 0.001, 0.999)
        v[rng.random(n_apps) < 0.2] = np.nan
        return np.round(v, 6)

    occ = rng.choice(["Laborers", "Sales staff", "Core staff", "Managers", "Drivers"], n_apps).astype(object)
    occ[rng.random(n_apps) < 0.3] = None
    app = pd.DataFrame({
        "SK_ID_CURR": ids,
        "TARGET": target,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n_apps, p=[0.9, 0.1]),
        "CODE_GENDER": rng.choice(["F", "M"], n_apps, p=[0.66, 0.34]),
        "FLAG_OWN_CAR": rng.choice(["Y", "N"], n_apps),
        "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n_apps),
        "CNT_CHILDREN": rng.poisson(0.4, n_apps),
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": np.round(credit * rng.uniform(0.8, 1.0, n_apps), -2),
        "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner", "Commercial associate"], n_apps),
        "NAME_EDUCATION_TYPE": rng.choice(["Secondary / secondary special", "Higher education"], n_apps),
        "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married", "Civil marriage"], n_apps),
        "REGION_RATING_CLIENT": rng.choice([1, 2, 3], n_apps, p=[0.1, 0.75, 0.15]),
        "DAYS_BIRTH": -rng.integers(7500, 25000, n_apps),
        "DAYS_EMPLOYED": employed,
        "DAYS_REGISTRATION": -rng.integers(0, 15000, n_apps).astype(float),
        "DAYS_ID_PUBLISH": -rng.integers(0, 7000, n_apps),
        "OWN_CAR_AGE": np.where(rng.random(n_apps) < 0.65, np.nan, rng.integers(0, 30, n_apps)),
        "OCCUPATION_TYPE": occ,
        "CNT_FAM_MEMBERS": rng.integers(1, 5, n_apps).astype(float),
        "ORGANIZATION_TYPE": rng.choice(["Business Entity Type 3", "XNA", "Self-employed", "Other"], n_apps),
        "EXT_SOURCE_1": ext(0.8),
        "EXT_SOURCE_2": ext(0.6),
        "EXT_SOURCE_3": ext(0.7),
        "DAYS_LAST_PHONE_CHANGE": -rng.integers(0, 4000, n_apps).astype(float),
    })

    # Owners for secondary tables: real applicants plus 5% orphan IDs (like application_test IDs).
    n_orph = max(3, n_apps // 20)
    owner_ids = np.concatenate([ids, np.arange(900001, 900001 + n_orph)])
    owner_z = np.concatenate([z, rng.normal(size=n_orph)])

    # bureau ---------------------------------------------------------------------------------
    b_cur, b_z = _repeat(owner_ids, owner_z, 3.0, rng)
    nb = len(b_cur)
    days_credit = -rng.integers(1, 2900, nb)
    active = rng.random(nb) < _sig(-0.5 + 0.6 * b_z)
    overdue_days = np.where(rng.random(nb) < _sig(-3.5 + 1.2 * b_z), rng.integers(1, 90, nb), 0)
    b_amt = np.round(rng.lognormal(12, 1, nb), 2)
    debt = np.where(active, np.round(b_amt * rng.uniform(0, 1, nb), 2), 0.0)
    bureau = pd.DataFrame({
        "SK_ID_CURR": b_cur,
        "SK_ID_BUREAU": np.arange(5000001, 5000001 + nb),
        "CREDIT_ACTIVE": np.where(active, "Active", np.where(rng.random(nb) < 0.02, "Bad debt", "Closed")),
        "CREDIT_CURRENCY": "currency 1",
        "DAYS_CREDIT": days_credit,
        "CREDIT_DAY_OVERDUE": overdue_days,
        "DAYS_CREDIT_ENDDATE": days_credit + rng.integers(100, 3000, nb).astype(float),  # often > 0
        "DAYS_ENDDATE_FACT": np.where(active, np.nan, days_credit + rng.integers(0, 2000, nb)),
        "AMT_CREDIT_MAX_OVERDUE": np.where(overdue_days > 0, np.round(b_amt * 0.05, 2), np.nan),
        "CNT_CREDIT_PROLONG": rng.poisson(0.05, nb),
        "AMT_CREDIT_SUM": b_amt,
        "AMT_CREDIT_SUM_DEBT": debt,
        "AMT_CREDIT_SUM_LIMIT": np.where(rng.random(nb) < 0.3, np.round(b_amt * 0.2, 2), 0.0),
        "AMT_CREDIT_SUM_OVERDUE": np.where(overdue_days > 0, np.round(debt * 0.1, 2), 0.0),
        "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card", "Car loan", "Mortgage"], nb),
        "DAYS_CREDIT_UPDATE": np.where(rng.random(nb) < 0.03, rng.integers(1, 30, nb), -rng.integers(0, 1000, nb)),
        "AMT_ANNUITY": np.where(rng.random(nb) < 0.6, np.nan, np.round(b_amt / 24, 2)),
    })
    bureau["DAYS_ENDDATE_FACT"] = bureau["DAYS_ENDDATE_FACT"].clip(upper=0)

    # bureau_balance: monthly status per bureau credit -----------------------------------------
    months = rng.integers(1, 24, nb)
    bb_id = np.repeat(bureau["SK_ID_BUREAU"].to_numpy(), months)
    bb_z = np.repeat(b_z, months)
    bb_month = -np.concatenate([np.arange(m) for m in months])
    late = rng.random(len(bb_id)) < _sig(-3.0 + 1.0 * bb_z)
    status = np.where(late, rng.choice(["1", "2", "3", "5"], len(bb_id), p=[0.7, 0.2, 0.07, 0.03]),
                      rng.choice(["0", "C", "X"], len(bb_id), p=[0.6, 0.3, 0.1]))
    bureau_balance = pd.DataFrame({"SK_ID_BUREAU": bb_id, "MONTHS_BALANCE": bb_month, "STATUS": status})

    # previous_application ----------------------------------------------------------------------
    p_cur, p_z = _repeat(owner_ids, owner_z, 2.5, rng)
    npv = len(p_cur)
    refused = rng.random(npv) < _sig(-1.6 + 0.9 * p_z)
    p_app = np.round(rng.lognormal(11.5, 0.9, npv), 1)
    p_credit = np.round(p_app * rng.uniform(0.8, 1.15, npv), 1)
    decision = -rng.integers(1, 2900, npv)
    status_prev = np.where(refused, "Refused", rng.choice(["Approved", "Canceled", "Unused offer"], npv, p=[0.8, 0.17, 0.03]))
    sentinel = rng.random(npv) < 0.4
    prev = pd.DataFrame({
        "SK_ID_PREV": np.arange(2000001, 2000001 + npv),
        "SK_ID_CURR": p_cur,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Consumer loans", "Revolving loans"], npv),
        "AMT_ANNUITY": np.round(p_credit / rng.uniform(6, 36, npv), 2),
        "AMT_APPLICATION": p_app,
        "AMT_CREDIT": p_credit,
        "AMT_DOWN_PAYMENT": np.where(rng.random(npv) < 0.5, np.nan, np.round(p_app * 0.1, 2)),
        "AMT_GOODS_PRICE": p_app,
        "RATE_DOWN_PAYMENT": np.where(rng.random(npv) < 0.5, np.nan, np.round(rng.uniform(0, 0.3, npv), 4)),
        "NAME_CONTRACT_STATUS": status_prev,
        "DAYS_DECISION": decision,
        "CNT_PAYMENT": rng.choice([6, 12, 24, 36], npv).astype(float),
        "NAME_YIELD_GROUP": rng.choice(["low_normal", "middle", "high", "XNA"], npv),
        "DAYS_FIRST_DUE": np.where(status_prev == "Approved", decision + 30, np.nan),
        "DAYS_LAST_DUE": np.where(status_prev == "Approved", np.where(sentinel, 365243, decision + rng.integers(30, 4000, npv)), np.nan),
        "DAYS_TERMINATION": np.where(status_prev == "Approved", np.where(sentinel, 365243, decision + rng.integers(40, 4100, npv)), np.nan),
    })

    # installments_payments (approved previous loans only) --------------------------------------
    appr = prev[prev["NAME_CONTRACT_STATUS"] == "Approved"]
    appr_z = p_z[(status_prev == "Approved")]
    k = rng.integers(1, 12, len(appr))
    i_prev = np.repeat(appr["SK_ID_PREV"].to_numpy(), k)
    i_cur = np.repeat(appr["SK_ID_CURR"].to_numpy(), k)
    i_z = np.repeat(appr_z, k)
    i_num = np.concatenate([np.arange(1, m + 1) for m in k])
    i_due = np.repeat(appr["DAYS_DECISION"].to_numpy(), k) + 30 * i_num
    ni = len(i_prev)
    i_late = np.where(rng.random(ni) < _sig(-2.2 + 1.0 * i_z), rng.integers(1, 60, ni), -rng.integers(0, 15, ni))
    i_amt = np.round(np.repeat(appr["AMT_ANNUITY"].to_numpy(), k), 2)
    short = rng.random(ni) < _sig(-3.0 + 0.8 * i_z)
    inst = pd.DataFrame({
        "SK_ID_PREV": i_prev,
        "SK_ID_CURR": i_cur,
        "NUM_INSTALMENT_VERSION": 1.0,
        "NUM_INSTALMENT_NUMBER": i_num,
        "DAYS_INSTALMENT": i_due.astype(float),  # can be > 0 (instalments due after the application)
        "DAYS_ENTRY_PAYMENT": (i_due + i_late).astype(float),
        "AMT_INSTALMENT": i_amt,
        "AMT_PAYMENT": np.where(short, np.round(i_amt * 0.5, 2), i_amt),
    })

    # POS_CASH_balance + credit_card_balance: monthly snapshots of approved previous loans ------
    def monthly(sub, sub_z, max_m):
        m = rng.integers(1, max_m, len(sub))
        return (np.repeat(sub["SK_ID_PREV"].to_numpy(), m), np.repeat(sub["SK_ID_CURR"].to_numpy(), m),
                np.repeat(sub_z, m), -np.concatenate([np.arange(1, x + 1) for x in m]))

    pos_mask = (appr["NAME_CONTRACT_TYPE"] != "Revolving loans").to_numpy()
    pp, pc, pz, pm = monthly(appr[pos_mask], appr_z[pos_mask], 15)
    npos = len(pp)
    dpd = np.where(rng.random(npos) < _sig(-3.0 + 1.0 * pz), rng.integers(1, 60, npos), 0)
    cnt = rng.choice([6.0, 12.0, 24.0], npos)
    pos = pd.DataFrame({
        "SK_ID_PREV": pp, "SK_ID_CURR": pc, "MONTHS_BALANCE": pm,
        "CNT_INSTALMENT": cnt, "CNT_INSTALMENT_FUTURE": np.floor(cnt * rng.uniform(0, 1, npos)),
        "NAME_CONTRACT_STATUS": rng.choice(["Active", "Completed"], npos, p=[0.9, 0.1]),
        "SK_DPD": dpd, "SK_DPD_DEF": np.where(dpd > 30, dpd - 30, 0),
    })

    cc_mask = ~pos_mask
    cp, cc_cur, cz, cm = monthly(appr[cc_mask], appr_z[cc_mask], 20)
    ncc = len(cp)
    limit = rng.choice([45000.0, 90000.0, 180000.0], ncc)
    util = np.clip(_sig(-0.5 + 0.9 * cz + rng.normal(0, 0.8, ncc)), 0, 1.2)
    draw = np.round(limit * rng.uniform(0, 0.2, ncc), 2)
    minpay = np.round(limit * util * 0.05, 2)
    pay = np.round(minpay * rng.uniform(0.3, 3, ncc), 2)
    cc_dpd = np.where(rng.random(ncc) < _sig(-3.2 + 1.0 * cz), rng.integers(1, 60, ncc), 0)
    cc = pd.DataFrame({
        "SK_ID_PREV": cp, "SK_ID_CURR": cc_cur, "MONTHS_BALANCE": cm,
        "AMT_BALANCE": np.round(limit * util, 2), "AMT_CREDIT_LIMIT_ACTUAL": limit,
        "AMT_DRAWINGS_ATM_CURRENT": np.round(draw * 0.5, 2), "AMT_DRAWINGS_CURRENT": draw,
        "AMT_INST_MIN_REGULARITY": minpay, "AMT_PAYMENT_CURRENT": pay, "AMT_PAYMENT_TOTAL_CURRENT": pay,
        "NAME_CONTRACT_STATUS": "Active", "SK_DPD": cc_dpd, "SK_DPD_DEF": np.where(cc_dpd > 30, 1, 0),
    })

    return {
        "application_train.csv": app,
        "bureau.csv": bureau,
        "bureau_balance.csv": bureau_balance,
        "previous_application.csv": prev,
        "installments_payments.csv": inst,
        "POS_CASH_balance.csv": pos,
        "credit_card_balance.csv": cc,
    }


def write_tables(out_dir: Path, n_apps: int = 300, seed: int = 0) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SYNTHETIC_DATA.txt").write_text(
        f"Generated by credit/synthetic.py (n_apps={n_apps}, seed={seed}). Not Home Credit data.\n", encoding="utf-8")
    for name, df in make_tables(n_apps, seed).items():
        df.to_csv(out_dir / name, index=False, lineterminator="\n")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Write synthetic Home-Credit-shaped CSVs (tests/CI only).")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    print(write_tables(a.out, a.n, a.seed))
