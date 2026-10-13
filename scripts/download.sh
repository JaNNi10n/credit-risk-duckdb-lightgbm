#!/usr/bin/env bash
# Download the Home Credit Default Risk competition data into data/raw/ (gitignored).
#
# Before running:
#   1. Sign in on kaggle.com and accept the rules at
#      https://www.kaggle.com/competitions/home-credit-default-risk/rules
#   2. Create an API token (kaggle.com -> Settings -> API) and save it as ~/.kaggle/kaggle.json
#      (or ~/.kaggle/access_token for new-style tokens), or export KAGGLE_USERNAME / KAGGLE_KEY.
#   3. pip install -r requirements.txt   (installs the kaggle CLI)
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw
PY=${PYTHON:-python}
mkdir -p "$RAW"

"$PY" -m kaggle competitions download -c home-credit-default-risk -p "$RAW"
"$PY" -m zipfile -e "$RAW/home-credit-default-risk.zip" "$RAW"
rm -f "$RAW/home-credit-default-risk.zip"

for f in application_train bureau bureau_balance previous_application installments_payments POS_CASH_balance credit_card_balance; do
  [ -f "$RAW/$f.csv" ] || { echo "missing $RAW/$f.csv" >&2; exit 1; }
done
echo "OK: raw CSVs in $RAW"
ls -l "$RAW"
