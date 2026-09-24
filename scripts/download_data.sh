#!/usr/bin/env bash
# Download the Kaggle credit-card fraud dataset into data/raw/.
# Requires the kaggle CLI and an API token at ~/.kaggle/kaggle.json.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$PROJECT_ROOT/data/raw"
DATASET="mlg-ulb/creditcardfraud"

mkdir -p "$RAW_DIR"

if [ -f "$RAW_DIR/creditcard.csv" ]; then
  echo "creditcard.csv already present in $RAW_DIR — nothing to do."
  exit 0
fi

if kaggle datasets download -d "$DATASET" -p "$RAW_DIR" --unzip; then
  echo "Downloaded $DATASET to $RAW_DIR"
  ls -lh "$RAW_DIR"
else
  cat <<EOF

Kaggle CLI download failed. Download manually:
  1. Open https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  2. Click "Download" (you must be logged in).
  3. Unzip and place creditcard.csv at:
       $RAW_DIR/creditcard.csv

To fix the CLI instead: kaggle.com -> Settings -> API -> Create New Token,
then save kaggle.json to ~/.kaggle/kaggle.json and re-run this script.
EOF
  exit 1
fi
