#!/usr/bin/env bash
# Upload Celeb-DF-v2 as a private Kaggle Dataset (parthivsuryakb/celeb-df-v2).
#
# One-shot. Re-running after success errors with "dataset already exists";
# in that case use `kaggle datasets version -p <dir> -m "<msg>"` instead.

set -euo pipefail

DATA_DIR="/Users/parthivsuryakb/Deepfake Detection/Celeb-DF-v2"
SLUG="parthivsuryakb/celeb-df-v2"
TITLE="celeb-df-v2"

# ----- sanity checks
[ -d "$DATA_DIR" ] || { echo "ERROR: $DATA_DIR not found"; exit 1; }
command -v kaggle >/dev/null || { echo "ERROR: kaggle CLI not installed (pip install kaggle)"; exit 1; }
[ -f "$HOME/.kaggle/access_token" ] || [ -n "${KAGGLE_API_TOKEN:-}" ] || {
    echo "ERROR: no Kaggle API token configured"; exit 1; }

# ----- clean macOS metadata that bloats uploads
echo "[1/4] removing .DS_Store files..."
find "$DATA_DIR" -name ".DS_Store" -type f -delete 2>/dev/null || true

# ----- write metadata (kaggle CLI requires this inside the staging dir)
META="$DATA_DIR/dataset-metadata.json"
cat > "$META" <<EOF
{
  "title": "$TITLE",
  "id": "$SLUG",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
echo "[2/4] wrote $META"

# Make sure we remove the metadata even if upload fails
trap 'rm -f "$META"; echo "cleaned up $META"' EXIT

# ----- summary
echo "[3/4] about to upload:"
echo "    source : $DATA_DIR"
echo "    size   : $(du -sh "$DATA_DIR" | awk '{print $1}')"
echo "    files  : $(find "$DATA_DIR" -type f | wc -l | xargs)"
echo "    target : https://www.kaggle.com/datasets/$SLUG"
echo ""

# ----- upload (kaggle CLI shows its own progress bar)
echo "[4/4] uploading — go grab a coffee, this takes 20-60 min depending on uplink..."
kaggle datasets create -p "$DATA_DIR"

echo ""
echo "Done. Dataset will appear at:"
echo "    https://www.kaggle.com/datasets/$SLUG"
echo "Kaggle processes the upload server-side for a few more minutes — "
echo "wait until 'Status: ready' on the dataset page before attaching it."
