#!/bin/bash
# Assemble <rollnumber>_assignment1.zip for the Moodle upload.
#
# Usage:  bash scripts/make_submission.sh [rollnumber]
#
# Contents:  src/, scripts/, outputs/ (results, plots, samples, tokenizers —
#            NOT the .pt checkpoints, which are hosted on HuggingFace),
#            README.md, report/ (tex + pdf).
set -euo pipefail
cd "$(dirname "$0")/.."
ROLL=${1:-63237038}
STAGE="/tmp/${ROLL}_assignment1"

rm -rf "$STAGE"
mkdir -p "$STAGE/src" "$STAGE/scripts" "$STAGE/outputs" "$STAGE/report"

cp -r src/models src/dataset.py src/train.py src/utils.py "$STAGE/src/"
cp scripts/*.sh scripts/*.py "$STAGE/scripts/" 2>/dev/null || true

for C in C1 C2 C3 C4 C5; do
  if [ -d "outputs/$C" ]; then
    mkdir -p "$STAGE/outputs/$C"
    # results, config, plots, samples, tokenizers — but no checkpoint .pt
    find "outputs/$C" -type f ! -name "*.pt" -exec cp --parents {} "$STAGE/" \;
  else
    echo "warning: outputs/$C missing"
  fi
done

cp README.md "$STAGE/" 2>/dev/null || true
cp IMPLEMENTATION.md "$STAGE/" 2>/dev/null || true
cp report/Report.tex "$STAGE/report/" 2>/dev/null || true
if [ -f report/Report.pdf ]; then
  cp report/Report.pdf "$STAGE/Report.pdf"
else
  echo "warning: report/Report.pdf missing (compile the LaTeX first)"
fi

ROOT="$(pwd)"
rm -f "${ROOT}/${ROLL}_assignment1.zip"
(cd "$STAGE" && zip -rq "${ROOT}/${ROLL}_assignment1.zip" .)
echo "wrote ${ROOT}/${ROLL}_assignment1.zip: $(du -h ${ROOT}/${ROLL}_assignment1.zip | cut -f1)"
