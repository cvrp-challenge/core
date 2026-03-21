#!/usr/bin/env bash
# Usage:
#   ./run_extract_table1.sh "/path/to/benchmark.pdf"
# Or set PDF path in BENCHMARK_PDF.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDF="${1:-${BENCHMARK_PDF:-}}"

if [[ -z "$PDF" ]]; then
  echo "Usage: $0 /path/to/benchmark.pdf" >&2
  exit 1
fi

python3 -m pip install -q -r "${SCRIPT_DIR}/requirements-table1-extract.txt"
python3 "${SCRIPT_DIR}/extract_table1_benchmark_pdf.py" --pdf "$PDF" --out "${SCRIPT_DIR}/../instances_characteristics.json"
