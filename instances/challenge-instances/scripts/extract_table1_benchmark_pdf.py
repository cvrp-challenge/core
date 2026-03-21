#!/usr/bin/env python3
"""
Extract Table 1 (XL instance characteristics) from benchmark.pdf and write
instances_characteristics.json.

Expected table columns (paper): #, Name, Dep, Cust, Dem, Q, r, BKS, Method
Output per instance (key = "XL-n...-k....vrp"):
  - size: int (from name, n)
  - min_routes: int (from name, k)
  - avg_route_length: float (column r)
  - capacity: int (column Q)
  - customer_distribution: str (Cust)
  - depot_position: str (Dep)
  - demand_distribution: str (Dem)

Install:
  pip install -r scripts/requirements-table1-extract.txt

Examples:
  python3 scripts/extract_table1_benchmark_pdf.py \\
    --pdf "/path/to/benchmark.pdf" \\
    --out ../instances_characteristics.json

If pdfplumber's vector table finder fails (common for LaTeX PDFs), the script
also scans page text line-by-line for Table 1 rows.

If both fail, copy the markdown pipe table from https://arxiv.org/html/2601.11467v1
into a file and use:
  python3 scripts/extract_table1_benchmark_pdf.py --md table1_paste.md --out ../instances_characteristics.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

NAME_RE = re.compile(r"(XL-n\d+-k\d+)")
NK_RE = re.compile(r"XL-n(\d+)-k(\d+)$")

# Table 1 method column (initial BKS attribution in the paper)
_METHOD_SUFFIXES = (
    "AILS-II",
    "FILO2",
    "FILO",
    "KGLSXXL",
    "HGS-CVRP",
    "SISRs",
    "LKH-3",
    "OR-Tools",
)


def _normalize_pdf_line(line: str) -> str:
    line = line.replace("\u2212", "-").replace("–", "-").replace("—", "-")
    line = re.sub(r"\s+", " ", line.strip())
    return line


def _split_cust_dem(middle: str) -> Tuple[str, str]:
    """Split middle string (after Dep) into Cust and Dem."""
    middle = middle.strip()
    m = re.search(r"\s+(SL|U|Q|\d+[-]\d+)\s*$", middle)
    if m:
        return middle[: m.start()].strip(), m.group(1)
    parts = middle.rsplit(None, 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return middle, ""


def _parse_table1_tail(name: str, dep: str, rest: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Parse trailing fields from Dep onward (Cust, Dem, Q, r, BKS, Method)."""
    method: Optional[str] = None
    for suf in sorted(_METHOD_SUFFIXES, key=len, reverse=True):
        if rest.endswith(suf) and (len(rest) == len(suf) or rest[-len(suf) - 1].isspace()):
            method = suf
            rest = rest[: -len(suf)].rstrip()
            break
    if method is None:
        toks = rest.rsplit(None, 1)
        if len(toks) != 2:
            return None
        rest, method = toks[0], toks[1]

    m = re.search(r"([\d,]+)\s*$", rest)
    if not m:
        return None
    rest = rest[: m.start()].rstrip()

    m = re.search(r"([\d.,]+)\s*$", rest)
    if not m:
        return None
    r_str = m.group(1).replace(",", ".")
    rest = rest[: m.start()].rstrip()

    m = re.search(r"([\d,]+)\s*$", rest)
    if not m:
        return None
    q_str = m.group(1)
    middle = rest[: m.start()].rstrip()

    cust, dem = _split_cust_dem(middle)
    nk = NK_RE.match(name)
    if not nk:
        return None
    try:
        capacity = _parse_int_loose(q_str)
        avg_route_length = float(r_str)
    except ValueError:
        return None

    key = f"{name}.vrp"
    return key, {
        "size": int(nk.group(1)),
        "min_routes": int(nk.group(2)),
        "avg_route_length": avg_route_length,
        "capacity": capacity,
        "customer_distribution": cust,
        "depot_position": dep,
        "demand_distribution": dem,
    }


def parse_table1_text_line(line: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Parse one text line: # Name Dep ... Cust Dem Q r BKS Method
    (same order as Table 1 in the benchmark paper).
    """
    line = _normalize_pdf_line(line)
    m = re.match(r"^\d+\s+(XL-n\d+-k\d+)\s+([ERC])\s+(.+)$", line)
    if m:
        return _parse_table1_tail(m.group(1), m.group(2), m.group(3))
    m = re.match(r"^(XL-n\d+-k\d+)\s+([ERC])\s+(.+)$", line)
    if m:
        return _parse_table1_tail(m.group(1), m.group(2), m.group(3))
    return None


def extract_table1_from_pdf_text(pdf_path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[int]]:
    """Scan all pages' text lines; return (instances dict, list of page numbers that contributed)."""
    import pdfplumber

    out: Dict[str, Dict[str, Any]] = {}
    pages_hit: List[int] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            page_added = False
            for raw in text.splitlines():
                if "XL-n" not in raw:
                    continue
                parsed = parse_table1_text_line(raw)
                if parsed:
                    out[parsed[0]] = parsed[1]
                    page_added = True
            if page_added:
                pages_hit.append(page_index + 1)
    return out, pages_hit


def _norm_header(cell: Optional[str]) -> str:
    if not cell:
        return ""
    s = cell.strip().lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    # arXiv HTML sometimes doubles single-letter column headers
    if s == "qq":
        return "q"
    if s == "rr":
        return "r"
    return s


def _parse_int_loose(s: Optional[str]) -> int:
    if s is None:
        raise ValueError("empty")
    t = str(s).strip().replace(",", "").replace(" ", "")
    t = re.sub(r"[^\d\-]", "", t)
    if not t or t == "-":
        raise ValueError(s)
    return int(t)


def _parse_float_loose(s: Optional[str]) -> float:
    if s is None:
        raise ValueError("empty")
    t = str(s).strip().replace(",", ".")
    t = re.sub(r"[^\d.\-]", "", t)
    if not t or t == "-":
        raise ValueError(s)
    return float(t)


def _header_to_indices(header_row: Sequence[Optional[str]]) -> Optional[Dict[str, int]]:
    """Map logical keys name, dep, cust, dem, q, r -> column indices."""
    keys: List[str] = [_norm_header(c) for c in header_row]
    idx: Dict[str, int] = {}
    for i, k in enumerate(keys):
        if not k:
            continue
        if k == "name":
            idx["name"] = i
        elif k == "dep":
            idx["dep"] = i
        elif k in ("cust", "customer", "customers"):
            idx["cust"] = i
        elif k == "dem":
            idx["dem"] = i
        elif k == "q":
            idx["q"] = i
        elif k == "r":
            idx["r"] = i
    if "name" not in idx:
        return None
    for req in ("dep", "cust", "dem", "q", "r"):
        if req not in idx:
            return None
    return idx


def _row_to_entry(
    row: Sequence[Optional[str]],
    col: Dict[str, int],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    name_cell = row[col["name"]] if col["name"] < len(row) else ""
    m = NAME_RE.search(str(name_cell or ""))
    if not m:
        return None
    stem = m.group(1)
    nk = NK_RE.match(stem)
    if not nk:
        return None
    size = int(nk.group(1))
    min_routes = int(nk.group(2))

    def cell(key: str) -> str:
        i = col[key]
        return str(row[i]).strip() if i < len(row) and row[i] is not None else ""

    try:
        capacity = _parse_int_loose(cell("q"))
        avg_route_length = _parse_float_loose(cell("r"))
    except ValueError:
        return None

    key = f"{stem}.vrp"
    return key, {
        "size": size,
        "min_routes": min_routes,
        "avg_route_length": avg_route_length,
        "capacity": capacity,
        "customer_distribution": cell("cust"),
        "depot_position": cell("dep"),
        "demand_distribution": cell("dem"),
    }


def parse_table_rows(rows: List[Sequence[Optional[str]]]) -> Dict[str, Dict[str, Any]]:
    if not rows:
        return {}
    header_idx: Optional[Dict[str, int]] = None
    start = 0
    for i, row in enumerate(rows):
        h = _header_to_indices(row)
        if h is not None:
            header_idx = h
            start = i + 1
            break
    if header_idx is None:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for row in rows[start:]:
        if not row:
            continue
        parsed = _row_to_entry(row, header_idx)
        if parsed:
            out[parsed[0]] = parsed[1]
    return out


def extract_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    import pdfplumber

    tables_out: List[Dict[str, Any]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            tables = page.extract_tables() or []
            for t_index, table in enumerate(tables):
                if not table:
                    continue
                parsed = parse_table_rows(table)
                if len(parsed) >= 50:
                    tables_out.append(
                        {
                            "page": page_index + 1,
                            "table_index": t_index,
                            "rows": len(table),
                            "instances": len(parsed),
                            "data": parsed,
                        }
                    )
    return tables_out


def parse_markdown_pipe_table(text: str) -> Dict[str, Dict[str, Any]]:
    rows: List[List[Optional[str]]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|\s*---", line):
            continue
        parts = line.split("|")
        cells = [p.strip() if p.strip() else None for p in parts[1:-1]]
        if not cells:
            continue
        rows.append(cells)
    return parse_table_rows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build instances_characteristics.json from Table 1.")
    ap.add_argument("--pdf", type=Path, help="Path to benchmark.pdf")
    ap.add_argument("--md", type=Path, help="Path to a file with Table 1 as markdown pipe rows (fallback)")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "instances_characteristics.json",
        help="Output JSON path (default: ../instances_characteristics.json)",
    )
    ap.add_argument(
        "--dump-candidates",
        action="store_true",
        help="With --pdf: print extract_tables candidates and text-line fallback counts, then exit",
    )
    ap.add_argument(
        "--min-instances",
        type=int,
        default=90,
        help="Minimum parsed instances required to write JSON (default: 90)",
    )
    args = ap.parse_args()

    if not args.pdf and not args.md:
        ap.error("Provide --pdf and/or --md")

    if args.pdf is not None:
        args.pdf = args.pdf.expanduser().resolve()
        if not args.pdf.is_file():
            raise SystemExit(
                f"--pdf is not a readable file:\n  {args.pdf}\n"
                "Use the full path to benchmark.pdf (copy-paste it; paths with '...' are not valid)."
            )

    if args.md is not None:
        args.md = args.md.expanduser().resolve()
        if not args.md.is_file():
            raise SystemExit(
                f"--md is not a readable file:\n  {args.md}"
            )

    chosen: Dict[str, Dict[str, Any]] = {}

    if args.md is not None:
        chosen = parse_markdown_pipe_table(args.md.read_text(encoding="utf-8", errors="replace"))

    if args.pdf is not None:
        text_dict, text_pages = extract_table1_from_pdf_text(args.pdf)
        candidates = extract_from_pdf(args.pdf)
        if args.dump_candidates:
            for c in candidates:
                print(
                    f"extract_tables: page={c['page']} table={c['table_index']} "
                    f"rows={c['rows']} instances={c['instances']}"
                )
            if not candidates:
                print(
                    "extract_tables: no candidates (LaTeX PDFs often lack detectable table lines)."
                )
            print(
                f"text_line_fallback: {len(text_dict)} instances "
                f"(pages with ≥1 hit: {text_pages})"
            )
            return

        best_tab = max(candidates, key=lambda x: x["instances"], default=None)
        tab_dict = best_tab["data"] if best_tab else {}
        pdf_pick = text_dict if len(text_dict) >= len(tab_dict) else tab_dict
        if len(pdf_pick) >= len(chosen):
            chosen = pdf_pick

    if len(chosen) < args.min_instances:
        raise SystemExit(
            f"Only {len(chosen)} instances parsed (need >= {args.min_instances}). "
            "Try --dump-candidates; if text_line_fallback is low, rows may be split across lines "
            "in the PDF — use --md with Table 1 from the paper HTML, or --min-instances."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(dict(sorted(chosen.items())), f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {len(chosen)} instances to {args.out}")


if __name__ == "__main__":
    main()
