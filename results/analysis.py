#!/usr/bin/env python
import argparse
import csv
import json
import re
import statistics
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
INSTANCES_DIR = ROOT / "instances" / "challenge-instances"
LOGS_DIR = ROOT / "results" / "out" / "logs"
SOLUTIONS_DIR = ROOT / "results" / "out" / "solutions"
TRAJECTORIES_DIR = ROOT / "results" / "out" / "trajectories"
SUMMARY_CSV = ROOT / "results" / "summary.csv"
OVERVIEW_MD = ROOT / "results" / "OVERVIEW.md"
CHALLENGE_BKS_PATH = INSTANCES_DIR / "challenge-bks.json"
INITIAL_BKS_PATH = INSTANCES_DIR / "initial-bks.json"

# Optional columns in summary.csv (gaps vs. paper Table 2 best costs per instance).
PAPER_GAP_COLS = (
    "gap_to_paper_filo_best_percent",
    "gap_to_paper_filo2_best_percent",
    "gap_to_paper_hgs_cvrp_best_percent",
)

# Gaps vs. paper Table 2 *mean* cost (60-run average) for FILO / FILO2 / HGS-CVRP.
PAPER_MEAN_GAP_COLS = (
    "gap_to_paper_filo_mean_percent",
    "gap_to_paper_filo2_mean_percent",
    "gap_to_paper_hgs_cvrp_mean_percent",
)

ARXIV_HTML_TABLE2_URL = "https://arxiv.org/html/2601.11467v1"


@dataclass
class InstanceLogSummary:
    instance: str
    best_cost: Optional[float]
    best_gap: Optional[float]
    iterations: int


INSTANCE_RE = re.compile(r"instance=(?P<name>[^ ]+)")
BEST_RE = re.compile(r"best_cost=(?P<cost>[0-9]+(?:\.[0-9]+)?)\s+\|\s+Gap:\s+(?P<gap>[0-9.]+)%")
SOL_COST_RE = re.compile(r"^Cost:\s*(?P<cost>[0-9]+(?:\.[0-9]+)?)\s*$")
RUNTIME_HEADER_RE = re.compile(r"RUNTIME SUMMARY BY STAGE")
RUNTIME_LINE_RE = re.compile(
    r"^\s*(?P<stage>[a-z_]+):\s*(?P<seconds>[0-9.]+)s\s*\(\s*(?P<pct>[0-9.]+)% of total runtime\)\s*$"
)


def list_instance_names() -> List[str]:
    """Return base names (without extension) of all challenge instances."""
    return sorted(
        p.stem
        for p in INSTANCES_DIR.glob("XL-*.vrp")
        if p.is_file()
    )


def list_solution_instance_names() -> List[str]:
    """Return base names (without `_probabilistic.sol`) for all final solutions."""
    result = []
    for p in SOLUTIONS_DIR.glob("XL-*_*probabilistic.sol"):
        if not p.is_file():
            continue
        name = p.name
        if name.endswith("_probabilistic.sol"):
            result.append(name[: -len("_probabilistic.sol")])
    return sorted(result)


def parse_solution_cost(path: Path) -> Optional[float]:
    """Return the absolute solution cost from a `_probabilistic.sol` file."""

    cost: Optional[float] = None
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                m = SOL_COST_RE.match(line.strip())
                if m:
                    cost = float(m.group("cost"))
    except OSError:
        return None

    return cost


def compute_coverage() -> Tuple[List[str], List[str]]:
    """Return (instances_with_solution, instances_without_solution)."""
    all_instances = list_instance_names()
    solved = set(list_solution_instance_names())
    with_solution = sorted(name for name in all_instances if name in solved)
    without_solution = sorted(name for name in all_instances if name not in solved)
    return with_solution, without_solution


def parse_log(path: Path) -> InstanceLogSummary:
    """Extract simple KPIs from a single log file."""
    best_cost: Optional[float] = None
    best_gap: Optional[float] = None
    iterations = 0
    instance_name: Optional[str] = None

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if instance_name is None:
                    m_inst = INSTANCE_RE.search(line)
                    if m_inst:
                        # e.g. instance=XL-n6168-k1922.vrp -> XL-n6168-k1922
                        instance_name = m_inst.group("name").split(".")[0]

                if "ITERATION " in line:
                    iterations += 1

                if "best_cost=" in line:
                    m = BEST_RE.search(line)
                    if m:
                        best_cost = float(m.group("cost"))
                        best_gap = float(m.group("gap"))
    except OSError:
        pass

    if instance_name is None:
        # Fall back to log file stem, assuming XL-*.log
        instance_name = path.stem

    return InstanceLogSummary(
        instance=instance_name,
        best_cost=best_cost,
        best_gap=best_gap,
        iterations=iterations,
    )


def iter_log_summaries() -> Iterable[InstanceLogSummary]:
    for log_path in LOGS_DIR.glob("XL-*.log"):
        if log_path.is_file():
            yield parse_log(log_path)


def load_bks_tables() -> Tuple[Dict[str, float], Dict[str, float]]:
    """Load current and initial BKS tables as instance -> cost."""

    with CHALLENGE_BKS_PATH.open("r", encoding="utf-8") as f:
        challenge_bks = json.load(f)
    with INITIAL_BKS_PATH.open("r", encoding="utf-8") as f:
        initial_bks = json.load(f)
    return challenge_bks, initial_bks


def _parse_paper_cost_cell(raw: str) -> Optional[float]:
    s = raw.strip()
    if s in ("–", "-", "—", "", "N/A", "n/a"):
        return None
    return float(s.replace(",", ""))


_INSTANCE_CELL_RE = re.compile(r"^XL-n\d+-k\d+$")


def _row_to_paper_means(cells: List[str]) -> Optional[Tuple[str, Tuple[float, float, float]]]:
    """If cells are a Table 2 data row, return (instance, (filo_mean, filo2_mean, hgs_mean))."""
    if not cells:
        return None
    idx = 0
    if cells[0].isdigit():
        idx = 1
    if idx >= len(cells):
        return None
    inst = cells[idx]
    if not _INSTANCE_CELL_RE.match(inst):
        return None
    vals = cells[idx + 1 :]
    if len(vals) < 16:
        return None
    filo_m = _parse_paper_cost_cell(vals[3])
    filo2_m = _parse_paper_cost_cell(vals[5])
    hgs_m = _parse_paper_cost_cell(vals[9])
    if filo_m is None or filo2_m is None or hgs_m is None:
        return None
    return inst, (filo_m, filo2_m, hgs_m)


def _parse_paper_table2_xl_pipe_markdown(text: str) -> Dict[str, Tuple[float, float, float]]:
    """Parse pipe-style table rows (arXiv HTML experiment page as markdown, or saved .md)."""

    out: Dict[str, Tuple[float, float, float]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("| XL-n"):
            continue
        if "Avg. gap" in line:
            break
        parts = line.split("|")
        if len(parts) < 3:
            continue
        cells = [p.strip() for p in parts[1:-1]]
        parsed = _row_to_paper_means(cells)
        if parsed is not None:
            inst, triple = parsed
            out[inst] = triple
    return out


def _parse_paper_table2_xl_arxiv_raw_html(html: str) -> Dict[str, Tuple[float, float, float]]:
    """
    Parse Table 2 from raw arXiv HTML (what urllib returns for /html/... URLs).

    Rows are HTML <tr> with <td> cells, not markdown pipes.
    """

    out: Dict[str, Tuple[float, float, float]] = {}
    for m in re.finditer(r"<tr\b[^>]*>(.*?)</tr>", html, flags=re.DOTALL | re.I):
        tr = m.group(1)
        raw_cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.DOTALL | re.I)
        cells: List[str] = []
        for raw in raw_cells:
            text = re.sub(r"<[^>]+>", "", raw)
            text = re.sub(r"\s+", " ", text).strip()
            cells.append(text)
        parsed = _row_to_paper_means(cells)
        if parsed is not None:
            inst, triple = parsed
            out[inst] = triple
    return out


def parse_paper_table2_xl_method_means(content: str) -> Dict[str, Tuple[float, float, float]]:
    """
    Parse arXiv Table 2 (XL instances) from either:
    - raw HTML from https://arxiv.org/html/2601.11467v1 (urllib), or
    - pipe-markdown table lines (saved page / copy-paste from browser “view source” tools
      that emit markdown).
    """

    pipe = _parse_paper_table2_xl_pipe_markdown(content)
    if len(pipe) >= 90:
        return pipe
    tagged = _parse_paper_table2_xl_arxiv_raw_html(content)
    return tagged if len(tagged) > len(pipe) else pipe


BUNDLED_PAPER_TABLE2_MD = Path(__file__).resolve().parent / "data" / "paper_table2_xl.md"


def _fetch_arxiv_html(url: str = ARXIV_HTML_TABLE2_URL, timeout: int = 60) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; core-results-analysis/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def cmd_paper_mean_gaps(args: argparse.Namespace) -> None:
    """Append / refresh mean-baseline gap columns on `summary.csv`."""

    if args.html:
        html = Path(args.html).read_text(encoding="utf-8", errors="replace")
    else:
        try:
            html = _fetch_arxiv_html()
        except (urllib.error.URLError, OSError) as exc:
            raise SystemExit(
                f"Could not fetch {ARXIV_HTML_TABLE2_URL} ({exc}). "
                "Save the arXiv HTML page and pass --html PATH."
            ) from exc

    paper = parse_paper_table2_xl_method_means(html)
    if len(paper) < 90 and BUNDLED_PAPER_TABLE2_MD.is_file():
        paper = parse_paper_table2_xl_method_means(
            BUNDLED_PAPER_TABLE2_MD.read_text(encoding="utf-8", errors="replace")
        )
    if len(paper) < 90:
        raise SystemExit(
            f"Parsed only {len(paper)} Table 2 XL rows (need ~100). "
            "urllib returns raw HTML (not markdown pipes). If this persists, save "
            f"`{ARXIV_HTML_TABLE2_URL}` in a browser and pass `--html PATH`, or add "
            f"pipe-style table rows to `{BUNDLED_PAPER_TABLE2_MD}`."
        )

    rows: List[Dict[str, str]] = []
    with SUMMARY_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        base_fields = list(reader.fieldnames or [])
        for r in reader:
            rows.append(dict(r))

    new_fields = list(base_fields)
    for col in PAPER_MEAN_GAP_COLS:
        if col not in new_fields:
            new_fields.append(col)

    missing = 0
    for r in rows:
        inst = r["instance"]
        if inst not in paper:
            missing += 1
            for col in PAPER_MEAN_GAP_COLS:
                r[col] = ""
            continue
        best = float(r["best_cost"])
        fm, f2m, hm = paper[inst]
        r[PAPER_MEAN_GAP_COLS[0]] = str(100.0 * (best - fm) / fm)
        r[PAPER_MEAN_GAP_COLS[1]] = str(100.0 * (best - f2m) / f2m)
        r[PAPER_MEAN_GAP_COLS[2]] = str(100.0 * (best - hm) / hm)

    if missing:
        print(f"warning: no Table 2 mean row for {missing} instance(s)")

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {SUMMARY_CSV} with mean-baseline gap columns ({len(rows)} rows)")


@dataclass(frozen=True)
class TrajectoryStats:
    instance: str
    improvements_total: int
    improvements_routing: int
    improvements_scp: int
    max_iteration_with_improvement: int
    avg_iteration_routing: Optional[float]
    avg_iteration_scp: Optional[float]


def _trajectory_points_csv_path(instance: str) -> Optional[Path]:
    """
    Prefer the newer output naming:
      results/out/trajectories/<instance>_trajectory_points.csv
    Fall back to legacy:
      results/out/trajectories/<instance>/trajectory_points.csv
    """
    p_new = TRAJECTORIES_DIR / f"{instance}_trajectory_points.csv"
    if p_new.is_file():
        return p_new

    p_legacy = TRAJECTORIES_DIR / instance / "trajectory_points.csv"
    if p_legacy.is_file():
        return p_legacy

    return None


def _load_trajectory_stats(instance: str, path: Path) -> TrajectoryStats:
    improvements_total = 0
    improvements_routing = 0
    improvements_scp = 0

    max_iteration_with_improvement = 0
    routing_iters: List[int] = []
    scp_iters: List[int] = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            try:
                iteration = int(float(row["iteration"]))
            except (KeyError, ValueError, TypeError):
                continue

            source = (row.get("source") or "").strip().lower()

            improvements_total += 1
            max_iteration_with_improvement = max(max_iteration_with_improvement, iteration)

            if source == "routing":
                improvements_routing += 1
                routing_iters.append(iteration)
            elif source == "scp":
                improvements_scp += 1
                scp_iters.append(iteration)
            else:
                # Ignore unknown sources (shouldn't happen).
                continue

    avg_iteration_routing = statistics.mean(routing_iters) if routing_iters else None
    avg_iteration_scp = statistics.mean(scp_iters) if scp_iters else None

    return TrajectoryStats(
        instance=instance,
        improvements_total=improvements_total,
        improvements_routing=improvements_routing,
        improvements_scp=improvements_scp,
        max_iteration_with_improvement=max_iteration_with_improvement,
        avg_iteration_routing=avg_iteration_routing,
        avg_iteration_scp=avg_iteration_scp,
    )


def cmd_summary_solutions(_: argparse.Namespace) -> None:
    """
    Build `results/summary.csv` from final solution files only.

    Columns:
    - instance
    - best_cost
    - gap_to_bks_percent
    - gap_to_initial_percent
    """

    challenge_bks, initial_bks = load_bks_tables()

    rows: List[Dict[str, object]] = []
    for name in list_solution_instance_names():
        sol_path = SOLUTIONS_DIR / f"{name}_probabilistic.sol"
        cost = parse_solution_cost(sol_path)
        if cost is None:
            continue

        bks = float(challenge_bks[name])
        init_bks = float(initial_bks[name])

        gap_bks = 100.0 * (cost - bks) / bks
        gap_init = 100.0 * (cost - init_bks) / init_bks

        rows.append(
            {
                "instance": name,
                "best_cost": cost,
                "gap_to_bks_percent": gap_bks,
                "gap_to_initial_percent": gap_init,
            }
        )

    rows.sort(key=lambda r: r["instance"])

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance",
                "best_cost",
                "gap_to_bks_percent",
                "gap_to_initial_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote summary for {len(rows)} instances to {SUMMARY_CSV}")


def _bucket_counts(gaps: List[float]) -> Dict[str, int]:
    """Bucket gaps into the same ranges as the original overview."""

    buckets = {
        "≤0.10": 0,
        "(0.10,0.20]": 0,
        "(0.20,0.30]": 0,
        "(0.30,0.40]": 0,
        "(0.40,0.50]": 0,
        "(0.50,0.60]": 0,
        "(0.60,0.70]": 0,
        "(0.70,0.80]": 0,
        "(0.80,0.90]": 0,
        "(0.90,1.00]": 0,
        "(1.00,1.25]": 0,
        "(1.25,1.50]": 0,
        "(1.50,1.75]": 0,
        "(1.75,2.00]": 0,
        "(2.00,2.25]": 0,
        ">2.25": 0,
    }

    for g in gaps:
        if g <= 0.10:
            buckets["≤0.10"] += 1
        elif g <= 0.20:
            buckets["(0.10,0.20]"] += 1
        elif g <= 0.30:
            buckets["(0.20,0.30]"] += 1
        elif g <= 0.40:
            buckets["(0.30,0.40]"] += 1
        elif g <= 0.50:
            buckets["(0.40,0.50]"] += 1
        elif g <= 0.60:
            buckets["(0.50,0.60]"] += 1
        elif g <= 0.70:
            buckets["(0.60,0.70]"] += 1
        elif g <= 0.80:
            buckets["(0.70,0.80]"] += 1
        elif g <= 0.90:
            buckets["(0.80,0.90]"] += 1
        elif g <= 1.00:
            buckets["(0.90,1.00]"] += 1
        elif g <= 1.25:
            buckets["(1.00,1.25]"] += 1
        elif g <= 1.50:
            buckets["(1.25,1.50]"] += 1
        elif g <= 1.75:
            buckets["(1.50,1.75]"] += 1
        elif g <= 2.00:
            buckets["(1.75,2.00]"] += 1
        elif g <= 2.25:
            buckets["(2.00,2.25]"] += 1
        else:
            buckets[">2.25"] += 1

    return buckets


def cmd_overview_solutions(_: argparse.Namespace) -> None:
    """Rebuild `OVERVIEW.md` using solution-based gaps and runtime summaries."""

    rows: List[Dict[str, object]] = []
    with SUMMARY_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    gap_bks = [float(r["gap_to_bks_percent"]) for r in rows]
    gap_init = [float(r["gap_to_initial_percent"]) for r in rows]

    avg_bks = statistics.mean(gap_bks)
    med_bks = statistics.median(gap_bks)
    avg_init = statistics.mean(gap_init)
    med_init = statistics.median(gap_init)

    buckets_bks = _bucket_counts(gap_bks)
    buckets_init = _bucket_counts(gap_init)

    # Aggregate runtime summaries across all logs.
    stage_seconds: Dict[str, float] = {}
    stage_pct_sum: Dict[str, float] = {}
    stage_counts: Dict[str, int] = {}
    total_runtime_seconds = 0.0

    for log_path in LOGS_DIR.glob("XL-*.log"):
        try:
            with log_path.open("r", encoding="utf-8") as f:
                in_runtime = False
                for line in f:
                    if not in_runtime and RUNTIME_HEADER_RE.search(line):
                        in_runtime = True
                        continue
                    if in_runtime:
                        if "total_iter:" in line:
                            m_total = RUNTIME_LINE_RE.search(line.split("|", maxsplit=2)[-1].strip())
                            if m_total:
                                total_runtime_seconds += float(m_total.group("seconds"))
                            break
                        m = RUNTIME_LINE_RE.search(line.split("|", maxsplit=2)[-1].strip())
                        if not m:
                            continue
                        stage = m.group("stage")
                        secs = float(m.group("seconds"))
                        pct = float(m.group("pct"))
                        stage_seconds[stage] = stage_seconds.get(stage, 0.0) + secs
                        stage_pct_sum[stage] = stage_pct_sum.get(stage, 0.0) + pct
                        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        except OSError:
            continue

    # Derived relative distribution across all runtime.
    stage_share: Dict[str, float] = {}
    for stage, secs in stage_seconds.items():
        if total_runtime_seconds > 0:
            stage_share[stage] = 100.0 * secs / total_runtime_seconds
        else:
            stage_share[stage] = 0.0

    lines: List[str] = []
    lines.append("## Results summary")
    lines.append("")
    lines.append(
        "- **Instances solved**: 100 / 100 challenge instances have final `_probabilistic.sol` solutions in `results/out/solutions`."
    )
    lines.append(
        f"- **Average gap to BKS**: {avg_bks:.4f}% (mean over all 100 instances, using costs from solution files vs. `challenge-bks.json`)."
    )
    lines.append(f"- **Median gap to BKS**: {med_bks:.4f}%.")
    lines.append(
        f"- **Average gap to initial BKS**: {avg_init:.4f}% (mean over all 100 instances, using costs from solution files vs. `initial-bks.json`)."
    )
    lines.append(f"- **Median gap to initial BKS**: {med_init:.4f}%.")
    if rows and all(c in rows[0] for c in PAPER_GAP_COLS):
        lines.append("")
        lines.append(
            "### Gaps vs. paper Table 2 baselines (best FILO / FILO2 / HGS-CVRP per instance, 100 instances)"
        )
        lines.append("")
        labels = {
            "gap_to_paper_filo_best_percent": "FILO (paper best)",
            "gap_to_paper_filo2_best_percent": "FILO2 (paper best)",
            "gap_to_paper_hgs_cvrp_best_percent": "HGS-CVRP (paper best)",
        }
        for col in PAPER_GAP_COLS:
            xs = [float(r[col]) for r in rows]
            lines.append(
                f"- **Average gap vs. {labels[col]}**: {statistics.mean(xs):.4f}%."
            )
            lines.append(f"- **Median gap vs. {labels[col]}**: {statistics.median(xs):.4f}%.")
    if rows and all(c in rows[0] for c in PAPER_MEAN_GAP_COLS):
        lines.append("")
        lines.append(
            "### Gaps vs. paper Table 2 baselines (mean FILO / FILO2 / HGS-CVRP per instance, 100 instances)"
        )
        lines.append("")
        mean_labels = {
            "gap_to_paper_filo_mean_percent": "FILO (paper mean)",
            "gap_to_paper_filo2_mean_percent": "FILO2 (paper mean)",
            "gap_to_paper_hgs_cvrp_mean_percent": "HGS-CVRP (paper mean)",
        }
        for col in PAPER_MEAN_GAP_COLS:
            xs = [float(r[col]) for r in rows if (r.get(col) or "").strip() != ""]
            if not xs:
                continue
            lines.append(
                f"- **Average gap vs. {mean_labels[col]}**: {statistics.mean(xs):.4f}%."
            )
            lines.append(f"- **Median gap vs. {mean_labels[col]}**: {statistics.median(xs):.4f}%.")
    lines.append("")
    lines.append("### Gap distribution (vs. BKS, 100 instances)")
    lines.append("")
    lines.append(f"- **gap ≤ 0.10%**:         {buckets_bks['≤0.10']}")
    lines.append(f"- **0.10% < gap ≤ 0.20%**: {buckets_bks['(0.10,0.20]']}")
    lines.append(f"- **0.20% < gap ≤ 0.30%**: {buckets_bks['(0.20,0.30]']}")
    lines.append(f"- **0.30% < gap ≤ 0.40%**: {buckets_bks['(0.30,0.40]']}")
    lines.append(f"- **0.40% < gap ≤ 0.50%**: {buckets_bks['(0.40,0.50]']}")
    lines.append(f"- **0.50% < gap ≤ 0.60%**: {buckets_bks['(0.50,0.60]']}")
    lines.append(f"- **0.60% < gap ≤ 0.70%**: {buckets_bks['(0.60,0.70]']}")
    lines.append(f"- **0.70% < gap ≤ 0.80%**: {buckets_bks['(0.70,0.80]']}")
    lines.append(f"- **0.80% < gap ≤ 0.90%**: {buckets_bks['(0.80,0.90]']}")
    lines.append(f"- **0.90% < gap ≤ 1.00%**: {buckets_bks['(0.90,1.00]']}")
    lines.append(f"- **1.00% < gap ≤ 1.25%**: {buckets_bks['(1.00,1.25]']}")
    lines.append(f"- **1.25% < gap ≤ 1.50%**: {buckets_bks['(1.25,1.50]']}")
    lines.append(f"- **1.50% < gap ≤ 1.75%**: {buckets_bks['(1.50,1.75]']}")
    lines.append(f"- **1.75% < gap ≤ 2.00%**: {buckets_bks['(1.75,2.00]']}")
    lines.append(f"- **2.00% < gap ≤ 2.25%**: {buckets_bks['(2.00,2.25]']}")
    lines.append(f"- **gap > 2.25%**:         {buckets_bks['>2.25']}")
    lines.append("")
    lines.append("### Gap distribution (vs. initial BKS, 100 instances)")
    lines.append("")
    lines.append(f"- **gap ≤ 0.10%**:         {buckets_init['≤0.10']}")
    lines.append(f"- **0.10% < gap ≤ 0.20%**: {buckets_init['(0.10,0.20]']}")
    lines.append(f"- **0.20% < gap ≤ 0.30%**: {buckets_init['(0.20,0.30]']}")
    lines.append(f"- **0.30% < gap ≤ 0.40%**: {buckets_init['(0.30,0.40]']}")
    lines.append(f"- **0.40% < gap ≤ 0.50%**: {buckets_init['(0.40,0.50]']}")
    lines.append(f"- **0.50% < gap ≤ 0.60%**: {buckets_init['(0.50,0.60]']}")
    lines.append(f"- **0.60% < gap ≤ 0.70%**: {buckets_init['(0.60,0.70]']}")
    lines.append(f"- **0.70% < gap ≤ 0.80%**: {buckets_init['(0.70,0.80]']}")
    lines.append(f"- **0.80% < gap ≤ 0.90%**: {buckets_init['(0.80,0.90]']}")
    lines.append(f"- **0.90% < gap ≤ 1.00%**: {buckets_init['(0.90,1.00]']}")
    lines.append(f"- **1.00% < gap ≤ 1.25%**: {buckets_init['(1.00,1.25]']}")
    lines.append(f"- **1.25% < gap ≤ 1.50%**: {buckets_init['(1.25,1.50]']}")
    lines.append(f"- **1.50% < gap ≤ 1.75%**: {buckets_init['(1.50,1.75]']}")
    lines.append(f"- **1.75% < gap ≤ 2.00%**: {buckets_init['(1.75,2.00]']}")
    lines.append(f"- **2.00% < gap ≤ 2.25%**: {buckets_init['(2.00,2.25]']}")
    lines.append(f"- **gap > 2.25%**:         {buckets_init['>2.25']}")
    lines.append("")
    lines.append("### Runtime summary by stage (all 100 instances)")
    lines.append("")
    lines.append(
        "| Stage | Total runtime [s] | Share of total runtime [%] | Average share per instance with stage [%] |"
    )
    lines.append("|-------|-------------------:|---------------------------:|-------------------------------------------:|")
    for stage in sorted(stage_seconds.keys()):
        secs = stage_seconds[stage]
        share = stage_share.get(stage, 0.0)
        avg_pct = 0.0
        if stage_counts.get(stage, 0) > 0:
            avg_pct = stage_pct_sum[stage] / stage_counts[stage]
        lines.append(
            f"| {stage} | {secs:.3f} | {share:.2f} | {avg_pct:.2f} |"
        )
    lines.append("")

    # Trajectory-based improvements (new-best events).
    challenge_bks, _ = load_bks_tables()
    instance_names = sorted(challenge_bks.keys())

    traj_missing: List[str] = []
    traj_stats: List[TrajectoryStats] = []
    for inst in instance_names:
        path = _trajectory_points_csv_path(inst)
        if path is None:
            traj_missing.append(inst)
            continue
        traj_stats.append(_load_trajectory_stats(inst, path))

    total_improvements = sum(ts.improvements_total for ts in traj_stats)
    total_routing = sum(ts.improvements_routing for ts in traj_stats)
    total_scp = sum(ts.improvements_scp for ts in traj_stats)

    avg_iterations_with_improvement = (
        statistics.mean([ts.max_iteration_with_improvement for ts in traj_stats]) if traj_stats else 0.0
    )
    avg_improvements = statistics.mean([ts.improvements_total for ts in traj_stats]) if traj_stats else 0.0
    avg_improvements_routing = (
        statistics.mean([ts.improvements_routing for ts in traj_stats]) if traj_stats else 0.0
    )
    avg_improvements_scp = statistics.mean([ts.improvements_scp for ts in traj_stats]) if traj_stats else 0.0

    share_routing = (100.0 * total_routing / total_improvements) if total_improvements > 0 else 0.0
    share_scp = (100.0 * total_scp / total_improvements) if total_improvements > 0 else 0.0

    avg_iter_routing_weighted = (
        sum(
            ts.avg_iteration_routing * ts.improvements_routing
            for ts in traj_stats
            if ts.avg_iteration_routing is not None
        )
        / total_routing
        if total_routing > 0
        else None
    )
    avg_iter_scp_weighted = (
        sum(
            ts.avg_iteration_scp * ts.improvements_scp
            for ts in traj_stats
            if ts.avg_iteration_scp is not None
        )
        / total_scp
        if total_scp > 0
        else None
    )

    lines.append("### Trajectory improvements (new best events)")
    lines.append("")
    lines.append(
        f"- **Average iterations (max iter with a new best)**: {avg_iterations_with_improvement:.2f}"
    )
    lines.append(f"- **Average #improvements**: {avg_improvements:.2f}")
    lines.append(f"- **Average #improvements via routing**: {avg_improvements_routing:.2f}")
    lines.append(f"- **Average #improvements via scp**: {avg_improvements_scp:.2f}")
    lines.append("")
    lines.append(
        f"- **Share of all improvements from routing/scp**: {share_routing:.2f}% routing, {share_scp:.2f}% scp"
    )
    if avg_iter_routing_weighted is not None and avg_iter_scp_weighted is not None:
        lines.append(
            f"- **Average iteration index of improvements**: routing {avg_iter_routing_weighted:.2f}, scp {avg_iter_scp_weighted:.2f}"
        )
    if traj_missing:
        lines.append("")
        lines.append(
            f"- **Warning**: missing trajectory CSVs for {len(traj_missing)} instance(s); averages computed over {len(traj_stats)} instance(s)."
        )

    OVERVIEW_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote overview to {OVERVIEW_MD}")


def cmd_coverage(_: argparse.Namespace) -> None:
    """Print which instances have / lack final `_probabilistic.sol` solutions."""
    with_solution, without_solution = compute_coverage()

    print("=== Coverage summary ===")
    print(f"Total instances: {len(with_solution) + len(without_solution)}")
    print(f"With solutions:  {len(with_solution)}")
    print(f"Without solutions: {len(without_solution)}")
    print()

    if without_solution:
        print("Instances without `_probabilistic.sol` solution:")
        for name in without_solution:
            print(f"  - {name}.vrp")


def cmd_summary(args: argparse.Namespace) -> None:
    """Summarize logs into a simple table (stdout or CSV)."""
    rows: List[Dict[str, object]] = []
    for s in iter_log_summaries():
        rows.append(
            {
                "instance": s.instance,
                "best_cost": s.best_cost if s.best_cost is not None else "",
                "best_gap_percent": s.best_gap if s.best_gap is not None else "",
                "iterations": s.iterations,
            }
        )

    rows.sort(key=lambda r: r["instance"])

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["instance", "best_cost", "best_gap_percent", "iterations"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote summary for {len(rows)} instances to {out_path}")
    else:
        # simple text table
        print(f"{'instance':30} {'best_cost':>12} {'gap[%]':>8} {'iters':>6}")
        print("-" * 60)
        for r in rows:
            inst = str(r["instance"])
            cost = "" if r["best_cost"] == "" else f"{r['best_cost']:.0f}"
            gap = "" if r["best_gap_percent"] == "" else f"{r['best_gap_percent']:.4f}"
            iters = str(r["iterations"])
            print(f"{inst:30} {cost:>12} {gap:>8} {iters:>6}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse challenge logs and solution coverage.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cov = sub.add_parser("coverage", help="Show which instances have / lack final solutions.")
    p_cov.set_defaults(func=cmd_coverage)

    p_sum = sub.add_parser("summary", help="Summarize log KPIs (best cost, gap, iterations).")
    p_sum.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Optional CSV output path. If omitted, prints a text table.",
    )
    p_sum.set_defaults(func=cmd_summary)

    p_sol = sub.add_parser(
        "solutions",
        help="Rebuild summary.csv and OVERVIEW.md from final solution files and BKS tables.",
    )
    p_sol.set_defaults(func=lambda args: (cmd_summary_solutions(args), cmd_overview_solutions(args)))

    p_over = sub.add_parser(
        "overview",
        help="Rebuild OVERVIEW.md from summary.csv and logs (does not rewrite summary.csv).",
    )
    p_over.set_defaults(func=cmd_overview_solutions)

    p_pmean = sub.add_parser(
        "paper_mean_gaps",
        help=(
            "Add gap columns vs. paper Table 2 FILO/FILO2/HGS-CVRP *mean* costs to summary.csv "
            "(fetches arXiv HTML unless --html)."
        ),
    )
    p_pmean.add_argument(
        "--html",
        metavar="PATH",
        help="Saved arXiv HTML for 2601.11467v1 if network fetch fails or you work offline.",
    )
    p_pmean.set_defaults(func=cmd_paper_mean_gaps)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

