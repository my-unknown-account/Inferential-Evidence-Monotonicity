"""Export paper-ready tables from generated result files."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


RETRIEVERS = ["bm25", "splade", "dpr", "colbert", "bge", "reasonir"]
RETRIEVER_LABELS = {
    "bm25": "BM25",
    "splade": "SPLADE",
    "dpr": "DPR",
    "colbert": "ColBERT",
    "bge": "BGE",
    "reasonir": "ReasonIR",
}

RQ1_METRIC = "rq1_emvr_overall"
RQ3_SCORE_UP_METRIC = "rq3_p_reader_up_given_score_up"
RQ3_SCORE_DOWN_METRIC = "rq3_p_reader_up_given_score_down"
RQ3_DIFFERENCE_METRIC = "rq3_alignment_difference"
TRANSITION_ORDER = ["1->2", "2->3", "3->4", "4->5"]
TRANSITION_LABELS = [r"$1\to2$", r"$2\to3$", r"$3\to4$", r"$4\to5$"]
CONVERGENCE_ORDER = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]


def read_bootstrap_metrics(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing bootstrap results: {path}")

    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    if not rows:
        raise ValueError(f"Bootstrap results are empty: {path}")

    metrics = {}
    for row in rows:
        metric = row["metric"]
        metrics[metric] = {
            "estimate": float(row["estimate"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
        }
    return metrics


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")

    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    if not rows:
        raise ValueError(f"Result file is empty: {path}")
    return rows


def require_metric(metrics: dict[str, dict[str, float]], metric: str, path: Path) -> dict[str, float]:
    if metric not in metrics:
        raise KeyError(f"Missing metric {metric!r} in {path}")
    return metrics[metric]


def percent(value: float) -> str:
    return f"{100 * value:.2f}"


def signed_percent(value: float) -> str:
    return f"{100 * value:+.2f}"


def ci(metric: dict[str, float]) -> str:
    return f"[{percent(metric['ci_low'])}, {percent(metric['ci_high'])}]"


def markdown_table(headers: list[str], rows: list[tuple[str, ...]], aligns: list[str]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(aligns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def require_rq2_files(results_dir: Path, retrievers: list[str]) -> None:
    missing = []
    for retriever in retrievers:
        rq2_dir = results_dir / "rq2" / retriever
        for filename in [
            "p_delta_lt_0_by_transition.csv",
            "p_delta_lt_0_by_added_hint_convergence.csv",
        ]:
            path = rq2_dir / filename
            if not path.exists():
                missing.append(path)

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Cannot export Figure 2 until all RQ2 outputs exist:\n"
            f"{formatted}"
        )


def load_retriever_bootstrap(results_dir: Path, retrievers: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    output = {}
    missing = []
    for retriever in retrievers:
        path = results_dir / "bootstrap" / retriever / "bootstrap_cis.csv"
        try:
            output[retriever] = read_bootstrap_metrics(path)
        except FileNotFoundError:
            missing.append(path)

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Cannot export paper tables until all bootstrap outputs exist:\n"
            f"{formatted}"
        )
    return output


def read_rate_series(path: Path, label_column: str, labels: list[str]) -> list[float]:
    by_label = {row[label_column]: float(row["p_delta_lt_0"]) * 100 for row in read_csv_rows(path)}
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise KeyError(f"Missing label(s) in {path}: {', '.join(missing)}")
    return [by_label[label] for label in labels]


def figure_2_series(
    results_dir: Path,
    retrievers: list[str],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    require_rq2_files(results_dir, retrievers)
    by_transition = {}
    by_convergence = {}
    for retriever in retrievers:
        rq2_dir = results_dir / "rq2" / retriever
        by_transition[retriever] = read_rate_series(
            rq2_dir / "p_delta_lt_0_by_transition.csv",
            "transition",
            TRANSITION_ORDER,
        )
        by_convergence[retriever] = read_rate_series(
            rq2_dir / "p_delta_lt_0_by_added_hint_convergence.csv",
            "added_hint_convergence_bin",
            CONVERGENCE_ORDER,
        )
    return by_transition, by_convergence


def table_1_rows(bootstrap_metrics: dict[str, dict[str, dict[str, float]]]) -> list[tuple[str, ...]]:
    rows = []
    for retriever in bootstrap_metrics:
        metric = require_metric(
            bootstrap_metrics[retriever],
            RQ1_METRIC,
            Path("results") / "bootstrap" / retriever / "bootstrap_cis.csv",
        )
        rows.append((RETRIEVER_LABELS[retriever], percent(metric["estimate"]), ci(metric)))
    return rows


def table_2_rows(bootstrap_metrics: dict[str, dict[str, dict[str, float]]]) -> list[tuple[str, ...]]:
    rows = []
    for retriever in bootstrap_metrics:
        metrics = bootstrap_metrics[retriever]
        score_down = require_metric(
            metrics,
            RQ3_SCORE_DOWN_METRIC,
            Path("results") / "bootstrap" / retriever / "bootstrap_cis.csv",
        )
        score_up = require_metric(
            metrics,
            RQ3_SCORE_UP_METRIC,
            Path("results") / "bootstrap" / retriever / "bootstrap_cis.csv",
        )
        difference = require_metric(
            metrics,
            RQ3_DIFFERENCE_METRIC,
            Path("results") / "bootstrap" / retriever / "bootstrap_cis.csv",
        )
        rows.append(
            (
                RETRIEVER_LABELS[retriever],
                percent(score_down["estimate"]),
                percent(score_up["estimate"]),
                f"{signed_percent(difference['estimate'])} {ci(difference)}",
            )
        )
    return rows


def table_1_markdown(rows: list[tuple[str, ...]]) -> str:
    table = markdown_table(
        ["Retriever", "EMVR (%)", "95% CI"],
        rows,
        ["---", "---:", "---"],
    )
    return (
        "Table 1: Evidence monotonicity violation rate (EMVR) across retrievers. "
        "Confidence intervals are obtained by bootstrap resampling at the question level.\n\n"
        f"{table}\n\n"
        "Note: Estimates and confidence intervals are read from "
        "`results/bootstrap/<retriever>/bootstrap_cis.csv`.\n"
    )


def table_2_markdown(rows: list[tuple[str, ...]]) -> str:
    table = markdown_table(
        [
            "Retriever",
            r"\(P(R\uparrow\mid S\downarrow)\)",
            r"\(P(R\uparrow\mid S\uparrow)\)",
            "Difference (95% CI)",
        ],
        rows,
        ["---", "---:", "---:", "---"],
    )
    return (
        "Table 2: Alignment between retrieval-score changes and downstream reader improvements. "
        r"The last column reports \(P(R\uparrow\mid S\uparrow)-P(R\uparrow\mid S\downarrow)\), "
        "with question-level bootstrap 95% confidence intervals.\n\n"
        f"{table}\n\n"
        r"Note: \(S\) means retrieval score and \(R\) means reader utility. "
        "Estimates and confidence intervals are read from "
        "`results/bootstrap/<retriever>/bootstrap_cis.csv`.\n"
    )


def save_figure_2(results_dir: Path, figures_dir: Path, retrievers: list[str]) -> list[Path]:
    matplotlib_cache = Path(os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"))
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    try:
        import matplotlib
    except ImportError as error:
        raise SystemExit("Figure export requires matplotlib. Install it with: pip install matplotlib") from error

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_transition, by_convergence = figure_2_series(results_dir, retrievers)
    all_values = [
        value
        for series_group in [by_transition, by_convergence]
        for series in series_group.values()
        for value in series
    ]
    y_max = min(100, max(all_values) + 5) if all_values else 100

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
        }
    )
    colors = plt.get_cmap("tab10").colors
    markers = ["o", "s", "^", "D", "P", "X"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4), sharey=True)

    for index, retriever in enumerate(retrievers):
        label = RETRIEVER_LABELS[retriever]
        style = {
            "marker": markers[index % len(markers)],
            "linewidth": 1.8,
            "markersize": 4.5,
            "color": colors[index % len(colors)],
            "label": label,
        }
        axes[0].plot(TRANSITION_LABELS, by_transition[retriever], **style)
        axes[1].plot(CONVERGENCE_ORDER, by_convergence[retriever], **style)

    axes[0].set_title("(a) EMVR by evidence accumulation")
    axes[0].set_xlabel("Evidence transition")
    axes[0].set_ylabel("EMVR (%)")
    axes[1].set_title("(b) EMVR by added-hint convergence")
    axes[1].set_xlabel("Added-hint convergence")

    for axis in axes:
        axis.set_ylim(0, y_max)
        axis.tick_params(axis="x", labelrotation=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(retrievers),
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / "figure_2_emvr.png"
    pdf_path = figures_dir / "figure_2_emvr.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def export_outputs(results_dir: Path, output_dir: Path, retrievers: list[str]) -> list[Path]:
    unknown = sorted(set(retrievers) - set(RETRIEVERS))
    if unknown:
        raise ValueError(f"Unknown retriever(s): {', '.join(unknown)}")

    bootstrap_metrics = load_retriever_bootstrap(results_dir, retrievers)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    written = [
        tables_dir / "table_1_rq1_emvr.md",
        tables_dir / "table_2_rq3_alignment.md",
    ]
    write_text(written[0], table_1_markdown(table_1_rows(bootstrap_metrics)))
    write_text(written[1], table_2_markdown(table_2_rows(bootstrap_metrics)))
    written.extend(save_figure_2(results_dir, figures_dir, retrievers))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create outputs/tables and outputs/figures from generated result files."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--retrievers", nargs="+", default=RETRIEVERS)
    args = parser.parse_args()

    written = export_outputs(args.results_dir, args.output_dir, args.retrievers)
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
