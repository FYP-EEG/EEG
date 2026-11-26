"""Generate the full public-MI source/performance/separation evidence report.

With processed archives restored:

    python tools/generate_public_mi_evidence.py

On a lightweight source-only checkout, the same command still writes the
source/literature/historical-audit table and explicitly records missing arrays;
it never fabricates CSP, tangent-space, covariance or confusion figures.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ML.motor_intent import MI_CATEGORIES, normalize_category
from ML.public_mi_evidence import DEFAULT_PIPELINES, generate_public_mi_evidence


def _categories(value):
    values = [part.strip() for part in str(value).split(",") if part.strip()]
    if not values or values == ["all"]:
        return MI_CATEGORIES
    wanted = tuple(normalize_category(value) for value in values)
    return tuple(category for category in MI_CATEGORIES if category in wanted)


def _pipelines(value):
    result = {}
    if not value:
        return result
    for item in str(value).split(","):
        if not item.strip():
            continue
        try:
            category, pipeline = item.split("=", 1)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--pipelines needs category=pipeline pairs") from exc
        category = normalize_category(category.strip())
        pipeline = pipeline.strip()
        if pipeline not in {"fbcsp_lda", "riemann_lr", "riemann_svm"}:
            raise argparse.ArgumentTypeError(
                f"unsupported evidence pipeline {pipeline!r}")
        result[category] = pipeline
    return result


def make_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Authoritative public-MI source/literature table plus grouped "
            "accuracy, CSP class colors, tangent PCA, confusion matrices and "
            "covariance summaries from actual standardized archives"))
    parser.add_argument(
        "--processed-root",
        default=str(ROOT / "dataset" / "public" / "processed"))
    parser.add_argument(
        "--model-root", default=str(ROOT / "models" / "public"))
    parser.add_argument(
        "--report-root", default=str(ROOT / "reports" / "public_mi" / "evidence"))
    parser.add_argument("--categories", default="all")
    parser.add_argument(
        "--pipelines", default=None,
        help=("comma-separated category=pipeline overrides; defaults: " +
              ", ".join(f"{key}={value}" for key, value in DEFAULT_PIPELINES.items())))
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hash-inputs", action="store_true",
                        help="compute full archive SHA-256 hashes (can be slow)")
    parser.add_argument("--inventory-only", action="store_true",
                        help="write metadata/source report but skip evaluation/plots")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        categories = _categories(args.categories)
        pipelines = _pipelines(args.pipelines)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    command_args = list(argv) if argv is not None else sys.argv[1:]
    command = "python tools/generate_public_mi_evidence.py"
    if command_args:
        command += " " + " ".join(shlex.quote(value) for value in command_args)
    report, markdown_path, json_path, csv_path = generate_public_mi_evidence(
        args.processed_root, args.report_root, model_root=args.model_root,
        categories=categories, n_splits=args.splits, seed=args.seed,
        pipelines=pipelines, content_hash=args.hash_inputs,
        inventory_only=args.inventory_only, command=command,
        project_root=ROOT)
    statuses = {category: row.get("status")
                for category, row in report["categories"].items()}
    print("PUBLIC MI EVIDENCE REPORT")
    print("  category status: " + ", ".join(
        f"{key}={value}" for key, value in statuses.items()))
    print(f"  Markdown -> {markdown_path}")
    print(f"  JSON     -> {json_path}")
    print(f"  CSV      -> {csv_path}")
    print(f"  Inventory-> {Path(args.report_root) / 'dataset_archive_inventory.csv'}")
    missing = [key for key, value in statuses.items()
               if value in {"missing", "failed"}]
    if missing:
        print("  No plot was synthesized for missing/failed categories: " +
              ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
