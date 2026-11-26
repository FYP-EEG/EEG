"""Compare public MI preprocessing/model hypotheses without replacing models.

Typical command after the normal preparation workflow:

    python tools/compare_public_mi.py

The existing 8--30 Hz archives can evaluate sensorimotor recipes.  To evaluate
Ofner low-frequency recipes, first prepare a separate 0.3--40 Hz archive root as
documented in ``docs/PUBLIC_MI_DATASETS.md``; never overwrite the normal root.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ML.motor_intent import MI_CATEGORIES, normalize_category
from ML.public_mi_experiments import run_public_mi_experiments

DEFAULT_PROCESSED_ROOT = ROOT / "dataset" / "public" / "processed"
DEFAULT_REPORT_ROOT = ROOT / "reports" / "public_mi" / "experiments"


def resolve_categories(value):
    requested = tuple(
        part.strip().lower() for part in str(value).split(",") if part.strip())
    if not requested or requested == ("all",):
        return MI_CATEGORIES
    categories = tuple(normalize_category(value) for value in requested)
    return tuple(value for value in MI_CATEGORIES if value in categories)


def make_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Report-only public MI comparison: category bands, harmonized "
            "labels, source-complete feet, FBCSP, Riemannian alignment, and "
            "subject/dataset-held-out evaluation"))
    parser.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument(
        "--categories", default="all",
        help="all or comma-separated finger,hand,upper_limb,feet")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    categories = resolve_categories(args.categories)
    report, json_path, csv_path = run_public_mi_experiments(
        args.processed_root, args.report_root, categories=categories,
        n_splits=args.splits, random_state=args.seed, verbose=True)
    successful = sum(
        value.get("status") == "success"
        for value in report["categories"].values())
    print("\nEXPERIMENT COMPLETE")
    print(f"  categories: {successful}/{len(categories)}")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print("  models/public: unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
