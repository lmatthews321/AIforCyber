import argparse
import csv
import itertools
import re
import sys
from pathlib import Path


WINDOW_SIZES = [4, 6, 8, 10, 12]
WINDOW_STEPS = [1, 2, 4, 6, 8]
HIDDEN_TENSORS = [
    [64, 64, 16],
    [64, 32, 16],
    [32, 32, 16],
    [32, 32, 8],
    [32, 16, 8],
    [16, 16, 8],
]
KERNEL_SIZES = [
    [(2, 2), (2, 2)],
    [(3, 3), (3, 3)],
    [(4, 4), (4, 4)],
]
DROPOUT_RATES = [0.05, 0.1, 0.15, 0.2, 0.25]

PARAMETER_SETS = list(
    itertools.product(
        WINDOW_SIZES,
        WINDOW_STEPS,
        HIDDEN_TENSORS,
        KERNEL_SIZES,
        DROPOUT_RATES,
    )
)
RESULT_FILE_RE = re.compile(r"result_(\d+)_(\d+)\.txt$")
METRIC_RE = re.compile(
    r"Model (accuracy|f1 score|detection rate|FPR):\s*([0-9.]+)\s*(%)?"
)
CONFUSION_MATRIX_RE = re.compile(
    r"\(TP, FP, TN, FN\):\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)"
)

CSV_FIELDS = [
    "n",
    "k",
    "split",
    "window_size",
    "window_step",
    "hidden_tensor",
    "kernel_size_0",
    "kernel_size_1",
    "dropout_rate",
    "accuracy_percent",
    "f1_score",
    "detection_rate_percent",
    "false_positive_rate_percent",
    "tp",
    "fp",
    "tn",
    "fn",
]


def parse_result_file(path):
    match = RESULT_FILE_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unexpected result filename: {path.name}")

    n, k = map(int, match.groups())
    if k >= len(PARAMETER_SETS):
        raise ValueError(f"{path.name}: k must be between 0 and {len(PARAMETER_SETS) - 1}")

    window_size, window_step, hidden_tensor, kernel_sizes, dropout_rate = (
        PARAMETER_SETS[k]
    )
    parameters = {
        "n": n,
        "k": k,
        "window_size": window_size,
        "window_step": window_step,
        "hidden_tensor": "[" + ",".join(map(str, hidden_tensor)) + "]",
        "kernel_size_0": f"{kernel_sizes[0][0]}x{kernel_sizes[0][1]}",
        "kernel_size_1": f"{kernel_sizes[1][0]}x{kernel_sizes[1][1]}",
        "dropout_rate": dropout_rate,
    }

    rows = []
    current_split = None
    metrics = {}
    confusion_matrix = None

    def finish_split():
        if current_split is None:
            return
        required_metrics = {"accuracy", "f1 score", "detection rate"}
        missing = required_metrics - metrics.keys()
        if missing or confusion_matrix is None:
            missing_items = sorted(missing)
            if confusion_matrix is None:
                missing_items.append("confusion matrix")
            raise ValueError(
                f"{path.name}: incomplete {current_split} set results; missing "
                + ", ".join(missing_items)
            )

        rows.append(
            {
                **parameters,
                "split": current_split,
                "accuracy_percent": metrics["accuracy"],
                "f1_score": metrics["f1 score"],
                "detection_rate_percent": metrics["detection rate"],
                "false_positive_rate_percent": metrics.get("FPR", ""),
                "tp": confusion_matrix[0],
                "fp": confusion_matrix[1],
                "tn": confusion_matrix[2],
                "fn": confusion_matrix[3],
            }
        )

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped in ("training set:", "test set:"):
            finish_split()
            current_split = stripped[:-1].split()[0]
            metrics = {}
            confusion_matrix = None
            continue

        if current_split is None:
            continue

        metric_match = METRIC_RE.search(stripped)
        if metric_match:
            name, value, _ = metric_match.groups()
            metrics[name] = float(value)
            continue
        if stripped == "FPR NaN!":
            metrics["FPR"] = ""
            continue

        matrix_match = CONFUSION_MATRIX_RE.search(stripped)
        if matrix_match:
            confusion_matrix = tuple(map(int, matrix_match.groups()))

    finish_split()
    if len(rows) != 2 or {row["split"] for row in rows} != {"training", "test"}:
        raise ValueError(
            f"{path.name}: expected complete training and test set results"
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Combine result_n_k.txt evaluation logs into one CSV file."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing result_n_k.txt files (default: results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/results.csv"),
        help="Output CSV file (default: results/results.csv)",
    )
    args = parser.parse_args()

    files = sorted(
        path
        for path in args.input_dir.glob("result_*.txt")
        if RESULT_FILE_RE.fullmatch(path.name)
    )
    if not files:
        parser.error(f"no result_n_k.txt files found in {args.input_dir}")

    try:
        rows = []
        for path in files:
            rows.extend(parse_result_file(path))

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.output} ({len(rows)} rows)")
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
