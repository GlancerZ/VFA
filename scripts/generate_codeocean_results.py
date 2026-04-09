import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from predictor import (
    DEFAULT_RESULTS_DIR,
    predict_task1_from_form,
    predict_task2_from_form,
)


def _load_form_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_markdown_report(
    results_dir: Path,
    task1: Dict[str, Any],
    task2: Dict[str, Any],
) -> None:
    task1_lines = []
    for diet_name, value in task1["all_results"].items():
        task1_lines.append("| {0} | {1:.2f} |".format(diet_name, value))

    report = """# Code Ocean Run Summary

This report was generated automatically by `run`.

## Task 1

- Mode: {task1_mode}
- Note: {task1_note}
- Recommended regimen: {task1_recommendation}
- Predicted VFA reduction: {task1_reduction:.2f} cm^2

| Dietary regimen | Predicted VFA reduction (cm^2) |
| --- | ---: |
{task1_rows}

## Task 2

- Mode: {task2_mode}
- Note: {task2_note}
- Predicted four-week VFA change: {task2_change:.2f} cm^2
- Interpretation: {task2_interpretation}

## What To Replace Before Publication

- Put the trained Task 1 model at `task1/best_model.pkl`.
- Put the trained Task 2 model at `task2/best_model_task2.pkl`.
- Add any extra package dependencies your serialized models require.
""".format(
        task1_mode=task1["prediction_mode"],
        task1_note=task1["prediction_message"],
        task1_recommendation=task1["recommended_diet_name"],
        task1_reduction=task1["vfa_reduction"],
        task1_rows="\n".join(task1_lines),
        task2_mode=task2["prediction_mode"],
        task2_note=task2["prediction_message"],
        task2_change=task2["results"]["current_diet_continuation"],
        task2_interpretation=task2["results"]["interpretation"],
    )

    (results_dir / "codeocean_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate reproducible Code Ocean outputs.")
    parser.add_argument(
        "--task1-form",
        default=str(ROOT_DIR / "sample_inputs" / "task1_form.json"),
        help="Path to the Task 1 form payload JSON.",
    )
    parser.add_argument(
        "--task2-form",
        default=str(ROOT_DIR / "sample_inputs" / "task2_form.json"),
        help="Path to the Task 2 form payload JSON.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Where to write reproducible outputs.",
    )
    args = parser.parse_args()

    task1_form = _load_form_payload(Path(args.task1_form).resolve())
    task2_form = _load_form_payload(Path(args.task2_form).resolve())
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    task1 = predict_task1_from_form(task1_form)
    task2 = predict_task2_from_form(task2_form)

    task1_frame = pd.DataFrame(
        [
            {"dietary_regimen": diet_name, "predicted_vfa_reduction_cm2": value}
            for diet_name, value in task1["all_results"].items()
        ]
    )
    task2_frame = pd.DataFrame(
        [
            {
                "metric": "predicted_vfa_change_cm2",
                "value": task2["results"]["current_diet_continuation"],
            }
        ]
    )

    task1_frame.to_csv(results_dir / "task1_predictions.csv", index=False)
    task2_frame.to_csv(results_dir / "task2_prediction.csv", index=False)
    _write_json(results_dir / "task1_summary.json", task1)
    _write_json(results_dir / "task2_summary.json", task2)
    _write_markdown_report(results_dir, task1, task2)

    summary = {
        "task1_mode": task1["prediction_mode"],
        "task2_mode": task2["prediction_mode"],
        "results_dir": str(results_dir),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
