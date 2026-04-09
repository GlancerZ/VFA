import argparse
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = Path("/results") if Path("/results").exists() else ROOT_DIR / "results"

DIET_NAME_MAP = {
    "Group_1": "Balanced Diet (100% Energy)",
    "Group_2": "TRF 16:8 (100% Energy)",
    "Group_3": "TRF 16:8 (75% Energy)",
    "Group_4": "alternate day fasting (75% Energy)",
    "Group_5": "5+2 (75% Energy)",
    "Group_6": "CR Only (75% Energy)",
    "Group_7": "CR Only (45% Energy)",
}

TASK_CONFIG = {
    "task1": {
        "model_path": ROOT_DIR / "task1" / "best_model.pkl",
    },
    "task2": {
        "model_path": ROOT_DIR / "task2" / "best_model_task2.pkl",
    },
}


@dataclass
class PredictionRun:
    predictions: List[float]
    mode: str
    message: str
    model_path: Path


def _required_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if value in (None, ""):
        raise ValueError("Missing required field: {0}".format(key))
    return float(value)


def _optional_float(payload: Mapping[str, Any], key: str, default: float) -> float:
    value = payload.get(key)
    if value in (None, ""):
        return default
    return float(value)


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if value in (None, ""):
        raise ValueError("Missing required field: {0}".format(key))
    return int(value)


def build_task1_records(form_data: Mapping[str, Any]) -> List[Dict[str, float]]:
    raw_metcar_rq = _required_float(form_data, "metcar_rq")
    base_data = {
        "Age": _required_float(form_data, "age"),
        "Sex": _required_int(form_data, "sex"),
        "Birthweight": _required_float(form_data, "birthweight"),
        "TBW_FFM": _required_float(form_data, "tbw_ffm"),
        "FFM_Trunk_percent": _required_float(form_data, "ffm_trunk_percent"),
        "BFM_Leg": _required_float(form_data, "bfm_leg"),
        "BC011": _required_float(form_data, "bc011"),
        "BC010": _required_float(form_data, "bc010"),
        "Metcar_RQ": raw_metcar_rq,
        "SH0018": _required_float(form_data, "sh0018"),
        "SH0024": _required_float(form_data, "sh0024"),
        "Naptime": _required_float(form_data, "naptime"),
    }

    records = []
    for group_num in range(1, 8):
        row = dict(base_data)
        row["Group"] = group_num
        records.append(row)
    return records


def build_task2_records(form_data: Mapping[str, Any]) -> List[Dict[str, float]]:
    raw_metcar_rq = _required_float(form_data, "metcar_rq")
    converted_metcar_rq = 0.01413 + 0.78413 * raw_metcar_rq
    record = {
        "Age": _required_float(form_data, "age"),
        "Sex": _required_int(form_data, "sex"),
        "Birthweight": _required_float(form_data, "birthweight"),
        "TBW_FFM": _required_float(form_data, "tbw_ffm"),
        "FFM_Trunk_percent": _required_float(form_data, "ffm_trunk_percent"),
        "BFM_Leg": _required_float(form_data, "bfm_leg"),
        "BC011": _required_float(form_data, "bc011"),
        "BC010": _required_float(form_data, "bc010"),
        "Metcar_RQ": converted_metcar_rq,
        "SH0018": _required_float(form_data, "sh0018"),
        "SH0024": _required_float(form_data, "sh0024"),
        "Naptime": _required_float(form_data, "naptime"),
        "VFA_change_w2": _optional_float(form_data, "vfa_change_w2", 0.0),
        "Group": 1,
    }
    return [record]


def _load_model(model_path: Path) -> Any:
    try:
        return joblib.load(model_path)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Unable to load {0} because a required dependency is missing: {1}".format(
                model_path.name, exc
            )
        ) from exc
    except Exception as joblib_exc:
        try:
            with model_path.open("rb") as handle:
                return pickle.load(handle)
        except Exception as pickle_exc:
            raise RuntimeError(
                "Unable to load {0} via joblib or pickle. joblib error: {1}; pickle error: {2}".format(
                    model_path.name, joblib_exc, pickle_exc
                )
            ) from pickle_exc


def _extract_estimator(model_bundle: Any) -> Any:
    if isinstance(model_bundle, dict):
        for key in ("model", "estimator", "pipeline", "predictor"):
            if key in model_bundle:
                return model_bundle[key]
    return model_bundle


def _extract_feature_order(model_bundle: Any) -> Optional[List[str]]:
    if not isinstance(model_bundle, dict):
        return None

    for key in ("feature_order", "features", "columns", "feature_cols"):
        value = model_bundle.get(key)
        if value:
            return list(value)

    return None


def _prepare_feature_frame(
    records: List[Dict[str, float]], feature_order: Optional[List[str]] = None
) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)

    if "Group" in frame.columns:
        group_values = frame["Group"].astype(int)
        for group_num in range(1, 8):
            column_name = "Group_{0}".format(group_num)
            frame[column_name] = (group_values == group_num).astype(float)
        if not feature_order or "Group" not in feature_order:
            frame = frame.drop(columns=["Group"])

    if feature_order:
        missing_columns = [column for column in feature_order if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                "Model feature list is missing columns from the input payload: {0}".format(
                    ", ".join(missing_columns)
                )
            )
        frame = frame[feature_order]

    return frame.astype(float)


def _coerce_predictions(raw_predictions: Any) -> List[float]:
    if isinstance(raw_predictions, pd.DataFrame):
        if "Pred" in raw_predictions.columns:
            series = raw_predictions["Pred"]
        else:
            series = raw_predictions.iloc[:, 0]
    elif isinstance(raw_predictions, pd.Series):
        series = raw_predictions
    else:
        series = pd.Series(raw_predictions)

    return [round(float(value), 4) for value in series.tolist()]


def _is_tiny_mlp_bundle(model_bundle: Any) -> bool:
    required_keys = {"model_spec", "scaler", "projection", "standardization", "feature_cols"}
    return isinstance(model_bundle, dict) and required_keys.issubset(model_bundle.keys())


def _predict_with_tiny_mlp_bundle(model_bundle: Dict[str, Any], records: List[Dict[str, float]]) -> List[float]:
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This model bundle requires PyTorch. Install `torch` before running inference."
        ) from exc

    class TinyMLP(nn.Module):
        def __init__(self, dim: int = 11, activation: str = "gelu", dropout: float = 0.2):
            super().__init__()
            if activation == "gelu":
                activation_layer = nn.GELU()
            elif activation == "relu":
                activation_layer = nn.ReLU()
            else:
                raise ValueError("Unsupported activation for TinyMLP: {0}".format(activation))

            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                activation_layer,
                nn.Dropout(dropout),
                nn.Linear(dim, 1),
            )

        def forward(self, inputs):
            return self.net(inputs)

    feature_order = list(model_bundle["feature_cols"])
    frame = _prepare_feature_frame(records, feature_order)

    scaler = model_bundle["scaler"]
    scaled = scaler.transform(frame.to_numpy(dtype=float))

    projection_matrix = np.asarray(model_bundle["projection"]["Kmat"], dtype=np.float32)
    projected = np.asarray(scaled, dtype=np.float32) @ projection_matrix

    stats = model_bundle["standardization"]
    x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
    x_std = np.asarray(stats["x_std"], dtype=np.float32)
    x_std = np.where(x_std == 0, 1.0, x_std)
    normalized = (projected - x_mean) / x_std

    model_spec = model_bundle["model_spec"]
    if model_spec.get("arch") != "TinyMLP":
        raise ValueError("Unsupported model bundle architecture: {0}".format(model_spec.get("arch")))

    model = TinyMLP(**model_spec.get("init_args", {}))
    model.load_state_dict(model_spec["state_dict"])
    model.eval()

    with torch.no_grad():
        predictions = model(torch.tensor(normalized, dtype=torch.float32)).cpu().numpy().reshape(-1)

    y_mean = float(stats["y_mean"])
    y_std = float(stats["y_std"])
    predictions = predictions * y_std + y_mean

    trim_range = model_bundle.get("trim_range")
    if trim_range and len(trim_range) == 2:
        predictions = np.clip(predictions, trim_range[0], trim_range[1])

    return [round(float(value), 4) for value in predictions.tolist()]


def _predict_with_model(task_name: str, records: List[Dict[str, float]]) -> PredictionRun:
    model_path = TASK_CONFIG[task_name]["model_path"]
    model_bundle = _load_model(model_path)

    if _is_tiny_mlp_bundle(model_bundle):
        predictions = _predict_with_tiny_mlp_bundle(model_bundle, records)
        return PredictionRun(
            predictions=predictions,
            mode="model",
            message="Generated predictions from {0} (TinyMLP bundle).".format(model_path.name),
            model_path=model_path,
        )

    estimator = _extract_estimator(model_bundle)
    feature_order = _extract_feature_order(model_bundle)
    frame = _prepare_feature_frame(records, feature_order)

    if hasattr(estimator, "predict"):
        raw_predictions = estimator.predict(frame)
    elif callable(estimator):
        raw_predictions = estimator(frame)
    else:
        raise TypeError("Loaded object does not expose a callable predictor.")

    predictions = _coerce_predictions(raw_predictions)
    return PredictionRun(
        predictions=predictions,
        mode="model",
        message="Generated predictions from {0}.".format(model_path.name),
        model_path=model_path,
    )


def _task1_demo_prediction(record: Mapping[str, Any]) -> float:
    group = int(record["Group"])
    group_offset = {
        1: 2.1,
        2: 3.4,
        3: 4.2,
        4: 3.0,
        5: 3.3,
        6: 2.6,
        7: 1.8,
    }[group]

    score = group_offset
    score += 0.10 * (float(record["Age"]) - 25.0)
    score += 0.08 * (float(record["TBW_FFM"]) - 70.0)
    score -= 0.06 * (float(record["FFM_Trunk_percent"]) - 42.0)
    score -= 0.35 * (float(record["BFM_Leg"]) - 5.0)
    score -= 0.55 * (float(record["BC011"]) - 0.35)
    score -= 0.05 * (float(record["BC010"]) - 35.0)
    score -= 2.50 * (float(record["Metcar_RQ"]) - 0.82)
    score -= 0.03 * (float(record["SH0018"]) - 75.0)
    score -= 0.12 * (float(record["SH0024"]) - 5.0)
    score -= 0.10 * (float(record["Naptime"]) - 0.5)
    if int(record["Sex"]) == 1:
        score += 0.3
    return round(max(min(score, 12.0), -5.0), 4)


def _task2_demo_prediction(record: Mapping[str, Any]) -> float:
    score = 0.75 * float(record["VFA_change_w2"])
    score += 0.04 * (float(record["Age"]) - 25.0)
    score -= 0.18 * (float(record["BFM_Leg"]) - 5.0)
    score -= 0.10 * (float(record["SH0024"]) - 5.0)
    score -= 1.80 * (float(record["Metcar_RQ"]) - 0.82)
    score += 0.03 * (float(record["TBW_FFM"]) - 70.0)
    return round(max(min(score, 20.0), -20.0), 4)


def _predict_with_demo(task_name: str, records: List[Dict[str, float]]) -> PredictionRun:
    if task_name == "task1":
        predictions = [_task1_demo_prediction(record) for record in records]
    elif task_name == "task2":
        predictions = [_task2_demo_prediction(record) for record in records]
    else:
        raise ValueError("Unsupported task: {0}".format(task_name))

    model_path = TASK_CONFIG[task_name]["model_path"]
    return PredictionRun(
        predictions=predictions,
        mode="demo",
        message=(
            "Model artifact not found at {0}. Using a deterministic demo fallback so the "
            "Capsule can still run end-to-end."
        ).format(model_path),
        model_path=model_path,
    )


def predict_records(task_name: str, records: List[Dict[str, float]]) -> PredictionRun:
    model_path = TASK_CONFIG[task_name]["model_path"]
    if model_path.exists():
        return _predict_with_model(task_name, records)
    return _predict_with_demo(task_name, records)


def predict_task1_from_form(form_data: Mapping[str, Any]) -> Dict[str, Any]:
    records = build_task1_records(form_data)
    run = predict_records("task1", records)

    raw_results = {}
    for index, prediction in enumerate(run.predictions, start=1):
        raw_results["Group_{0}".format(index)] = round(float(prediction), 2)

    best_key = max(raw_results, key=raw_results.get)
    converted_results = {
        DIET_NAME_MAP[group_key]: value for group_key, value in raw_results.items()
    }

    return {
        "task_type": "task1",
        "prediction_mode": run.mode,
        "prediction_message": run.message,
        "recommended_diet_name": DIET_NAME_MAP[best_key],
        "vfa_reduction": raw_results[best_key],
        "all_results": converted_results,
        "raw_results": raw_results,
    }


def _task2_interpretation(prediction: float) -> str:
    if prediction > 0:
        return "Positive values indicate a modeled decrease in VFA after four weeks on the current diet."
    if prediction < 0:
        return "Negative values indicate a modeled increase in VFA after four weeks on the current diet."
    return "A value close to zero indicates little modeled change in VFA after four weeks on the current diet."


def predict_task2_from_form(form_data: Mapping[str, Any]) -> Dict[str, Any]:
    records = build_task2_records(form_data)
    run = predict_records("task2", records)
    prediction_value = round(float(run.predictions[0]), 2)

    return {
        "task_type": "task2",
        "prediction_mode": run.mode,
        "prediction_message": run.message,
        "results": {
            "current_diet_continuation": prediction_value,
            "interpretation": _task2_interpretation(prediction_value),
        },
    }


def load_records_from_json(json_path: Path) -> Any:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        return payload

    raise ValueError("Input JSON must be a list, a single record object, or an object with a 'records' key.")


def write_predictions_csv(predictions: List[float], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"Pred": predictions})
    frame.to_csv(output_path, index=False)
    return output_path


def run_inference_from_json(
    task_name: str, json_path: Path, output_path: Optional[Path] = None
) -> Dict[str, Any]:
    payload = load_records_from_json(json_path)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        if task_name == "task1":
            records = build_task1_records(payload)
        elif task_name == "task2":
            records = build_task2_records(payload)
        else:
            raise ValueError("Unsupported task: {0}".format(task_name))
    else:
        raise ValueError("Input JSON does not contain any records.")

    if not records:
        raise ValueError("Input JSON does not contain any records.")

    run = predict_records(task_name, records)

    if output_path is None:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = ROOT_DIR / task_name / "infer_preds_{0}.csv".format(stamp)

    write_predictions_csv(run.predictions, output_path)
    return {
        "task_name": task_name,
        "mode": run.mode,
        "message": run.message,
        "output_path": str(output_path),
        "predictions": [round(float(value), 2) for value in run.predictions],
    }


def cli_infer_main(task_name: str) -> int:
    parser = argparse.ArgumentParser(description="Run JSON-based inference for {0}.".format(task_name))
    parser.add_argument("--pkl", help="Optional explicit path to a model artifact.")
    parser.add_argument("--json", required=True, help="Path to the input JSON payload.")
    parser.add_argument("--output", help="Optional output CSV path.")
    args = parser.parse_args()

    if args.pkl:
        TASK_CONFIG[task_name]["model_path"] = Path(args.pkl).resolve()

    output_path = Path(args.output).resolve() if args.output else None
    result = run_inference_from_json(task_name, Path(args.json).resolve(), output_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
