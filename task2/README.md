Place the trained Task 2 artifact here as `best_model_task2.pkl`.

The generic `infer_from_pkl.py` loader supports common `joblib` and `pickle`
serializations, including direct estimators and dictionaries containing:

- `model`, `estimator`, `pipeline`, or `predictor`
- optional `feature_order`, `features`, or `columns`
- the bundled `TinyMLP` format currently used by this repository
