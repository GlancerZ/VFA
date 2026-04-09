# Code Ocean Checklist

## Recommended Setup

1. Import this repository into a new Code Ocean Capsule.
2. Choose a Python starter environment.
3. Install the packages from `requirements.txt` including `torch` and `scikit-learn 1.6.1`.
4. Keep the repository root as `/code` and use the included `run` file for the Reproducible Run.

## Before You Release

- Add the trained model file `task1/best_model.pkl`.
- Add the trained model file `task2/best_model_task2.pkl`.
- Run the Capsule once and confirm that outputs are written to `/results`.

## If The Model Files Are Missing

The repository now falls back to a deterministic demo predictor so the Capsule can
still execute end-to-end. This is useful for environment testing and App Panel
development, but it should not be treated as the final scientific model output.
