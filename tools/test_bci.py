"""Synthetic motor-imagery tests for the local model workflow.

    python tools/test_bci.py
    python tools/test_bci.py --layer 4

All layers generate their own data and run without a headset or recordings.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ok = 0
failed = []


def check(condition, name, detail=""):
    global ok
    if condition:
        ok += 1
        print(f"  PASS  {name}")
    else:
        failed.append(name)
        print(f"  FAIL  {name}" + (f"\n        {detail}" if detail else ""))


def make_trials(seed=7, blocks=3, trials_per_class=8):
    rng = np.random.default_rng(seed)
    trials = []
    labels = []
    for _ in range(blocks):
        for label in (0, 1):
            for _ in range(trials_per_class):
                trial = rng.normal(0, 10.0, (8, 750))
                trial[2] *= 0.55 if label == 1 else 1.0
                trial[4] *= 1.0 if label == 1 else 0.55
                trials.append(trial)
                labels.append(f"target_{label}")
    return np.asarray(trials), np.asarray(labels)


def write_recordings(root, seed=7):
    user_dir = Path(root) / "demo"
    user_dir.mkdir(parents=True, exist_ok=True)
    trials, labels = make_trials(seed=seed)
    for block in range(3):
        section = slice(block * 16, (block + 1) * 16)
        np.savez_compressed(
            user_dir / f"calib_{block}.npz",
            X=trials[section],
            labels=labels[section],
            fs=250,
            montage="cyton8_mi",
        )
    return user_dir, trials


def train_demo(root):
    write_recordings(root)
    from ML.train_local import train_user

    return train_user("demo", root=root, verbose=False)


def layer1():
    print("\nLAYER 1 - Calibration data loading")
    from ML.train_local import load_user_data

    with tempfile.TemporaryDirectory() as temp:
        write_recordings(temp)
        data = load_user_data("demo", root=temp)
        check(data["X"].shape == (48, 8, 750), "loads all trial windows")
        check(np.bincount(data["y"]).tolist() == [24, 24],
              "loads balanced target classes")
        check(len(np.unique(data["groups"])) == 6,
              "keeps recording blocks separate for validation")
        check(data["fs"] == 250 and data["montage"] == "cyton8_mi",
              "preserves sample rate and montage metadata")


def layer2():
    print("\nLAYER 2 - Motor-imagery feature pipelines")
    from ML.mi_advanced import CSP, FilterBankCSP, TangentSpaceMapper, PIPELINES

    trials, labels = make_trials()
    y = np.asarray([int(label[-1]) for label in labels])

    csp = CSP(n_components=4).fit(trials, y)
    features = csp.transform(trials)
    check(features.shape == (48, 4), "CSP returns the configured feature count")
    check(np.isfinite(features).all(), "CSP features are finite")

    fbcsp = FilterBankCSP(sample_freq=250).fit(trials, y)
    bank_features = fbcsp.transform(trials)
    check(bank_features.shape == (48, 24), "filter-bank CSP combines band features")

    tangent = TangentSpaceMapper(sample_freq=250).fit(trials, y)
    tangent_features = tangent.transform(trials)
    check(tangent_features.shape == (48, 36),
          "covariance features have expected dimension")
    check(np.isfinite(tangent_features).all(), "covariance features are finite")

    for name, factory in PIPELINES.items():
        try:
            pipeline = factory(sample_freq=250).fit(trials, y)
            predictions = pipeline.predict(trials)
            check(predictions.shape == (48,) and set(predictions) <= {0, 1},
                  f"{name} fits and predicts two classes")
        except Exception as exc:
            check(False, f"{name} fits and predicts two classes", repr(exc))


def layer3():
    print("\nLAYER 3 - Training and model persistence")
    import joblib

    with tempfile.TemporaryDirectory() as temp:
        user_dir, _ = write_recordings(temp)
        result = train_demo(temp)
        check(result is not None, "training selects a pipeline")
        model_path = user_dir / "model.joblib"
        check(model_path.exists(), "training writes model.joblib")
        if model_path.exists():
            bundle = joblib.load(model_path)
            check(bundle.get("kind") == "MILocalModel",
                  "saved bundle has the local model format")
            check(bundle["pipeline"] == result["pipeline"],
                  "saved pipeline metadata matches")
            check(hasattr(bundle["model"], "predict"),
                  "saved estimator reloads and predicts")


def layer4():
    print("\nLAYER 4 - Local model inference")
    from ML.local_model import LocalMIModel, NO_ACTION

    class ProbabilityModel:
        def __init__(self, probabilities):
            self.probabilities = np.asarray(probabilities, dtype=float)

        def predict_proba(self, X):
            return np.tile(self.probabilities, (len(X), 1))

    left_model = LocalMIModel(ProbabilityModel([0.9, 0.1]),
                              confidence_threshold=0.65)
    command, confidence = left_model.predict_command(np.zeros((8, 750)))
    check(command == "LEFT" and confidence == 0.9,
          "maps confident class to a command")

    right_model = LocalMIModel(ProbabilityModel([0.1, 0.9]),
                               confidence_threshold=0.65)
    command, confidence = right_model.predict_command(np.zeros((8, 750)))
    check(command == "RIGHT" and confidence == 0.9,
          "maps the other class to its command")

    reject_model = LocalMIModel(ProbabilityModel([0.55, 0.45]),
                                confidence_threshold=0.65)
    label, confidence = reject_model.predict_proba(np.zeros((8, 750)))
    check(label == NO_ACTION and confidence == 0.55,
          "rejects a below-threshold prediction")
    check(reject_model.predict_command(np.zeros((8, 750)))[0] is None,
          "rejected prediction produces no command")

    try:
        left_model.predict_command(np.zeros(750))
        check(False, "rejects a window with the wrong dimensions")
    except ValueError:
        check(True, "rejects a window with the wrong dimensions")

    with tempfile.TemporaryDirectory() as temp:
        user_dir, trials = write_recordings(temp)
        result = train_demo(temp)
        model = LocalMIModel.load("demo", root=temp, confidence_threshold=0.0)
        check(result is not None and model.source == "personal",
              "loads the trained personal model")
        check(model.describe()["pipeline"] == result["pipeline"],
              "loaded model retains pipeline metadata")
        label, confidence = model.predict_proba(trials[0])
        check(label in (0, 1) and 0.0 <= confidence <= 1.0,
              "trained model predicts a class and confidence")
        check(model.predict_command(trials[0])[0] in ("LEFT", "RIGHT"),
              "trained model returns a command")
        check((user_dir / "model.joblib").exists(),
              "inference uses the saved model")
        try:
            LocalMIModel.load("missing-user", root=temp)
            check(False, "missing model raises FileNotFoundError")
        except FileNotFoundError:
            check(True, "missing model raises FileNotFoundError")


LAYERS = {1: layer1, 2: layer2, 3: layer3, 4: layer4}


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic motor-imagery model tests")
    parser.add_argument("--layer", type=int, choices=sorted(LAYERS))
    args = parser.parse_args()

    selected = [args.layer] if args.layer else sorted(LAYERS)
    for number in selected:
        LAYERS[number]()

    print(f"\nRESULT: {ok} passed, {len(failed)} failed")
    if failed:
        print("Failed checks:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
