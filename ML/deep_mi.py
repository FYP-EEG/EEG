"""Optional Braindecode EEGConformer/TCFormer sklearn-style classifiers.

Edited by Anson
Date: 10/11/2026
Purpose: make Transformer comparison and public-to-personal weight transfer
         optional without making PyTorch a core runtime dependency.
"""

from pathlib import Path

import numpy as np

SUPPORTED_DEEP_MODELS = ("eegconformer", "tcformer")


class BraindecodeMIClassifier:
    """Small fit/predict wrapper around maintained Braindecode MI models.

    Inputs are normalized with training-set channel statistics. A public model
    can initialize matching layers for personal fine-tuning, but its classifier
    head and normalization are refit on the personal labels/data.
    """

    def __init__(self, model_name="eegconformer", sample_freq=250,
                 max_epochs=40, batch_size=32, learning_rate=1e-3,
                 weight_decay=1e-4, random_state=42, device="auto",
                 pretrained_path=None, verbose=False):
        name = str(model_name).strip().lower()
        if name not in SUPPORTED_DEEP_MODELS:
            raise ValueError(f"model_name must be one of {SUPPORTED_DEEP_MODELS}")
        self.model_name = name
        self.sample_freq = int(sample_freq)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self.device = str(device)
        self.pretrained_path = str(pretrained_path) if pretrained_path else None
        self.verbose = bool(verbose)

    @staticmethod
    def _dependencies():
        try:
            import torch
            from braindecode.models import EEGConformer, TCFormer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Transformer MI training needs PyTorch and Braindecode: "
                "python -m pip install '.[transformer]'") from exc
        return torch, EEGConformer, TCFormer

    def _device_name(self, torch):
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _make_model(self, n_outputs, n_chans, n_times):
        torch, EEGConformer, TCFormer = self._dependencies()
        cls = EEGConformer if self.model_name == "eegconformer" else TCFormer
        model = cls(n_outputs=int(n_outputs), n_chans=int(n_chans),
                    n_times=int(n_times), sfreq=float(self.sample_freq))
        return torch, model

    def _load_matching_pretrained_layers(self, model, torch):
        if not self.pretrained_path:
            return {"loaded": 0, "skipped": 0}
        import joblib

        bundle = joblib.load(Path(self.pretrained_path))
        if not isinstance(bundle, dict) or bundle.get("kind") != "PublicMICategoryModel":
            raise ValueError(f"{self.pretrained_path} is not a public MI category model")
        source = bundle.get("model")
        if not isinstance(source, BraindecodeMIClassifier) or not hasattr(source, "model_"):
            raise ValueError("public pretraining bundle does not contain a fitted Braindecode model")
        if source.model_name != self.model_name:
            raise ValueError(
                f"pretrained architecture {source.model_name!r} does not match {self.model_name!r}")
        incoming = source.model_.state_dict()
        current = model.state_dict()
        compatible = {key: value.detach().cpu() for key, value in incoming.items()
                      if key in current and tuple(value.shape) == tuple(current[key].shape)}
        current.update(compatible)
        model.load_state_dict(current)
        return {"loaded": len(compatible), "skipped": len(incoming) - len(compatible)}

    def fit(self, X, y):
        torch, model = self._make_model(len(np.unique(y)), np.asarray(X).shape[1],
                                        np.asarray(X).shape[2])
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 3 or len(X) != len(y) or not len(X):
            raise ValueError("X/y must be non-empty (trials, channels, samples) arrays")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        self.classes_, y_index = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("deep MI classification needs at least two classes")
        self.n_channels_in_ = int(X.shape[1])
        self.n_times_in_ = int(X.shape[2])
        self.mean_ = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.scale_ = X.std(axis=(0, 2), keepdims=True).astype(np.float32)
        self.scale_[self.scale_ < 1e-8] = 1.0
        Xn = ((X - self.mean_) / self.scale_).astype(np.float32)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self.transfer_report_ = self._load_matching_pretrained_layers(model, torch)
        device = self._device_name(torch)
        model.to(device)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(Xn), torch.from_numpy(y_index.astype(np.int64)))
        generator = torch.Generator().manual_seed(self.random_state)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True,
            generator=generator)
        counts = np.bincount(y_index, minlength=len(self.classes_)).astype(np.float32)
        weights = counts.sum() / (len(counts) * np.maximum(counts, 1.0))
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.as_tensor(weights, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay)
        self.loss_history_ = []
        model.train()
        for epoch in range(self.max_epochs):
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu()) * len(xb)
            epoch_loss = total / len(dataset)
            self.loss_history_.append(epoch_loss)
            if self.verbose:
                print(f"  {self.model_name} epoch {epoch + 1}/{self.max_epochs}: loss={epoch_loss:.5f}")
        self.model_ = model.to("cpu")
        self.model_.eval()
        self.device_used_ = device
        return self

    def _prepare(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3 or X.shape[1:] != (self.n_channels_in_, self.n_times_in_):
            raise ValueError(
                f"expected (*, {self.n_channels_in_}, {self.n_times_in_}), got {X.shape}")
        return ((X - self.mean_) / self.scale_).astype(np.float32)

    def predict_proba(self, X):
        torch, _, _ = self._dependencies()
        Xn = self._prepare(X)
        self.model_.eval()
        rows = []
        with torch.no_grad():
            for start in range(0, len(Xn), self.batch_size):
                logits = self.model_(torch.from_numpy(Xn[start:start + self.batch_size]))
                if isinstance(logits, tuple):
                    logits = logits[0]
                rows.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(rows)

    def predict(self, X):
        indices = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[indices]

    def transfer_summary(self):
        return dict(getattr(self, "transfer_report_", {"loaded": 0, "skipped": 0}))
