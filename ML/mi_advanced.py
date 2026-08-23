"""
Author: Brian
Created on: 16/8/2026
Purpose: Step 6 -- Motor Imagery upgrades: Filter Bank CSP (FBCSP) and Riemannian
         tangent-space mapping, replacing the plain CSP+LDA that scored 61.4%.

Implemented in numpy/scipy/sklearn only -- no mne, no pyriemann. That keeps the SDK
dependency surface small and means the same code runs in Colab and on the demo laptop.

WHY THE BASELINE FAILED
-----------------------
The Colab CSP+LDA pipeline reached 61.40% cross-subject (near chance for 2 classes).
Two reasons, and neither is fixed by "a better classifier":
  1. CSP uses ONE broad band (8-30 Hz). Individual mu/beta peaks vary by several Hz
     between people, so a fixed band is wrong for most subjects. FBCSP addresses this
     by extracting CSP features in several narrow bands and selecting the informative
     ones per subject.
  2. Cross-subject covariance structure differs wildly. Riemannian methods respect the
     curved geometry of SPD covariance matrices instead of treating them as flat
     vectors, and are the current state of the art for small-sample MI.

HONEST EXPECTATION
------------------
These are better estimators, but the dominant problem is cross-subject variability, so
neither will turn 61% into 85% without per-subject calibration. The recommendation from
docs/accuracy_strategy.md stands: keep MI OUT of the live demo path and ship SSVEP+blink.
This module exists so the claim is measured rather than assumed.
"""

import numpy as np
from scipy.linalg import eigh, sqrtm
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------- CSP
def _cov(X, reg=0.05):
    """Regularised, trace-normalised covariance of (n_channels, n_times)."""
    X = X - X.mean(axis=1, keepdims=True)
    C = X @ X.T / max(1, X.shape[1] - 1)
    tr = np.trace(C)
    if tr > 0:
        C = C / tr                       # trace normalisation: scale invariance
    if reg > 0:
        C = (1 - reg) * C + reg * np.eye(C.shape[0]) / C.shape[0]
    return C


class CSP(BaseEstimator, TransformerMixin):
    """Common Spatial Patterns with log-variance features (binary)."""

    def __init__(self, n_components=4, reg=0.05):
        self.n_components = n_components
        self.reg = reg

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP here supports exactly 2 classes")
        C = [np.mean([_cov(t, self.reg) for t in X[y == c]], axis=0)
             for c in classes]
        vals, vecs = eigh(C[0], C[0] + C[1])
        order = np.argsort(vals)
        k = self.n_components // 2
        idx = np.concatenate([order[:k], order[-k:]])   # most extreme both ends
        self.filters_ = vecs[:, idx].T
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        out = np.empty((len(X), self.filters_.shape[0]))
        for i, trial in enumerate(X):
            proj = self.filters_ @ trial
            v = np.var(proj, axis=1)
            v = np.maximum(v, 1e-20)
            out[i] = np.log(v / v.sum())
        return out


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """CSP features extracted in several narrow bands, concatenated.

    Handles the individual mu/beta frequency differences that a single 8-30 Hz band
    cannot. Pair with SelectKBest to pick the informative bands per subject.
    """

    def __init__(self, bands=((4, 8), (8, 12), (12, 16), (16, 20), (20, 26), (26, 32)),
                 sample_freq=250, n_components=4, reg=0.05):
        self.bands = [tuple(b) for b in bands]
        self.sample_freq = sample_freq
        self.n_components = n_components
        self.reg = reg

    def _sos(self):
        return [butter(4, list(b), btype='band', fs=self.sample_freq, output='sos')
                for b in self.bands]

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.sos_ = self._sos()
        self.csps_ = []
        for sos in self.sos_:
            Xb = sosfiltfilt(sos, X, axis=-1)
            self.csps_.append(CSP(self.n_components, self.reg).fit(Xb, y))
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        feats = [csp.transform(sosfiltfilt(sos, X, axis=-1))
                 for sos, csp in zip(self.sos_, self.csps_)]
        return np.hstack(feats)


# ------------------------------------------------------------- Riemannian
def _logm_spd(C):
    """Matrix logarithm of a symmetric positive-definite matrix."""
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-12)
    return vecs @ np.diag(np.log(vals)) @ vecs.T


def _invsqrtm_spd(C):
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-12)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def riemannian_mean(covs, tol=1e-8, max_iter=50):
    """Fréchet (geometric) mean of SPD matrices via gradient descent.

    The arithmetic mean is wrong for covariance matrices -- they live on a curved
    manifold, not a flat vector space. This is the whole point of the method.
    """
    covs = np.asarray(covs, float)
    M = covs.mean(axis=0)                      # arithmetic mean as the seed
    for _ in range(max_iter):
        M_isqrt = _invsqrtm_spd(M)
        M_sqrt = np.real(sqrtm(M))
        T = np.mean([_logm_spd(M_isqrt @ C @ M_isqrt) for C in covs], axis=0)
        norm = np.linalg.norm(T)
        vals, vecs = np.linalg.eigh(T)
        expT = vecs @ np.diag(np.exp(vals)) @ vecs.T
        M = M_sqrt @ expT @ M_sqrt
        M = (M + M.T) / 2
        if norm < tol:
            break
    return M


class TangentSpaceMapper(BaseEstimator, TransformerMixin):
    """Project trial covariances into the tangent space at their Riemannian mean.

    Output is a flat vector, so any standard classifier (LDA, SVM, LogReg) can be used
    downstream -- while the geometry of the covariance manifold has been respected.
    """

    def __init__(self, sample_freq=250, band=(8.0, 30.0), reg=0.05):
        self.sample_freq = sample_freq
        self.band = band
        self.reg = reg

    def _covs(self, X):
        X = np.asarray(X, float)
        if self.band:
            sos = butter(4, list(self.band), btype='band',
                         fs=self.sample_freq, output='sos')
            X = sosfiltfilt(sos, X, axis=-1)
        return np.array([_cov(t, self.reg) for t in X])

    def fit(self, X, y=None):
        covs = self._covs(X)
        self.mean_ = riemannian_mean(covs)
        self.mean_isqrt_ = _invsqrtm_spd(self.mean_)
        n = self.mean_.shape[0]
        self.idx_ = np.triu_indices(n)
        # off-diagonal terms count twice in the inner product
        w = np.sqrt(2.0) * np.ones((n, n))
        np.fill_diagonal(w, 1.0)
        self.w_ = w[self.idx_]
        return self

    def transform(self, X):
        covs = self._covs(X)
        out = []
        for C in covs:
            S = _logm_spd(self.mean_isqrt_ @ C @ self.mean_isqrt_)
            out.append(S[self.idx_] * self.w_)
        return np.array(out)


# ---------------------------------------------------------------- pipelines
def make_csp_lda(sample_freq=250, n_components=4):
    """Baseline for comparison: single-band CSP + shrinkage LDA."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    class _Band(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            self.sos_ = butter(4, [8.0, 30.0], btype='band',
                               fs=sample_freq, output='sos')
            return self
        def transform(self, X):
            return sosfiltfilt(self.sos_, np.asarray(X, float), axis=-1)

    return Pipeline([
        ("band", _Band()),
        ("csp", CSP(n_components)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def make_fbcsp(sample_freq=250, n_components=4, k_features=12):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return Pipeline([
        ("fbcsp", FilterBankCSP(sample_freq=sample_freq, n_components=n_components)),
        ("select", SelectKBest(mutual_info_classif, k=k_features)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def make_riemannian(sample_freq=250, C=1.0):
    from sklearn.svm import SVC
    return Pipeline([
        ("ts", TangentSpaceMapper(sample_freq=sample_freq)),
        ("scale", StandardScaler()),
        ("svm", SVC(kernel="linear", C=C, probability=True)),
    ])


def make_riemannian_lr(sample_freq=250):
    from sklearn.linear_model import LogisticRegression
    return Pipeline([
        ("ts", TangentSpaceMapper(sample_freq=sample_freq)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000)),
    ])


PIPELINES = {
    "csp_lda": make_csp_lda,
    "fbcsp": make_fbcsp,
    "riemann_svm": make_riemannian,
    "riemann_lr": make_riemannian_lr,
}
