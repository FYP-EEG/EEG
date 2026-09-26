"""
Author: Anson
Created on: 26/9/2026
Purpose: runnable diagnostic tests for each of the five failure layers.

    python tools/layer_tests.py --layer 1 --recording profiles/anson/calib_*.npz
    python tools/layer_tests.py --layer 2 --recording profiles/anson/calib_*.npz
    python tools/layer_tests.py --layer 3 --user anson
    python tools/layer_tests.py --all --user anson

WHY LAYERED
-----------
A BCI can fail at five independent layers and every one of them looks identical
from the outside: nothing happens when the user tries. Testing them separately is
the only way to attribute a failure instead of guessing.

Run them IN ORDER. A layer-3 result computed on layer-1-broken data is noise.
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.signal import butter, sosfiltfilt, welch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def _v(tag, name, detail=""):
    mark = {PASS: "[PASS]", WARN: "[WARN]", FAIL: "[FAIL]"}[tag]
    print(f"  {mark} {name}")
    if detail:
        for line in str(detail).split("\n"):
            print(f"         {line}")
    return tag


def _load(pattern):
    hits = sorted(glob.glob(str(pattern)))
    if not hits:
        raise FileNotFoundError(f"no recording matched {pattern}")
    d = np.load(hits[-1], allow_pickle=True)
    return (d["X"].astype(np.float64), d["labels"].astype(str),
            int(d["fs"]), str(d["montage"]), Path(hits[-1]).name)


def bandpower(x, fs, lo, hi):
    """Average power in a band. Welch is used because it averages over
    sub-windows, which is far less noisy than a single periodogram."""
    f, p = welch(x, fs=fs, nperseg=min(256, x.shape[-1]))
    m = (f >= lo) & (f <= hi)
    return float(np.mean(p[..., m]))


# =====================================================================
# LAYER 1 — HARDWARE
# =====================================================================
def layer1(X, labels, fs, montage):
    """Is the rig recording brain activity at all?

    Covers: dead channels, saturation, bad contact, mains interference,
    flat-lining, channel cross-talk, and the definitive alpha check.
    """
    print("\n" + "=" * 68)
    print("LAYER 1 · HARDWARE AND SIGNAL QUALITY")
    print("=" * 68)
    res = []

    from ML.montage import get
    m = get(montage)
    names = m.get("labels", [f"ch{i}" for i in range(X.shape[1])])
    flat = X.transpose(1, 0, 2).reshape(X.shape[1], -1)

    # 1.1 amplitude range -------------------------------------------------
    print("\n1.1 Per-channel amplitude (RMS)")
    rms = np.std(flat, axis=1)
    bad = []
    for i, r in enumerate(rms):
        state = "ok"
        if r < 1.0:
            state, tag = "DEAD - not connected", FAIL
        elif r > 200:
            state, tag = "SATURATED - bad contact or movement", FAIL
        elif r > 100:
            state, tag = "noisy", WARN
        else:
            tag = PASS
        if tag != PASS:
            bad.append(f"{names[i]}={r:.1f}uV ({state})")
        print(f"      {names[i]:<5} {r:7.1f} uV   {state}")
    res.append(_v(FAIL if any("DEAD" in b or "SAT" in b for b in bad)
                  else (WARN if bad else PASS),
                  "amplitude in 1-200 uV range",
                  "\n".join(bad) if bad else "all channels plausible"))

    # 1.2 flat-line / stuck ADC ------------------------------------------
    print("\n1.2 Flat-line detection")
    stuck = [names[i] for i in range(len(flat))
             if len(np.unique(np.round(flat[i], 3))) < 10]
    res.append(_v(FAIL if stuck else PASS, "no stuck/constant channels",
                  f"stuck: {stuck}" if stuck else "all channels varying"))

    # 1.3 mains interference ---------------------------------------------
    print("\n1.3 Mains interference (50 Hz in HK)")
    worst = []
    for i in range(len(flat)):
        mains = bandpower(flat[i], fs, 48, 52)
        sig = bandpower(flat[i], fs, 8, 30)
        ratio = mains / (sig + 1e-12)
        if ratio > 2.0:
            worst.append(f"{names[i]}: mains/signal = {ratio:.1f}x")
    res.append(_v(FAIL if len(worst) > 2 else (WARN if worst else PASS),
                  "mains not dominating the 8-30 Hz band",
                  "\n".join(worst) if worst else "50 Hz under control"))

    # 1.4 channel cross-talk / bridging -----------------------------------
    print("\n1.4 Electrode bridging (channels too similar)")
    C = np.corrcoef(flat)
    bridged = [f"{names[i]}~{names[j]} r={C[i, j]:.2f}"
               for i in range(len(C)) for j in range(i + 1, len(C))
               if C[i, j] > 0.98]
    res.append(_v(WARN if bridged else PASS, "no bridged electrode pairs",
                  "\n".join(bridged) if bridged
                  else "channels are electrically distinct"))

    # 1.5 alpha blocking — the definitive test ----------------------------
    print("\n1.5 Eyes-closed alpha (the definitive proof of real EEG)")
    print("      requires a recording with eyes-open and eyes-closed blocks")
    has = [l for l in np.unique(labels) if "close" in l.lower() or "open" in l.lower()]
    if len(has) >= 2:
        oc = X[np.array([("close" in l.lower()) for l in labels])]
        op = X[np.array([("open" in l.lower()) for l in labels])]
        post = m.get("ssvep", [len(names) - 2, len(names) - 1])
        a_c = np.mean([bandpower(t[post], fs, 8, 13) for t in oc])
        a_o = np.mean([bandpower(t[post], fs, 8, 13) for t in op])
        ratio = a_c / (a_o + 1e-12)
        res.append(_v(PASS if ratio > 1.5 else FAIL,
                      f"alpha increases with eyes closed ({ratio:.2f}x)",
                      "expect >1.5x. If not, electrodes are not on scalp "
                      "properly or are not posterior enough."))
    else:
        res.append(_v(WARN, "alpha test skipped",
                      "record 15 s eyes-open then 15 s eyes-closed and label "
                      "them 'eyes_open' / 'eyes_closed' to enable this"))
    return res


# =====================================================================
# LAYER 2 — SIGNAL / SEPARABILITY
# =====================================================================
def layer2(X, labels, fs, montage):
    """Is there a measurable difference between the two intentions,
    BEFORE any classifier is involved?

    Covers: ERD presence, lateralisation, effect size, per-channel
    contribution, class balance, and idle separability.
    """
    print("\n" + "=" * 68)
    print("LAYER 2 · SIGNAL SEPARABILITY  (no classifier involved)")
    print("=" * 68)
    res = []

    from ML.montage import get
    m = get(montage)
    names = m.get("labels", [f"ch{i}" for i in range(X.shape[1])])

    t0 = X[labels == "target_0"]
    t1 = X[labels == "target_1"]
    idle = X[np.isin(labels, ["idle", "idle_distracted"])]

    # 2.1 class balance ---------------------------------------------------
    print("\n2.1 Class balance")
    n0, n1 = len(t0), len(t1)
    bal = min(n0, n1) / max(n0, n1, 1)
    res.append(_v(PASS if bal > 0.8 and min(n0, n1) >= 8 else
                  (WARN if min(n0, n1) >= 8 else FAIL),
                  f"balanced classes (target_0={n0}, target_1={n1})",
                  "need >=8 per class; imbalance biases every metric"))
    if min(n0, n1) < 2:
        return res

    # 2.2 band power per channel ------------------------------------------
    print("\n2.2 Mu/beta band power by channel  (log scale)")
    sos = butter(4, [8, 30], btype="band", fs=fs, output="sos")
    f0 = sosfiltfilt(sos, t0, axis=-1)
    f1 = sosfiltfilt(sos, t1, axis=-1)
    p0 = np.log(np.var(f0, axis=2) + 1e-12)
    p1 = np.log(np.var(f1, axis=2) + 1e-12)

    print(f"      {'ch':<6}{'left-imag':>11}{'right-imag':>12}{'diff':>9}"
          f"{'Cohen d':>10}")
    ds = []
    for i in range(X.shape[1]):
        pooled = np.sqrt((p0[:, i].var() + p1[:, i].var()) / 2) + 1e-12
        d = (p0[:, i].mean() - p1[:, i].mean()) / pooled
        ds.append(d)
        print(f"      {names[i]:<6}{p0[:, i].mean():>11.3f}"
              f"{p1[:, i].mean():>12.3f}{p0[:, i].mean()-p1[:, i].mean():>9.3f}"
              f"{d:>10.2f}")

    # 2.3 lateralisation --------------------------------------------------
    print("\n2.3 Contralateral lateralisation")
    lat = m.get("motor_lateral")
    if lat and len(lat) == 2:
        c3, c4 = lat
        # imagining LEFT hand should REDUCE power at the RIGHT cortex (C4)
        left_c4 = p0[:, c4].mean()
        right_c4 = p1[:, c4].mean()
        left_c3 = p0[:, c3].mean()
        right_c3 = p1[:, c3].mean()
        ok = (left_c4 < right_c4) and (right_c3 < left_c3)
        res.append(_v(PASS if ok else WARN,
                      "ERD appears contralaterally",
                      f"{names[c4]}: left-imag {left_c4:.3f} vs right-imag "
                      f"{right_c4:.3f}  (expect left LOWER)\n"
                      f"{names[c3]}: right-imag {right_c3:.3f} vs left-imag "
                      f"{left_c3:.3f}  (expect right LOWER)\n"
                      "wrong direction can mean swapped electrodes or "
                      "swapped cue labels"))
    else:
        res.append(_v(WARN, "lateralisation skipped",
                      "montage has no motor_lateral pair"))

    # 2.4 overall effect size — THE GO/NO-GO GATE -------------------------
    print("\n2.4 Overall separability  (THE GO/NO-GO GATE)")
    best = max(abs(np.array(ds)))
    bi = int(np.argmax(np.abs(ds)))
    # multivariate distance across all motor channels
    mu0, mu1 = p0.mean(0), p1.mean(0)
    pooled_cov = (np.cov(p0.T) + np.cov(p1.T)) / 2
    try:
        maha = float(np.sqrt((mu0 - mu1) @ np.linalg.pinv(pooled_cov) @ (mu0 - mu1)))
    except Exception:
        maha = float("nan")
    tag = PASS if best > 1.5 else (WARN if best > 0.8 else FAIL)
    res.append(_v(tag, f"best single-channel Cohen d = {best:.2f} ({names[bi]})",
                  f"multivariate Mahalanobis distance = {maha:.2f}\n"
                  "d > 1.5 strongly separable -> proceed to layer 3\n"
                  "d 0.8-1.5 marginal -> more trials, check placement\n"
                  "d < 0.8 NOT separable -> fix layer 1 or reconsider paradigm"))

    # 2.5 statistical significance ---------------------------------------
    print("\n2.5 Is the difference statistically real?")
    t, pv = stats.ttest_ind(p0[:, bi], p1[:, bi])
    res.append(_v(PASS if pv < 0.05 else FAIL,
                  f"t-test on best channel: p = {pv:.4g}",
                  "p < 0.05 means the difference is unlikely to be chance"))

    # 2.6 idle separability ----------------------------------------------
    print("\n2.6 Can rest be told apart from imagery?")
    if len(idle) >= 5:
        fi = sosfiltfilt(sos, idle, axis=-1)
        pi = np.log(np.var(fi, axis=2) + 1e-12)
        act = np.concatenate([p0[:, bi], p1[:, bi]])
        pooled = np.sqrt((act.var() + pi[:, bi].var()) / 2) + 1e-12
        d_idle = abs(act.mean() - pi[:, bi].mean()) / pooled
        res.append(_v(PASS if d_idle > 0.8 else WARN,
                      f"idle vs active Cohen d = {d_idle:.2f}",
                      "needed so the system can output NO_ACTION; "
                      "low d means it cannot tell resting from trying"))
    else:
        res.append(_v(WARN, "idle test skipped", "no idle windows in recording"))
    return res


# =====================================================================
# LAYER 3 — MODEL
# =====================================================================
def layer3(user, n_splits=4, n_perm=200):
    """Can a classifier learn the difference, and is the score real?

    Covers: leakage-safe CV, significance vs chance, confidence intervals,
    pipeline comparison, learning curve, and cross-session stability.
    """
    print("\n" + "=" * 68)
    print("LAYER 3 · OFFLINE MODEL ACCURACY")
    print("=" * 68)
    res = []

    from ML.train_local import load_user_data
    from ML.evaluate import grouped_folds
    from ML.mi_advanced import PIPELINES

    d = load_user_data(user)
    X, y, g = d["X"], d["y"], d["groups"]
    fs = d["fs"]
    print(f"\n  {len(X)} trials, {len(np.unique(g))} blocks, "
          f"{len(d['files'])} session(s)")

    # 3.1 enough independent blocks ---------------------------------------
    print("\n3.1 Enough independent recording blocks for honest CV?")
    bpc = min(len(np.unique(g[y == c])) for c in (0, 1))
    res.append(_v(PASS if bpc >= 3 else FAIL,
                  f"{bpc} block(s) per class",
                  "need >=3 separate blocks per class, otherwise CV trains and\n"
                  "tests on single blocks and the score is meaningless\n"
                  "(observed: 100% and 26.6% from exactly this situation)"))

    # 3.2 grouped CV for every pipeline -----------------------------------
    print("\n3.2 Grouped cross-validation (no leakage)")
    print(f"      {'pipeline':<14}{'accuracy':>20}")
    scores = {}
    for name, fac in PIPELINES.items():
        accs = []
        for tr, te in grouped_folds(g, y, n_splits=n_splits):
            try:
                p = fac(sample_freq=fs); p.fit(X[tr], y[tr])
                accs.append(float(np.mean(p.predict(X[te]) == y[te])))
            except Exception:
                pass
        if accs:
            scores[name] = accs
            print(f"      {name:<14}{np.mean(accs):>12.1%} +/-{np.std(accs):<6.1%}")
    if not scores:
        return res + [_v(FAIL, "no pipeline trained", "check the recording")]

    best = max(scores, key=lambda k: np.mean(scores[k]))
    acc = float(np.mean(scores[best]))
    res.append(_v(PASS if acc > 0.65 else (WARN if acc > 0.55 else FAIL),
                  f"best pipeline {best} at {acc:.1%}",
                  "chance = 50%; >65% is the usable threshold"))

    # 3.3 significance vs chance ------------------------------------------
    print("\n3.3 Is the accuracy significantly above chance?")
    n_test = int(len(X) / n_splits)
    n_correct = int(round(acc * n_test))
    bt = stats.binomtest(n_correct, n_test, 0.5, alternative="greater")
    res.append(_v(PASS if bt.pvalue < 0.05 else FAIL,
                  f"binomial test p = {bt.pvalue:.4g}",
                  f"{n_correct}/{n_test} correct. p<0.05 means the result is\n"
                  "unlikely to come from a coin flip"))

    # 3.4 permutation test -------------------------------------------------
    print(f"\n3.4 Permutation test ({n_perm} label shuffles)")
    null = []
    rng = np.random.default_rng(0)
    fac = PIPELINES[best]
    for _ in range(n_perm):
        yp = rng.permutation(y)
        a = []
        for tr, te in grouped_folds(g, yp, n_splits=n_splits):
            try:
                p = fac(sample_freq=fs); p.fit(X[tr], yp[tr])
                a.append(float(np.mean(p.predict(X[te]) == yp[te])))
            except Exception:
                pass
        if a:
            null.append(np.mean(a))
    if null:
        pv = (np.sum(np.array(null) >= acc) + 1) / (len(null) + 1)
        res.append(_v(PASS if pv < 0.05 else FAIL,
                      f"permutation p = {pv:.4g}",
                      f"null distribution mean {np.mean(null):.1%}, "
                      f"95th pct {np.percentile(null,95):.1%}\n"
                      "this is the strictest test: it rebuilds the whole\n"
                      "pipeline on shuffled labels"))

    # 3.5 bootstrap CI ----------------------------------------------------
    print("\n3.5 Confidence interval (bootstrap)")
    a = np.array(scores[best])
    boot = [np.mean(rng.choice(a, len(a), replace=True)) for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    res.append(_v(PASS if lo > 0.5 else WARN,
                  f"{acc:.1%} (95% CI {lo:.1%} - {hi:.1%})",
                  "report the interval, not the point estimate;\n"
                  "if the lower bound touches 50% the result is not solid"))

    # 3.6 learning curve --------------------------------------------------
    print("\n3.6 Learning curve — is more data still helping?")
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = max(2, int(len(np.unique(g)) * frac))
        keep = np.isin(g, np.unique(g)[:k])
        a2 = []
        for tr, te in grouped_folds(g[keep], y[keep], n_splits=min(3, k)):
            try:
                p = fac(sample_freq=fs)
                p.fit(X[keep][tr], y[keep][tr])
                a2.append(float(np.mean(p.predict(X[keep][te]) == y[keep][te])))
            except Exception:
                pass
        if a2:
            print(f"      {int(frac*100):>3}% of blocks -> {np.mean(a2):.1%}")
    res.append(_v(PASS, "learning curve computed",
                  "still rising = record more; flat = at the ceiling"))

    # 3.7 cross-session -----------------------------------------------------
    print("\n3.7 Cross-session stability")
    if len(d["files"]) >= 2:
        res.append(_v(WARN, "manual check required",
                      "train on session 1, test on session 2.\n"
                      "Within ~10 points = electrodes are repeatable;\n"
                      "a big drop means recalibration is needed every session"))
    else:
        res.append(_v(WARN, "only one session",
                      "record on a second day to test repeatability"))
    return res


# =====================================================================
# LAYER 4 / 5 — protocols (require a live human)
# =====================================================================
def layer4_protocol():
    print("\n" + "=" * 68)
    print("LAYER 4 · ONLINE PERFORMANCE   (live session required)")
    print("=" * 68)
    print("""
4.1 Cued online accuracy
    Run:  python EEG/data_record.py --file realistic_ui --trials 20 --cue 6
    Metric: proportion of trials where the cued button was selected first
    PASS: >= 70%
    Catches: model works offline but not with feedback or fatigue

4.2 False positives per minute of IDLE      <-- the headline metric
    Protocol: wear the headset, do nothing for 5 minutes.
              Rest, read, look around, blink normally.
    Metric: commands emitted per minute
    PASS: < 0.5 / min
    Catches: threshold too low. A forced-argmax classifier measured 100%
             accuracy on attentive trials while firing on 141 of 141 idle
             windows - accuracy alone would have called that a success.

4.3 Time to selection
    Metric: seconds from cue to correct selection, including retries
    Report: median and 90th percentile
    Catches: technically accurate but unusably slow

4.4 Information Transfer Rate (Wolpaw ITR)
    Metric: bits/min, combining accuracy and speed
    Catches: a system that trades one for the other without net gain

4.5 Blink and artifact rejection
    Protocol: 20 deliberate blinks, 10 jaw clenches, 10 head turns during rest
    PASS: >= 90% flagged, 0 producing a command
    Catches: artifacts being read as intent

4.6 Error recovery
    Protocol: after a wrong selection, can the user correct within 2 attempts?
    Catches: a UI with no way back

4.7 Sustained performance (drift)
    Protocol: repeat 4.1 at minute 0, 10 and 20 of one session
    PASS: accuracy drop < 15 points
    Catches: electrode drying, fatigue, gel drift
""")


def layer5_protocol():
    print("\n" + "=" * 68)
    print("LAYER 5 · USABILITY AND PRODUCT   (participants required)")
    print("=" * 68)
    print("""
5.1 Task completion
    Task: select 5 specific actions in the game HUD in a given order
    Metrics: completion rate, errors, total time
    PASS: >= 80% completion unaided

5.2 Subjective workload
    Instrument: NASA-TLX, or a 1-5 fatigue scale after each block
    PASS: mental demand <= 3/5 after 10 minutes
    Catches: works for 2 minutes, exhausting after 10. No objective
             metric captures this, and for MI it is the usual limit.

5.3 Learning effect
    Protocol: same task on 3 separate days
    Expected: time-to-selection improves; BCI skill is trainable
    Catches: a system only the developer can drive

5.4 Multi-user validation
    Participants: 3-5, each with their own calibration
    Report: per-user accuracy and the spread
    Catches: a single-subject result presented as a general library.
             ~15-30% of people show little usable MI ("BCI illiteracy"),
             so a 3-subject study may legitimately include a failure.

5.5 Developer integration
    Protocol: someone unfamiliar builds a brain-controlled button using
              only the public API and the README
    PASS: working app without reading library source

5.6 Installation
    Run: pip install into a clean venv, then import bci_sdk
    PASS: imports and runs with no source checkout
    CURRENT STATUS: FAILS - the wheel omits EEG/ and pygame_lib/

5.7 Safety and comfort
    Checks: session length limits, electrode pressure, skin condition after
            removal, clear stop instruction
""")


def main():
    ap = argparse.ArgumentParser(description="Layered BCI diagnostics")
    ap.add_argument("--layer", type=int, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--recording", default=None)
    ap.add_argument("--user", default=None)
    ap.add_argument("--perm", type=int, default=200)
    a = ap.parse_args()

    layers = [1, 2, 3, 4, 5] if a.all else ([a.layer] if a.layer else [])
    if not layers:
        ap.print_help()
        return 1

    allres = []
    for L in layers:
        try:
            if L in (1, 2):
                pat = a.recording or (ROOT / "profiles" / str(a.user or "") /
                                      "calib_*.npz")
                X, labels, fs, montage, fn = _load(pat)
                print(f"\nrecording: {fn}   {X.shape}  fs={fs}  montage={montage}")
                allres += layer1(X, labels, fs, montage) if L == 1 else \
                          layer2(X, labels, fs, montage)
            elif L == 3:
                if not a.user:
                    print("layer 3 needs --user"); continue
                allres += layer3(a.user, n_perm=a.perm)
            elif L == 4:
                layer4_protocol()
            else:
                layer5_protocol()
        except FileNotFoundError as e:
            print(f"\n  cannot run layer {L}: {e}")
        except Exception as e:
            print(f"\n  layer {L} error: {type(e).__name__}: {e}")

    if allres:
        print("\n" + "=" * 68)
        n_f = allres.count(FAIL); n_w = allres.count(WARN); n_p = allres.count(PASS)
        print(f"SUMMARY: {n_p} pass, {n_w} warn, {n_f} fail")
        if n_f:
            print("Fix the FAILs at the lowest layer first - higher layers")
            print("computed on broken data are meaningless.")
        print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
