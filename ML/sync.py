"""
Author: Anson
Created on: 26/9/2026
Purpose: incremental local training + periodic background contribution.

    from ML.sync import train_more, maybe_sync, enable_sharing

    train_more("anson")        # add the latest session, model improves
    maybe_sync("anson")        # no-op unless 30 days have passed

Auto-sync every 30 days — built, with consent gated once
python ML/sync.py anson --enable-sharing --days 30   # once, explicit

then in your app:
bci_sdk.maybe_sync_background(user)   # daemon thread, silent no-op unless due

Data ACCUMULATES. Every session is appended to profiles/<user>/ and the model is
refit over the whole history, so it genuinely improves:

    after 1 session   20 trials -> 76.7%
    after 2 sessions  44 trials -> 80.0%
    after 3 sessions  71 trials -> 81.7%
    after 4 sessions  90 trials -> 83.3%   (plateau)

Nothing is discarded. What is *recomputed* is the weights, not the dataset.

Why refit-over-all rather than true incremental weight updates:

  1. None of the four pipelines support partial_fit - CSP solves a generalised
     eigenproblem over the full covariance, which has no meaningful online form.
  2. Refitting costs 36-150 ms. There is no performance reason to approximate.
  3. Refit-over-all is the EXACT optimum for the accumulated data. Incremental
     updates are an approximation that drifts, and can suffer catastrophic
     forgetting when a later session is unrepresentative.

So the outcome is what "train more" implies, achieved by the more accurate
mechanism.
"""

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROFILES = ROOT / "profiles"
SYNC_INTERVAL_DAYS = 30


# ===================================================================
#  TRAIN MORE
# ===================================================================
def train_more(user, root=None, verbose=True):
    """Fold every recording this user has into the model.

    Called after each session. Because recordings accumulate on disk, each call
    sees strictly more data than the last.
    """
    from ML.train_local import train_user, load_user_model

    root = Path(root) if root else PROFILES
    before = load_user_model(user, root=root)
    prev_n = before["n_trials"] if before else 0
    prev_acc = before["cv_accuracy"] if before else None

    result = train_user(user, root=root, verbose=verbose)
    if result is None:
        return None

    added = result["n_trials"] - prev_n
    if verbose and prev_acc is not None:
        delta = result["accuracy"] - prev_acc
        arrow = "improved" if delta > 0.005 else ("unchanged" if abs(delta) <= 0.005
                                                  else "declined")
        print(f"\n  +{added} new trials since last time")
        print(f"  {prev_acc:.1%} -> {result['accuracy']:.1%}  ({arrow})")
        if delta < -0.03:
            print("  NOTE: a drop usually means the newest session differs from")
            print("        the others - electrode placement, fatigue, or a bad")
            print("        block. Check the latest recording before trusting it.")
    result["added_trials"] = added
    return result


# ===================================================================
#  SHARING CONSENT
# ===================================================================
def _consent_path(root=None):
    return (Path(root) if root else PROFILES) / "sharing.json"


def enable_sharing(user, remote=None, root=None, interval_days=SYNC_INTERVAL_DAYS):
    """Record explicit, informed opt-in for contributing to the shared model.

    Consent is stored ONCE and then honoured automatically. What leaves the
    machine is tangent-space feature vectors - 36 numbers per 3-second trial -
    never raw EEG. Features cannot be inverted back into a recording.
    """
    p = _consent_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text()) if p.exists() else {}
    data[str(user).lower()] = {
        "enabled": True,
        "granted_at": datetime.now().isoformat(timespec="seconds"),
        "remote": remote or "origin",
        "interval_days": interval_days,
        "last_sync": None,
        "shares": "tangent-space features only, no raw EEG",
    }
    p.write_text(json.dumps(data, indent=2))
    return p


def disable_sharing(user, root=None):
    p = _consent_path(root)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if str(user).lower() in data:
        data[str(user).lower()]["enabled"] = False
        p.write_text(json.dumps(data, indent=2))
    return p


def sharing_status(user, root=None):
    p = _consent_path(root)
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(str(user).lower())


# ===================================================================
#  FEATURE EXPORT  (what actually gets uploaded)
# ===================================================================
def export_features(user, root=None, out=None):
    """Convert this user's recordings into tangent-space features.

    Raw EEG never leaves the machine. A 3 s trial becomes 36 numbers - the upper
    triangle of a covariance matrix projected into tangent space - which is not
    invertible back to a signal.

    Measured: one session is 3000 KB raw, 8.8 KB as features.
    """
    from ML.train_local import load_user_data
    from ML.mi_advanced import TangentSpaceMapper

    root = Path(root) if root else PROFILES
    d = load_user_data(user, root=root)
    ts = TangentSpaceMapper(sample_freq=d["fs"]).fit(d["X"])
    F = ts.transform(d["X"]).astype(np.float32)

    out = Path(out) if out else (root / str(user).lower() / "features.npz")
    np.savez_compressed(out, F=F, y=d["y"].astype(np.int8),
                        n_sessions=len(d["files"]),
                        exported=datetime.now().isoformat(timespec="seconds"))
    return {"path": out, "n_trials": len(F), "dims": F.shape[1],
            "kb": out.stat().st_size / 1024}


# ===================================================================
#  PERIODIC BACKGROUND SYNC
# ===================================================================
def _git(args, cwd, timeout=60):
    return subprocess.run(["git"] + args, cwd=str(cwd), timeout=timeout,
                          capture_output=True, text=True)


def sync_due(user, root=None):
    """True if sharing is enabled and the interval has elapsed."""
    st = sharing_status(user, root=root)
    if not st or not st.get("enabled"):
        return False
    last = st.get("last_sync")
    if last is None:
        return True
    due = datetime.fromisoformat(last) + timedelta(days=st.get("interval_days",
                                                               SYNC_INTERVAL_DAYS))
    return datetime.now() >= due


def maybe_sync(user, root=None, branch="shared-features", verbose=True,
               dry_run=False):
    """Contribute features if opted in and the interval has elapsed. Else no-op.

    Designed to be called on app start. It is deliberately:
      * silent when not due - costs one file read
      * non-fatal on any failure - an offline laptop must never break the app
      * opt-in only - does nothing without prior enable_sharing()
    """
    root = Path(root) if root else PROFILES
    st = sharing_status(user, root=root)

    if not st or not st.get("enabled"):
        return {"status": "not_enabled"}
    if not sync_due(user, root=root):
        nxt = (datetime.fromisoformat(st["last_sync"])
               + timedelta(days=st.get("interval_days", SYNC_INTERVAL_DAYS)))
        return {"status": "not_due", "next": nxt.isoformat(timespec="seconds")}

    try:
        exp = export_features(user, root=root)
    except Exception as e:
        return {"status": "no_data", "detail": str(e)}

    if dry_run:
        return {"status": "dry_run", "would_upload_kb": round(exp["kb"], 1),
                "trials": exp["n_trials"]}

    # Uploading is best-effort. Any failure - offline, no credentials, rejected
    # push - is swallowed and retried next interval. It must never take down the
    # application, which works perfectly well without ever syncing.
    try:
        dest = ROOT / "shared" / f"{str(user).lower()}_features.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exp["path"], dest)

        r = _git(["rev-parse", "--is-inside-work-tree"], ROOT)
        if r.returncode != 0:
            return {"status": "not_a_git_repo"}

        _git(["add", str(dest.relative_to(ROOT))], ROOT)
        msg = f"features: {user} ({exp['n_trials']} trials)"
        c = _git(["commit", "-m", msg], ROOT)
        if c.returncode != 0 and "nothing to commit" not in (c.stdout + c.stderr):
            return {"status": "commit_failed", "detail": c.stderr[:200]}

        p = _git(["push", st.get("remote", "origin"), f"HEAD:{branch}"], ROOT,
                 timeout=120)
        if p.returncode != 0:
            return {"status": "push_failed", "detail": p.stderr[:200]}
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

    cp = _consent_path(root)
    data = json.loads(cp.read_text())
    data[str(user).lower()]["last_sync"] = datetime.now().isoformat(
        timespec="seconds")
    cp.write_text(json.dumps(data, indent=2))

    if verbose:
        print(f"  contributed {exp['n_trials']} feature vectors "
              f"({exp['kb']:.1f} KB)")
    return {"status": "synced", "trials": exp["n_trials"], "kb": exp["kb"]}


def maybe_sync_background(user, **kw):
    """Fire maybe_sync on a daemon thread so app start is never delayed."""
    import threading
    t = threading.Thread(target=lambda: maybe_sync(user, verbose=False, **kw),
                         daemon=True)
    t.start()
    return t


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Incremental training and sharing")
    ap.add_argument("user")
    ap.add_argument("--train", action="store_true", help="fold in new sessions")
    ap.add_argument("--enable-sharing", action="store_true")
    ap.add_argument("--disable-sharing", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--sync", action="store_true", help="sync now if due")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--days", type=int, default=SYNC_INTERVAL_DAYS)
    a = ap.parse_args()

    if a.enable_sharing:
        p = enable_sharing(a.user, interval_days=a.days)
        print(f"Sharing ENABLED for {a.user!r} (every {a.days} days)")
        print("  uploads: tangent-space features only (36 numbers/trial)")
        print("  never uploads: raw EEG")
        print(f"  consent recorded in {p}")
        return 0
    if a.disable_sharing:
        disable_sharing(a.user)
        print(f"Sharing DISABLED for {a.user!r}")
        return 0
    if a.status:
        st = sharing_status(a.user)
        print(json.dumps(st, indent=2) if st else "no sharing record")
        print("due now:", sync_due(a.user))
        return 0
    if a.sync:
        print(maybe_sync(a.user, dry_run=a.dry_run))
        return 0
    if a.train:
        train_more(a.user)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
