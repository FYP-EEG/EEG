"""
Author: Anson
Created on: 6/10/2026
Purpose: the whole test suite, in one file, organised by failure layer.

    python tests/test_bci.py              # everything
    python tests/test_bci.py --layer 3    # one layer

WHY LAYERS AND NOT FEATURES
---------------------------
A BCI can fail at five independent layers and every one of them presents
identically: nothing happens when the user tries. Grouping the tests by layer
means a failure already tells you where to look, and that the layers must be
read IN ORDER - a layer-3 result computed on layer-1-broken data is noise.

TWO MODES, SAME FIVE LAYERS
---------------------------
    python tests/test_bci.py                     code tests, all layers
    python tests/test_bci.py --layer 3           code tests, one layer
    python tests/test_bci.py --user anson        diagnostics on real data
    python tests/test_bci.py --recording "p/*.npz"   same, explicit file
    python tests/test_bci.py --user anson --code     both

CODE mode fabricates its own data and asks "is the software correct?" - it is
deterministic, needs no hardware, and a failure is a bug. DATA mode reads an
actual recording and asks "is THIS person\'s signal usable?" - a failure there
is not a bug, it is a finding, which is why it reports PASS/WARN/FAIL verdicts
rather than assertions and does not set a non-zero exit code on its own.

Layer 2 in DATA mode is the go/no-go gate: a recording that cannot clear it
will not be rescued by a better classifier.
"""
import argparse
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.signal import butter, sosfiltfilt, welch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# The original suites each pushed their own paths; several modules are still
# imported bare (stream_bridge, stimulus) because EEG/ and pygame_lib/ have no
# __init__.py. Declared once here instead of ten times.
for _sub in ("", "EEG", "pygame_lib"):
    _p = str(ROOT / _sub) if _sub else str(ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

ok = 0
fail = []
_layer_start = 0


def _record(n, c, detail=""):
    global ok
    if c:
        ok += 1
        print(f"  PASS  {n}")
    else:
        fail.append(n)
        print(f"  FAIL  {n}" + (f"\n        {detail}" if detail else ""))


def check(n, c, detail=""):
    """Standard order, used by nine of the ten original suites."""
    _record(n, c, detail)


def check_cond_first(c, n, detail=""):
    """Reversed order, as the realtime section was originally written.

    Calls _record directly. Routing it through check() would recurse, because
    `check` is rebound to THIS function for the duration of that section.
    """
    _record(n, c, detail)


def banner(title, why):
    global _layer_start
    _layer_start = ok + len(fail)
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    for line in why.split("\n"):
        print("  " + line)
    print()


def layer_total(title):
    n = ok + len(fail) - _layer_start
    print(f"\n  -- {title.split(' - ')[0]}: {n} checks")


def layer1():
    global check
    banner('LAYER 1 - ACQUISITION AND SIGNAL QUALITY', 'Can the system get clean, correctly-timed samples off the hardware at all?')

    # --------------------------------------------------------------
    # from test_step4_bridge.py
    # --------------------------------------------------------------
    import sys, time, glob
    from pathlib import Path
    import numpy as np

    from stream_bridge import (RingBuffer, StreamBridge, SyntheticBackend,
                               ReplayBackend)          # noqa
    from ML.bci_engine import BCIEngine                # noqa
    from ML.hybrid_classifier import HybridSSVEPClassifier  # noqa

    print("\n== RingBuffer ==")
    rb = RingBuffer(3, 100)
    check("empty buffer returns None", rb.latest(10) is None)
    rb.write(np.ones((3,50)))
    check("partial fill: not enough samples yet", rb.latest(60) is None)
    check("partial fill: enough samples", rb.latest(50).shape == (3,50))
    rb.write(np.full((3,80), 2.0))
    got = rb.latest(100)
    check("wraps around capacity", got.shape == (3,100))
    check("wrap keeps newest values", np.allclose(got[:, -80:], 2.0))
    check("len caps at capacity", len(rb) == 100)
    rb2 = RingBuffer(2, 50)
    rb2.write(np.arange(200).reshape(1,-1).repeat(2,0))
    check("oversized write keeps newest", rb2.latest(50)[0,-1] == 199)
    try:
        rb2.write(np.ones((5,10))); check("rejects wrong channel count", False)
    except ValueError:
        check("rejects wrong channel count", True)

    print("\n== RingBuffer reconstructs a continuous signal exactly ==")
    fs, total = 250, 4000
    sig = np.vstack([np.sin(2*np.pi*15*np.arange(total)/fs)]*2)
    rb3 = RingBuffer(2, 2000); i=0; rng=np.random.default_rng(0)
    while i < total:
        n = min(int(rng.integers(1,97)), total-i)
        rb3.write(sig[:, i:i+n]); i += n
    check("ragged chunks reconstruct exactly",
          np.allclose(rb3.latest(750), sig[:, total-750:total]))

    print("\n== Windowing cadence & overlap ==")
    be = SyntheticBackend(seed=2); be.set_active(0)
    br = StreamBridge(backend=be, window=750, hop=250, max_pending=100)
    br.start(); time.sleep(6.0); br.stop()
    r = br.report()
    # 6s elapsed - 3s priming = 3s steady-state at 1 window/s
    expected_w = int((6.0 - 750/250) * 250/250)
    check(f"windows produced ({r['windows_made']}, expected ~{expected_w})",
          r["windows_made"] >= expected_w - 1)
    check(f"steady-state rate ~{r['expected_windows_per_s']:.1f}/s (got {r['windows_per_s']:.2f})",
          abs(r["windows_per_s"] - r["expected_windows_per_s"]) < 0.35)
    check(f"effective fs within 3% (err {r['fs_error_pct']:+.2f}%)",
          abs(r["fs_error_pct"]) < 3.0)
    ws = br.poll_all()
    check("all windows are (8,750)", all(w.shape == (8,750) for w in ws))
    if len(ws) >= 2:
        a, b = ws[0], ws[1]
        check("consecutive windows overlap by window-hop",
              np.allclose(a[:, 250:], b[:, :500]))

    print("\n== Backpressure: slow consumer drops oldest, never blocks ==")
    be2 = SyntheticBackend(seed=3)
    br2 = StreamBridge(backend=be2, window=750, hop=250, max_pending=2)
    br2.start(); time.sleep(6.5)
    pend = len(br2._pending)
    br2.stop()
    check("pending never exceeds max_pending", pend <= 2)
    check("drops were counted", br2.report()["windows_dropped"] > 0)
    w = br2.poll()
    check("still serves a valid window after drops", w is not None and w.shape==(8,750))

    print("\n== poll() is non-blocking ==")
    be3 = SyntheticBackend(seed=4)
    br3 = StreamBridge(backend=be3); br3.start()
    t0=time.time(); [br3.poll() for _ in range(2000)]; dt=time.time()-t0
    check(f"2000 polls took {dt*1000:.0f}ms (non-blocking)", dt < 0.5)
    check("bridge thread is alive", br3.is_running)
    br3.stop()
    check("stops cleanly", not br3.is_running)

    print("\n== Live-path classification (the whole point of Step 4) ==")
    clf = HybridSSVEPClassifier()
    for tgt in (0,1):
        be4 = SyntheticBackend(seed=1); be4.set_active(tgt)
        br4 = StreamBridge(backend=be4); br4.start()
        w = br4.wait_for_window(timeout=8); br4.stop()
        sc = clf.fbcca_scores(clf.apply_filter(w))
        check(f"gaze target {tgt} classified correctly through the bridge",
              int(np.argmax(sc)) == tgt and sc.max() > clf.confidence_threshold)

    be5 = SyntheticBackend(seed=1); be5.set_idle(True)
    br5 = StreamBridge(backend=be5); br5.start()
    w = br5.wait_for_window(timeout=8); br5.stop()
    check("idle stays below threshold through the bridge",
          clf.fbcca_scores(clf.apply_filter(w)).max() < clf.confidence_threshold)

    print("\n== Phase continuity regression (bug found by Step 4) ==")
    be6 = SyntheticBackend(seed=5); be6.set_active(0); be6.start()
    chunks = []
    for _ in range(12):                       # accumulate >= 750 samples in many chunks
        time.sleep(0.3)
        c = be6.read_available()
        if c is not None:
            chunks.append(c)
        if sum(c.shape[1] for c in chunks) >= 900:
            break
    joined = np.concatenate(chunks, axis=1)
    check(f"gathered {len(chunks)} separate chunks", len(chunks) >= 3)
    sc = clf.fbcca_scores(clf.apply_filter(joined[:, -750:]))
    check(f"stitched chunks preserve SSVEP (conf {sc.max():.2f}, phase continuous)",
          int(np.argmax(sc)) == 0 and sc.max() > clf.confidence_threshold)

    print("\n== End-to-end with BCIEngine ==")
    be7 = SyntheticBackend(seed=1); be7.set_active(0)
    eng = BCIEngine(mode="SSVEP", debounce_window=3, refractory_ms=0)
    br7 = StreamBridge(backend=be7); br7.start()
    fired=[]; lat=[]
    end=time.time()+9
    while time.time()<end:
        w = br7.poll()
        if w is not None:
            t0=time.time(); d,f = eng.process_frame(w); lat.append((time.time()-t0)*1000)
            if f: fired.append(d)
        time.sleep(1/60)
    br7.stop()
    check(f"engine dispatched a command ({fired})", len(fired) >= 1)
    check("dispatched the correct target", all(d=="SSVEP_0" for d in fired))
    check(f"classify latency < hop budget (mean {np.mean(lat):.0f}ms vs 1000ms)",
          np.mean(lat) < 1000)
    check("no windows dropped at 60fps", br7.report()["windows_dropped"] == 0)

    print("\n== Blink through the live path ==")
    be8 = SyntheticBackend(seed=6); be8.set_active(0)
    eng8 = BCIEngine(mode="SSVEP", debounce_window=2, artifact_lockout_ms=800)
    br8 = StreamBridge(backend=be8); br8.start()
    br8.wait_for_window(timeout=8)
    be8.trigger_blink()
    seen=[]
    end=time.time()+4
    while time.time()<end:
        w=br8.poll()
        if w is not None: seen.append(eng8.process_frame(w)[0])
        time.sleep(1/60)
    br8.stop()
    check(f"blink detected in live stream ({seen})", "BLINK" in seen or "LOCKOUT" in seen)

    print("\n== ReplayBackend (real recorded EEG through the live path) ==")
    recs = sorted(glob.glob(str(ROOT/"dataset"/"calib_*.npz")))
    if recs:
        rp = ReplayBackend(recs[-1], speed=4.0)
        info = rp.info()
        check("replay reports duration", info["duration_s"] > 0)
        br9 = StreamBridge(backend=rp); br9.start()
        w = br9.wait_for_window(timeout=10); br9.stop()
        check("replay yields a valid window", w is not None and w.shape[0]==8)
        check("replay window has real signal variance", w is not None and w.std() > 0)
    else:
        print("  SKIP  no calibration recording found (run Step 3 first)")

    # --------------------------------------------------------------
    # from test_step5_stimulus.py
    # --------------------------------------------------------------
    import os, sys
    os.environ.setdefault("SDL_VIDEODRIVER","dummy")
    from pathlib import Path
    import numpy as np
    import pygame; pygame.init(); pygame.display.set_mode((200,200))
    from stimulus import (FlickerPlan, StimulusEngine, open_display,
                          suggest_frequencies, detect_refresh_rate)  # noqa

    print("\n== The bug: naive half-period rounding ==")
    def naive(f, refresh=60):
        hp = max(1, round(refresh/(2*f)))
        return refresh/(2*hp)
    check("naive rounding collapses 20Hz onto 15Hz @60Hz", naive(20)==15.0 and naive(15)==15.0)
    p15 = FlickerPlan(15.0, 60); p20 = FlickerPlan(20.0, 60)
    check("period-quantised 15Hz is exact", p15.actual==15.0 and p15.is_exact)
    check("period-quantised 20Hz is exact", p20.actual==20.0 and p20.is_exact)
    check("20Hz uses an asymmetric 3-frame period",
          p20.period_frames==3 and p20.on_frames==2 and not p20.is_symmetric)
    check("15/20 no longer collide", p15.period_frames != p20.period_frames)

    print("\n== FlickerPlan frame patterns ==")
    seq15 = [p15.is_on(f) for f in range(8)]
    seq20 = [p20.is_on(f) for f in range(6)]
    check(f"15Hz pattern 2on/2off {seq15[:4]}", seq15[:4]==[True,True,False,False])
    check(f"20Hz pattern 2on/1off {seq20[:3]}", seq20[:3]==[True,True,False])
    check("patterns repeat exactly by period",
          all(p15.is_on(f)==p15.is_on(f+4) for f in range(40)) and
          all(p20.is_on(f)==p20.is_on(f+3) for f in range(40)))
    check("duty cycles reported", abs(p15.duty-0.5)<1e-9 and abs(p20.duty-2/3)<1e-9)
    po = FlickerPlan(15.0, 60, phase_offset=2)
    check("phase offset shifts the pattern", po.is_on(0) != p15.is_on(0))

    print("\n== FFT proves the rendered fundamental ==")
    eng = StimulusEngine((15.0,20.0), 60.0)
    for i, want in enumerate((15.0, 20.0)):
        r = eng.spectrum_check(i, n_frames=1200)
        check(f"target {i}: FFT peak {r['fft_peak']:.2f}Hz == {want}Hz",
              r["matches"] and abs(r["fft_peak"]-want) < 0.1)

    print("\n== Non-integer refresh rates are reported honestly ==")
    e144 = StimulusEngine((15.0,20.0), 144.0)
    check("144Hz: 15Hz is NOT exact and says so", not e144.plans[0].is_exact)
    check("144Hz: reports actual 14.40Hz", abs(e144.plans[0].actual-14.4)<0.01)
    check("144Hz: error percentage computed", abs(e144.plans[0].error_pct+4.0)<0.1)
    e75 = StimulusEngine((15.0,20.0), 75.0)
    check("75Hz: 15Hz exact, 20Hz becomes 18.75Hz",
          e75.plans[0].is_exact and abs(e75.plans[1].actual-18.75)<0.01)

    print("\n== suggest_frequencies gives exactly-renderable targets ==")
    s60 = suggest_frequencies(60.0)
    check(f"60Hz suggestions {s60}", 15.0 in s60 and 20.0 in s60 and 30.0 in s60)
    check("all suggestions are exactly renderable @60",
          all(FlickerPlan(f,60).is_exact for f in s60))
    check("suggestions avoid the alpha band", all(f > 13.0 for f in s60))
    s144 = suggest_frequencies(144.0)
    # suggestions are rounded to 4dp for display, so allow sub-0.01Hz error
    check(f"144Hz suggestions are renderable {s144}",
          all(abs(FlickerPlan(f,144).error_hz) < 0.01 for f in s144))
    check("144Hz suggestions differ from 60Hz ones", set(s144) != set(s60))

    print("\n== V-synced display opens and stays blit-compatible ==")
    surf, info = open_display((320,240), refresh_hint=60.0)
    check("display opened", surf is not None)
    check("uses SCALED (not OPENGL)", "SCALED" in info["flags"] or info["flags"]=="DOUBLEBUF")
    tmp = pygame.Surface((10,10)); tmp.fill((255,0,0))
    try:
        surf.blit(tmp,(0,0)); blit_ok=True
    except Exception:
        blit_ok=False
    check("blit works (OPENGL would break this)", blit_ok)
    check("refresh reported", info["refresh"] > 0)

    print("\n== Frame counter & timing audit ==")
    eng2 = StimulusEngine((15.0,20.0), 60.0)
    for _ in range(200):
        pygame.display.flip(); eng2.tick()
    check("frame counter advances", eng2.frame==200)
    r = eng2.timing_report()
    check("timing report produced", r is not None and r["frames"]==200)
    check("reports jitter and drops", "jitter_pct" in r and "dropped" in r)
    eng2.reset()
    check("reset clears state", eng2.frame==0 and eng2.timing_report() is None)

    print("\n== Collision detection ==")
    ecol = StimulusEngine((20.0, 20.5), 60.0)   # both round to 3 frames
    periods = {p.period_frames for p in ecol.plans}
    check("detects two targets sharing a frame period", len(periods)==1)

    print("\n== Speller integration ==")
    from example.bci_speller import Speller  # noqa
    app = Speller(source="keyboard", flicker=True, refresh=60.0)
    check("speller opens a vsync display", app.display_info["refresh"]==60.0)
    check("speller exposes ACTUAL rendered freqs", app.actual_freqs==(15.0,20.0))
    before = [t.flicker_on for row in app.tiles for t in row]
    for _ in range(3):
        app._refresh_states(); app._apply_flicker(); app.draw()
    after = [t.flicker_on for row in app.tiles for t in row]
    check("tiles actually toggle", before != after)
    check("frame counter ticks with draw()", app.stim.frame >= 3)
    check("tiles carry their flicker frequency",
          all(t.freq in app.actual_freqs for row in app.tiles for t in row))

    print("\n== Classifier tracks ACTUAL, not requested, frequency ==")
    app2 = Speller(source="keyboard", flicker=True, refresh=144.0)
    check("144Hz speller reports shifted freqs",
          abs(app2.actual_freqs[0]-14.4)<0.01)
    from ML.bci_engine import BCIEngine  # noqa
    eng3 = BCIEngine(mode="SSVEP", target_freqs=app2.actual_freqs)
    check("engine reference signals match the rendered frequency",
          eng3.ssvep_clf.target_freqs == list(app2.actual_freqs))

    pygame.quit()
    layer_total('LAYER 1 - ACQUISITION AND SIGNAL QUALITY')


def layer2():
    global check
    banner('LAYER 2 - CALIBRATION AND SEPARABILITY', 'Is there a measurable difference between the two intentions, before any\nclassifier is involved, and can a per-user threshold be fitted?')

    # --------------------------------------------------------------
    # from test_step3_calibration.py
    # --------------------------------------------------------------
    import glob, json, sys, tempfile
    from pathlib import Path
    import numpy as np

    from ML.hybrid_classifier import HybridSSVEPClassifier, NO_ACTION  # noqa
    from ML.bci_engine import BCIEngine  # noqa
    from ML import calibration  # noqa
    import importlib.util  # noqa
    spec = importlib.util.spec_from_file_location("calrec", ROOT/"EEG"/"calibration_record.py")
    calrec = importlib.util.module_from_spec(spec); spec.loader.exec_module(calrec)

    print("\n== Protocol design ==")
    names = [b for b,_,_,_,_ in calrec.DEFAULT_PROTOCOL]
    check("has both SSVEP target blocks", "target_0" in names and "target_1" in names)
    check("has an idle block", "idle" in names)
    check("has a distraction block", "idle_distracted" in names)
    check("has an artifact/blink block", "artifact" in names)
    idle_s = sum(r*s for b,_,r,s,_ in calrec.DEFAULT_PROTOCOL if b.startswith("idle"))
    tgt_s  = sum(r*s for b,_,r,s,_ in calrec.DEFAULT_PROTOCOL if b.startswith("target"))
    check(f"idle time ({idle_s:.0f}s) exceeds target time ({tgt_s:.0f}s) - realistic",
          idle_s > tgt_s)
    total = sum(r*s for _,_,r,s,_ in calrec.DEFAULT_PROTOCOL)/60
    check(f"protocol is a reasonable length ({total:.1f} min)", 3 <= total <= 40)

    print("\n== Simulate backend ==")
    be = calrec.SimulateBackend(seed=0)
    be.start(); be.set_state("target_0", 0)
    w = be.read(750)
    check("emits (8,750) uV windows", w.shape == (8,750))
    be.set_state("artifact", 2); wa = be.read(750)
    check("blink block produces >100uV frontal peaks", np.max(np.abs(wa[[0,1]])) > 100)
    be.set_state("idle", -1); wi = be.read(750)
    check("idle block stays low-voltage frontally", np.max(np.abs(wi[[0,1]])) < 100)

    print("\n== Artifact-band regression (bug found by Step 3) ==")
    clf = HybridSSVEPClassifier()
    check("blink caught on RAW data", clf.artifact_detection(wa, is_raw=True))
    check("blink NOT caught after SSVEP highpass (why the bug existed)",
          not clf.artifact_detection(clf.apply_filter(wa), is_raw=False))
    lab,_,_ = clf.predict_proba(wa)
    check("predict_proba returns BLINK for a blink window", lab == "BLINK")
    check("idle window is not flagged as artifact", not clf.artifact_detection(wi, is_raw=True))

    print("\n== Recording -> analysis -> profile round trip ==")
    be2 = calrec.SimulateBackend(seed=7); be2.start()
    quick = [(b,c,1,min(s,6.0),i) for b,c,r,s,i in calrec.DEFAULT_PROTOCOL]
    X,y,labels,aborted = calrec.run_protocol(be2, None, quick, 250, 750, 250, verbose=False)
    check("recorded windows", len(X) > 0 and not aborted)
    check("labels cover all blocks", set(labels) == set(names))
    tmp = Path(tempfile.mkdtemp())
    p = calrec.save(X,y,labels,"TEST",250,"cyton8_ssvep",[15.0,20.0],tmp)
    check("npz saved", p.exists())
    Xl,yl,ll,fs,mont,tf = calibration.load(p)
    check("round-trips shape", Xl.shape == np.asarray(X,dtype=np.float32).shape)
    check("round-trips montage/fs", mont=="cyton8_ssvep" and fs==250)

    print("\n== Threshold fitting from the user's own idle data ==")
    prof = calibration.analyse(str(p), percentile=95.0, out=tmp/"prof.json")
    check("profile produced", prof is not None)
    check("threshold is subject-fitted (not the 0.15 default)",
          prof["confidence_threshold"] != 0.15)
    check("profile records idle minutes", prof["idle_minutes"] > 0)
    check("json written", (tmp/"prof.json").exists())

    print("\n== Engine loads the profile ==")
    eng = BCIEngine.from_profile(tmp/"prof.json")
    check("engine threshold matches profile",
          abs(eng.ssvep_clf.confidence_threshold - prof["confidence_threshold"]) < 1e-9)
    check("engine targets match profile", eng.ssvep_clf.target_freqs == [15.0,20.0])
    be3 = calrec.SimulateBackend(seed=11); be3.start(); be3.set_state("idle",-1)
    res = [eng.process_frame(be3.read(750)) for _ in range(15)]
    check("calibrated engine emits no commands while idle", not any(t for _,t in res))

    print("\n== ITR metric ==")
    check("ITR zero at chance", calibration.itr_bits_per_min(2, 0.5, 10) == 0.0)
    check("ITR positive when accurate", calibration.itr_bits_per_min(2, 0.95, 10) > 5)
    layer_total('LAYER 2 - CALIBRATION AND SEPARABILITY')


def layer3():
    global check
    banner('LAYER 3 - OFFLINE MODEL', 'Does a model trained on that data generalise, under leakage-safe CV?')

    # --------------------------------------------------------------
    # from test_step6_models.py
    # --------------------------------------------------------------
    import sys
    from pathlib import Path
    import numpy as np
    from ML.trca import trca_weights, corr2, TRCAClassifier, HybridTRCAClassifier  # noqa
    from ML.hybrid_classifier import NO_ACTION  # noqa
    from ML.mi_advanced import (CSP, FilterBankCSP, TangentSpaceMapper,
                                riemannian_mean, PIPELINES, _cov)  # noqa
    from ML.evaluate import infer_groups, subdivide_groups, grouped_folds  # noqa

    FS,N=250,750
    rng=np.random.default_rng(0)

    def ssvep_trials(freq,n,n_ch=6,occ=(2,3,4,5),snr=1.0,mix=None,phase=0.4):
        if mix is None: mix=np.eye(n_ch)
        out=[]
        for _ in range(n):
            t=np.arange(N)/FS; x=rng.standard_normal((n_ch,N))*2.0
            for ch in range(n_ch): x[ch]+=3.0*np.sin(2*np.pi*10.2*t+rng.uniform(0,6.28))
            for ch in occ:
                for h,a in ((1,1.0),(2,0.4)): x[ch]+=snr*a*np.sin(2*np.pi*h*freq*t+phase)
            out.append(mix@x)
        return np.array(out)

    print("\n== TRCA core ==")
    tr_in = ssvep_trials(15.0, 12)
    w = trca_weights(tr_in)
    check("returns one weight per channel", w.shape == (6,))
    check("filter is unit-norm", abs(np.linalg.norm(w)-1.0) < 1e-9)
    try:
        trca_weights(tr_in[:1]); check("rejects <2 trials", False)
    except ValueError: check("rejects <2 trials", True)
    try:
        trca_weights(np.zeros((5,6))); check("rejects wrong ndim", False)
    except ValueError: check("rejects wrong ndim", True)
    check("corr2 self-correlation is 1", abs(corr2(np.arange(10),np.arange(10))-1)<1e-9)
    check("corr2 anti-correlation is -1", abs(corr2(np.arange(10),-np.arange(10))+1)<1e-9)
    check("corr2 handles constant input", corr2(np.ones(10), np.arange(10)) == 0.0)

    print("\n== TRCA classifier ==")
    mix = np.eye(6)+0.4*rng.standard_normal((6,6))
    X = np.concatenate([ssvep_trials(15.0,20,mix=mix), ssvep_trials(20.0,20,mix=mix)])
    y = np.array([0]*20+[1]*20)
    clf = TRCAClassifier(target_freqs=(15.0,20.0), sample_freq=FS)
    clf.fit(X,y)
    check("fit marks fitted", clf.is_fitted and clf.n_train_==40)
    check("templates per band and class", len(clf.templates_)==3 and len(clf.templates_[0])==2)
    Xte = np.concatenate([ssvep_trials(15.0,15,mix=mix), ssvep_trials(20.0,15,mix=mix)])
    yte = np.array([0]*15+[1]*15)
    pred = np.array([int(np.argmax(clf.scores(x))) for x in Xte])
    acc = np.mean(pred==yte)
    check(f"held-out accuracy {acc:.0%} > chance", acc > 0.7)
    lab,conf,sc = clf.predict_proba(Xte[0])
    check("predict_proba returns triple", len(sc)==2 and isinstance(conf,float))
    clf.confidence_threshold = 99.0
    check("low confidence -> NO_ACTION", clf.predict_proba(Xte[0])[0] == NO_ACTION)
    clf.confidence_threshold = 0.2
    try:
        TRCAClassifier().fit(X[:1], y[:1]); check("errors clearly on too-few trials", False)
    except ValueError as e: check("errors clearly on too-few trials", "TRCA needs" in str(e))
    idle = ssvep_trials(15.0, 10, snr=0.0, mix=mix)
    t = clf.calibrate_threshold(idle, percentile=95)
    check("calibrate_threshold from idle", t > 0)

    print("\n== HybridTRCA falls back safely ==")
    h = HybridTRCAClassifier(target_freqs=(15.0,20.0), sample_freq=FS)
    check("untrained -> uses FBCCA", h.active == "fbcca")
    w8 = np.zeros((8,N)); w8[6]=np.sin(2*np.pi*15*np.arange(N)/FS)*8
    lab,_,_ = h.predict_proba(w8)
    check("works untrained", lab in (0,1,NO_ACTION,"BLINK"))
    X8 = np.concatenate([ssvep_trials(15.0,20,n_ch=8,occ=(6,7)),
                         ssvep_trials(20.0,20,n_ch=8,occ=(6,7))])
    h.fit(X8, y)
    check("trained -> switches to TRCA", h.active == "trca")
    h.prefer="fbcca"
    check("prefer overrides", h.active=="fbcca")
    blink = np.zeros((8,N)); blink[0,300:340]=400.0
    check("blink still detected", h.predict_proba(blink)[0]=="BLINK")

    print("\n== MI: CSP ==")
    def mi_trials(n, erd_ch, n_ch=8):
        out=[]
        for _ in range(n):
            x=rng.standard_normal((n_ch,N))
            for ch in range(n_ch): x[ch]*=1.0
            x[erd_ch]*=0.4                       # band-power reduction
            out.append(x)
        return np.array(out)
    Xm=np.concatenate([mi_trials(30,2), mi_trials(30,4)]); ym=np.array([0]*30+[1]*30)
    csp=CSP(n_components=4).fit(Xm,ym)
    F=csp.transform(Xm)
    check("CSP outputs n_components features", F.shape==(60,4))
    check("CSP features are finite", np.all(np.isfinite(F)))
    try:
        CSP().fit(Xm, np.array([0]*20+[1]*20+[2]*20)); check("CSP rejects 3 classes", False)
    except ValueError: check("CSP rejects 3 classes", True)
    fb=FilterBankCSP(sample_freq=FS).fit(Xm,ym)
    Ffb=fb.transform(Xm)
    check(f"FBCSP concatenates bands ({Ffb.shape[1]} features)", Ffb.shape[1]==6*4)

    print("\n== MI: Riemannian geometry ==")
    covs=np.array([_cov(x) for x in Xm[:20]])
    M=riemannian_mean(covs)
    check("Riemannian mean is symmetric", np.allclose(M,M.T,atol=1e-8))
    check("Riemannian mean is positive definite", np.all(np.linalg.eigvalsh(M)>0))
    single=riemannian_mean(covs[:1])
    check("mean of one matrix is itself", np.allclose(single,covs[0],atol=1e-6))
    ts=TangentSpaceMapper(sample_freq=FS).fit(Xm)
    T=ts.transform(Xm)
    check(f"tangent space dim = n(n+1)/2 = 36 ({T.shape[1]})", T.shape[1]==36)
    check("tangent features finite", np.all(np.isfinite(T)))

    print("\n== MI pipelines run end to end ==")
    for name,fac in PIPELINES.items():
        try:
            p=fac(sample_freq=FS); p.fit(Xm,ym)
            a=np.mean(p.predict(Xm)==ym)
            check(f"{name} fits and predicts (train acc {a:.0%})", 0.0<=a<=1.0)
        except Exception as e:
            check(f"{name} fits and predicts", False); print("      ",e)

    print("\n== Leakage-safe grouping (the important part) ==")
    labels=np.array(["target_0"]*9+["target_1"]*9+["idle"]*6)
    g=infer_groups(labels)
    check("groups follow label blocks", len(np.unique(g))==3)
    yb=np.array([0]*9+[1]*9+[-1]*6)
    sub,keep=subdivide_groups(g[:18],yb[:18],n_sub=3)
    check("subdivides into more groups", len(np.unique(sub))==6)
    check("drops overlapping seam windows", (~keep).sum()>0)
    gg=np.repeat(np.arange(8),5); yy=np.repeat([0,1],20)
    seen_test=[]
    for tr,te in grouped_folds(gg,yy,n_splits=4):
        check_overlap = set(gg[tr]) & set(gg[te])
        if check_overlap: seen_test.append(check_overlap)
    check("no group appears in both train and test", not seen_test)
    folds=list(grouped_folds(gg,yy,n_splits=4))
    check(f"produced {len(folds)} usable folds", len(folds)>=3)
    check("every fold has both classes in test",
          all(len(np.unique(yy[te]))==2 for _,te in folds))

    # --------------------------------------------------------------
    # from test_step8_datasets_models.py
    # --------------------------------------------------------------
    import os, sys, tempfile
    os.environ.setdefault("SDL_VIDEODRIVER","dummy")
    from pathlib import Path
    import numpy as np
    import bci_sdk
    from ML.datasets import (load_calibration_npz, pick_channels, resample_to,
                             crop_or_pad, combine, load_benchmark_mat)
    from ML.trca import TRCAClassifier
    from ML.hybrid_classifier import HybridSSVEPClassifier

    print("\n== Dataset loaders ==")
    X,y,meta = load_calibration_npz(str(ROOT/"dataset"/"calib_*.npz"))
    check("own recording loads (trials,ch,samples)", X.ndim==3 and X.shape[1:]==(8,750))
    check("labels balanced", set(np.unique(y))=={0,1})
    check("groups exposed for leakage-safe CV", len(np.unique(meta["groups"]))>=2)
    check("idle windows exposed", meta["idle"].shape[0]>0)
    check("artifact windows exposed", meta["artifact"].shape[0]>0)
    check("montage + freqs in meta", meta["montage"]=="cyton8_ssvep" and len(meta["target_freqs"])==2)

    check("pick_channels 64->8", pick_channels(np.zeros((4,64,750)),[53,54,55,56,57,60,61,62]).shape==(4,8,750))
    check("resample 1000->250", resample_to(np.zeros((2,8,1000)),1000,250).shape==(2,8,250))
    check("crop_or_pad up", crop_or_pad(np.zeros((2,8,600)),750).shape==(2,8,750))
    check("crop_or_pad down", crop_or_pad(np.zeros((2,8,900)),750).shape==(2,8,750))
    try:
        pick_channels(np.zeros((2,8,750)),[53]); check("pick_channels guards range", False)
    except ValueError: check("pick_channels guards range", True)

    Xc,yc,src = combine((np.zeros((5,8,750)),[0]*5),(np.ones((3,8,750)),[1]*3))
    check("combine public+own", Xc.shape==(8,8,750) and list(np.bincount(src))==[5,3])
    try:
        combine((np.zeros((2,8,750)),[0]*2),(np.zeros((2,64,750)),[1]*2))
        check("combine rejects shape mismatch", False)
    except ValueError: check("combine rejects shape mismatch", True)
    try:
        load_benchmark_mat("/nope/*.mat",freq_map={10:0}); check("missing .mat guarded", False)
    except FileNotFoundError: check("missing .mat guarded", True)
    try:
        load_benchmark_mat(str(ROOT/"dataset"/"calib_*.npz")); check("freq_map required", False)
    except (ValueError,Exception): check("freq_map required", True)

    print("\n== Model persistence ==")
    fb=HybridSSVEPClassifier(target_freqs=meta["target_freqs"],montage=meta["montage"])
    Xf=np.array([fb.apply_filter(x) for x in X])
    t=TRCAClassifier(target_freqs=tuple(meta["target_freqs"]),channels=fb.ssvep_channels)
    t.fit(Xf,y)
    tmp=Path(tempfile.mkdtemp())/"m.joblib"
    t.save(tmp)
    check("save() writes a file", tmp.exists())
    t2=TRCAClassifier.load(tmp)
    check("load() restores fitted state", t2.is_fitted)
    a=[int(np.argmax(t.scores(x))) for x in Xf]
    b=[int(np.argmax(t2.scores(x))) for x in Xf]
    check("round-trip predictions identical", a==b)
    check("hyper-params preserved",
          t2.target_freqs==t.target_freqs and t2.channels==t.channels
          and t2.ensemble==t.ensemble)
    try:
        TRCAClassifier().save(tmp); check("refuses to save unfitted", False)
    except RuntimeError: check("refuses to save unfitted", True)
    bad=Path(tempfile.mkdtemp())/"bad.joblib"
    import joblib; joblib.dump({"kind":"something_else"},bad)
    try:
        TRCAClassifier.load(bad); check("rejects foreign model file", False)
    except ValueError: check("rejects foreign model file", True)

    print("\n== Session uses a trained model ==")
    t.calibrate_threshold([fb.apply_filter(w) for w in meta["idle"][:40]])
    t.save(tmp)
    s=bci_sdk.BCISession(source="sim",model=str(tmp))
    check("session accepts model=", s.model is not None)
    check("engine classifier was swapped", "TRCA" in s.report()["classifier"])
    check("report records model path", s.report()["model"]==str(tmp))
    import time
    s.start(); e=time.time()+5
    while time.time()<e: s.poll(); time.sleep(1/60)
    s.stop()
    check("runs without error using trained model", s.bridge.report()["windows_made"]>=1)
    try:
        bci_sdk.BCISession(source="sim",model=str(tmp),target_freqs=(10.,12.))
        check("frequency mismatch raises", False)
    except bci_sdk.CalibrationError: check("frequency mismatch raises", True)
    try:
        bci_sdk.BCISession(source="sim",model="/nonexistent/m.joblib")
        check("missing model raises CalibrationError", False)
    except bci_sdk.CalibrationError: check("missing model raises CalibrationError", True)
    s2=bci_sdk.BCISession(source="sim")
    check("no model -> FBCCA fallback", "TRCA" not in s2.report()["classifier"])

    print("\n== SDK exports ==")
    for n in ["load_benchmark_mat","load_physionet_edf","load_calibration_npz",
              "pick_channels","combine","datasets"]:
        check(f"exports {n}", hasattr(bci_sdk,n))
    layer_total('LAYER 3 - OFFLINE MODEL')


def layer4():
    global check
    banner('LAYER 4 - ONLINE DECISION', 'With feedback and idle periods, does it emit the right command and stay\nsilent when it should? This is where false activations are caught.')

    # --------------------------------------------------------------
    # from test_gap4_idle.py
    # --------------------------------------------------------------
    import sys, time
    from pathlib import Path
    import numpy as np

    from ML.hybrid_classifier import HybridSSVEPClassifier, NO_ACTION  # noqa
    from ML.bci_engine import BCIEngine  # noqa
    from ML.sim_stream import SyntheticSSVEPStream  # noqa
    from ML import montage  # noqa

    print("\n== Montage / channel-map fix (blocker B1) ==")
    clf = HybridSSVEPClassifier()
    check("defaults to 8-ch Cyton montage", clf.ssvep_channels == [6,7])
    check("default targets are 15/20 Hz", clf.target_freqs == [15.0,20.0])
    check("sub-bands start above alpha", min(clf.sub_band_starts) >= 12.0)
    try:
        HybridSSVEPClassifier(montage="benchmark64").predict_proba(np.zeros((8,750)))
        check("64-ch config on 8-ch data raises clear error", False)
    except ValueError as e:
        check("64-ch config on 8-ch data raises clear error", "8" in str(e))

    print("\n== Confidence + NO_ACTION (blocker B4) ==")
    s = SyntheticSSVEPStream(seed=3)
    lab, conf, scores = clf.predict_proba(s.next_window())
    check("predict_proba returns 3-tuple", isinstance(scores, np.ndarray) and len(scores)==2)
    check("attentive window is confident", conf > clf.confidence_threshold)
    check("attentive window classified correctly", lab == 0)

    idle_stream = SyntheticSSVEPStream(seed=99, idle=True)
    idle_windows = [idle_stream.next_window() for _ in range(25)]
    idle_labels = [clf.predict_proba(w)[0] for w in idle_windows]
    n_noact = sum(1 for l in idle_labels if l == NO_ACTION)
    check(f"idle -> NO_ACTION ({n_noact}/25)", n_noact >= 20)

    print("\n== Threshold calibration from user's own idle data ==")
    c2 = HybridSSVEPClassifier()
    thr = c2.calibrate_threshold(idle_windows, percentile=95.0)
    check("calibrate_threshold sets a value", thr > 0)
    s2 = SyntheticSSVEPStream(seed=5)
    check("still detects target after calibration", c2.predict_proba(s2.next_window())[0] in (0,1))

    print("\n== BCIEngine: consecutive debounce + lockout + refractory ==")
    eng = BCIEngine(mode="SSVEP", debounce_window=3, refractory_ms=0)
    st = SyntheticSSVEPStream(seed=1)
    states=[]
    for i in range(3):
        states.append(eng.process_frame(st.next_window()))
    check("needs 3 consecutive windows", states[0][1] is False and states[1][1] is False)
    check("fires on 3rd consecutive window", states[2][1] is True and states[2][0]=="SSVEP_0")

    eng2 = BCIEngine(mode="SSVEP", debounce_window=3, refractory_ms=0)
    idle2 = SyntheticSSVEPStream(seed=42, idle=True)
    res = [eng2.process_frame(idle2.next_window()) for _ in range(20)]
    check("idle never dispatches a command", not any(t for _,t in res))
    check("idle reports NO_ACTION", sum(1 for d,_ in res if d=="NO_ACTION") >= 15)

    eng3 = BCIEngine(mode="SSVEP", debounce_window=2, artifact_lockout_ms=800, refractory_ms=0)
    blink = st.next_window(); blink = st.inject_blink(blink)
    d,t = eng3.process_frame(blink)
    check("blink detected", d=="BLINK" and not t)
    check("engine is locked out after blink", eng3.is_locked_out)
    d2,t2 = eng3.process_frame(st.next_window())
    check("output suppressed during lockout", d2=="LOCKOUT" and not t2)

    eng4 = BCIEngine(mode="SSVEP", debounce_window=2, refractory_ms=5000)
    st4 = SyntheticSSVEPStream(seed=2)
    fires = [eng4.process_frame(st4.next_window())[1] for _ in range(10)]
    check("refractory prevents key repeat (<=1 fire in 10)", sum(fires) <= 1)

    print("\n== BCIEngine: filter bypass fixed (blocker B3) ==")
    import inspect
    src = inspect.getsource(BCIEngine._classify)
    check("engine uses predict_proba (filtered path)", "predict_proba" in src)
    check("engine no longer calls raw cca_ssvep_detection", "cca_ssvep_detection" not in src)

    print("\n== End-to-end: realistic distracted session ==")
    eng5 = BCIEngine(mode="SSVEP", debounce_window=3, refractory_ms=1200)
    att = SyntheticSSVEPStream(seed=7); idl = SyntheticSSVEPStream(seed=8, idle=True)
    rng = np.random.default_rng(0)
    false_pos = 0; true_pos = 0
    for i in range(150):
        if i % 10 < 3:                       # 30% attentive, in runs (realistic gaze)
            att.set_active(0)
            d,t = eng5.process_frame(att.next_window())
            if t: true_pos += 1
        else:
            d,t = eng5.process_frame(idl.next_window())
            if t: false_pos += 1
    rep = eng5.report()
    print(f"    dispatched={rep['dispatched']} true={true_pos} false_idle={false_pos} "
          f"no_action_rate={rep['no_action_rate']:.1%}")
    check("zero false commands during idle/distraction", false_pos == 0)
    check("still issues real commands", true_pos >= 1)
    check("majority of session is NO_ACTION", rep["no_action_rate"] > 0.5)

    # --------------------------------------------------------------
    # from test_step10_realtime.py
    # --------------------------------------------------------------
    _std = check
    check = check_cond_first
    import shutil
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np


    from bci_sdk.selection import SelectionTree, LEFT, RIGHT          # noqa: E402
    from bci_sdk.runtime import (RealtimeDecider, ArtifactDetector,    # noqa: E402
                                 BLINK, LOCKOUT, BUILDING, REFRACTORY,
                                 REJECTED, DISPATCHED)
    from ML.local_model import LocalMIModel, NO_ACTION                 # noqa: E402



    class Btn:
        def __init__(self, name):
            self.name, self.fired = name, 0

        def trigger(self):
            self.fired += 1

        def __repr__(self):
            return f"Btn({self.name})"


    class FakeModel:
        """Scripted classifier so the gates can be tested without EEG."""

        def __init__(self, script):
            self.script = list(script)        # list of (command|None, confidence)
            self.i = 0
            self.sample_freq = 250
            self.montage = None

        def predict_command(self, window):
            v = self.script[min(self.i, len(self.script) - 1)]
            self.i += 1
            return v


    class Clock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            return self.t

        def advance(self, ms):
            self.t += ms


    # =====================================================================
    print("\n1. SelectionTree geometry")
    # =====================================================================
    btns = [Btn(f"b{i}") for i in range(5)]
    t = SelectionTree(btns)
    check(t.max_depth() == 3, "5 buttons need at most 3 decisions",
          f"got {t.max_depth()}")
    check(abs(t.mean_depth() - 2.4) < 1e-9, "mean depth is 2.4",
          f"got {t.mean_depth()}")
    check(sorted(t.depths()) == [2, 2, 2, 3, 3], "depths are 2,2,2,3,3",
          f"got {sorted(t.depths())}")

    for n, want in ((2, 1), (3, 2), (4, 2), (8, 3), (9, 4), (16, 4)):
        tt = SelectionTree([Btn(str(i)) for i in range(n)])
        check(tt.max_depth() == want, f"{n} buttons -> ceil(log2) = {want}",
              f"got {tt.max_depth()}")

    # =====================================================================
    print("\n2. Both classes select; every button is reachable")
    # =====================================================================
    reached = set()
    for target in range(5):
        tr = SelectionTree([Btn(f"b{i}") for i in range(5)])
        # walk down to `target` by choosing the half that contains it
        guard = 0
        while len(tr.candidates) > 1 and guard < 10:
            lo, _ = tr._split(tr.candidates)
            tr.decide(LEFT if target in lo else RIGHT)
            guard += 1
        reached.add(target)
    check(reached == set(range(5)), "all 5 buttons reachable by LEFT/RIGHT alone")

    tr = SelectionTree(btns)
    out = [tr.decide(RIGHT), tr.decide(RIGHT)]
    check(out[-1] is not None, "RIGHT,RIGHT resolves a leaf in 2 decisions")
    check(sum(b.fired for b in btns) == 1, "exactly one button fired",
          f"fired={[b.fired for b in btns]}")
    check(len(tr.candidates) == 5, "tree resets itself after a selection")

    tr2 = SelectionTree([Btn(f"b{i}") for i in range(8)])
    tr2.decide(LEFT)
    before = list(tr2.candidates)
    tr2.decide(RIGHT)
    check(tr2.candidates != before, "second decision narrows further")
    tr2.undo()
    check(tr2.candidates == before, "undo() steps back exactly one decision")

    tr3 = SelectionTree([Btn(f"b{i}") for i in range(5)])
    tr3.decide(RIGHT); tr3.decide(RIGHT)          # resolves and fires
    check(tr3.undo() == list(range(5)),
          "undo() after a selection cannot un-fire it; tree stays reset")

    # =====================================================================
    print("\n3. Gates: consecutive agreement")
    # =====================================================================
    clk = Clock()
    tree = SelectionTree([Btn(f"b{i}") for i in range(4)])
    d = RealtimeDecider(FakeModel([("LEFT", 0.9)] * 10), tree, clock=clk,
                        debounce_window=3, refractory_ms=0)
    states = [d.step(np.zeros((8, 750)))["state"] for _ in range(3)]
    check(states == [BUILDING, BUILDING, DISPATCHED],
          "3 agreeing windows required before dispatch", f"got {states}")

    clk = Clock()
    tree = SelectionTree([Btn(f"b{i}") for i in range(4)])
    d = RealtimeDecider(FakeModel([("LEFT", .9), ("RIGHT", .9), ("LEFT", .9),
                                   ("RIGHT", .9), ("LEFT", .9), ("RIGHT", .9)]),
                        tree, clock=clk, debounce_window=3, refractory_ms=0)
    states = [d.step(np.zeros((8, 750)))["state"] for _ in range(6)]
    check(DISPATCHED not in states, "alternating predictions never dispatch",
          f"got {states}")

    # =====================================================================
    print("\n4. Gates: confidence rejection (the reject option)")
    # =====================================================================
    clk = Clock()
    tree = SelectionTree([Btn(f"b{i}") for i in range(4)])
    d = RealtimeDecider(FakeModel([(None, 0.4)] * 20), tree, clock=clk,
                        refractory_ms=0)
    states = {d.step(np.zeros((8, 750)))["state"] for _ in range(20)}
    check(states == {REJECTED}, "sub-threshold windows are all rejected")
    check(d.stats["dispatched"] == 0, "no command from 20 rejected windows")
    check(d.report()["reject_rate"] == 1.0, "reject_rate reports 1.0")

    # a run broken by one rejection must restart from zero
    clk = Clock()
    tree = SelectionTree([Btn(f"b{i}") for i in range(4)])
    d = RealtimeDecider(FakeModel([("LEFT", .9), ("LEFT", .9), (None, .1),
                                   ("LEFT", .9), ("LEFT", .9)]),
                        tree, clock=clk, debounce_window=3, refractory_ms=0)
    states = [d.step(np.zeros((8, 750)))["state"] for _ in range(5)]
    check(DISPATCHED not in states, "a rejected window breaks the run",
          f"got {states}")

    # =====================================================================
    print("\n5. Gates: refractory period")
    # =====================================================================
    clk = Clock()
    tree = SelectionTree([Btn(f"b{i}") for i in range(4)])
    d = RealtimeDecider(FakeModel([("LEFT", 0.9)] * 20), tree, clock=clk,
                        debounce_window=1, refractory_ms=1200)
    r1 = d.step(np.zeros((8, 750)))
    clk.advance(500)
    r2 = d.step(np.zeros((8, 750)))
    clk.advance(800)                       # 1300 ms total
    r3 = d.step(np.zeros((8, 750)))
    check(r1["state"] == DISPATCHED, "first command dispatches")
    check(r2["state"] == REFRACTORY, "second at +500 ms is suppressed")
    check(r3["state"] == DISPATCHED, "third at +1300 ms dispatches")

    # =====================================================================
    print("\n6. Artifact detection runs on RAW data")
    # =====================================================================
    fs = 250
    tt = np.arange(750) / fs
    clean = np.random.default_rng(0).normal(0, 5, (8, 750))
    blink = clean.copy()
    # a blink: large, slow, frontal
    blink[0] += 220 * np.exp(-((tt - 1.5) ** 2) / (2 * 0.12 ** 2))
    blink[1] += 200 * np.exp(-((tt - 1.5) ** 2) / (2 * 0.12 ** 2))

    det = ArtifactDetector(channels=(0, 1), sample_freq=fs)
    check(not det.is_artifact(clean), "clean window is not flagged",
          f"peak {det.peak_uv(clean):.1f} uV")
    check(det.is_artifact(blink), "220 uV blink is flagged",
          f"peak {det.peak_uv(blink):.1f} uV")

    # the reason the check must come first: the motor band removes the blink
    from scipy.signal import butter, sosfiltfilt                      # noqa: E402
    sos = butter(4, [8, 30], btype="band", fs=fs, output="sos")
    after = float(np.max(np.abs(sosfiltfilt(sos, blink[:2], axis=-1))))
    check(after < 100.0,
          f"8-30 Hz filtering hides the blink ({det.peak_uv(blink):.0f} uV raw "
          f"-> {after:.0f} uV filtered, under the 100 uV threshold)")

    clk = Clock()
    tree = SelectionTree([Btn(f"b{i}") for i in range(4)])
    d = RealtimeDecider(FakeModel([("LEFT", 0.95)] * 20), tree, artifact=det,
                        clock=clk, debounce_window=1, artifact_lockout_ms=800,
                        refractory_ms=0)
    s1 = d.step(blink)
    s2 = d.step(clean)
    clk.advance(900)
    s3 = d.step(clean)
    check(s1["state"] == BLINK, "blink window reports BLINK")
    check(s2["state"] == LOCKOUT, "next window is locked out")
    check(s3["state"] == DISPATCHED, "output resumes after the 800 ms lockout")

    # =====================================================================
    print("\n7. Full loop: train -> save -> load -> decide")
    # =====================================================================
    tmp = Path(tempfile.mkdtemp())
    try:
        rng = np.random.default_rng(7)
        C, T = 8, 750

        def epoch(label, n):
            """ERD: class 1 suppresses power on ch2, class 0 on ch4."""
            out = []
            for _ in range(n):
                s = rng.normal(0, 10.0, (C, T))
                s[2] *= 0.55 if label == 1 else 1.0
                s[4] *= 1.0 if label == 1 else 0.55
                out.append(s)
            return out

        X, y, lab = [], [], []
        for _ in range(3):                    # 3 blocks so grouped CV is valid
            for cls in (0, 1):
                for w in epoch(cls, 8):
                    X.append(w); y.append(cls)
                    lab.append(f"target_{cls}")
        X = np.array(X)
        udir = tmp / "demo"
        udir.mkdir(parents=True)
        for b in range(3):
            sl = slice(b * 16, (b + 1) * 16)
            np.savez_compressed(udir / f"calib_{b}.npz", X=X[sl],
                                labels=np.array(lab[sl]), fs=250,
                                montage="cyton8_motor")

        from ML.train_local import train_user
        res = train_user("demo", root=tmp, verbose=False)
        check(res is not None, "train_user produced a model")
        check((udir / "model.joblib").exists(), "model.joblib written to disk")

        m = LocalMIModel.load("demo", root=tmp, confidence_threshold=0.6)
        check(m.source == "personal", "LocalMIModel loads the personal model")
        check(m.sample_freq == 250, "sample rate carried through")
        check(m.describe()["pipeline"] == res["pipeline"],
              "pipeline name round-trips")

        # scale invariance: _cov() trace-normalises, so overall gain cannot move
        # the features. This is why electrode gain drift between sessions is not a
        # source of error, and why the uV/volts question is moot.
        probe = X[0]
        lab0, conf0 = m.predict_proba(probe)
        lab1, conf1 = m.predict_proba(probe * 1e-6)
        lab2, conf2 = m.predict_proba(probe * 1e3)
        check(lab0 == lab1 == lab2 and abs(conf0 - conf1) < 1e-9
              and abs(conf0 - conf2) < 1e-9,
              "predictions are invariant to overall signal scale",
              f"uV {conf0:.6f}  x1e-6 {conf1:.6f}  x1e3 {conf2:.6f}")

        kb = (udir / "model.joblib").stat().st_size / 1024
        check(kb < 100, f"model is small enough to ship ({kb:.1f} KB)")

        # drive the real-time path with class-1 windows
        buttons = [Btn(f"b{i}") for i in range(5)]
        tree = SelectionTree(buttons)
        clk = Clock()
        dec = RealtimeDecider(m, tree, artifact=ArtifactDetector(sample_freq=250),
                              clock=clk, debounce_window=3, refractory_ms=1200)
        fired = 0
        for i in range(40):
            w = epoch(1, 1)[0]
            r = dec.step(w)
            clk.advance(1000)               # 1 s hop
            if r["selected"] is not None:
                fired += 1
        check(dec.stats["windows"] == 40, "all 40 windows processed")
        check(fired >= 1, f"at least one button selected end-to-end (got {fired})")
        check(sum(b.fired for b in buttons) == fired,
              "button.trigger() fired exactly once per selection")
        print(f"         {dec.report()}")

        # no-model case must fail loudly
        try:
            LocalMIModel.load("nobody", root=tmp)
            check(False, "missing model raises FileNotFoundError")
        except FileNotFoundError:
            check(True, "missing model raises FileNotFoundError")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    check = _std

    layer_total('LAYER 4 - ONLINE DECISION')


def layer5():
    global check
    banner('LAYER 5 - PRODUCT AND USABILITY', 'Does the library a developer actually touches behave correctly?')

    # --------------------------------------------------------------
    # from test_speller.py
    # --------------------------------------------------------------
    import os
    import sys
    from pathlib import Path

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.insert(0, str(ROOT / "pygame_lib"))

    from ML.text_buffer import TextBuffer  # noqa: E402



    print("\n== TextBuffer ==")
    b = TextBuffer()
    for ch in "HELLO":
        b.accept(ch)
    check("letters accumulate + auto-case", b.text == "Hello")
    b.accept(TextBuffer.SPACE)
    b.accept(TextBuffer.SPACE)
    check("double space collapsed", b.text == "Hello ")
    b.accept("W"); b.accept("O"); b.accept("R"); b.accept("L"); b.accept("D")
    check("second word", b.text == "Hello world")
    b.accept(TextBuffer.BACKSPACE)
    check("backspace", b.text == "Hello worl")
    b.accept(TextBuffer.UNDO)
    check("undo restores", b.text == "Hello world")
    b.accept(".")
    b.accept(TextBuffer.SPACE)
    b.accept("i")
    check("capitalise after sentence end", b.text == "Hello world. I")

    spoken = {}
    b2 = TextBuffer(on_speak=lambda t: spoken.setdefault("t", t))
    b2.accept("Yes. ")
    b2.accept(TextBuffer.SPEAK)
    check("phrase tile inserts whole phrase", b2.text == "Yes. ")
    check("speak callback fires", spoken.get("t") == "Yes. ")
    b2.accept(TextBuffer.CLEAR)
    check("clear empties", b2.text == "")
    check("selection log recorded", len(b2.selection_log) == 3)

    b3 = TextBuffer()
    for c in "the quick brown fox jumps over the lazy dog again and again":
        b3.accept(c)
    w = b3.wrapped(chars_per_line=20, max_lines=3)
    check("wrapped returns <= max_lines", len(w) <= 3)
    check("wrapped respects width", all(len(l) <= 20 for l in w))

    print("\n== Grid + navigation ==")
    import pygame  # noqa: E402
    pygame.init()
    pygame.display.set_mode((1024, 700))
    from example.bci_speller import Speller, build_layout  # noqa: E402

    rows = build_layout()
    cells = [c for r in rows for c in r]
    labels = [c[0] for c in cells]
    check("alphabet present", all(ch in labels for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    check("digits present", all(d in labels for d in "0123456789"))
    check("phrase tiles present", "HELP" in labels and "WATER" in labels)
    check("action tiles present", {"SPACE", "DEL", "CLEAR", "SPEAK"} <= set(labels))
    check("grid is 7 wide", all(len(r) <= 7 for r in rows))

    app = Speller(source="keyboard")
    check("tiles built", sum(len(r) for r in app.tiles) == len(cells))
    check("no tile overlaps panel", all(t.rect.top > 0 for r in app.tiles for t in r))
    check("tiles fit window", all(t.rect.right <= 1024 and t.rect.bottom <= 700
                                  for r in app.tiles for t in r))

    # spell "HI" by scanning: find H then I
    def select_label(app, label):
        for r, row in enumerate(app.tiles):
            for c, t in enumerate(row):
                if t.label == label:
                    app.stage, app.row_idx, app.col_idx = "ROW", 0, 0
                    for _ in range(r):
                        app.cmd_next()
                    app.cmd_select()               # enter row
                    for _ in range(c):
                        app.cmd_next()
                    app.cmd_select()               # commit tile
                    return True
        return False

    select_label(app, "H")
    select_label(app, "I")
    check("row-column scanning spells 'Hi'", app.buffer.text == "Hi")
    check("returns to ROW stage after commit", app.stage == "ROW")

    app.stage, app.row_idx, app.col_idx = "COL", 0, len(app.tiles[0])
    app.cmd_select()
    check("escape cell returns to row scan", app.stage == "ROW")

    select_label(app, "SPACE")
    select_label(app, "HELP")
    check("phrase tile via scanning", app.buffer.text == "Hi I need help. ")

    app.draw()
    check("renders without error", True)

    print("\n== Synthetic stream ==")
    from ML.sim_stream import SyntheticSSVEPStream  # noqa: E402
    s = SyntheticSSVEPStream()
    w0 = s.next_window()
    check("window shape (8,750) - Cyton", w0.shape == (8, 750))
    s.set_active(1)
    check("active target switchable", s.active == 1)

    pygame.quit()

    # --------------------------------------------------------------
    # from test_step7_sdk.py
    # --------------------------------------------------------------
    import os, re, sys, time
    os.environ.setdefault("SDL_VIDEODRIVER","dummy")
    from pathlib import Path
    import numpy as np
    import bci_sdk

    print("\n== Package surface ==")
    check("has a version", bool(bci_sdk.__version__))
    check("info() reports capabilities", set(bci_sdk.info()) >=
          {"version","ui_available","hardware_available","montages"})
    for name in ["BCISession","Command","BCIEngine","StreamBridge","TextBuffer",
                 "HybridSSVEPClassifier","TRCAClassifier","BrainButton","ButtonGroup",
                 "Scanner","SyntheticBackend","ReplayBackend","NO_ACTION"]:
        check(f"exports {name}", hasattr(bci_sdk, name))
    check("__all__ matches real attributes",
          all(hasattr(bci_sdk,n) for n in bci_sdk.__all__))

    print("\n== Typed errors ==")
    check("error hierarchy", issubclass(bci_sdk.HardwareError, bci_sdk.BCIError)
          and issubclass(bci_sdk.NotCalibratedError, bci_sdk.CalibrationError))
    try:
        bci_sdk.BCISession(source="sim", profile="/nonexistent/*.json")
        check("missing profile raises CalibrationError", False)
    except bci_sdk.CalibrationError:
        check("missing profile raises CalibrationError", True)
    try:
        bci_sdk.BCISession(source="hardware", fallback_to_sim=False)
        check("hardware failure raises HardwareError", False)
    except bci_sdk.HardwareError:
        check("hardware failure raises HardwareError", True)
    s = bci_sdk.BCISession(source="hardware", fallback_to_sim=True)
    check("fallback_to_sim degrades gracefully", "sim" in s.source)

    print("\n== BrainButton callbacks (Gap 3 item 2) ==")
    import pygame; pygame.init(); pygame.display.set_mode((400,300))
    b = bci_sdk.BrainButton("YES", rect=(0,0,120,60))
    fired=[]
    @b.on_brain_select
    def _sel(btn): fired.append(btn.label)
    check("on_brain_select works as a decorator", callable(_sel))
    b.select()
    check("select() fires callback", fired==["YES"])
    enter,leave=[],[]
    b.on_focus(lambda x: enter.append(1)); b.on_blur(lambda x: leave.append(1))
    b.arm(); check("arm() fires on_focus", len(enter)==1)
    check("is_armed", b.is_armed)
    b.disarm(); check("disarm() fires on_blur", len(leave)==1)
    b2 = bci_sdk.BrainButton("GO", rect=(0,0,120,60), dwell=0.05)
    hits=[]; b2.on_brain_select(lambda x: hits.append(1))
    b2.arm(); time.sleep(0.08)
    check("dwell auto-selects after timeout", b2.update() and hits==[1])
    check("progress reaches 0 after commit", b2.progress==0.0)
    multi=[]
    b3=bci_sdk.BrainButton("M",rect=(0,0,10,10))
    b3.on_brain_select(lambda x: multi.append(1)); b3.on_brain_select(lambda x: multi.append(2))
    b3.select(); check("multiple callbacks all fire", multi==[1,2])
    bad=bci_sdk.BrainButton("B",rect=(0,0,10,10))
    bad.on_brain_select(lambda x: 1/0)
    bad.select(); check("a throwing callback cannot crash the app", True)
    dis=bci_sdk.BrainButton("D",rect=(0,0,10,10)); dis.enabled=False
    d=[]; dis.on_brain_select(lambda x: d.append(1)); dis.select()
    check("disabled button does not fire", d==[])

    print("\n== ButtonGroup ==")
    btns=[bci_sdk.BrainButton(l,rect=(i*60,0,55,40)) for i,l in enumerate("ABCD")]
    g=bci_sdk.ButtonGroup(btns)
    check("len works", len(g)==4)
    g.assign_frequencies([15.0,20.0])
    check("frequencies assigned round-robin",
          [b.freq for b in g]==[15.0,20.0,15.0,20.0])
    cmd=bci_sdk.Command(name="SSVEP_1",index=1)
    check("routes command to matching button", g.handle_command(cmd) is btns[1])
    check("out-of-range command returns None",
          g.handle_command(bci_sdk.Command(name="SSVEP_9",index=9)) is None)
    check("at() hit-tests", g.at((10,10)) is btns[0] and g.at((5000,5000)) is None)
    stim=bci_sdk.StimulusEngine(target_freqs=(15.0,20.0),refresh_hz=60.0)
    g.apply_flicker(stim)
    check("apply_flicker sets tile state", all(isinstance(b.flicker_on,bool) for b in g))

    print("\n== Scanner (2 commands -> many buttons) ==")
    rows=[[bci_sdk.BrainButton(f"{r}{c}",rect=(c*60,r*50,55,40)) for c in range(3)]
          for r in range(2)]
    sc=bci_sdk.Scanner(rows)
    picked=[]
    for row in rows:
        for b in row: b.on_brain_select(lambda x: picked.append(x.label))
    check("starts in ROW stage", sc.stage=="ROW")
    sc.next(); check("next() advances row", sc.row_idx==1)
    sc.select(); check("select() descends to COL", sc.stage=="COL")
    sc.next(); check("next() advances column", sc.col_idx==1)
    sc.select()
    check("select() commits the button", picked==["11"] and sc.stage=="ROW")
    sc.stage="COL"; sc.row_idx=0; sc.col_idx=len(rows[0])
    sc.select(); check("escape cell returns to ROW", sc.stage=="ROW")
    stages=[]; sc.on_stage_change(lambda st: stages.append(st))
    sc.select(); check("on_stage_change fires", stages==["COL"])
    sc.stage="ROW"
    sc.handle_command(bci_sdk.Command(name="SSVEP_0",index=0))
    check("handle_command(0) == next()", sc.row_idx==1)
    sc.handle_command(bci_sdk.Command(name="SSVEP_1",index=1))
    check("handle_command(1) == select()", sc.stage=="COL")
    sc.update_highlight()
    check("highlight marks the scanned row",
          any(b.state=="scan" for b in rows[sc.row_idx]))

    print("\n== BCISession lifecycle ==")
    sess=bci_sdk.BCISession(source="sim")
    check("not running before start", not sess.is_running)
    check("poll() before start returns None", sess.poll() is None)
    sess.start(); check("running after start", sess.is_running)
    cmds=[]; states=[]; raws=[]
    sess.on_command(lambda c: cmds.append(c))
    sess.on_state(lambda s: states.append(s))
    sess.on_raw_window(lambda w: raws.append(w.shape))
    end=time.time()+8
    while time.time()<end:
        sess.poll(); time.sleep(1/60)
    sess.stop()
    check("not running after stop", not sess.is_running)
    check(f"delivered commands ({[str(c) for c in cmds]})", len(cmds)>=1)
    check("Command carries a positive confidence", all(c.confidence>0 for c in cmds))
    check("Command index parsed from name", all(c.index in (0,1) for c in cmds))
    check("state callback fired", len(states)>=1)
    check("raw window callback got (8,750)", raws and raws[0]==(8,750))
    r=sess.report()
    check("report has engine+stream keys",
          any(k.startswith("engine_") for k in r) and any(k.startswith("stream_") for k in r))

    print("\n== Context manager ==")
    with bci_sdk.BCISession(source="sim") as s2:
        check("context manager starts", s2.is_running)
    check("context manager stops", not s2.is_running)

    print("\n== Replay source ==")
    import glob
    if glob.glob(str(ROOT/"dataset"/"calib_*.npz")):
        s3=bci_sdk.BCISession(source="replay")
        check("replay session builds", s3.bridge.backend.name=="replay")
        s3.start(); time.sleep(4.5); s3.poll(); s3.stop()
        check("replay produced windows", s3.bridge.report()["windows_made"]>=1)
    else:
        print("  SKIP  no recording")

    print("\n== Third-party app uses NO internals (the real Gap 3 test) ==")
    demo=(ROOT/"example"/"sdk_demo.py").read_text()
    internal=re.findall(r"^\s*(?:from|import)\s+(ML|EEG|pygame_lib|stream_bridge|TileButton|stimulus)\b",
                        demo, re.M)
    check(f"sdk_demo.py imports no internal modules {internal}", not internal)
    check("sdk_demo.py imports bci_sdk", "import bci_sdk" in demo)
    uses=[a for a in ["BCISession","BrainButton","Scanner","ButtonGroup",
                      "StimulusEngine","open_display"] if f"bci_sdk.{a}" in demo]
    check(f"demo exercises {len(uses)} public APIs", len(uses)>=5)

    pygame.quit()

    # --------------------------------------------------------------
    # from test_step9_profiles_app.py
    # --------------------------------------------------------------
    import os, sys, json, tempfile, time
    os.environ.setdefault("SDL_VIDEODRIVER","dummy")
    from pathlib import Path
    import numpy as np
    import bci_sdk
    from bci_sdk.profiles import UserProfile, list_users, _slug
    from ML.datasets import load_freq_phase, build_freq_map

    tmp=Path(tempfile.mkdtemp())

    print("\n== Freq_Phase-derived label map (bug fix) ==")
    import scipy.io
    freqs=[]
    for j in range(5):
        for i in range(8): freqs.append(round(8.0+i*1.0+j*0.2,1))
    fp=tmp/"Freq_Phase.mat"
    scipy.io.savemat(fp,{'freqs':np.array(freqs),'phases':np.zeros(40)})
    f,ph=load_freq_phase(fp)
    check("loads 40 targets", len(f)==40 and abs(min(f)-8.0)<1e-9 and abs(max(f)-15.8)<1e-9)
    m,miss=build_freq_map(fp,(15.0,20.0))
    check("15 Hz maps to index 7 (NOT 35)", m=={7:0})
    check("20 Hz correctly reported missing", miss==[20.0])
    m2,miss2=build_freq_map(fp,(10.0,12.0))
    check("10/12 Hz both present", m2=={2:0,4:1} and miss2==[])
    try:
        load_freq_phase(tmp/"nope.mat"); check("missing file guarded", False)
    except FileNotFoundError: check("missing file guarded", True)
    bad=tmp/"bad.mat"; scipy.io.savemat(bad,{'nothing':np.zeros(3)})
    try:
        load_freq_phase(bad); check("non-FreqPhase file rejected", False)
    except ValueError: check("non-FreqPhase file rejected", True)

    print("\n== UserProfile ==")
    p=UserProfile("Anson", root=tmp/"profiles")
    check("slug is filesystem-safe", p.slug=="anson")
    check("weird names sanitised", UserProfile("A n/s..on!",root=tmp).slug.replace('_','')!="" )
    check("new user is uncalibrated", not p.is_calibrated and not p.exists)
    check("uncalibrated -> no session kwargs", p.as_session_kwargs()=={})
    p.update_from_analysis({"confidence_threshold":0.0812,"target_freqs":[15.0,20.0],
                            "montage":"cyton8_ssvep","sample_freq":250,
                            "debounce_window":3,"metrics":{"precision":1.0},
                            "idle_minutes":3.7})
    check("saved to disk", p.path.exists())
    check("threshold stored", abs(p.threshold-0.0812)<1e-9)
    check("now calibrated", p.is_calibrated)
    check("session kwargs point at profile", "profile" in p.as_session_kwargs())
    p2=UserProfile("Anson", root=tmp/"profiles")
    check("reloads from disk", abs(p2.threshold-0.0812)<1e-9)
    check("fresh profile is not stale", not p2.is_stale(max_age_days=7))
    p2.data["calibrated_at"]=time.time()-10*86400; p2.save()
    check("old profile flagged stale", UserProfile("Anson",root=tmp/"profiles").is_stale(7))
    rp=p.recording_path()
    check("recording path inside user dir", rp.parent.name=="anson" and rp.suffix==".npz")

    UserProfile("Brian",root=tmp/"profiles").update_from_analysis({"confidence_threshold":0.05})
    users=list_users(root=tmp/"profiles")
    check(f"lists both users {[u.name for u in users]}", len(users)==2)
    check("users are independent",
          abs(UserProfile("Anson",root=tmp/"profiles").threshold-0.0812)<1e-9 and
          abs(UserProfile("Brian",root=tmp/"profiles").threshold-0.05)<1e-9)

    print("\n== Profile drives the session threshold ==")
    real=UserProfile("anson")
    if real.is_calibrated:
        s=bci_sdk.BCISession(source="sim", **real.as_session_kwargs())
        check("session uses THIS user's threshold",
              abs(s.engine.ssvep_clf.confidence_threshold-real.threshold)<1e-9)
        check("session reports calibrated", s.report()["calibrated"])
    else:
        print("  SKIP  no real calibrated profile yet")
    s2=bci_sdk.BCISession(source="sim")
    check("uncalibrated session uses default", s2.engine.ssvep_clf.confidence_threshold==0.15)

    print("\n== App page flow ==")
    import pygame
    from example.bci_app import App
    a=App(source="sim")
    check("opens on USERS page", a.page==App.PAGE_USERS)
    a2=App(source="sim", user="__brand_new__")
    check("new user routed to CALIBRATE", a2.page==App.PAGE_CALIBRATE)
    if real.is_calibrated:
        a3=App(source="sim", user="anson")
        check("calibrated user routed to READY", a3.page==App.PAGE_READY)
    check("uncalibrated user CANNOT reach speller directly",
          App(source="sim", user="__brand_new2__").page != App.PAGE_SPELLER)
    a.win.fill((0,0,0)); a.draw_users()
    a.profile=UserProfile("anson"); a.page=App.PAGE_CALIBRATE
    a.win.fill((0,0,0)); a.draw_calibrate()
    if real.is_calibrated:
        a.page=App.PAGE_READY; a.win.fill((0,0,0)); a.draw_ready()
    check("all pages render without error", True)

    class E: key=None; unicode=""
    e=E(); e.key=pygame.K_DOWN
    a.page=App.PAGE_USERS; a.users=list_users(); a.sel=0
    a.users_key(e)
    check("DOWN moves selection", a.sel==min(1,len(a.users)))

    pygame.quit()
    layer_total('LAYER 5 - PRODUCT AND USABILITY')


# ====================================================================
#  DATA MODE - diagnostics on a real recording
#  (was tools/layer_tests.py)
# ====================================================================
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
def data_layer1(X, labels, fs, montage):
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
def data_layer2(X, labels, fs, montage):
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
def data_layer3(user, n_splits=4, n_perm=200):
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
def data_layer4():
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


def data_layer5():
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

CODE_LAYERS = {1: layer1, 2: layer2, 3: layer3, 4: layer4, 5: layer5}


def main():
    ap = argparse.ArgumentParser(
        description="Layered BCI tests. Bare = test the CODE; "
                    "--user/--recording = diagnose real DATA.")
    ap.add_argument("--layer", type=int, choices=sorted(CODE_LAYERS))
    ap.add_argument("--user", help="diagnose this user\'s latest recording")
    ap.add_argument("--recording", help="diagnose this .npz (glob allowed)")
    ap.add_argument("--code", action="store_true",
                    help="also run the code tests when diagnosing data")
    ap.add_argument("--perm", type=int, default=200,
                    help="label shuffles for the layer-3 permutation test")
    a = ap.parse_args()

    data_mode = bool(a.user or a.recording)
    want = [a.layer] if a.layer else sorted(CODE_LAYERS)

    if not data_mode or a.code:
        for n in want:
            CODE_LAYERS[n]()
        print("\n" + "=" * 70)
        print(f"  CODE: {ok} passed, {len(fail)} failed")
        if fail:
            print("  failed:", fail)
        print("=" * 70)

    if data_mode:
        pattern = a.recording or str(ROOT / "profiles" / a.user.lower()
                                     / "calib_*.npz")
        res = []
        if any(n in want for n in (1, 2)):
            X, labels, fs, montage, name = _load(pattern)
            print(f"\nrecording: {name}   {X.shape}  fs={fs}  "
                  f"montage={montage}")
            if 1 in want:
                res += data_layer1(X, labels, fs, montage)
            if 2 in want:
                res += data_layer2(X, labels, fs, montage)
        if 3 in want:
            if not a.user:
                print("\n  layer 3 needs --user (it trains on every session)")
            else:
                res += data_layer3(a.user, n_perm=a.perm)
        if 4 in want:
            data_layer4()
        if 5 in want:
            data_layer5()

        n_fail = sum(1 for r in res if r == FAIL)
        n_warn = sum(1 for r in res if r == WARN)
        print("\n" + "=" * 70)
        print(f"  DATA: {sum(1 for r in res if r == PASS)} pass, "
              f"{n_warn} warn, {n_fail} fail")
        if n_fail:
            print("  A data failure is a finding, not a bug. Re-record before "
                  "training.")
        print("=" * 70)

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
