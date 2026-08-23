# Step 4 — Acquisition → engine bridge

**Covers:** blocker **B5** from `docs/instructions_verification.md` — *"There is no acquisition
→ engine bridge"*. This appears in **none** of the five gaps in `instructions.md`, yet it is
the missing half of Deliverable 1 ("a pipeline for real-time EEG data streaming, noise
reduction, and intent classification").
**Status:** ✅ complete. `--source engine` now runs.
**Tests:** 33 assertions in `tests/test_step4_bridge.py`, all passing.

---

## The problem

`data_receive.py` streams and plots. `bci_engine.py` classifies a window *handed to it*. But
nothing turned a continuous sample stream into the overlapping 750-sample windows the engine
expects. The speller's `sim` mode faked it by calling `SyntheticSSVEPStream.next_window()`
directly — no buffering, no timing, no relationship to a real board.

## Design

Acquisition runs on **its own thread**, writing into a lock-protected ring buffer. The UI
thread calls `poll()` once per frame — non-blocking, usually returns `None`.

```
backend.read_available()  ──►  RingBuffer  ──►  window cutter  ──►  poll()  ──►  BCIEngine
      [acquisition thread, ~30 Hz]                                  [UI thread, 60 fps]
```

Why threaded matters doubly here: a blocking classifier would stall the render loop, and UI
jitter corrupts the very SSVEP signal being measured. Measured: **472 UI frames rendered while
only 5 windows were classified, 0 dropped.**

**Backpressure.** If the consumer falls behind, windows are dropped **oldest-first** and
counted, rather than queueing — so the speller responds to *current* intent, never stale.

**One code path, three backends** (identical 4-method contract, so the speller can't tell them
apart):

| Backend | Use |
|---|---|
| `SyntheticBackend` | generated EEG at wall-clock rate, no hardware |
| `ReplayBackend` | replays a Step 3 `.npz` through the live path — regression-test the streaming stack against **real recorded EEG** without a headset |
| `BrainFlowBackend` | live OpenBCI Cyton (imported lazily) |

---

## Demo output

```
  [  5.0s] SSVEP_0    ( 13.4 ms)  <== DISPATCH
        -- user looks away --
  [  9.0s] NO_ACTION  ( 13.4 ms)
        -- user gazes at 20 Hz --
  [ 13.1s] SSVEP_1    ( 13.0 ms)  <== DISPATCH
        -- user blinks --
  [ 14.0s] BLINK      (  0.8 ms)

  effective fs       : 248.0 Hz (nominal 250, error -0.79%)
  windows made       : 13 (1.00/s steady-state, expected 1.00/s)
  windows dropped    : 0 (0.0%)
  classify latency   : mean 11.7 ms, max 15.2 ms (budget 1000 ms/hop)
```

**Latency headroom is ~70×**: classification takes ~14 ms against a 1000 ms hop budget. Nothing
about the current pipeline is compute-bound, so Step 6's heavier TRCA/FBCSP has ample room.

---

## 🐛 Bug this step uncovered

The first integration run classified **every** window as `NO_ACTION`, even with a simulated
user staring straight at a target.

I first suspected the ring buffer, so I tested it directly: feed a known 15 Hz sine in ragged
1–96 sample chunks and demand exact reconstruction → **`max abs err: 0.0`**. The buffer was
provably correct, which pointed at the backend.

**Cause:** `SyntheticBackend.read_available()` drew a **fresh random phase on every chunk**.
Each chunk was individually fine, but stitching them produced a phase-discontinuous signal —
destroying the periodicity that CCA exists to detect.

```
scores through bridge (before): [0.076, 0.106]   -> below threshold, NO_ACTION
scores through bridge (after) : [0.623, 0.053]   -> correct, confident
dominant freqs (before): 7.0, 9.0, 9.3, 12.0, 14.7 Hz   <- smeared, no 15 Hz peak
```

**Fix:** oscillator phases (`_alpha_ph`, `_ssvep_ph`) now persist across reads.

This was a bug in test infrastructure, not shipped code — but it's exactly the class of bug
that makes a demo mysteriously fail on the day, and it only appears once you stream
*continuously* rather than generating independent windows. A regression test now stitches 12
separate chunks and asserts the SSVEP survives.

Two smaller fixes: `_emit_ready()`'s window slicing was rewritten (the old expression
double-counted `lag`), and `windows_per_s` now excludes the 3 s priming period during which no
window can exist — it was reporting 0.75/s when the true steady-state rate was exactly 1.00/s.

---

## Speller integration

`--source engine` works, and `sim`/`replay`/`engine` share one path:

```bash
python example/bci_speller.py --source sim
python example/bci_speller.py --source replay
python example/bci_speller.py --source engine --serial-port COM4
python example/bci_speller.py --source engine --profile dataset/profile_calib_S01_*.json
```

`--profile` loads the Step 3 per-subject threshold. Without hardware, `engine` degrades
gracefully to keyboard with a clear message rather than crashing.

---

## Files

| File | Change |
|---|---|
| `EEG/stream_bridge.py` | **new** — `RingBuffer`, `StreamBridge`, 3 backends, demo CLI |
| `example/bci_speller.py` | consumes the bridge; `--source replay/engine`, `--profile` |
| `tests/test_step4_bridge.py` | **new** — 33 assertions |

`ML/sim_stream.py` is now superseded by `SyntheticBackend` but kept, since Step 1–3 tests use it.

## Run
```bash
python EEG/stream_bridge.py --backend synthetic --seconds 16
python EEG/stream_bridge.py --backend replay
python tests/test_step4_bridge.py
```

## Carried forward
- Sample-rate error is ~1–2% because the synthetic backend derives samples from wall-clock
  time. A real board clocks itself; re-measure `effective_fs` on hardware.
- `BrainFlowBackend` is written against the documented API but **has never run against a
  physical Cyton** — expect to adjust `board_id` and channel ordering on first contact.
- Next: **Step 5**, V-synced frame-accurate flicker. The stimulus UI in
  `EEG/calibration_record.py` already demonstrates the correct `DOUBLEBUF | SCALED, vsync=1`
  approach; Step 5 brings it to the speller and logs frame deltas to prove low jitter.
