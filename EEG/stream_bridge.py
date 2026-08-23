"""
Author: Anson
Created on: 16/8/2026
Purpose: Step 4 -- the acquisition -> engine bridge (blocker B5).
Location: project_dir/EEG/stream_bridge.py

WHY THIS EXISTS
---------------
Nothing in the codebase ever fed live samples into BCIEngine.process_frame().
`data_receive.py` streams and plots; `bci_engine.py` classifies a window handed to
it; but no component turned a continuous sample stream into the overlapping
750-sample windows the engine expects. This is the missing half of Deliverable 1
("a pipeline for real-time EEG data streaming ... and intent classification") and
it appears in none of the five gaps in instructions.md.

DESIGN
------
Acquisition runs on its OWN THREAD writing into a lock-protected ring buffer. The
UI thread calls `poll()` once per frame, which is non-blocking and returns at most
one ready window. Consequences:

  * A slow classifier can never stall the render loop (which matters doubly for
    SSVEP: UI jitter corrupts the very signal we are measuring).
  * A slow UI can never drop samples -- the ring buffer keeps filling.
  * If the consumer falls behind, windows are dropped OLDEST-FIRST and counted,
    rather than queueing up and making the speller respond to stale intent.

Backends all satisfy the same 4-method contract (start/stop/read_available/info),
so the speller cannot tell them apart:

  SyntheticBackend  -- generated EEG, no hardware (default)
  ReplayBackend     -- replays a Step 3 .npz recording in real time; this is how
                       you regression-test the live path against REAL data
  BrainFlowBackend  -- live OpenBCI Cyton

Usage
    bridge = StreamBridge(backend=SyntheticBackend(), window=750, hop=250)
    bridge.start()
    while running:
        w = bridge.poll()                 # None most frames -- that is expected
        if w is not None:
            decision, fired = engine.process_frame(w)
    bridge.stop()
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================== ring buffer
class RingBuffer:
    """Fixed-capacity circular buffer of shape (n_channels, capacity)."""

    def __init__(self, n_channels, capacity):
        self.n_channels = n_channels
        self.capacity = int(capacity)
        self._buf = np.zeros((n_channels, self.capacity), dtype=np.float64)
        self._write = 0          # next write index
        self.total_written = 0   # lifetime sample count
        self._lock = threading.Lock()

    def write(self, chunk):
        """chunk: (n_channels, n) -- newest samples appended."""
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[0] != self.n_channels:
            raise ValueError(
                f"chunk must be ({self.n_channels}, n), got {chunk.shape}")
        n = chunk.shape[1]
        if n == 0:
            return
        if n >= self.capacity:                     # keep only the newest
            chunk = chunk[:, -self.capacity:]
            n = self.capacity
        with self._lock:
            end = self._write + n
            if end <= self.capacity:
                self._buf[:, self._write:end] = chunk
            else:
                split = self.capacity - self._write
                self._buf[:, self._write:] = chunk[:, :split]
                self._buf[:, :end - self.capacity] = chunk[:, split:]
            self._write = end % self.capacity
            self.total_written += n

    def latest(self, n):
        """Return the most recent n samples as (n_channels, n), or None."""
        with self._lock:
            if self.total_written < n or n > self.capacity:
                return None
            start = (self._write - n) % self.capacity
            if start + n <= self.capacity:
                return self._buf[:, start:start + n].copy()
            split = self.capacity - start
            out = np.empty((self.n_channels, n), dtype=np.float64)
            out[:, :split] = self._buf[:, start:]
            out[:, split:] = self._buf[:, :n - split]
            return out

    def __len__(self):
        return min(self.total_written, self.capacity)


# ================================================================= backends
class SyntheticBackend:
    """Generated EEG at wall-clock rate. Same signal model as Step 3."""

    name = "synthetic"

    def __init__(self, fs=250, n_channels=8, target_freqs=(15.0, 20.0),
                 occipital=(6, 7), frontal=(0, 1), seed=None, snr=2.0):
        self.fs = fs
        self.n_channels = n_channels
        self.target_freqs = list(target_freqs)
        self.occipital = list(occipital)
        self.frontal = list(frontal)
        self.snr = snr
        self.rng = np.random.default_rng(seed)
        self.active = 0          # gaze target index, or None for idle
        self.blink = False
        self._t = 0.0
        self._last = None
        # Phases MUST persist across chunks. Re-randomising them per read made the
        # stitched stream phase-discontinuous, destroying the very SSVEP the
        # classifier looks for -- found by the Step 4 integration demo.
        self._alpha_f = self.rng.uniform(9.5, 10.5)
        self._alpha_ph = self.rng.uniform(0, 2 * np.pi)
        self._ssvep_ph = self.rng.uniform(0, 2 * np.pi)

    # -- test/demo controls
    def set_active(self, idx):
        self.active = None if idx is None else int(idx) % len(self.target_freqs)

    def set_idle(self, flag=True):
        self.active = None if flag else 0

    def trigger_blink(self):
        self.blink = True

    def start(self):
        self._last = time.time()
        self._t = 0.0

    def stop(self):
        pass

    def info(self):
        return {"fs": self.fs, "n_channels": self.n_channels, "backend": self.name}

    def read_available(self):
        """Generate however many samples wall-clock time says are due."""
        now = time.time()
        n = int((now - self._last) * self.fs)
        if n <= 0:
            return None
        self._last += n / self.fs
        t = self._t + np.arange(n) / self.fs
        self._t += n / self.fs

        x = self.rng.standard_normal((self.n_channels, n)) * 3.0
        for ch in self.occipital:
            x[ch] += 4.5 * np.sin(2 * np.pi * self._alpha_f * t + self._alpha_ph)

        if self.active is not None:
            f = self.target_freqs[self.active]
            for ch in self.occipital:
                for h, amp in ((1, 1.0), (2, 0.45), (3, 0.2)):
                    x[ch] += self.snr * amp * np.sin(
                        2 * np.pi * h * f * t + self._ssvep_ph)

        if self.blink:
            self.blink = False
            w = min(int(0.15 * self.fs), n)
            if w > 0:
                shape = np.hanning(w) * 220.0
                for ch in self.frontal:
                    x[ch, :w] += shape
        return x


class ReplayBackend:
    """Replays a Step 3 calibration .npz through the live path in real time.

    This is the important one for validation: it lets the whole streaming stack be
    exercised against REAL recorded EEG, so a regression can be caught without
    putting a headset on someone.
    """

    name = "replay"

    def __init__(self, npz_path, loop=True, speed=1.0, hop=250):
        d = np.load(npz_path, allow_pickle=True)
        X = d["X"]                                  # (n_epochs, ch, samples)
        self.fs = int(d["fs"])
        self.labels = d["labels"].astype(str)
        self.y = d["y"]
        # stitch overlapping epochs back into one continuous signal using the hop
        self.n_channels = X.shape[1]
        chunks = [X[0]] + [X[i][:, -hop:] for i in range(1, len(X))]
        self.signal = np.concatenate(chunks, axis=1).astype(np.float64)
        self.loop = loop
        self.speed = speed
        self.pos = 0
        self._last = None

    def start(self):
        self._last = time.time()
        self.pos = 0

    def stop(self):
        pass

    def info(self):
        return {"fs": self.fs, "n_channels": self.n_channels, "backend": self.name,
                "duration_s": self.signal.shape[1] / self.fs}

    def read_available(self):
        now = time.time()
        n = int((now - self._last) * self.fs * self.speed)
        if n <= 0:
            return None
        self._last += n / (self.fs * self.speed)
        total = self.signal.shape[1]
        if self.pos >= total:
            if not self.loop:
                return None
            self.pos = 0
        end = min(self.pos + n, total)
        out = self.signal[:, self.pos:end]
        self.pos = end
        return out if out.shape[1] else None


class BrainFlowBackend:
    """Live OpenBCI Cyton. BrainFlow imported lazily so this module always loads."""

    name = "brainflow"

    def __init__(self, board_id=0, serial_port="COM4", n_channels=8,
                 ip_address="225.1.1.1", ip_port=6677, master_board=-1,
                 scale_to_uv=True):
        from brainflow.board_shim import BoardShim, BrainFlowInputParams  # noqa
        self.BoardShim = BoardShim
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        params.ip_address = ip_address
        params.ip_port = ip_port
        params.master_board = master_board
        self.board = BoardShim(board_id, params)
        self.board_id = board_id
        self.n_channels = n_channels
        self.scale_to_uv = scale_to_uv
        self.fs = 250
        self.eeg_channels = None

    def start(self):
        self.board.prepare_session()
        self.board.start_stream(45000)
        self.fs = self.BoardShim.get_sampling_rate(self.board_id)
        self.eeg_channels = self.BoardShim.get_eeg_channels(
            self.board_id)[:self.n_channels]
        time.sleep(1.0)

    def stop(self):
        try:
            if self.board.is_prepared():
                self.board.stop_stream()
                self.board.release_session()
        except Exception:
            pass

    def info(self):
        return {"fs": self.fs, "n_channels": self.n_channels, "backend": self.name}

    def read_available(self):
        """Non-blocking: drain whatever the board has buffered."""
        count = self.board.get_board_data_count()
        if count <= 0:
            return None
        data = self.board.get_board_data(count)
        return data[self.eeg_channels, :]          # BrainFlow already returns uV


# ============================================================ the bridge
class StreamBridge:
    """Turns a continuous sample stream into overlapping windows for BCIEngine."""

    def __init__(self, backend=None, window=750, hop=250, buffer_seconds=10.0,
                 max_pending=2, on_window=None):
        """
        :param window: samples per epoch (750 = 3 s @ 250 Hz, matches training)
        :param hop:    samples between epoch starts (250 = 1 s, 67% overlap)
        :param max_pending: windows allowed to queue before dropping oldest.
                       Keeps the speller responding to CURRENT intent, not stale.
        :param on_window: optional callback(window) fired on the acquisition thread
        """
        self.backend = backend or SyntheticBackend()
        info = self.backend.info()
        self.fs = info["fs"]
        self.n_channels = info["n_channels"]
        self.window = int(window)
        self.hop = int(hop)
        self.max_pending = int(max_pending)
        self.on_window = on_window

        cap = max(int(buffer_seconds * self.fs), self.window * 3)
        self.ring = RingBuffer(self.n_channels, cap)

        self._thread = None
        self._stop = threading.Event()
        self._pending = []                  # ready windows awaiting poll()
        self._plock = threading.Lock()
        self._next_at = None                # total_written index of next window

        self.stats = {"samples": 0, "windows_made": 0, "windows_dropped": 0,
                      "windows_consumed": 0, "reads": 0, "empty_reads": 0,
                      "backend_errors": 0}
        self._t_start = None
        self._last_window_time = None

    # ------------------------------------------------------------ lifecycle
    def start(self):
        if self._thread is not None:
            return self
        self.backend.start()
        self._stop.clear()
        self._next_at = self.window          # first window once `window` samples exist
        self._t_start = time.time()
        self._thread = threading.Thread(target=self._run, name="StreamBridge",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.backend.stop()
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -------------------------------------------------------- acquisition
    def _run(self):
        poll_s = max(0.002, (self.hop / self.fs) / 8.0)
        while not self._stop.is_set():
            try:
                chunk = self.backend.read_available()
                self.stats["reads"] += 1
            except Exception:
                self.stats["backend_errors"] += 1
                time.sleep(poll_s)
                continue

            if chunk is None or chunk.shape[1] == 0:
                self.stats["empty_reads"] += 1
                time.sleep(poll_s)
                continue

            if chunk.shape[0] != self.n_channels:
                chunk = chunk[:self.n_channels, :]
            self.ring.write(chunk)
            self.stats["samples"] += chunk.shape[1]
            self._emit_ready()
            time.sleep(poll_s)

    def _emit_ready(self):
        """Cut every window whose end-sample has now arrived.

        `_next_at` is the lifetime sample index at which the current window ENDS.
        A window that ended `lag` samples ago is `latest(window + lag)` with the
        trailing `lag` samples removed.
        """
        while self.ring.total_written >= self._next_at:
            lag = self.ring.total_written - self._next_at
            need = self.window + lag
            if need > self.ring.capacity:
                # consumer/thread stalled so long the window aged out of the buffer
                self.stats["windows_dropped"] += 1
                self._next_at += self.hop
                continue
            block = self.ring.latest(need)
            if block is None:
                break
            w = block[:, :self.window] if lag == 0 else block[:, :-lag][:, -self.window:]
            self._next_at += self.hop
            self.stats["windows_made"] += 1
            self._last_window_time = time.time()

            if self.on_window:
                try:
                    self.on_window(w)
                except Exception:
                    pass

            with self._plock:
                self._pending.append(w)
                while len(self._pending) > self.max_pending:
                    self._pending.pop(0)
                    self.stats["windows_dropped"] += 1

    # ------------------------------------------------------------ consumer
    def poll(self):
        """Non-blocking. Return the newest ready window, or None."""
        with self._plock:
            if not self._pending:
                return None
            w = self._pending.pop(0)
            self.stats["windows_consumed"] += 1
            return w

    def poll_all(self):
        with self._plock:
            out, self._pending = self._pending, []
            self.stats["windows_consumed"] += len(out)
            return out

    def wait_for_window(self, timeout=5.0):
        """Blocking convenience for scripts/tests (never use in a render loop)."""
        end = time.time() + timeout
        while time.time() < end:
            w = self.poll()
            if w is not None:
                return w
            time.sleep(0.005)
        return None

    # --------------------------------------------------------- diagnostics
    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def report(self):
        el = (time.time() - self._t_start) if self._t_start else 0.0
        s = dict(self.stats)
        s["elapsed_s"] = el
        s["effective_fs"] = (self.stats["samples"] / el) if el > 0 else 0.0
        s["fs_error_pct"] = ((s["effective_fs"] - self.fs) / self.fs * 100
                             if el > 0 else 0.0)
        # Windowing cannot begin until the first full window exists, so measure
        # the rate over the STEADY-STATE period, not including that priming time.
        prime_s = self.window / self.fs
        steady = max(1e-9, el - prime_s)
        s["prime_s"] = prime_s
        s["windows_per_s"] = (self.stats["windows_made"] / steady) if el > prime_s else 0.0
        s["expected_windows_per_s"] = self.fs / self.hop
        s["drop_rate"] = (self.stats["windows_dropped"]
                          / max(1, self.stats["windows_made"]))
        s["buffer_fill"] = len(self.ring) / self.ring.capacity
        return s

    def print_report(self):
        r = self.report()
        print(f"  backend            : {self.backend.name}")
        print(f"  elapsed            : {r['elapsed_s']:.1f} s")
        print(f"  samples            : {r['samples']}")
        print(f"  effective fs       : {r['effective_fs']:.1f} Hz "
              f"(nominal {self.fs}, error {r['fs_error_pct']:+.2f}%)")
        print(f"  windows made       : {r['windows_made']} "
              f"({r['windows_per_s']:.2f}/s steady-state, "
              f"expected {r['expected_windows_per_s']:.2f}/s; "
              f"{r['prime_s']:.1f}s priming excluded)")
        print(f"  windows consumed   : {r['windows_consumed']}")
        print(f"  windows dropped    : {r['windows_dropped']} "
              f"({r['drop_rate']:.1%} -- consumer too slow)")
        print(f"  backend errors     : {r['backend_errors']}")


# ------------------------------------------------------------------ demo
def _demo(seconds=12.0, backend_name="synthetic", npz=None):
    from ML.bci_engine import BCIEngine

    if backend_name == "replay":
        import glob
        matches = sorted(glob.glob(npz or str(ROOT / "dataset" / "calib_*.npz")))
        if not matches:
            print("No calibration .npz found; run Step 3 first.")
            return 1
        be = ReplayBackend(matches[-1])
        print(f"Replaying {matches[-1]}  ({be.info()['duration_s']:.0f}s of EEG)")
    else:
        be = SyntheticBackend(seed=1)

    engine = BCIEngine(mode="SSVEP", debounce_window=3, refractory_ms=1200)
    bridge = StreamBridge(backend=be, window=750, hop=250)

    print(f"\nStreaming for {seconds:.0f}s ...\n")
    events = []
    with bridge:
        end = time.time() + seconds
        frame = 0
        while time.time() < end:
            frame += 1
            w = bridge.poll()
            if w is not None:
                t0 = time.time()
                decision, fired = engine.process_frame(w)
                dt = (time.time() - t0) * 1000
                events.append((decision, fired, dt))
                flag = "  <== DISPATCH" if fired else ""
                print(f"  [{time.time()-bridge._t_start:5.1f}s] {decision:<10} "
                      f"({dt:5.1f} ms){flag}")
                # drive the synthetic user around a bit
                if backend_name == "synthetic" and isinstance(be, SyntheticBackend):
                    n = len(events)
                    if n == 4:
                        be.set_idle(True); print("        -- user looks away --")
                    elif n == 8:
                        be.set_idle(False); be.set_active(1)
                        print("        -- user gazes at 20 Hz --")
                    elif n == 11:
                        be.trigger_blink(); print("        -- user blinks --")
            time.sleep(1 / 60)                 # simulate a 60 fps UI loop

    print("\nBridge report:")
    bridge.print_report()
    if events:
        lat = [e[2] for e in events]
        print(f"  classify latency   : mean {np.mean(lat):.1f} ms, "
              f"max {np.max(lat):.1f} ms (budget {1000*250/250:.0f} ms/hop)")
    print("\nEngine report:")
    for k, v in engine.report().items():
        print(f"  {k:<22} {v:.3f}" if isinstance(v, float) else f"  {k:<22} {v}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Step 4 acquisition->engine bridge demo")
    ap.add_argument("--backend", choices=["synthetic", "replay"], default="synthetic")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--npz", default=None)
    a = ap.parse_args()
    sys.exit(_demo(a.seconds, a.backend, a.npz))
