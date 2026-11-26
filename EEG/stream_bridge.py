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
  OpenBCICSVBackend   -- replays an OpenBCI GUI CSV with metadata parsing
  OpenBCIGUIUDPBackend -- receives GUI Networking Time Series Raw JSON
  BrainFlowBackend    -- direct Cyton or BrainFlow Streaming Board

Usage
    bridge = StreamBridge(backend=SyntheticBackend(), window=750, hop=250)
    bridge.start()
    while running:
        w = bridge.poll()                 # None most frames -- that is expected
        if w is not None:
            decision, fired = engine.process_frame(w)
    bridge.stop()
"""

import csv
import json
import re
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from EEG.cyton_commands import (
    cyton_profile_settings,
    live_channel_commands,
    normalize_electrode_mode,
    send_cyton_command,
)


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


_OPENBCI_EEG_COLUMN = re.compile(
    r"^(?:exg|eeg)\s*(?:channel|ch)?\s*[-_#:]?\s*(\d+)\b",
    re.IGNORECASE)
# Also accept exports whose EEG columns are named by standard 10-20/10-10
# electrode labels rather than OpenBCI's ``EXG Channel N`` convention.
_ELECTRODE_COLUMN = re.compile(
    r"^(?:FP|AF|F|FT|FC|T|C|TP|CP|P|PO|O|IZ|A|M)(?:Z|\d{1,2})$",
    re.IGNORECASE)
_OPENBCI_SAMPLE_RATE = re.compile(
    r"sample\s*rate\s*(?:=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:hz)?",
    re.IGNORECASE)
_OPENBCI_CHANNEL_COUNT = re.compile(
    r"number\s*of\s*channels\s*(?:=|:)?\s*(\d+)", re.IGNORECASE)


def _normalise_csv_heading(value):
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _infer_csv_sample_rate(timestamps):
    """Infer Hz from numeric timestamps expressed in seconds/ms/us/ns."""
    values = np.asarray(timestamps, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if not len(diffs):
        return None
    dt = float(np.median(diffs))
    candidates = [1.0 / dt, 1000.0 / dt, 1_000_000.0 / dt,
                  1_000_000_000.0 / dt]
    plausible = [value for value in candidates if 20.0 <= value <= 5000.0]
    if not plausible:
        return None
    common = (125.0, 200.0, 250.0, 256.0, 500.0, 512.0, 1000.0)
    return min(plausible, key=lambda value: min(abs(value - ref) for ref in common))


def _openbci_unit_scale(headings, metadata):
    """Return (factor-to-microvolts, declared-unit). GUI defaults to µV."""
    unit_text = " ".join([*headings, *metadata])
    match = re.search(
        r"(?:data\s*)?units?\s*(?:=|:)?\s*(uv|µv|μv|mv|volts?|v)\b",
        unit_text, flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"[\[(]\s*(uv|µv|μv|mv|v)\s*[\])]",
                          unit_text, flags=re.IGNORECASE)
    unit = (match.group(1).casefold() if match else "uv")
    if unit in {"uv", "µv", "μv"}:
        return 1.0, "uV"
    if unit == "mv":
        return 1000.0, "mV"
    return 1_000_000.0, "V"


def load_openbci_gui_csv(csv_path, n_channels=8, fs=None, scale_to_uv=None):
    """Load an OpenBCI GUI CSV as an 8-channel, channel-major µV signal.

    OpenBCI GUI files normally begin with ``%`` metadata and use columns named
    ``EXG Channel 0`` ... ``EXG Channel 7``. Comparison exports with standard
    electrode headings such as ``Fp1,Fp2,Fz,Cz,...`` are also accepted. Column
    positions are not hardcoded. Sample rate comes from metadata or from a
    numeric ``Timestamp``/``session_time`` column when metadata is absent.
    """
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"OpenBCI GUI CSV not found: {path}")
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    metadata = [line.strip().lstrip("%").strip() for line in lines
                if line.lstrip().startswith("%")]
    content = [(line_number, line) for line_number, line in enumerate(lines, 1)
               if line.strip() and not line.lstrip().startswith("%")]
    if not content:
        raise ValueError(f"OpenBCI GUI CSV has no header/data rows: {path}")

    header_number, header_line = content[0]
    try:
        dialect = csv.Sniffer().sniff(header_line, delimiters=",;\t")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    headings = next(csv.reader([header_line], delimiter=delimiter))
    headings = [heading.strip() for heading in headings]
    indexed = []
    electrode_columns = []
    for index, heading in enumerate(headings):
        normalised_heading = _normalise_csv_heading(heading)
        match = _OPENBCI_EEG_COLUMN.match(normalised_heading)
        if match:
            indexed.append((int(match.group(1)), index, heading))
        elif _ELECTRODE_COLUMN.fullmatch(heading.strip()):
            electrode_columns.append((len(electrode_columns), index, heading))
    indexed.sort(key=lambda item: item[0])
    requested = int(n_channels)
    if requested < 1:
        raise ValueError("n_channels must be at least 1")
    candidates = indexed if len(indexed) >= requested else electrode_columns
    column_schema = ("openbci_exg" if candidates is indexed else
                     "electrode_labels")
    if len(candidates) < requested:
        found_rows = indexed or electrode_columns
        found = ", ".join(heading for _, _, heading in found_rows) or "none"
        raise ValueError(
            f"expected at least {requested} EXG/EEG or standard electrode "
            f"columns, found {len(found_rows)} ({found}) in header line "
            f"{header_number}")
    selected = candidates[:requested]

    normalised = [_normalise_csv_heading(heading) for heading in headings]
    timestamp_names = {
        "timestamp", "session_time", "session time", "time",
        "time_s", "time_ms", "time_us", "time_ns",
    }
    timestamp_index = next(
        (index for index, heading in enumerate(normalised)
         if heading in timestamp_names), None)
    if timestamp_index is None:
        timestamp_index = next(
            (index for index, heading in enumerate(normalised)
             if "timestamp" in heading and "formatted" not in heading), None)

    columns = [[] for _ in selected]
    timestamps = []
    dropped_rows = 0
    max_column = max([column for _, column, _ in selected] +
                     ([timestamp_index] if timestamp_index is not None else []))
    for line_number, line in content[1:]:
        row = next(csv.reader([line], delimiter=delimiter))
        if len(row) <= max_column:
            dropped_rows += 1
            continue
        try:
            values = [float(row[column].strip()) for _, column, _ in selected]
        except ValueError:
            # Tolerate an accidentally repeated GUI header, but never reinterpret
            # another numeric field as EEG.
            dropped_rows += 1
            continue
        if not np.isfinite(values).all():
            dropped_rows += 1
            continue
        for target, value in zip(columns, values):
            target.append(value)
        if timestamp_index is not None:
            try:
                timestamps.append(float(row[timestamp_index].strip()))
            except ValueError:
                timestamps.append(float("nan"))
    if not columns or len(columns[0]) < 2:
        raise ValueError(f"OpenBCI GUI CSV contains fewer than 2 valid EEG rows: {path}")

    metadata_rate = None
    for line in metadata:
        match = _OPENBCI_SAMPLE_RATE.search(line)
        if match:
            metadata_rate = float(match.group(1))
            break
    inferred_rate = _infer_csv_sample_rate(timestamps)
    resolved_fs = float(fs) if fs is not None else metadata_rate or inferred_rate
    if resolved_fs is None or not np.isfinite(resolved_fs) or resolved_fs <= 0:
        raise ValueError(
            "could not determine OpenBCI CSV sample rate from metadata or numeric "
            "Timestamp column; pass fs explicitly")

    declared_channels = None
    for line in metadata:
        match = _OPENBCI_CHANNEL_COUNT.search(line)
        if match:
            declared_channels = int(match.group(1))
            break
    unit_factor, input_unit = _openbci_unit_scale(headings, metadata)
    if scale_to_uv is not None:
        unit_factor = float(scale_to_uv)
        input_unit = "custom"
    if not np.isfinite(unit_factor) or unit_factor <= 0:
        raise ValueError("scale_to_uv must be finite and greater than zero")
    signal = np.asarray(columns, dtype=np.float64) * unit_factor
    return {
        "path": str(path.resolve()),
        "data": signal,
        "fs": resolved_fs,
        "n_channels": requested,
        "channel_labels": [heading for _, _, heading in selected],
        "channel_numbers": [number for number, _, _ in selected],
        "column_schema": column_schema,
        "input_unit": input_unit,
        "scale_to_uv": unit_factor,
        "samples": int(signal.shape[1]),
        "duration_s": float(signal.shape[1] / resolved_fs),
        "rows_dropped": dropped_rows,
        "metadata": metadata,
        "metadata_channel_count": declared_channels,
        "sample_rate_source": ("argument" if fs is not None else
                               "metadata" if metadata_rate is not None else
                               "timestamp"),
    }


class OpenBCICSVBackend:
    """Replay OpenBCI GUI CSV EEG at wall-clock rate without live hardware."""

    name = "openbci_csv"

    def __init__(self, csv_path, loop=True, speed=1.0, n_channels=8,
                 fs=None, scale_to_uv=None):
        loaded = load_openbci_gui_csv(
            csv_path, n_channels=n_channels, fs=fs, scale_to_uv=scale_to_uv)
        self.path = loaded["path"]
        self.signal = loaded.pop("data")
        self.fs = loaded["fs"]
        self.n_channels = loaded["n_channels"]
        self.channel_labels = loaded["channel_labels"]
        self.loop = bool(loop)
        self.speed = float(speed)
        if not np.isfinite(self.speed) or self.speed <= 0:
            raise ValueError("CSV replay speed must be finite and greater than zero")
        self._metadata = loaded
        self.pos = 0
        self._last = None

    def start(self):
        self.pos = 0
        self._last = time.monotonic()

    def stop(self):
        self._last = None

    def info(self):
        return {**self._metadata, "fs": self.fs,
                "n_channels": self.n_channels, "backend": self.name,
                "loop": self.loop, "speed": self.speed}

    def read_available(self):
        if self._last is None:
            return None
        now = time.monotonic()
        n = int((now - self._last) * self.fs * self.speed)
        if n <= 0:
            return None
        self._last += n / (self.fs * self.speed)
        total = self.signal.shape[1]
        if not self.loop:
            if self.pos >= total:
                return None
            end = min(self.pos + n, total)
            out = self.signal[:, self.pos:end]
            self.pos = end
            return out if out.shape[1] else None

        pieces = []
        remaining = n
        while remaining > 0:
            take = min(remaining, total - self.pos)
            if take:
                pieces.append(self.signal[:, self.pos:self.pos + take])
                self.pos += take
                remaining -= take
            if self.pos >= total:
                self.pos = 0
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=1)


# Friendly alias for scripts whose source selector is simply called ``csv``.
CSVReplayBackend = OpenBCICSVBackend


class OpenBCIGUIUDPBackend:
    """Receive the OpenBCI GUI Networking Widget's raw UDP JSON stream.

    Configure the GUI Networking Widget as:

    * protocol: UDP;
    * data type: Time Series Raw (not Time Series Filtered);
    * destination IP: 127.0.0.1;
    * destination port: 12345 (or the matching ``port=`` value).

    The GUI sends ``{"type":"timeSeriesRaw","data":[channel][sample]}``
    datagrams in microvolts.  This is the GUI's actual networking schema; it is
    not BrainFlow's ``STREAMING_BOARD`` wire format.  The GUI owns the Cyton COM
    port in this route, while this backend owns only the UDP listener.
    """

    name = "openbci_gui_udp"

    def __init__(self, bind_ip="127.0.0.1", port=12345, fs=250,
                 n_channels=8, startup_timeout=6.0, receive_bytes=65535):
        self.bind_ip = str(bind_ip)
        self.port = int(port)
        self.fs = float(fs)
        self.n_channels = int(n_channels)
        self.startup_timeout = float(startup_timeout)
        self.receive_bytes = int(receive_bytes)
        if not (1 <= self.port <= 65535):
            raise ValueError("GUI UDP port must be in 1..65535")
        if self.fs <= 0 or not np.isfinite(self.fs):
            raise ValueError("GUI UDP sampling rate must be finite and positive")
        if self.n_channels < 1:
            raise ValueError("GUI UDP channel count must be positive")
        self.sock = None
        self._primed = []
        self._reset_diagnostics()

    def _reset_diagnostics(self):
        self.packet_count = 0
        self.samples_received = 0
        self.malformed_packets = 0
        self.wrong_type_packets = 0
        self.wrong_shape_packets = 0
        self.nonfinite_packets = 0
        self.packet_sample_counts = []
        self.source_addresses = set()
        self.interarrival_gaps_s = []
        self._first_packet_time = None
        self._last_packet_time = None
        self._last_error = None

    def reset_raw_diagnostics(self):
        """Start a fresh diagnostic interval without closing the UDP socket."""
        self._reset_diagnostics()

    def start(self):
        if self.sock is not None:
            return
        self._reset_diagnostics()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        except OSError:
            pass
        try:
            sock.bind((self.bind_ip, self.port))
            sock.setblocking(False)
        except Exception:
            sock.close()
            raise
        self.sock = sock
        self._primed = []
        deadline = time.monotonic() + max(0.0, self.startup_timeout)
        while time.monotonic() < deadline:
            chunks = self._drain_socket()
            if chunks:
                self._primed.extend(chunks)
                return
            time.sleep(.01)
        self.stop()
        raise RuntimeError(
            "UDP listener opened but no valid OpenBCI GUI Time Series Raw "
            f"packets arrived on {self.bind_ip}:{self.port}; start the GUI "
            "data stream and Networking UDP stream, and select Time Series Raw")

    def stop(self):
        sock, self.sock = self.sock, None
        self._primed = []
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def info(self):
        return {
            "fs": self.fs, "n_channels": self.n_channels,
            "backend": self.name, "bind_ip": self.bind_ip,
            "port": self.port, "units": "uV",
            "packet_schema": "OpenBCI GUI UDP JSON timeSeriesRaw",
            "filtered": False,
        }

    def _decode_packet(self, payload, address):
        now = time.monotonic()
        try:
            message = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError) as exc:
            self.malformed_packets += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            return None
        if message.get("type") != "timeSeriesRaw":
            self.wrong_type_packets += 1
            self._last_error = (
                f"expected timeSeriesRaw, received {message.get('type')!r}")
            return None
        try:
            chunk = np.asarray(message["data"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            self.malformed_packets += 1
            self._last_error = f"invalid data array: {exc}"
            return None
        if (chunk.ndim != 2 or chunk.shape[0] < self.n_channels or
                chunk.shape[1] < 1):
            self.wrong_shape_packets += 1
            self._last_error = (
                f"expected at least ({self.n_channels},1), got {chunk.shape}")
            return None
        chunk = chunk[:self.n_channels]
        if not np.isfinite(chunk).all():
            self.nonfinite_packets += 1
            self._last_error = "packet contains NaN/inf"
            return None
        self.packet_count += 1
        self.samples_received += int(chunk.shape[1])
        self.packet_sample_counts.append(int(chunk.shape[1]))
        if len(self.packet_sample_counts) > 4096:
            self.packet_sample_counts = self.packet_sample_counts[-4096:]
        self.source_addresses.add(f"{address[0]}:{address[1]}")
        if self._last_packet_time is not None:
            self.interarrival_gaps_s.append(now - self._last_packet_time)
            if len(self.interarrival_gaps_s) > 4096:
                self.interarrival_gaps_s = self.interarrival_gaps_s[-4096:]
        if self._first_packet_time is None:
            self._first_packet_time = now
        self._last_packet_time = now
        return chunk

    def _drain_socket(self):
        if self.sock is None:
            return []
        chunks = []
        while True:
            try:
                payload, address = self.sock.recvfrom(self.receive_bytes)
            except BlockingIOError:
                break
            except OSError as exc:
                if self.sock is not None:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                break
            chunk = self._decode_packet(payload, address)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def read_available(self):
        chunks, self._primed = self._primed, []
        chunks.extend(self._drain_socket())
        if not chunks:
            return None
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=1)

    def raw_diagnostics(self):
        elapsed = (self._last_packet_time - self._first_packet_time
                   if self._first_packet_time is not None and
                   self._last_packet_time is not None else 0.0)
        gaps = np.asarray(self.interarrival_gaps_s, dtype=float)
        sample_counts = np.asarray(self.packet_sample_counts, dtype=float)
        return {
            **self.info(),
            "valid_packets": int(self.packet_count),
            "samples_received": int(self.samples_received),
            "effective_fs_since_first_packet": (
                float(self.samples_received / elapsed) if elapsed > 0 else None),
            "malformed_packets": int(self.malformed_packets),
            "wrong_type_packets": int(self.wrong_type_packets),
            "wrong_shape_packets": int(self.wrong_shape_packets),
            "nonfinite_packets": int(self.nonfinite_packets),
            "packet_samples_median": (
                float(np.median(sample_counts)) if sample_counts.size else None),
            "packet_samples_min": (
                int(np.min(sample_counts)) if sample_counts.size else None),
            "packet_samples_max": (
                int(np.max(sample_counts)) if sample_counts.size else None),
            "interarrival_gap_max_s": (
                float(np.max(gaps)) if gaps.size else None),
            "source_addresses": sorted(self.source_addresses),
            "packet_continuity_exact": False,
            "packet_continuity_note": (
                "OpenBCI GUI UDP JSON has no sequence number; exact packet loss "
                "cannot be reconstructed. Validate sample rate and gap statistics."),
            "last_error": self._last_error,
        }


class BrainFlowBackend:
    """Direct Cyton or BrainFlow Streaming Board with raw transfer diagnostics.

    BrainFlow already returns EEG rows in microvolts.  The full board matrix is
    retained long enough to inspect package numbers and timestamps before only
    the declared EEG rows are returned to the recorder.  This detects wrong row
    selection and packet discontinuities without filtering the EEG.

    ``board_id=-2`` is BrainFlow's own Streaming Board protocol.  It is not the
    OpenBCI GUI Networking Widget's JSON format; use ``OpenBCIGUIUDPBackend`` for
    a GUI-owned Cyton session.
    """

    name = "brainflow"

    def __init__(self, board_id=0, serial_port="COM8", n_channels=8,
                 ip_address="225.1.1.1", ip_port=6677, master_board=-1,
                 scale_to_uv=True, reset_channels=True,
                 electrode_mode="thinkpulse", cyton_gain=None,
                 cyton_profile=None, stream_buffer_size=450000):
        from brainflow.board_shim import BoardShim, BrainFlowInputParams
        self.BoardShim = BoardShim
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        params.ip_address = ip_address
        params.ip_port = ip_port
        # BrainFlow's STREAMING_BOARD (-2) has no channel descriptor of its
        # own. It must be told which real board produced the packets, and
        # sampling/channel metadata must be queried from that master board.
        if board_id == -2 and master_board == -1:
            master_board = 0
        params.master_board = master_board
        self.board = BoardShim(board_id, params)
        self.board_id = int(board_id)
        self.master_board = int(master_board)
        self.descriptor_board_id = (self.master_board
                                    if self.board_id == -2 else self.board_id)
        self.n_channels = int(n_channels)
        self.scale_to_uv = bool(scale_to_uv)  # compatibility; BrainFlow is uV
        self.reset_channels = bool(reset_channels)
        self.stream_buffer_size = int(stream_buffer_size)
        if self.stream_buffer_size < 1:
            raise ValueError("stream_buffer_size must be positive")
        settings = cyton_profile_settings(
            cyton_profile, electrode_mode=electrode_mode, gain=cyton_gain)
        self.cyton_profile = settings["name"]
        self.electrode_mode = normalize_electrode_mode(
            settings["electrode_mode"])
        self.cyton_gain = int(settings["gain"])
        self.profile_settings = settings
        self._live_channel_commands = live_channel_commands(
            gain=self.cyton_gain, profile=self.cyton_profile)
        self.fs = 250
        self.eeg_channels = None
        self.package_num_channel = None
        self.timestamp_channel = None
        self.num_board_rows = None
        self.command_results = []
        self._reset_diagnostics()

    def _reset_diagnostics(self):
        self._diag_samples = 0
        self._diag_reads = 0
        self._diag_missing_packets = 0
        self._diag_duplicate_packets = 0
        self._diag_out_of_order_packets = 0
        self._diag_nonmonotonic_timestamps = 0
        self._diag_last_package = None
        self._diag_last_timestamp = None
        self._diag_first_read_time = None
        self._diag_last_read_time = None
        self._diag_min = None
        self._diag_max = None
        self._diag_sum = None
        self._diag_sumsq = None

    def reset_raw_diagnostics(self):
        """Start a fresh diagnostic interval while leaving the board stream open."""
        self._reset_diagnostics()

    def _descriptor(self, method_name, default=None):
        method = getattr(self.BoardShim, method_name, None)
        if method is None:
            return default
        try:
            return method(self.descriptor_board_id)
        except Exception:  # noqa: BLE001 - optional BrainFlow descriptor
            return default

    def start(self):
        self._reset_diagnostics()
        self.command_results = []
        self.board.prepare_session()
        try:
            # Cyton test-signal and powered-down settings persist. Reset, then
            # apply every field of the selected explicit profile before stream.
            if self.board_id == 0 and self.reset_channels:
                self.command_results.append({
                    "command": "d", **send_cyton_command(self.board, "d")})
                time.sleep(.25)
                for command in self._live_channel_commands:
                    self.command_results.append({
                        "command": command,
                        **send_cyton_command(self.board, command),
                    })
                time.sleep(.25)
            # Match the current OpenBCI GUI BrainFlow ring-buffer size.  This
            # affects retention capacity, not the raw decoder or signal scale.
            self.board.start_stream(getattr(self, "stream_buffer_size", 450000))
            self.fs = self.BoardShim.get_sampling_rate(
                self.descriptor_board_id)
            self.eeg_channels = list(self.BoardShim.get_eeg_channels(
                self.descriptor_board_id)[:self.n_channels])
            if len(self.eeg_channels) != self.n_channels:
                raise RuntimeError(
                    f"descriptor exposes {len(self.eeg_channels)} EEG rows, "
                    f"expected {self.n_channels}")
            self.package_num_channel = self._descriptor(
                "get_package_num_channel")
            self.timestamp_channel = self._descriptor("get_timestamp_channel")
            self.num_board_rows = self._descriptor("get_num_rows")

            # Opening a socket/COM handle is not enough. Refuse to start a timed
            # recording unless samples actually arrive.
            deadline = time.monotonic() + 6.0
            while time.monotonic() < deadline:
                if self.board.get_board_data_count() > 0:
                    return
                time.sleep(.05)
            mode = ("BrainFlow Streaming Board: start a compatible BrainFlow "
                    "producer (not the GUI JSON Networking stream)"
                    if self.board_id == -2 else
                    "direct Cyton: check power, dongle, radio and serial port")
            raise RuntimeError(f"board opened but no EEG samples arrived ({mode})")
        except Exception:  # cleanup then preserve the original hardware error
            try:
                if self.board.is_prepared():
                    self.board.stop_stream()
                    self.board.release_session()
            except Exception as cleanup_error:  # noqa: BLE001
                self._last_cleanup_error = (
                    f"{type(cleanup_error).__name__}: {cleanup_error}")
            raise

    def stop(self):
        try:
            if self.board.is_prepared():
                self.board.stop_stream()
                self.board.release_session()
        except Exception as cleanup_error:  # noqa: BLE001
            self._last_cleanup_error = (
                f"{type(cleanup_error).__name__}: {cleanup_error}")

    def info(self):
        route = ("brainflow_streaming_board" if self.board_id == -2
                 else "direct_cyton" if self.board_id == 0
                 else "brainflow_board")
        return {
            "fs": self.fs, "n_channels": self.n_channels,
            "backend": self.name, "acquisition_route": route,
            "board_id": self.board_id, "master_board": self.master_board,
            "descriptor_board_id": self.descriptor_board_id,
            "electrode_mode": self.electrode_mode,
            "cyton_profile": self.cyton_profile,
            "cyton_gain": self.cyton_gain,
            "profile_settings": dict(self.profile_settings),
            "eeg_channel_rows": (list(self.eeg_channels)
                                 if self.eeg_channels is not None else None),
            "package_num_row": self.package_num_channel,
            "timestamp_row": self.timestamp_channel,
            "board_rows": self.num_board_rows,
            "units": "uV", "filtered": False,
            "stream_buffer_size": getattr(self, "stream_buffer_size", 450000),
            "last_cleanup_error": getattr(self, "_last_cleanup_error", None),
        }

    def _update_packet_continuity(self, data):
        row = self.package_num_channel
        if row is not None and 0 <= int(row) < data.shape[0]:
            values = np.asarray(np.rint(data[int(row)]), dtype=np.int64)
            for current in values:
                current = int(current) % 256
                previous = self._diag_last_package
                if previous is not None:
                    delta = (current - previous) % 256
                    if delta == 0:
                        self._diag_duplicate_packets += 1
                    elif 1 < delta <= 128:
                        self._diag_missing_packets += delta - 1
                    elif delta > 128:
                        self._diag_out_of_order_packets += 1
                self._diag_last_package = current
        row = self.timestamp_channel
        if row is not None and 0 <= int(row) < data.shape[0]:
            values = np.asarray(data[int(row)], dtype=float)
            for current in values[np.isfinite(values)]:
                if (self._diag_last_timestamp is not None and
                        current < self._diag_last_timestamp):
                    self._diag_nonmonotonic_timestamps += 1
                self._diag_last_timestamp = float(current)

    def _update_signal_diagnostics(self, eeg):
        now = time.monotonic()
        if self._diag_first_read_time is None:
            self._diag_first_read_time = now
        self._diag_last_read_time = now
        self._diag_reads += 1
        self._diag_samples += int(eeg.shape[1])
        current_min = np.min(eeg, axis=1)
        current_max = np.max(eeg, axis=1)
        current_sum = np.sum(eeg, axis=1)
        current_sumsq = np.sum(np.square(eeg), axis=1)
        self._diag_min = (current_min if self._diag_min is None else
                          np.minimum(self._diag_min, current_min))
        self._diag_max = (current_max if self._diag_max is None else
                          np.maximum(self._diag_max, current_max))
        self._diag_sum = (current_sum if self._diag_sum is None else
                          self._diag_sum + current_sum)
        self._diag_sumsq = (current_sumsq if self._diag_sumsq is None else
                            self._diag_sumsq + current_sumsq)

    def read_available(self):
        """Drain raw board data, inspect transport rows, and return EEG rows."""
        count = self.board.get_board_data_count()
        if count <= 0:
            return None
        data = np.asarray(self.board.get_board_data(count), dtype=np.float64)
        if data.ndim != 2:
            raise RuntimeError(f"BrainFlow returned unexpected board shape {data.shape}")
        if self.eeg_channels is None:
            raise RuntimeError("BrainFlow EEG row metadata was not initialized")
        if max(self.eeg_channels, default=-1) >= data.shape[0]:
            raise RuntimeError(
                f"EEG row map {self.eeg_channels} exceeds board data {data.shape}")
        self._update_packet_continuity(data)
        eeg = data[self.eeg_channels, :]
        if not np.isfinite(eeg).all():
            raise RuntimeError("BrainFlow returned non-finite raw EEG")
        self._update_signal_diagnostics(eeg)
        return eeg

    def raw_diagnostics(self):
        n = int(self._diag_samples)
        elapsed = (self._diag_last_read_time - self._diag_first_read_time
                   if self._diag_first_read_time is not None and
                   self._diag_last_read_time is not None else 0.0)
        if n > 0:
            mean = self._diag_sum / n
            variance = np.maximum(self._diag_sumsq / n - np.square(mean), 0.0)
            std = np.sqrt(variance)
        else:
            mean = std = np.full(self.n_channels, np.nan)
        return {
            **self.info(),
            "reads": int(self._diag_reads),
            "samples_received": n,
            "effective_fs_since_first_read": (
                float(n / elapsed) if elapsed > 0 else None),
            "channel_mean_uv": np.asarray(mean).tolist(),
            "channel_std_uv": np.asarray(std).tolist(),
            "channel_min_uv": (np.asarray(self._diag_min).tolist()
                               if self._diag_min is not None else []),
            "channel_max_uv": (np.asarray(self._diag_max).tolist()
                               if self._diag_max is not None else []),
            "missing_packets_estimate": int(self._diag_missing_packets),
            "duplicate_packages": int(self._diag_duplicate_packets),
            "out_of_order_packages": int(self._diag_out_of_order_packets),
            "nonmonotonic_timestamps": int(
                self._diag_nonmonotonic_timestamps),
            "packet_continuity_exact": self.package_num_channel is not None,
            "command_results": list(self.command_results),
        }


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
            except Exception as backend_error:  # noqa: BLE001
                self.stats["backend_errors"] += 1
                self._last_backend_error = (
                    f"{type(backend_error).__name__}: {backend_error}")
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
                except Exception as callback_error:  # noqa: BLE001
                    self._last_callback_error = (
                        f"{type(callback_error).__name__}: {callback_error}")

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
