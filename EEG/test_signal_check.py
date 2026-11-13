"""
Author: Anson
Created on: 10/11/2026
Purpose: minimal OpenBCI Cyton-to-laptop connection and sample-flow test.

Run from the project root after closing OpenBCI GUI and serial monitors:
    py EEG/test_cyton_connection.py --port COM4

This does not train, classify or save EEG. It only opens the Cyton, drains live
8-channel samples for a few seconds, reports basic values, and releases the port.
"""

import argparse
import sys
import time

import numpy as np

from stream_bridge import BrainFlowBackend


def _serial_ports():
    try:
        from serial.tools import list_ports
        return [(item.device, item.description) for item in list_ports.comports()]
    except Exception:
        return []


def _print_hints(port):
    print("\nConnection checklist:")
    print("  1. Confirm the Cyton and USB dongle are powered.")
    print("  2. Confirm Windows Device Manager shows the dongle's COM port.")
    print(f"  3. Replace {port!r} if Windows assigned a different COM number.")
    print("  4. Close OpenBCI GUI, Arduino Serial Monitor and other serial apps.")
    print("  5. Unplug/replug the dongle, power-cycle Cyton, then retry.")
    print("  6. Keep board-id 0 for the standard 8-channel Cyton.")


def run(port="COM4", seconds=5.0, board_id=0):
    seconds = max(1.0, float(seconds))
    print("OpenBCI Cyton connection test")
    print(f"  board_id : {board_id}")
    print(f"  port     : {port}")
    print(f"  duration : {seconds:g} s")
    ports = _serial_ports()
    if ports:
        print("  detected serial ports:")
        for device, description in ports:
            marker = "  <== requested" if device.upper() == port.upper() else ""
            print(f"    {device:<8} {description}{marker}")
    else:
        print("  serial-port listing unavailable (connection test will still run)")

    backend = None
    chunks = []
    started = None
    try:
        backend = BrainFlowBackend(board_id=int(board_id), serial_port=str(port),
                                   n_channels=8)
        print("\nOpening board...", flush=True)
        backend.start()
        info = backend.info()
        print(f"Connected: fs={info['fs']} Hz, channels={info['n_channels']}")

        # BrainFlowBackend.start() intentionally waits one second for its buffer
        # to prime. Discard that initial buffer so the measured rate below only
        # covers the requested collection interval.
        backend.read_available()
        started = time.monotonic()
        deadline = started + seconds
        while time.monotonic() < deadline:
            chunk = backend.read_available()
            if chunk is not None and chunk.size:
                chunks.append(np.asarray(chunk, dtype=np.float64))
            time.sleep(.02)

        elapsed = time.monotonic() - started
        if not chunks:
            raise RuntimeError("board opened but returned no EEG samples")
        data = np.concatenate(chunks, axis=1)
        effective_fs = data.shape[1] / max(elapsed, 1e-9)
        finite = bool(np.isfinite(data).all())
        channel_std = np.std(data, axis=1)
        channel_peak = np.max(np.abs(data), axis=1)

        print("\nSample report")
        print(f"  shape        : {data.shape}")
        print(f"  samples      : {data.shape[1]}")
        print(f"  elapsed      : {elapsed:.2f} s")
        print(f"  effective fs : {effective_fs:.1f} Hz (nominal {info['fs']} Hz)")
        print(f"  finite data  : {'YES' if finite else 'NO'}")
        print("  per-channel standard deviation / absolute peak (uV):")
        labels = ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "Pz")
        for index, (std, peak) in enumerate(zip(channel_std, channel_peak)):
            name = labels[index] if index < len(labels) else f"CH{index + 1}"
            print(f"    {index + 1:>2} {name:<3}  std={std:>10.3f}  peak={peak:>10.3f}")

        minimum = int(info["fs"] * seconds * .70)
        passed = (data.shape[0] == 8 and data.shape[1] >= minimum and finite)
        if not passed:
            raise RuntimeError(
                f"sample stream failed sanity check: expected 8 channels and "
                f"at least {minimum} finite samples")
        print("\nCYTON CONNECTION: PASS")
        print("This proves serial connection and sample flow only; it does not "
              "prove electrode quality or motor-imagery accuracy.")
        return 0
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("brainflow"):
            print("\nCYTON CONNECTION: FAIL")
            print("BrainFlow is not installed in this Python environment.")
            print("Install it with:  py -m pip install brainflow")
        else:
            print(f"\nCYTON CONNECTION: FAIL\n{type(exc).__name__}: {exc}")
        _print_hints(port)
        return 1
    except Exception as exc:
        print("\nCYTON CONNECTION: FAIL")
        print(f"{type(exc).__name__}: {exc}")
        _print_hints(port)
        return 1
    finally:
        if backend is not None:
            try:
                backend.stop()
                print("Board session released.")
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Test OpenBCI Cyton serial connection and 8-channel samples")
    parser.add_argument("--port", default="COM4",
                        help="Windows serial port shown in Device Manager")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="seconds of live samples to inspect")
    parser.add_argument("--board-id", type=int, default=0,
                        help="BrainFlow board ID; standard Cyton is 0")
    args = parser.parse_args()
    return run(args.port, args.seconds, args.board_id)


if __name__ == "__main__":
    sys.exit(main())
