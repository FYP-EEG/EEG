# Cyton raw-transfer parity with OpenBCI GUI

## Finding

The OpenBCI GUI and this project's direct route both use BrainFlow for Cyton acquisition. The GUI reference inspected at commit `e23869e7b5cc621e733d8fa0d81f05d477264306` calls BrainFlow `prepare_session`, `start_stream`, `get_board_data`, and board-specific EEG-channel lookup. Its waveform smoothing is downstream of acquisition and cannot repair a truly flat raw stream.

The meaningful parity controls are therefore:

1. which process owns the Cyton COM port;
2. ADS1299 channel gain/input/BIAS/SRB settings;
3. physical sensor power, reference and BIAS wiring;
4. BrainFlow EEG row extraction and units;
5. sample flow, packet continuity and raw variance before filtering.

## Explicit profiles

`EEG/cyton_commands.py` exposes two profiles and does not silently exchange them.

| Profile | Intended hardware | Gain default | Input | BIAS | SRB2 | SRB1 |
|---|---|---:|---|---|---|---|
| `openbci_gui_default` | ordinary passive electrodes / GUI Reset defaults | ×24 | Normal | included | connected | disconnected |
| `thinkpulse_active` | powered ThinkPulse active electrodes | ×8; override ×8 or lower | Normal | excluded | connected | disconnected |

For ThinkPulse, power every active sensor, connect the earlobe reference to SRB2, and leave the physical BIAS lead disconnected. The software cannot verify those physical facts. Do not apply the GUI passive ×24 profile to ThinkPulse simply to obtain a larger-looking trace; use `--gain 8`, `4`, `2`, or `1` and diagnose the raw range.

## Route A — direct Python to Cyton

Only Python owns the COM port. Close OpenBCI GUI first.

```powershell
python tools/cyton_transfer_check.py capture `
  --route direct `
  --port COM8 `
  --profile thinkpulse_active `
  --gain 4 `
  --seconds 12 `
  --out reports/cyton_transfer/direct_gain4
```

The direct backend:

- resets persistent Cyton test/channel state with `d`;
- reapplies all eight declared profile commands;
- asks BrainFlow for the Cyton EEG row indices rather than assuming rows 0–7;
- preserves BrainFlow's physical microvolt units;
- inspects package-number and timestamp rows before extracting EEG;
- reports missing/duplicate/out-of-order package estimates;
- reports each channel's raw min, max, variance, unique-value count, rails and flatness;
- applies no EEG filter or classifier.

Use the same route in the recorder:

```powershell
python EEG/data_record.py --universal anson `
  --acquisition-route direct `
  --serial-port COM8 `
  --cyton-profile thinkpulse_active `
  --cyton-gain 4
```

## Route B — OpenBCI GUI owns Cyton and streams raw JSON to Python

This route is useful when the GUI displays valid raw data but direct acquisition does not. It is an actual GUI Networking stream, not BrainFlow's unrelated `STREAMING_BOARD` packet protocol.

1. Open OpenBCI GUI and connect to Cyton.
2. Apply the GUI gain/channel settings that produce non-railed data.
3. Open the **Networking** widget.
4. Select **UDP**.
5. Select **Time Series Raw**, not Time Series Filtered.
6. Set destination IP `127.0.0.1` and port `12345`.
7. Start the GUI data stream and Networking stream.
8. Run:

```powershell
python tools/cyton_transfer_check.py capture `
  --route gui-udp `
  --udp-ip 127.0.0.1 `
  --udp-port 12345 `
  --seconds 12 `
  --out reports/cyton_transfer/gui_raw
```

Or record through the same route:

```powershell
python EEG/data_record.py --universal anson `
  --acquisition-route gui-udp `
  --gui-udp-ip 127.0.0.1 `
  --gui-udp-port 12345
```

The GUI UDP schema is:

```json
{"type":"timeSeriesRaw","data":[[channel_1_samples],[channel_2_samples]]}
```

The GUI output is in microvolts. The receiver rejects `timeSeriesFilt`, malformed, non-finite, or wrong-shape packets. The GUI JSON packet does not contain a sequence number, so exact UDP packet loss cannot be reconstructed. The report instead records valid/malformed packet counts, per-packet sample counts, receive gaps, source addresses, total samples and effective sample rate.

## Matched comparison procedure

Use unchanged sensor placement, sensor power and SRB2 reference. Capture the two routes sequentially because only one process can own the COM port.

```powershell
python tools/cyton_transfer_check.py compare `
  reports/cyton_transfer/direct_gain4.npz `
  reports/cyton_transfer/gui_raw.npz `
  --out reports/cyton_transfer/direct_vs_gui
```

The comparison deliberately does not correlate waveforms: the recordings are sequential, not time-aligned. Compare instead:

- `units = uV` in both captures;
- nominal and effective sample rate near 250 Hz;
- eight EEG rows and the intended channel order;
- raw SD and span per channel;
- flat, rail, amplitude and jump flags;
- direct package continuity;
- the exact direct profile/gain and command transcript;
- the matching GUI amplifier/gain setting recorded by the operator.

If the GUI capture passes but direct capture fails with unchanged wiring, compare profile/gain and direct board rows first. If both fail, fix power/reference/contact/hardware before filtering or model use. A filtered trace must never be used to override a failed raw-transfer result.
