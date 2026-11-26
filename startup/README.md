# Cyton headset startup gate

Run this before using the headset outside the recorder:

```bat
python startup/check_headband.py --port COM8
```

`EEG/data_record.py` calls the same gate automatically before a real direct-Cyton MI recording. Startup does **not** ask the operator to type `YES`. Software cannot inspect physical connectors, so S0 records the required wiring contract without claiming that it verified it:

- **ThinkPulse (default):** powered active sensors, one earlobe reference on SRB2, and no physical BIAS connection.
- **Passive electrodes:** firm SRB/reference and physical BIAS connections.

The startup decision is `READY` only when:

1. the selected wiring contract has been stated (physical wiring remains the operator's responsibility);
2. all 8 channels transfer the internal-ground test;
3. all 8 channels transfer the known square-wave test;
4. the selected Normal-input channel profile is applied and all 8 live channels pass connection, finite-value, flatness, adaptive amplitude/jump and clipping checks.

A failed stage does not terminate the calling recorder/example. Each distinct failure state is printed once with stage-specific fixes and a ready-to-run `tools/debug_bci_pipeline.py` command. Identical failed polls update one `Silent repeated failures: N` counter line in place instead of printing more lines. The script keeps checking without starting cues or data capture until every stage passes. `--retry-delay` controls polling, and `Ctrl+C` aborts deliberately.

If Cyton is disconnected or produces no samples, startup reports one explicit board/connection failure; dependent square-wave and Normal-input stages are marked `SKIP`. BrainFlow can occasionally accept a Cyton ASCII command but fail while decoding its optional response when that response starts with a raw byte such as `A0`. The shared command helper ignores only that post-command response-text decoding error and validates success from captured ground/square/Normal signals; all command-status and hardware exceptions still propagate. BrainFlow 5.22.0 or newer remains required.

Failed polls do not write HTML/JSON. One startup report is written only after `READY`. A pass proves acquisition-path and gross raw quality only; it does not prove motor-imagery accuracy.

## ThinkPulse gain and amplitude handling

ThinkPulse is the current default. It requires gain 8 or lower, BIAS generation off, SRB1 off and SRB2 on. Gains above 8 are rejected. When all eight channels show genuine clipping/railing, startup automatically retries at gain **8 → 4 → 2 → 1** and returns the selected gain to the recording backend.

Amplitude acceptance is not based on a replacement fixed microvolt threshold. The preflight measures robust per-channel variation, peer-channel scale, sample-step scale and repeated extreme plateaus. A stable common active-electrode scale or DC offset can pass; a local gross outlier, flat channel, abrupt discontinuity or clipped plateau fails.

A trace that OpenBCI GUI calls “out of range” can look flat later because this code median-centres signals for variation measurements and downstream filters remove DC. If the ADC was saturated, many raw samples have the same minimum/maximum; after centring/filtering that plateau has almost no variation. Reports therefore retain the raw median and clipping fraction and test repeated extrema before treating a trace as merely flat.

## Current motor montage and recording

The `cyton8_motor` physical channel order is:

```text
CH1 F3 · CH2 C3 · CH3 Cz · CH4 P3 · CH5 Pz · CH6 F4 · CH7 C4 · CH8 P4
```

Record and automatically train the personal finger/hand/feet category models with:

```bat
python EEG/data_record.py --universal anson
```

## Check with an OpenBCI GUI CSV (no hardware)

An OpenBCI GUI CSV can exercise parsing, preflight, cueing, capture, epoching and report generation without opening a board:

```bat
python EEG/data_record.py --universal anson --eeg-csv "path\OpenBCI-RAW.csv" --no-auto-train
```

The parser reads `%Sample Rate`, numeric timestamps when metadata is absent, and the named `EXG Channel 0` through `EXG Channel 7` columns rather than assuming fixed column positions. Replay loops at wall-clock rate. Output is marked as an offline pipeline check and automatic model training is disabled because old replayed EEG is not temporally related to the new cues.

Preflight a CSV directly:

```bat
python EEG/signal_preflight.py --csv "path\OpenBCI-RAW.csv" --out reports/signal_preflight/openbci
```

SDK scripts can use `OpenBCICSVBackend`, or `BCISession(source="csv", eeg_csv="path/OpenBCI-RAW.csv")`.

For an independent real-board check:

```bat
python tools/test_signal_check.py --port COM8 --seconds 6
python tools/test_signal_check.py --board cyton --serial-port COM8 --seconds 6
```

The full ThinkPulse-aware hardware debugger is:

```bat
python tools/debug_bci_pipeline.py --port COM8
```
