# Step 1 complete — Gap 1: BCI Speller (EEG-to-Text)

## Files added
| File | Purpose |
|---|---|
| `pygame_lib/TileButton.py` | Rectangular, text-labelled key. Sibling of `Button.py` (see note). Has `flicker_on` + `freq` fields already, so Gap 5 can drive it without touching the UI. |
| `ML/text_buffer.py` | `TextBuffer` — pygame-free sentence accumulator. Letters, phrases, SPACE/DEL/UNDO/CLEAR/SPEAK, auto-capitalisation, double-space collapse, word wrap, selection log. |
| `example/bci_speller.py` | The speller app: 53-tile grid + row-column scanning + live output panel. |
| `ML/sim_stream.py` | Synthetic SSVEP windows so the app can be demoed through the *real* `BCIEngine` with no headset. |
| `tests/test_speller.py` | 27 headless assertions. All passing. |
| `docs/speller_screenshot.png` | Rendered proof. |

## Why `TileButton` instead of editing `Button.py`
`Button` is circular and **requires** an image icon (`pygame.image.load(icon)`), and its
hit-test is a radius check. A 53-key text keyboard needs rectangular text tiles. Editing
`Button` in place would have broken `rockpaperscissors.py` and `realistic_ui.py`.
`TileButton` keeps the identical public API (`update_layout / draw / check_within / click`)
so the SDK in Gap 3 can expose them behind one interface.

## Selection paradigm — a design decision the instructions left open
The classifier gives **two** reliable commands (SSVEP 10 Hz / 12 Hz) but the grid has 53 tiles.
I used standard **row–column scanning**:

- `NEXT` (10 Hz / LEFT arrow) — advance the highlight
- `SELECT` (12 Hz / RIGHT arrow) — enter the highlighted row, then commit the tile
- Each row ends in an **escape cell** so a wrong row is always recoverable
- Confirmation uses a **dwell bar** (default 1.0 s) drawn on the tile before commit

Worst case ≈ 8 row-steps + 7 col-steps. If you later expand to 4–6 flicker frequencies this
drops sharply — the grid code doesn't care, only `cmd_next`/`cmd_select` would change.

## Run it
```bash
cd bci_project
python example/bci_speller.py                 # keyboard/mouse, no hardware
python example/bci_speller.py --source sim    # synthetic EEG through the real BCIEngine
python example/bci_speller.py --source engine # live BrainFlow (needs the bridge, see B5)
python tests/test_speller.py                  # 27 assertions
```

## Verified behaviour
- Scanning spells `"Hi I need help. "` end-to-end through the grid.
- `sim` mode: 10 Hz windows advance the row (`row_idx 0→4`), 12 Hz windows arm the dwell,
  injected 180 µV frontal spike is caught by `artifact_detection` → `BLINK` (ignored, not typed).
- Resizing rebuilds the grid; no tile overlaps the output panel at any size tested.

## Known limitation carried forward
`sim` mode works because `SyntheticSSVEPStream` emits **64-channel** windows to match
`HybridSSVEPClassifier`'s default `ssvep_channels=[54,55,60,61,62]`. On the real 8-channel
Cyton those indices raise `IndexError`. That channel-map fix is blocker **B1** in
`instructions_verification.md` and must be done before `--source engine` will run.
