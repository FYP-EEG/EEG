# Step 7 — Standardised developer SDK

**Covers:** `instructions.md` Gap 3, all three items + blocker **B10** (packaging metadata).
**Status:** ✅ complete — the final gap. **Tests:** 63 assertions in `tests/test_step7_sdk.py`.

---

## Gap 3 checklist

| Instruction | Delivered |
|---|---|
| "Package core ML logic and UI into a cohesive SDK (`import bci_sdk`)" | `bci_sdk/` — 29 exports over `ML/`, `EEG/`, `pygame_lib/` |
| "Extend Button to support `button.on_brain_select(callback)`" | `BrainButton.on_brain_select()`, usable as a decorator, plus `on_focus` / `on_blur` |
| "Docstrings and a README for third-party developers" | `README.md`, module docstrings, `pyproject.toml` |

---

## Design: a façade, not a move

Nothing was relocated. `bci_sdk` is a **stable public surface** that re-exports the
existing modules and adds the event layer on top. Two reasons:

1. Moving files would have broken all 185 existing assertions from Steps 1–6.
2. It establishes the boundary the instructions actually asked for: integrators import
   `bci_sdk` and never touch internals, so we stay free to refactor underneath.

Three layers, each usable alone:

```
BCISession          high   -- backend -> bridge -> engine, delivers Commands
BrainButton/Scanner mid    -- event-driven widgets
BCIEngine/...       low    -- the components themselves, unchanged
```

**pygame and brainflow are optional.** `import bci_sdk` works without either; only the
affected features are withheld, with an actionable error if you touch them.
`bci_sdk.info()` reports what's available.

---

## The acceptance test that matters

Gap 3 is only satisfied if someone *outside* the project can build an app. So
`example/sdk_demo.py` is a complete 4-button smart-home controller written **only**
against the public API, and a test enforces that:

```python
internal = re.findall(r"^\s*(?:from|import)\s+(ML|EEG|pygame_lib|stream_bridge|...)", demo)
check("sdk_demo.py imports no internal modules", not internal)   # PASS  []
```

The whole app is ~60 lines of logic:

```python
session = bci_sdk.BCISession(source="sim", profile=args.profile)
scanner = bci_sdk.Scanner([buttons[:2], buttons[2:]])

for b in buttons:
    @b.on_brain_select
    def activated(btn):
        log.append(f"Activated: {btn.label}")

session.on_command(scanner.handle_command)
```

---

## Two selection paradigms, one callback API

The SDK can't assume how many commands a user's classifier produces, so both are
supported and application code doesn't change between them:

| | `ButtonGroup` (direct) | `Scanner` (row/column) |
|---|---|---|
| Commands needed | one flicker frequency per button | **two** total |
| Practical max | ~4–6 buttons at 60 Hz | unlimited |
| Trade-off | fast | slower, scales to a keyboard |

---

## 🐛 Bug this step caught

`Command.confidence` was always `0.000`. `BCIEngine` computed a confidence for every
window but never stored it, so the SDK's dataclass had nothing to read — a public API
promising a field it could never populate. Fixed by exposing
`BCIEngine.last_confidence`; now `SSVEP_0(conf=0.616)`. A test asserts every delivered
command carries a positive confidence.

Also hardened: **a throwing callback cannot crash the host app.** Every user callback is
invoked inside a try/except, since a third-party bug in an event handler shouldn't take
down someone's assistive device mid-sentence.

---

## Typed errors

`BCIError` → `HardwareError`, `CalibrationError` → `NotCalibratedError`. Integrators can
distinguish "headset unplugged" from "bad profile path" from "model needs calibration".
`BCISession(source="hardware", fallback_to_sim=True)` degrades to synthetic data instead
of crashing; `fallback_to_sim=False` raises `HardwareError` for CI.

---

## Files

| File | Purpose |
|---|---|
| `bci_sdk/__init__.py` | **new** — public surface, optional-dependency handling, `info()` |
| `bci_sdk/session.py` | **new** — `BCISession`, `Command` |
| `bci_sdk/widgets.py` | **new** — `BrainButton`, `ButtonGroup`, `Scanner` |
| `bci_sdk/errors.py` | **new** — typed exception hierarchy |
| `example/sdk_demo.py` | **new** — third-party app using only public API |
| `README.md` | **new** — developer documentation |
| `pyproject.toml` | **new** — packaging, extras, console scripts |
| `ML/bci_engine.py` | exposes `last_confidence` |
| `tests/test_step7_sdk.py` | **new** — 63 assertions |

## Run
```bash
pip install -e ".[ui]"
python example/sdk_demo.py
python tests/test_step7_sdk.py
```

## Carried forward
- The README's "Things that will bite you" section is the highest-value part for a new
  integrator: `NO_ACTION` is normal, match stimulus to monitor refresh, avoid 8–13 Hz,
  never block the render loop.
- `pyproject.toml` is valid and installs, but the package has **not** been published or
  installed into a clean venv from a built wheel — do that before sharing externally.
- Version is `0.7.0` (alpha). The API should be considered unstable until it has been
  exercised against real hardware.
