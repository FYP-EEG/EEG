# Your Hardware: Cyton + EEG Headband Kit

**Verdict: this hardware works for your project — but you must NOT wire it the way OpenBCI's
default tutorial says.** Their default headband layout has almost no occipital coverage, and
occipital is the only place SSVEP exists.

Sources: [OpenBCI Headband Kit Guide](https://docs.openbci.com/AddOns/Headwear/HeadBand/) ·
[Headband Kit shop page](https://shop.openbci.com/products/openbci-eeg-headband-kit) ·
[Cyton EEG setup](https://docs.openbci.com/GettingStarted/Biosensing-Setups/EEGSetup/)

---

## ✅ What's already correct

| Thing | Status |
|---|---|
| Board = **Cyton**, `board_id = 0` | ✅ your code's default |
| Sampling rate **250 Hz** | ✅ matches `sample_freq=250` everywhere |
| 8 channels | ✅ matches `cyton8_ssvep` montage |
| Window 750 samples = 3.0 s | ✅ correct at 250 Hz |
| Connection | USB dongle → a COM port |

No sample-rate surprise. The Daisy risk I flagged earlier does not apply.

### Acquisition profile and route are part of the hardware contract

Do not treat OpenBCI GUI defaults and the active ThinkPulse setup as the same
wiring profile:

| Setup | Cyton gain | BIAS | SRB2 | Code profile |
|---|---:|---|---|---|
| Standard passive OpenBCI/GUI reference | ×24 | included | connected | `openbci_gui_default` |
| Active ThinkPulse (current main setup) | **×8 or lower** | excluded | connected | `thinkpulse_active` |

For an exact GUI-versus-direct checklist, commands, packet format and raw-only
comparison workflow, see **[`CYTON_OPENBCI_GUI_PARITY.md`](CYTON_OPENBCI_GUI_PARITY.md)**.
Use `tools/cyton_transfer_check.py` before filtering or classification whenever the
GUI shows a changing signal but `data_record.py` appears flat.

> The passive ear-reference/BIAS wiring below describes the standard dry-headband
> SSVEP arrangement. With active ThinkPulse sensors, follow their powered-sensor
> wiring and use `thinkpulse_active`; do not copy the passive BIAS connection
> blindly. Only one program may own the Cyton serial port: close the GUI for the
> direct route, or let the GUI own it and send **Time Series Raw** over UDP.

---

## 🔴 The problem: the default headband wiring has no O1/O2

OpenBCI's Headband–Cyton tutorial specifies this:

| Ch | Their default | Electrode |
|---|---|---|
| 0 | Fp1 | flat snap |
| 1 | Fp2 | flat snap |
| 2 | Fpz | flat snap |
| 3 | TP7 | comb |
| 4 | TP8 | comb |
| 5 | P7 | comb |
| 6 | P8 | comb |
| 7 | **Oz** | comb |

That is **one** occipital electrode (Oz) out of eight. Your classifier reads channels 6 and 7
expecting **O1 and O2**:

```python
ssvep_channels = [6, 7]     # ML/montage.py -> cyton8_ssvep
```

With the default wiring, channel 6 would be **P8** — a parietal site with weak SSVEP. You'd
be feeding the classifier mostly-blind data and the result would look like "the algorithm
doesn't work."

**The headband strap is adjustable and the kit explicitly supports `PO7/PO8`, `O1/O2` and
`Oz`.** So this is a placement decision, not a hardware limitation. You just have to make it
deliberately.

---

## ✅ Recommended wiring for SSVEP

Use the comb electrodes across the back of the strap:

| Ch | Cyton pin | Electrode | Type | Purpose |
|---|---|---|---|---|
| 0 | N1P | **Fp1** | flat snap | blink detection |
| 1 | N2P | **Fp2** | flat snap | blink detection |
| 2 | N3P | PO7 | comb | extra occipital *(bonus)* |
| 3 | N4P | PO3 | comb | extra occipital *(bonus)* |
| 4 | N5P | POz | comb | extra occipital *(bonus)* |
| 5 | N6P | PO4 | comb | extra occipital *(bonus)* |
| 6 | N7P | **O1** | comb | **SSVEP — required** |
| 7 | N8P | **O2** | comb | **SSVEP — required** |
| — | SRB | A1 | ear clip | reference |
| — | BIAS | A2 | ear clip | ground |

**Channels 0, 1, 6, 7 are what your code actually uses today.** Channels 2–5 cost nothing
extra (the electrodes are in the kit) and they're the exact upgrade the progress report
recommends — they would make TRCA viable later. Wire them even if unused for now.

### Finding O1 and O2
Feel for the **inion** — the bony bump at the base of your skull. Per OpenBCI's own guide:
**1 inch above the inion, 1 inch to the left = O1**; same on the right = **O2**. Part the hair
so the comb teeth reach scalp.

---

## ⚠️ Dry comb electrodes — set expectations

The headband kit is **dry** (no gel). That means:

- Higher impedance and a noisier signal than wet electrodes
- **Hair is the enemy** — comb teeth must reach scalp, especially at O1/O2
- More sensitive to movement; sit still
- Expect to spend 5–10 minutes re-seating electrodes before the signal looks clean

This is normal for dry systems, not a fault. Budget the time.

> Your professor may also have gold-cup wet electrodes and paste. **If offered, take them for
> O1/O2.** The signal quality difference at occipital sites is substantial, and SSVEP
> amplitude is what your whole project depends on.

---

## 📋 Pre-session actions

### 1. Confirm the wiring you'll actually use
Decide before you arrive. If someone else set up the headband, **verify which physical
electrode is on which pin** — don't assume.

### 2. If you cannot get O1/O2, change the montage instead of the hardware
If the strap only reaches Oz comfortably, wire Oz to channel 6 or 7 and note it. Oz sits
between O1 and O2 and picks up SSVEP well — often better than either alone. Just make sure
whatever you wire to channels 6 and 7 **is occipital**.

### 3. Bring the checklist
`docs/LAB_SESSION_CHECKLIST.md` — especially **Step C, the eyes-closed alpha test**. With dry
electrodes it becomes even more important: it's the fastest way to tell "bad contact" from
"bad algorithm."

---

## 🎯 What "good" looks like on this hardware

| Check | Target |
|---|---|
| Per-channel RMS | 5–50 µV (dry may run higher) |
| Eyes-closed alpha at O1/O2 | **clear increase in 10 Hz power** |
| Fp1/Fp2 blink amplitude | 150–300 µV |
| `backend errors` | 0 |
| Cohen's d after calibration | > 1.5 good · 0.8–1.5 marginal · < 0.8 investigate |

**If eyes-closed alpha doesn't appear at O1/O2, stop and fix the electrodes.** On a dry
headband that's the most likely single point of failure, and no software setting compensates
for it.

---

## Summary

| Question | Answer |
|---|---|
| Right board? | ✅ Cyton, `board_id=0`, 250 Hz — matches your code exactly |
| Enough channels? | ✅ 8 is enough for 2-command SSVEP |
| Right electrodes in the kit? | ✅ comb electrodes reach O1/O2/Oz |
| Will the default wiring work? | ❌ **no — it puts P8 on your SSVEP channel** |
| Fix required? | Wiring only. **No code change.** |
| Dry electrodes OK? | ⚠️ workable, noisier, allow setup time |

The single thing to get right: **channels 6 and 7 must be occipital (O1/O2).** Everything else
in the setup is already aligned with your code.
