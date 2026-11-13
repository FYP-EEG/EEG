"""
Author: Anson
Created on: 19/10/2026
Purpose: separate live dashboard for gameplay EEG and prediction performance.

Start a game, choose GAME, then run in another terminal:
    python example/performance_monitor.py

Optional live modes:
    python example/performance_monitor.py --app rockpaperscissors
    python example/performance_monitor.py --session runtime_logs/.../<stamp>

Training-data mode:
    python example/performance_monitor.py --training "dataset/motor_intent/calib_brian_*.npz"
    python example/performance_monitor.py --training dataset/motor_intent --model profiles/.../intent_model.joblib

The dashboard only reads local files. It does not modify the model or
automatically feed manual labels into training. Model results shown for saved
training epochs are explicitly previews, not held-session validation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _resolve_session(root, app=None, explicit=None):
    if explicit:
        return Path(explicit)
    pointer = root / (str(app) if app else "") / "latest_session.json"
    if not app:
        pointer = root / "latest_session.json"
    data = _read_json(pointer)
    return Path(data["session_dir"]) if data and data.get("session_dir") else None


def _draw_card(surface, rect, title, value, color, font, small):
    pygame.draw.rect(surface, (16, 31, 42), rect, border_radius=10)
    pygame.draw.rect(surface, (49, 78, 91), rect, 1, border_radius=10)
    surface.blit(small.render(title, True, (145, 170, 179)), (rect.x + 12, rect.y + 9))
    text = font.render(str(value), True, color)
    surface.blit(text, (rect.x + 12, rect.y + 31))


def _draw_eeg(surface, rect, eeg, channels, fs, font, small,
              title_text="LATEST 3-SECOND EEG WINDOW"):
    pygame.draw.rect(surface, (10, 23, 32), rect, border_radius=12)
    pygame.draw.rect(surface, (49, 78, 91), rect, 1, border_radius=12)
    title = font.render(title_text, True, (220, 232, 235))
    surface.blit(title, (rect.x + 14, rect.y + 10))
    if eeg is None or np.asarray(eeg).ndim != 2:
        waiting = font.render("Waiting for live EEG windows...", True, (244, 201, 93))
        surface.blit(waiting, waiting.get_rect(center=rect.center))
        return

    x = np.asarray(eeg, dtype=float)
    n_channels, n_samples = x.shape
    plot = pygame.Rect(rect.x + 68, rect.y + 42, rect.width - 84, rect.height - 58)
    row_h = plot.height / max(1, n_channels)
    # Decimate only for drawing; saved latest_eeg.npy retains all samples.
    step = max(1, int(np.ceil(n_samples / max(1, plot.width))))
    indices = np.arange(0, n_samples, step)
    for channel in range(n_channels):
        centre = plot.y + (channel + .5) * row_h
        pygame.draw.line(surface, (31, 52, 63),
                         (plot.x, int(centre)), (plot.right, int(centre)), 1)
        signal = x[channel]
        signal = signal - np.nanmedian(signal)
        scale = float(np.nanpercentile(np.abs(signal), 98)) if signal.size else 1.0
        scale = max(scale, 1e-6)
        values = np.clip(signal[indices] / scale, -1, 1)
        points = []
        for pos, value in enumerate(values):
            px = plot.x + pos * (plot.width - 1) / max(1, len(values) - 1)
            py = centre - value * row_h * .39
            points.append((int(px), int(py)))
        if len(points) >= 2:
            pygame.draw.aalines(surface, (71, 215, 208), False, points)
        name = channels[channel] if channel < len(channels) else f"CH{channel + 1}"
        label = small.render(name, True, (190, 211, 216))
        surface.blit(label, label.get_rect(midright=(plot.x - 8, int(centre))))
    duration = n_samples / max(1, fs)
    note = small.render(
        f"{n_channels} channels · {n_samples} samples · {duration:.2f}s · auto-scale per channel",
        True, (120, 148, 158))
    surface.blit(note, note.get_rect(bottomright=(rect.right - 12, rect.bottom - 7)))


def _run_training_dashboard(training):
    """Interactive inspection of saved labelled EEG trials."""
    pygame.init()
    window = pygame.display.set_mode((1100, 720), pygame.RESIZABLE)
    pygame.display.set_caption("BCI Training Data Performance Monitor")
    clock = pygame.time.Clock()
    big = pygame.font.SysFont("dejavusans,arial", 25, bold=True)
    font = pygame.font.SysFont("dejavusans,arial", 18, bold=True)
    small = pygame.font.SysFont("dejavusans,arial", 13)
    report = training.report()
    channels = report.get("channel_labels", [])
    index = 0
    kept_only = False
    running = True

    def available_indices():
        if kept_only:
            values = np.flatnonzero(training.keep).tolist()
            return values or list(range(len(training.X)))
        return list(range(len(training.X)))

    while running:
        indices = available_indices()
        if index not in indices:
            index = indices[0]
        position = indices.index(index)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and
                    event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE)):
                running = False
            elif event.type == pygame.VIDEORESIZE:
                window = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    index = indices[(position + 1) % len(indices)]
                elif event.key == pygame.K_LEFT:
                    index = indices[(position - 1) % len(indices)]
                elif event.key == pygame.K_HOME:
                    index = indices[0]
                elif event.key == pygame.K_END:
                    index = indices[-1]
                elif event.key == pygame.K_k:
                    kept_only = not kept_only
                    index = available_indices()[0]

        trial = training.trial(index)
        width, height = window.get_size()
        window.fill((7, 16, 24))
        heading = big.render("TRAINING DATA PERFORMANCE", True, (235, 245, 247))
        window.blit(heading, (22, 16))
        source = f"{report['sessions']} file(s) · {', '.join(report['subjects'])} · {report['montage']}"
        window.blit(small.render(source[-125:], True, (125, 151, 160)), (23, 49))

        train_acc = report.get("training_balanced_accuracy")
        values = [
            ("TRIALS", report["trials"], (220, 232, 235)),
            ("KEPT", report["kept"], (125, 219, 155)),
            ("DROPPED", report["dropped"], (239, 165, 139)),
            ("CLASSES", len(report["class_counts"]), (244, 201, 93)),
            ("MODEL PREVIEW", "--" if train_acc is None else f"{train_acc:.1%}",
             (71, 215, 208)),
        ]
        gap = 10
        card_w = (width - 44 - gap * 4) // 5
        for card_index, (title, value, color) in enumerate(values):
            _draw_card(window,
                       pygame.Rect(22 + card_index * (card_w + gap), 76,
                                   card_w, 72),
                       title, value, color, font, small)

        intended = trial["intended"].replace("_", " ").upper()
        predicted = ("--" if trial["predicted"] is None else
                     trial["predicted"].replace("_", " ").upper())
        quality = "KEEP" if trial["keep"] else "DROP"
        reasons = "none" if not trial["reasons"] else ", ".join(trial["reasons"])
        prediction_color = ((125, 219, 155) if trial["correct"] is True else
                            (255, 128, 111) if trial["correct"] is False else
                            (160, 184, 192))
        line = (f"TRIAL {index + 1}/{len(training.X)} · intended={intended} · "
                f"predicted={predicted} · quality={quality}")
        window.blit(font.render(line, True, prediction_color), (24, 160))
        window.blit(small.render(
            f"subject={trial['subject']} · rejection reasons={reasons}",
            True, (190, 211, 216)), (25, 190))
        counts = "  |  ".join(
            f"{name.replace('intent::', '')}: {report['retained_counts'][name]}/{count}"
            for name, count in report["class_counts"].items())
        window.blit(small.render(counts[:150], True, (145, 170, 179)), (25, 211))
        reason_totals = "  |  ".join(
            f"{name}: {count}" for name, count
            in report.get("drop_reason_counts", {}).items() if count)
        if not reason_totals:
            reason_totals = "drop reasons: none"
        window.blit(small.render(reason_totals[:150], True, (239, 165, 139)),
                    (25, 229))

        _draw_eeg(window, pygame.Rect(20, 255, width - 40,
                                      max(160, height - 295)),
                  trial["window"], channels, report["fs"], font, small,
                  title_text="SELECTED LABELLED TRAINING EPOCH")
        controls = ("Left/Right: trial · Home/End: first/last · "
                    f"K: {'show all' if kept_only else 'kept only'} · Esc: close")
        window.blit(small.render(controls, True, (145, 170, 179)),
                    (24, height - 27))
        warning = small.render(
            "MODEL PREVIEW uses training data; it is not held-session accuracy.",
            True, (239, 165, 139))
        window.blit(warning, warning.get_rect(bottomright=(width - 22, height - 8)))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    return 0


def main():
    parser = argparse.ArgumentParser(description="Live EEG/model performance dashboard")
    parser.add_argument("--root", default=str(ROOT / "runtime_logs"),
                        help="live gameplay log root")
    parser.add_argument("--app", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--training", nargs="+", default=None,
                        help="training NPZ path(s), directories or glob patterns")
    parser.add_argument("--model", default=None,
                        help="optional intent_model.joblib for training preview")
    parser.add_argument("--user", default=None,
                        help="resolve model from this runtime profile user")
    parser.add_argument("--intents", default=None,
                        help="comma-separated intents used with --user")
    parser.add_argument("--amp-uv", type=float, default=400.0,
                        help="gross-amplitude screening threshold")
    parser.add_argument("--jump-uv", type=float, default=75.0,
                        help="sample-to-sample jump threshold")
    parser.add_argument("--flat-uv", type=float, default=.05,
                        help="per-channel standard-deviation floor")
    parser.add_argument("--z-max", type=float, default=4.0,
                        help="within-class robust band-power z threshold")
    args = parser.parse_args()
    log_root = Path(args.root)

    if args.training:
        from bci_sdk.performance import TrainingDataPerformance
        model_path = args.model
        if model_path is None and args.user and args.intents:
            from ML.motor_intent import intent_profile_dir
            intents = [value.strip() for value in args.intents.split(",")
                       if value.strip()]
            model_path = intent_profile_dir(args.user, intents) / "intent_model.joblib"
        training = TrainingDataPerformance(
            args.training, model_path=model_path,
            screening_kw={"amp_uv": args.amp_uv, "jump_uv": args.jump_uv,
                          "flat_uv": args.flat_uv, "z_max": args.z_max})
        return _run_training_dashboard(training)

    pygame.init()
    window = pygame.display.set_mode((1100, 720), pygame.RESIZABLE)
    pygame.display.set_caption("BCI Prediction Performance Monitor")
    clock = pygame.time.Clock()
    big = pygame.font.SysFont("dejavusans,arial", 25, bold=True)
    font = pygame.font.SysFont("dejavusans,arial", 18, bold=True)
    small = pygame.font.SysFont("dejavusans,arial", 13)

    session = None
    status = None
    eeg = None
    status_mtime = eeg_mtime = None
    last_resolve = 0.0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and
                    event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE)):
                running = False
            elif event.type == pygame.VIDEORESIZE:
                window = pygame.display.set_mode(event.size, pygame.RESIZABLE)

        now = time.monotonic()
        if session is None or now - last_resolve > 1.0:
            session = _resolve_session(log_root, args.app, args.session)
            last_resolve = now
        if session:
            status_path = session / "status.json"
            try:
                modified = status_path.stat().st_mtime_ns
                if modified != status_mtime:
                    status = _read_json(status_path)
                    status_mtime = modified
            except OSError:
                pass
            eeg_path = session / "latest_eeg.npy"
            try:
                modified = eeg_path.stat().st_mtime_ns
                if modified != eeg_mtime:
                    eeg = np.load(eeg_path, allow_pickle=False)
                    eeg_mtime = modified
            except (OSError, ValueError):
                pass

        width, height = window.get_size()
        window.fill((7, 16, 24))
        heading = big.render("BCI PREDICTION PERFORMANCE", True, (235, 245, 247))
        window.blit(heading, (22, 16))
        source = (str(session) if session else
                  "Waiting for runtime_logs/latest_session.json...")
        source_text = small.render(source[-125:], True, (125, 151, 160))
        window.blit(source_text, (23, 49))

        if status:
            stats = status.get("stats", {})
            accuracy = status.get("manual_accuracy")
            values = [
                ("WINDOWS", stats.get("windows", 0), (220, 232, 235)),
                ("CLASSIFIED", stats.get("classified", 0), (244, 201, 93)),
                ("DISPATCHED", stats.get("dispatched", 0), (125, 219, 155)),
                ("REJECTED", stats.get("rejected", 0), (239, 165, 139)),
                ("MANUAL ACCURACY", "--" if accuracy is None else f"{accuracy:.1%}",
                 (71, 215, 208)),
            ]
            gap = 10
            card_w = (width - 44 - gap * 4) // 5
            for index, (title, value, color) in enumerate(values):
                _draw_card(window, pygame.Rect(22 + index * (card_w + gap), 76,
                                               card_w, 72),
                           title, value, color, font, small)

            prediction = status.get("latest_prediction") or {}
            feedback = status.get("last_feedback") or {}
            pred_text = (f"LATEST: {prediction.get('command', '--')}  |  "
                         f"selected={prediction.get('selected') or '--'}  |  "
                         f"confidence={prediction.get('confidence', 0):.2f}  |  "
                         f"state={prediction.get('state', '--')}")
            window.blit(font.render(pred_text, True, (220, 232, 235)), (24, 162))
            feedback_text = ("FEEDBACK: none" if not feedback else
                             f"FEEDBACK: {'CORRECT' if feedback.get('correct') else 'INCORRECT'} "
                             f"for prediction #{feedback.get('prediction_id')}")
            feedback_color = ((125, 219, 155) if feedback.get("correct")
                              else (239, 165, 139))
            window.blit(small.render(feedback_text, True, feedback_color), (25, 190))
            pause_text = "PAUSED" if status.get("paused") else "LIVE"
            window.blit(font.render(pause_text, True,
                                    (244, 201, 93) if status.get("paused")
                                    else (125, 219, 155)), (width - 110, 160))
            channels = status.get("channels", [])
            fs = int(status.get("sample_freq", 250))
        else:
            channels, fs = [], 250
            waiting = font.render("Start a demo and select GAME to create a session.",
                                  True, (244, 201, 93))
            window.blit(waiting, (24, 105))

        eeg_top = 220
        _draw_eeg(window, pygame.Rect(20, eeg_top, width - 40,
                                      max(160, height - eeg_top - 20)),
                  eeg, channels, fs, font, small)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
