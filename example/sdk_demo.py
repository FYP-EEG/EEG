"""
A complete brain-controlled app written ONLY against the public bci_sdk API.

This file is the acceptance test for Gap 3: it touches no internals (no ML.*,
no EEG.*, no pygame_lib.*) and is roughly 60 lines of real logic. If a third-party
developer can write this, the SDK is usable.

    python example/sdk_demo.py                 # synthetic EEG
    python example/sdk_demo.py --source replay
    python example/sdk_demo.py --profile dataset/profile_*.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame
import bci_sdk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sim",
                    choices=["sim", "replay", "hardware"])
    ap.add_argument("--profile", default=None)
    ap.add_argument("--seconds", type=float, default=None)
    args = ap.parse_args()

    pygame.init()
    screen, display_info = bci_sdk.open_display((760, 460))
    pygame.display.set_caption("bci_sdk demo - smart home control")
    font = pygame.font.SysFont("dejavusans,arial", 22, bold=True)
    small = pygame.font.SysFont("dejavusans,arial", 15)

    # ---- 1. build brain-controlled buttons -------------------------------
    labels = [("LIGHTS", "lights"), ("MUSIC", "music"),
              ("FAN", "fan"), ("HELP", "help")]
    buttons = []
    for i, (label, value) in enumerate(labels):
        col, row = i % 2, i // 2
        buttons.append(bci_sdk.BrainButton(
            label, rect=(60 + col * 330, 150 + row * 120, 300, 100),
            value=value, dwell=0.8))
    group = bci_sdk.ButtonGroup(buttons)

    # ---- 2. two-command scanning over them -------------------------------
    scanner = bci_sdk.Scanner([buttons[:2], buttons[2:]])

    log = ["Ready."]

    # ---- 3. attach behaviour with decorators -----------------------------
    for b in buttons:
        @b.on_brain_select
        def activated(btn):
            log.append(f"Activated: {btn.label}")
            del log[:-4]

    @scanner.on_stage_change
    def stage_changed(stage):
        log.append(f"scanning {'rows' if stage == 'ROW' else 'buttons'}")
        del log[:-4]

    # ---- 4. one session object supplies the brain commands ---------------
    session = bci_sdk.BCISession(source=args.source, profile=args.profile)

    @session.on_command
    def got_command(cmd):
        scanner.handle_command(cmd)

    stim = bci_sdk.StimulusEngine(target_freqs=session.target_freqs,
                                  refresh_hz=display_info["refresh"])
    clock = pygame.time.Clock()

    with session:
        running = True
        import time
        end = None if args.seconds is None else time.time() + args.seconds
        while running and (end is None or time.time() < end):
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (
                        e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_LEFT:
                    scanner.next()
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_RIGHT:
                    scanner.select()
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    hit = group.at(pygame.mouse.get_pos())
                    if hit:
                        hit.select()

            session.poll()                 # non-blocking
            scanner.update()               # dwell timers
            scanner.update_highlight()

            screen.fill((20, 22, 30))
            title = font.render("Smart Home  -  brain controlled", True, (235, 240, 250))
            screen.blit(title, (60, 40))
            sub = small.render(
                f"src={session.source}   state={session.last_state}   "
                f"LEFT=next  RIGHT=select  ESC=quit", True, (150, 158, 180))
            screen.blit(sub, (60, 78))
            scanner.draw(screen)
            for i, line in enumerate(log[-3:]):
                screen.blit(small.render(line, True, (140, 200, 160)),
                            (60, 400 + i * 18))
            pygame.display.flip()
            stim.tick()
            clock.tick(60)

    session.print_report()
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
