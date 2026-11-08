"""
Author: Anson
Created on: 26/10/2026
Purpose: internet-sourced CC0 hand/foot outlines for motor-intent teaching.

The body shapes are not hand-drawn by this project. They are local, recolourable
CC0 assets from FreeSVG/OpenClipart and Wikimedia Commons. See
``bci_sdk/assets/body_map/ASSET_SOURCES.md`` for source and licence details.
"""

from pathlib import Path


_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "body_map"
_ASSET_CACHE = None
_SPRITE_CACHE = {}

# Exact, mutually exclusive masks derived from the downloaded hand silhouette.
# Each mask was isolated at its anatomical web; no finger contains any pixel
# owned by another finger or by the palm. This prevents cue colours spilling
# into an adjacent digit or continuing as a wedge across the palm.
_FINGER_MASK_ASSETS = {
    "thumb": "finger_thumb_mask.png",
    "index": "finger_index_mask.png",
    "middle": "finger_middle_mask.png",
    "ring": "finger_ring_mask.png",
    "pinky": "finger_pinky_mask.png",
}


def _load_assets(pygame):
    global _ASSET_CACHE
    if _ASSET_CACHE is not None:
        return _ASSET_CACHE

    names = {
        "hand_outline": "hand_outline_cc0.png",
        "hand_mask": "hand_silhouette_mask.png",
        "foot_mask": "left_footprint_cc0.png",
        "foot_outline": "left_footprint_outline.png",
        **{f"finger_{finger}": filename
           for finger, filename in _FINGER_MASK_ASSETS.items()},
    }
    missing = [name for name in names.values()
               if not (_ASSET_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "body-map CC0 asset(s) missing: " + ", ".join(missing)
            + f"; expected under {_ASSET_DIR}")

    loaded = {}
    for key, name in names.items():
        image = pygame.image.load(str(_ASSET_DIR / name))
        if pygame.display.get_surface() is not None:
            image = image.convert_alpha()
        loaded[key] = image
    _ASSET_CACHE = loaded
    return loaded


def _tint_mask(pygame, mask_surface, color):
    """Return ``mask_surface`` recoloured while preserving its exact alpha."""
    result = pygame.Surface(mask_surface.get_size(), pygame.SRCALPHA)
    result.fill((*color, 255))
    result.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return result


def _render_hand(pygame, side, selected, whole, colors):
    assets = _load_assets(pygame)
    base_color, outline_color, active_color = colors
    silhouette = assets["hand_mask"]

    hand = _tint_mask(
        pygame, silhouette, active_color if whole else base_color)
    if not whole and selected in _FINGER_MASK_ASSETS:
        # The selected asset is already a complete, source-contour-clipped,
        # mutually exclusive finger region. Unlike the old broad polygons, it
        # cannot include an adjacent finger or any part of the palm.
        finger_mask = assets[f"finger_{selected}"]
        hand.blit(_tint_mask(pygame, finger_mask, active_color), (0, 0))

    hand.blit(_tint_mask(pygame, assets["hand_outline"], outline_color),
              (0, 0))
    if side == "right":
        hand = pygame.transform.flip(hand, True, False)
    return hand


def _render_foot(pygame, side, active_now, colors):
    assets = _load_assets(pygame)
    base_color, outline_color, active_color = colors
    foot = _tint_mask(
        pygame, assets["foot_mask"],
        active_color if active_now else base_color)
    foot.blit(_tint_mask(pygame, assets["foot_outline"], outline_color),
              (0, 0))
    if side == "right":
        foot = pygame.transform.flip(foot, True, False)
    return foot


def _sprite(kind, side, selected, whole, size, colors, pygame):
    key = (kind, side, selected, bool(whole), tuple(size), tuple(colors))
    cached = _SPRITE_CACHE.get(key)
    if cached is not None:
        return cached
    source = (_render_hand(pygame, side, selected, whole, colors)
              if kind == "hand" else
              _render_foot(pygame, side, whole, colors))
    sprite = pygame.transform.smoothscale(source, size)
    if len(_SPRITE_CACHE) >= 96:
        _SPRITE_CACHE.clear()
    _SPRITE_CACHE[key] = sprite
    return sprite


def draw_body_intent_map(surface, rect, highlighted=None, compact=False):
    """Draw two CC0 outline hands and two CC0 footprint outlines.

    ``highlighted`` may be any of the ten finger labels, ``left_hand``,
    ``right_hand``, ``both_feet``, or ``idle``/``None``. Left/right counterparts
    use the same downloaded source mirrored horizontally. Finger highlights use
    mutually exclusive masks, so they fill only the selected finger and cannot
    overlap another finger or the palm.
    """
    import pygame

    rect = pygame.Rect(rect)
    hi = str(highlighted or "idle").strip().lower()
    ink = (220, 232, 235)
    muted = (91, 119, 130)
    base = (34, 62, 75)
    active = (244, 201, 93)
    active2 = (71, 215, 208)
    colors = (base, ink, active)

    font_size = 12 if compact else 17
    small_size = 9 if compact else 12
    font = pygame.font.SysFont("dejavusans,arial", font_size, bold=True)
    small = pygame.font.SysFont("dejavusans,arial", small_size)
    x, y, w, h = rect

    status_y = y + max(10, int(h * .035))
    if hi == "idle":
        status_text, status_color = "REST / NO MOVEMENT", active2
    elif highlighted:
        status_text, status_color = hi.replace("_", " ").upper(), active
    else:
        status_text, status_color = "BODY INTENT", ink
    status = font.render(status_text, True, status_color)
    surface.blit(status, status.get_rect(center=(x + w // 2, status_y)))

    # The downloaded hand is tall and narrow (458:600), avoiding the previous
    # oversized/fat procedural palm. Thumbs face inward after mirroring.
    hand_top = y + int(h * .075)
    hand_h = max(70, int(h * .50))
    hand_w = max(54, round(hand_h * 458 / 600))
    max_hand_w = max(52, int(w * .37))
    if hand_w > max_hand_w:
        hand_w = max_hand_w
        hand_h = max(70, round(hand_w * 600 / 458))
    centers = {"left": x + int(w * .27), "right": x + int(w * .73)}

    for side in ("left", "right"):
        whole = hi == side + "_hand"
        selected = hi[len(side) + 1:] if hi.startswith(side + "_") else None
        hand = _sprite("hand", side, selected, whole,
                       (hand_w, hand_h), colors, pygame)
        hand_rect = hand.get_rect(midtop=(centers[side], hand_top))
        surface.blit(hand, hand_rect)

        side_active = whole or hi.startswith(side + "_")
        title = font.render(side.upper() + " HAND", True,
                            active if side_active else ink)
        title_y = min(y + int(h * .64), hand_rect.bottom + font_size)
        surface.blit(title, title.get_rect(center=(centers[side], title_y)))
        if not compact:
            orientation = small.render("PALM", True, muted)
            surface.blit(orientation,
                         orientation.get_rect(center=(centers[side],
                                                      title_y + font_size)))

    feet_on = hi == "both_feet"
    feet_top = y + int(h * .68)
    foot_h = max(42, int(h * .215))
    foot_w = max(27, round(foot_h * 960 / 1219))
    foot_gap = max(5, round(foot_w * .10))
    foot_centers = (x + w // 2 - foot_w // 2 - foot_gap,
                    x + w // 2 + foot_w // 2 + foot_gap)
    for side, center_x in zip(("left", "right"), foot_centers):
        foot = _sprite("foot", side, None, feet_on,
                       (foot_w, foot_h), colors, pygame)
        surface.blit(foot, foot.get_rect(midtop=(center_x, feet_top)))

    feet_label_y = min(y + h - font_size,
                       feet_top + foot_h + font_size)
    feet_label = font.render("BOTH FEET", True, active if feet_on else ink)
    surface.blit(feet_label,
                 feet_label.get_rect(center=(x + w // 2, feet_label_y)))
