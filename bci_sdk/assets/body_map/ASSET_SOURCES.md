# Body-map asset sources

These images are stored locally so the pygame calibration screen does not need
internet access at runtime.

## Open-palm outline

- Project file: `hand_outline_cc0.png`
- Derived clipping mask: `hand_silhouette_mask.png`
- Derived non-overlapping finger masks:
  - `finger_thumb_mask.png`
  - `finger_index_mask.png`
  - `finger_middle_mask.png`
  - `finger_ring_mask.png`
  - `finger_pinky_mask.png`
- Source title: **Open palm outline vector graphics**
- Source page: https://freesvg.org/open-palm-outline-vector-graphics
- FreeSVG/OpenClipart SVG ID: **24847**
- Downloaded PNG: https://freesvg.org/img/Simple-Palm.png
- Licence stated by source: **Public Domain (CC0)**; commercial use and
  modification permitted; attribution not required.
- Local changes: transparent margins cropped; line alpha retained; a closed
  silhouette mask was derived from the outline. Each finger was then isolated
  from that silhouette at its anatomical web into a mutually exclusive mask.
  The five masks have zero shared pixels, preventing a selected cue from
  spilling into an adjacent finger or the palm. The right hand is produced at
  runtime by mirroring the left-hand assets.

## Left footprint

- Project file: `left_footprint_cc0.png`
- Derived outline: `left_footprint_outline.png`
- Source title: **Left footprint.svg**
- Source page: https://commons.wikimedia.org/wiki/File:Left_footprint.svg
- Original file: https://upload.wikimedia.org/wikipedia/commons/5/55/Left_footprint.svg
- Author listed by Wikimedia Commons: **Benji Park**
- Licence stated and reviewed by Wikimedia Commons: **CC0 1.0 / public-domain
  dedication** (originally from the Open Clip Art Library).
- Local changes: source alpha retained and recoloured at runtime; an outline
  alpha was derived from the silhouette. The right foot is produced at runtime
  by mirroring the left-foot asset.

No Shutterstock, stock-preview, or watermarked image is included.
