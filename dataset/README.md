# Public EEG data boundary

Edited by Anson  
Date: 10/11/2026  
Purpose: keep downloaded and derived public EEG outside source control.

- Put manually retained source downloads in `raw/<dataset_id>/`.
- The automated workflow uses `work/<run-id>/` for resumable temporary downloads and removes only that workflow-owned directory after successful verification.
- Import standardized archives into `processed/<dataset_id>/<category>/`.
- Do not place personal Cyton calibration here; it remains under `dataset/motor_intent/` and `profiles/`.
- Do not redistribute a processed archive unless its source license allows it. In particular, the BNCI2014-001 registry entry is CC-BY-ND and marks derived redistribution as false.

See `../../docs/PUBLIC_MI_DATASETS.md` for commands and citations.
