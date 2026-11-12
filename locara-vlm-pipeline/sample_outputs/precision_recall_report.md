Scorer: offline:offline-cv-baseline; clips: 10; positive class: defect (1-5 dims <= 2; framing = 0; duplicate >= 0.5)

| Dimension | Defects in truth | TP | FP | FN | Precision | Recall | F1 | Exact | Within 1 |
|---|---|---|---|---|---|---|---|---|---|
| egocentric_framing | 1 | 1 | 1 | 0 | 0.50 | 1.00 | 0.67 | 0.90 | n/a |
| motion_quality | 1 | 1 | 2 | 0 | 0.33 | 1.00 | 0.50 | 0.30 | 0.70 |
| scene_diversity | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.10 | 0.80 |
| hand_visibility | 3 | 3 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.60 | 0.90 |
| lighting_exposure | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.60 | 1.00 |
| duplicate_risk | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| overall_usability | 5 | 4 | 0 | 1 | 1.00 | 0.80 | 0.89 | 0.60 | 0.80 |

Tier accuracy 0.70; reject detection P 1.00 R 1.00; auto-accept precision n/a (0 auto-accepted, 5 sent to review)

| Truth \ Predicted | usable | borderline | reject |
|---|---|---|---|
| usable | 0 | 3 | 0 |
| borderline | 0 | 2 | 0 |
| reject | 0 | 0 | 5 |
