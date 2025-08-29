
# HelmNet SafeGuard — Computer Vision for PPE Compliance

## Summary
I built a deep‑learning system to detect industrial workers **with vs without helmets** from CCTV streams. Four models
(Simple CNN and three VGG‑16 variants) all achieved **100%** metrics; we selected **VGG‑16 + FFNN with Data Augmentation**
for robustness to lighting, camera angles, and occlusions.

### Why it matters
Helmet compliance prevents head injuries and fines. Automating detection enables real‑time alerts and continuous
compliance analytics without adding headcount.

### Highlights
- **Final model:** VGG‑16 (frozen) + FFNN head + Augmentation → **100% Accuracy/Recall/Precision/F1** on test.
- **Metric focus:** **Recall** to minimize missed violations (FN=0 across splits).
- **Data:** 631 RGB images, balanced classes, 200×200.
- **Augmentations:** rotation/shift/shear/zoom; no horizontal flip (safety badges/graphics may be orientation‑sensitive).
- **Ops:** Edge deployment + dashboards (weekly compliance heat‑maps).

### Tech & Workflow
- TensorFlow/Keras, OpenCV, scikit‑learn, Albumentations
- EDA → preprocessing → baselines → transfer learning → augmentation → selection by **Recall** + robustness

### Deliverables
- Notebook with pipeline & visuals
- PDF slides summarizing results
- GitHub repo (README, requirements, training/inference scripts)
