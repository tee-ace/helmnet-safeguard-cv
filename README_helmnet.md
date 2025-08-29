
# HelmNet SafeGuard — Computer Vision for PPE Compliance

## Project Overview
HelmNet SafeGuard detects whether industrial workers are **With Helmet** or **Without Helmet** using deep learning.
We built and compared four models — **Simple CNN**, **VGG‑16**, **VGG‑16 + FFNN**, and **VGG‑16 + FFNN + Data Augmentation** —
and selected the augmented VGG‑16 variant as the production model for its robustness while retaining **100%** metrics.

## Business Problem
Manual helmet checks on factory floors and construction sites are error‑prone and costly. Missing violations increases injury risk
and regulatory exposure. HelmNet automates PPE monitoring on CCTV feeds and triggers near‑real‑time alerts.

## Data
- **Images:** 631 RGB images, **200×200** pixels
- **Labels:** Binary (`With Helmet`, `Without Helmet`) — balanced (311 vs 320)
- **Splits:** Train 80%, Val 20%, Test 20% (approx. 504 / 63 / 64)
- **Diversity:** Indoor/outdoor scenes, varied angles/lighting/occlusions

## Methods
1. **EDA:** Sample visualization, label balance checks, grayscale experiment, pixel normalization to [0,1].
2. **Preprocessing:** RGB retained for modeling; grayscale explored; `train/val/test` split; augmentation on final model only.
3. **Models:**
   - **Model 1 — Simple CNN (baseline):** 3×Conv (32→64→128, 3×3, ReLU) → 2×MaxPool (4×4) → Flatten → Dense(4, ReLU) → Sigmoid.
   - **Model 2 — VGG‑16 (frozen backbone):** Flatten → Sigmoid head.
   - **Model 3 — VGG‑16 + FFNN:** Dense(256, ReLU) → Dropout(0.5) → Dense(128, ReLU) → Sigmoid.
   - **Model 4 — VGG‑16 + FFNN + Augmentation:** real‑time augmentation during training.
4. **Training Config (typical):** Adam(lr=1e‑3), BCE loss, batch=32, epochs=10–15; metric emphasis: **Recall** to minimize FN.
5. **Augmentation (Model 4):** rotation=20°, width/height_shift=0.2, shear=0.2, zoom=0.2, horizontal_flip=False.

## Results
All four models reached **100% Accuracy, Recall, Precision, and F1** on train, validation, and test within 10–15 epochs.
- **Final choice:** **Model 4 (VGG‑16 + FFNN + Aug)** — identical metrics with improved robustness to lighting, angles, and occlusions.
- **Confusion Matrices:** 0 FP / 0 FN across splits — perfectly meets safety goal of **never missing a no‑helmet case**.

## Actionable Recommendations
- **Deploy at the Edge:** Run Model 4 on edge devices/CCTV for instant detection; send “no‑helmet” alerts to supervisors.
- **Ops Dashboards:** Weekly compliance heat‑maps and per‑zone trend reports.
- **Continuous Learning:** Capture edge cases (new helmet colors, lighting) for quarterly re‑training.
- **Extend PPE Coverage:** Fine‑tune to detect gloves, vests, face shields with additional labeled data.

## How to Run
```bash
git clone https://github.com/yourusername/helmnet-safeguard-cv.git
cd helmnet-safeguard-cv
pip install -r requirements.txt

# Train final model
python src/train_vgg16_aug.py  --data_dir ./data  --epochs 15  --batch_size 32

# Single-image inference
python src/inference.py --image ./samples/test.jpg

# (Optional) Evaluate on test set
python src/inference.py --eval --data_dir ./data/test
```

## Repository Structure
```
helmnet-safeguard-cv/
├─ data/                 # placeholder; do not commit real images
├─ notebooks/
│  └─ HelmNet_Safety_Computer_Vision_Project.ipynb
├─ reports/
│  └─ HelmNet_Safeguard_Computer_Vision_Project.pdf
├─ src/
│  ├─ train_cnn.py
│  ├─ train_vgg16.py
│  ├─ train_vgg16_ffnn.py
│  ├─ train_vgg16_aug.py
│  └─ inference.py
├─ README.md
├─ requirements.txt
└─ LICENSE
```

## Notes
- **Metric choice:** **Recall** prioritized to avoid **False Negatives** (never miss unsafe “no‑helmet” workers).
- Perfect scores suggest strong separability or a small/very clean dataset — monitor carefully in production and add hard samples over time.
