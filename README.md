# Scene-Referred HDR Radiometric Analysis

Reconstruction and statistical evaluation of HDR radiance maps from the Fairchild HDR Image Dataset.

---

## Overview

This project reconstructs scene-referred High Dynamic Range (HDR) radiance maps from RAW exposure brackets and performs quantitative radiometric analysis across 105 scenes.

Rather than focusing on tone mapping or display rendering, this work investigates HDR at the **radiometric level**:

- How much dynamic range do scenes actually contain?
- Does exposure span predict effective HDR?
- How does luminance distribution structure influence dynamic range?
- Are 9 exposure brackets necessary?

The repository provides a structured and reproducible pipeline from RAW (.NEF) files to statistical evaluation and visualization.

---

## Dataset: Fairchild HDR Image Dataset

Each scene contains:

- 9 exposure-bracketed RAW images (.NEF)
- Identical viewpoint
- Varying shutter times
- Fixed aperture and ISO

Total scenes processed: **105**

Directory structure:

```bash
data/raw/
    Scene_Name/
        exposure_1.nef
        ...
        exposure_9.nef
```

Exposure spans vary across scenes (~6–15 EV).

---

## Pipeline Overview

The complete radiometric workflow:

1. RAW decoding  
2. Linear radiance recovery  
3. GPU-accelerated HDR merging  
4. EXR export (floating-point radiance maps)  
5. Dynamic range computation  
6. Statistical analysis  

---

## RAW → Linear Radiance

RAW files are decoded using `rawpy` with:

- No automatic white balance  
- No automatic brightness scaling  
- Gamma = (1,1) for linear output  
- 16-bit precision  

Output images are normalized to [0,1] and treated as linear RGB.

This ensures HDR merging operates in **scene-referred linear space**.

---

## HDR Merge Method

HDR merging follows a weighted radiance model:

L = Σ [ w(z) · (z / t) ] / Σ w(z)

Where:

- \( z \) = normalized pixel intensity  
- \( t \) = exposure time  
- \( w(z) \) = triangular weighting function  

Implementation details:

- GPU acceleration via PyTorch  
- Float32 precision  
- Output stored as OpenEXR (.exr)

Each scene produces:

```bash
data/hdr_exr/Scene_Name.exr
```

---

## Dynamic Range Measurement

Dynamic range is computed robustly using percentile-based luminance:

DR = log₁₀( P₉₉.₉ / P₀.₁ )

Where luminance is:

\[
Y = 0.2126R + 0.7152G + 0.0722B
\]

Percentiles reduce sensitivity to extreme outliers.

---

## Metrics Computed Per Scene

For each of 105 scenes:

- Dynamic Range (log10)
- Log-luminance standard deviation
- Exposure span (EV)
- Theoretical DR derived from EV span

---

## Statistical Analysis

### Correlation

- **Corr(DR, EV_span)** = 0.2716  
- **Corr(DR, LogSpread)** = 0.7358  

Interpretation:

- Exposure span weakly predicts effective dynamic range.
- Luminance distribution structure strongly predicts dynamic range.

---

### Linear Regression (DR ~ EV_span)

- **R²** = 0.0738  
- **Slope** = 0.2778  
- **Intercept** = 0.3467  

Exposure span explains only ~7% of DR variance across scenes.

---

## Exposure Redundancy Study

Representative scenes were reconstructed using:

- 9 exposures  
- 7 exposures  
- 5 exposures  
- 3 exposures  

Mean DR change:

- Δ (3 vs 9 exposures) ≈ 0.0376 log10  
- Δ (5 vs 9 exposures) ≈ −0.0308 log10  

Observation:

Reducing exposure count produces minimal change in effective dynamic range for most scenes.

---

## Generated Visualizations

All figures are automatically generated in:

```bash
results/figures/
```

Generated outputs:

- `dr_distribution.png`
- `dr_vs_ev.png`
- `dr_vs_logspread.png`
- `correlation_matrix.png`
- `correlation_summary.png`
- `redundancy_table.png`
- `final_summary.png`

---

## Repository Structure

```bash
scene-referred-hdr-radiometry/

├── src/
│   ├── analysis.py
│   ├── hdr_merge.py
│   ├── nef2exr.py
│   ├── metrics.py
│   ├── raw_utils.py
│   └── exr_utils.py
│
├── scripts/
│   ├── run_convert_nef2exr.py
│   └── run_full_analysis.py
│
├── config.py
└── requirements.txt
```

---

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Convert RAW to EXR:

```bash
python scripts/run_convert_nef2exr.py
```

Run full analysis:

```bash
python scripts/run_full_analysis.py
```

All outputs are generated automatically.

---

## Conclusions

This project demonstrates:

- Scene-referred HDR reconstruction from RAW exposure brackets  
- Robust percentile-based dynamic range measurement  
- Weak predictive power of exposure span  
- Strong dependence on luminance distribution structure  
- Redundancy in exposure bracketing for many scenes  

The repository provides a structured, reproducible framework for HDR radiometric analysis.

---

##  Author

Adhavan Murugaiyan  
