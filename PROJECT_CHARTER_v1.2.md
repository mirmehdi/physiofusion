# Project Charter & AI Kickoff Prompt
## A Reusable Multimodal Physiological-Signal ML Stack, Applied to Glucose

**Version:** 1.2  
**Owner:** Project Lead (AI Lead)  
**Status:** Initiation  
**One-line:** Build a reusable, uncertainty-aware ML stack for multimodal physiological signals, and demonstrate it across three glucose problems of increasing data richness — CGM forecasting, non-invasive estimation, and optical-spectral modeling.

### Changelog (v1.1 → v1.2)
Re-centered after a priorities update: **portfolio + skillset are the primary objective; a paper is an optional byproduct.** Changes: (1) the deliverable is now a **reusable component library** demonstrated across multiple tasks, not a single benchmark; (2) restructured into **three tracks sharing that library**; (3) **CGM glucose forecasting promoted to the primary, data-rich skill-building engine** (the modern stack — SSL, state-space models, ensembles — needs the data volume the wearable sets lack); (4) the **CGM dataset ecosystem** added (OhioT1DM, DiaData, T1DiabetesGranada, OpenAPS, Shanghai, JAEB); (5) **all license-gating and "verify license" hedging removed** — these are openly downloadable research datasets and this is a non-commercial portfolio project; (6) the **MLOps/cloud layer promoted from optional to core scope**, because building it is itself a target skill. Rigor (cross-subject validation, conformal uncertainty, leakage-proof harness) is retained as engineering best practice that makes the portfolio credible.

---

### 0. How to use this document
Both the project charter and the context prompt to bootstrap an AI assistant (Claude / Claude Code) in a fresh session. Paste it whole; it is self-contained. It defines *what* and *why*; the assistant helps build the *how*.

### 0a. Primary objective and what "done" means
**Primary objective: maximize demonstrated skill breadth and depth, and produce a portfolio centerpiece.** The centerpiece is a clean, reusable ML stack (Stages 1–8 below) applied across the three tracks, with the engineering rigor that signals seniority. A paper is welcome if a track yields one, but it does not drive scope.

Run the tracks **sequentially**, Track 1 first; each reuses the shared library, which is what keeps this one coherent project rather than three.

---

### 1. Business / domain context
- **Two glucose problems.** *Forecasting* (predict near-future glucose from CGM history + context — what diabetes-tech companies ship today) and *estimation* (infer current glucose from non-invasive physiology — the harder, more differentiated frontier).
- **Why this is good portfolio.** Track 1 maps to Dexcom / Abbott / Medtronic / Tidepool / Loop; Track 2 maps to DiaMonTech / Apple / wearables / biosensing; Track 3 maps to spectroscopy / AI-for-science. One library, three industry-relevant demonstrations.
- **Honesty guardrail (kept, lightweight).** This is a research/portfolio project, not a product. No clinical or diagnostic claim. Context: the FDA's Feb-2024 safety communication states it has not authorized any smartwatch or smart ring to measure glucose on its own — so non-invasive estimation is framed as research, not deployment.

### 2. The shared component library (Stages 1–8) — the centerpiece
A modular stack reused across tracks. Each module is a learnable skill and a portfolio artifact.
1. **Signal front-end** — windowing/alignment, filtering, signal-quality scoring, feature extraction. (SciPy, NeuroKit2, HeartPy; cgmquantify for glycemic-variability metrics.)
2. **Temporal encoders** — TCN, LSTM/GRU, temporal Transformer (PatchTST-style), state-space (Mamba/S4), optional Kalman head.
3. **Modality encoders** — per-signal embeddings (CGM/insulin/meal/activity for Track 1; PPG/EDA/temp/accel for Track 2; spectra for Track 3).
4. **Fusion** — late fusion (default) → cross-attention → uncertainty-weighted.
5. **Self-supervised learning** — masked modeling (CGM or spectral), contrastive, cross-modal alignment. *(Now first-class: Track 1's data volume makes it real.)*
6. **Uncertainty & OOD** — deep ensembles, MC-dropout, **conformal prediction**, evidential/heteroscedastic; OOD via embedding distance, reconstruction error, signal-quality gating.
7. **Personalization** — population (cross-subject) → subject embeddings / mixed-effects → adapter/LoRA → meta-learning (MAML) → test-time adaptation. The accuracy-vs-label-budget curve is the signature result.
8. **Distillation & edge** — cloud teacher → on-device student (soft outputs + embeddings + uncertainty); export ONNX → LiteRT.

### 3. The three tracks
**Track 1 — CGM glucose forecasting (PRIMARY skill engine; data-rich; start here).**
Predict glucose at 30/60-min horizons from CGM + insulin + meals + activity. This is where the full modern stack is learned and demonstrated at scale, against published benchmarks. Exercises Stages 1–8 with enough data for SSL, state-space models, and ensembles to be meaningful.

**Track 2 — Non-invasive estimation from wearable physiology (your signature; differentiated).**
PPG/EDA/temp/accel → glucose, CGM as ground truth. The calibration-free, cross-subject, multimodal-fusion story tied to the DiaMonTech background. Smaller data; reuses the library; this is the differentiation.

**Track 3 — Optical spectral modeling (optional stretch; most novel).**
NIR/MIR (and Raman if available) spectra → glucose: chemometrics → deep spectral encoders → physics-informed band attention + interferent unmixing → masked-spectral SSL. Lowest data availability; highest novelty.

### 4. Data sources (all openly downloadable)
| Track | Dataset | Scale / contents | Role |
|---|---|---|---|
| **1** | **OhioT1DM** | 12 patients, 8 wks: CGM 5-min, insulin, meals, fitness-band; **predefined splits + published baselines** (~6.45 mg/dL RMSE @30 min) | **Start here** — benchmark + harness proof |
| **1** | **DiaData** | **1,720 subjects, 149M CGM values** (integrates 13–15 sets); Zenodo + GitHub | Scale + **SSL pretraining** |
| **1** | **T1DiabetesGranada** | **736 patients, 257,780 days** over 4 yrs + clinical | Generalization / personalization |
| **1** | **OpenAPS Data Commons** | 184 people, 23 countries, free-living DIY, ~1.6M samples | Real-world robustness |
| **1** | **ShanghaiT1DM / T2DM** | 12 T1D + 100 T2D, CGM + diet + clinical (figshare) | Easy open grab; rare T2D |
| **1** | **JAEB public** (CITY, ReplaceBG, RT-CGM, DCLP) | clinical-trial scale, public.jaeb.org | Cross-dataset transfer |
| **2** | **BIG IDEAs Lab** | PPG/EDA/temp/accel + Dexcom G6, ~26k readings | Wearable estimation core |
| **2** | **PhysioCGM** | + ECG/respiration, 10 T1D, ≤17 days | External validation |
| **2** | **D1NAMO** | ECG/breathing/accel + glucose | Fallback / ablation |
| **3** | In-vitro NIR/MIR glucose + large NIR corpus | spectra (+ unlabeled corpus) | Spectral track + pretraining |

**Reproducibility references (not dependencies):** DIAX (JSON standard for CGM/insulin/meal) and the diabetes-dataset harmonization "time-aligned tabular format" guidelines — useful citations if you write anything up.

**Data discipline (kept — this is what makes results credible):** split by subject/patient, always; a subject is in exactly one of {train, val, test}; enforce with `LeaveOneGroupOut`/`GroupKFold` and a **CI test that fails the build if any subject crosses splits.** Most inflated glucose results come from same-subject leakage.

### 5. Methodology — earn complexity
Simple → complex; every model evaluated through the same leakage-proof harness; the evaluator is the single source of truth (one schema, one script generates all tables).

**Model ladder (per track):** classical baseline (Elastic Net + XGBoost/LightGBM on engineered features — often hard to beat) → TCN/1D-CNN → PatchTST-style Transformer → (Track 1) TS2Vec-style SSL pretraining + Mamba/S4 ablation. Fusion: late → cross-attention. Uncertainty: ensemble/dropout + conformal. Personalization: population → subject-embed → LoRA → (optional) MAML/TTA.

### 6. Evaluation & metrics
- **Validation:** leave-one-subject-out / grouped CV; report mean ± spread across subjects. For Track 1, also use OhioT1DM's predefined splits to compare against published baselines.
- **Forecasting (Track 1):** RMSE, MAE at 30/60-min horizons; Clarke/Parkes Error Grid; hypo/hyper event detection.
- **Estimation (Track 2):** MARD, RMSE, MAE; Clarke/Parkes.
- **Uncertainty:** conformal coverage vs. nominal (±2%), interval width, calibration plots.
- **Ablations:** modality-alone vs. fused; classical vs. deep; personalization budget; with/without SSL; cross-dataset transfer.

### 7. Tech stack (MLOps/cloud now core — it's a target skill)
- **ML:** Python, SciPy, scikit-learn, XGBoost/LightGBM; PyTorch + Lightning. NeuroKit2/HeartPy; cgmquantify.
- **Experiment/versioning:** MLflow (tracking + registry), DVC (data + pipeline), Hydra (configs/sweeps).
- **Data layer (minimal):** windowed Parquet; DuckDB optional for analytics. No heavier DB at this scale.
- **Cloud MLOps (build this — portfolio + skill):** GCP Vertex AI — Pipelines (ingest → preprocess → train → eval → register → deploy), managed GPU training, Model Registry, Model Monitor (drift = §6 OOD in production). BigQuery for the feature layer. CI/CD via GitHub Actions → Vertex (incl. the no-leakage test). Keep the core on portable tools (MLflow/DVC/ONNX) so it runs anywhere.
- **Edge:** ONNX → LiteRT (the distilled student).

### 8. Compute & cost (estimates)
- **Local dev:** Mac M3, 24 GB, PyTorch MPS — assert device placement (avoid silent CPU fallback). Good for classical baselines, prototyping, small models.
- **Cloud (Vertex):** single mid-tier GPU (T4/L4/A10) for most models; burst to A100/several L4s for SSL pretraining on DiaData, ensembles, and sweeps. **Use spot/preemptible** (up to ~91% off) with checkpointing.
- **Total:** still modest for Tracks 1–2 (low hundreds of GPU-hours); SSL pretraining on the 149M-value corpus is the main cost lever. Rule: cut hyperparameter search before cutting ablations.
- **Storage:** tens of GB. Object-storage bucket as DVC remote.

### 9. Plan
| Phase | Track | Goal | Exit gate |
|---|---|---|---|
| **P0 — Setup** | — | Repo, Hydra, DVC, MLflow, CI, **leakage-proof splitter first**; ingest OhioT1DM | One LOSO/preset-split fold runs end-to-end; CI green |
| **P1 — Forecasting baseline** | 1 | GBT/Elastic Net + TCN; evaluator; match a published OhioT1DM baseline | Benchmarked result reproduced; library v0 |
| **P2 — Forecasting depth + scale** | 1 | PatchTST; scale to DiaData/Granada; SSL pretraining; Mamba/S4 ablation | Cross-dataset + SSL results; library v1 |
| **P3 — Trustworthiness + personalization** | 1 | Conformal uncertainty; population→LoRA personalization curve | Coverage validated; budget curve |
| **P4 — Distillation + MLOps** | 1 | Teacher→student, ONNX/LiteRT; Vertex pipeline + Model Monitor | Edge student + reproducible cloud pipeline |
| **P5 — Estimation (signature)** | 2 | Reuse library on BIG IDEAs/PhysioCGM; fusion; calibration-free cross-subject | Per-modality vs fused; calibration-free result |
| **P6 — Spectral (optional)** | 3 | Chemometrics → deep spectral → physics-informed → spectral SSL | Spectral model with band attention |
| **P7 — Package** | — | Repo polish, model cards, results, optional paper draft | Portfolio-ready; (optional) submission |

### 10. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Same-subject leakage inflates results | LOSO enforced in code + CI gate |
| Scope sprawl across three tracks | Sequential; shared library; Track 1 to a finished result before opening the next |
| Over-reaching on high-risk methods | GBT→TCN→PatchTST is the spine; Mamba/MAML/TTA are ablations |
| MPS silent CPU fallback | Device asserts + benchmarks; heavy jobs on Vertex |
| Cloud cost | Spot, checkpointing, budget alerts; prototype locally |
| Modest accuracy read as failure | Portfolio value is the engineered stack + rigor + breadth, not a single SOTA number |

### 11. Deliverables
1. **Reusable component library** (Stages 1–8) — the centerpiece.
2. Public, clean monorepo: Hydra configs, DVC pipeline, MLflow logging, grouped-split safeguards, one-command repro.
3. Three task demonstrations (forecasting / estimation / optional spectral) with results packages.
4. Vertex MLOps pipeline + ONNX/LiteRT edge student.
5. Model cards + (optional) one paper draft from whichever track yields the cleanest claim.

### 12. First tasks (P0 — start here)
1. Scaffold the monorepo + env + Hydra + DVC + MLflow + GitHub Actions, structured so the library is reused across tracks.
2. Build the **subject-grouped, leakage-proof splitter and its CI test first.**
3. Ingest **OhioT1DM**; build windowing + the forecasting target (30/60-min horizons).
4. Run one fold end-to-end with the GBT baseline against OhioT1DM's preset split to prove the harness.
5. Stand up the evaluator (RMSE/MAE + error grid) so every later model and track is measured identically.

> Build the library on Track 1 (data-rich CGM forecasting), prove it against published baselines, then reuse it for the differentiated estimation track and the optional spectral track. Optimize for a clean, reusable stack and demonstrated breadth — accuracy is secondary to engineering quality and rigor.
