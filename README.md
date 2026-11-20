# Yelp review fraud detection: learned detectors, a calibrated stacker and LLM investigation

BT4012 Group 18. We predict Yelp's review filter on the YelpZip dataset with four detectors combined by a calibrated
logistic stacker and a small post-stacker correction (B). We also test whether an LLM investigation agent adds value
beyond conventional models, either in its decisions or in the explanations it gives moderators.

**Headline result (locked test split, 91,737 reviews):** stacker + B reaches F1 0.5061 and PR-AUC 0.5037, against
0.411 F1 for the rule benchmark and 0.4972 for the stacker alone (ΔF1 +0.0090, 95% CI +0.0054 to +0.0126). No LLM
design improved decisions; the LLM is used only to write cited moderator briefs.

## Files

| File | Purpose |
| --- | --- |
| `pipeline.ipynb` | **Main notebook.** Preprocessing and reviewer-level splits, detectors D1–D4, the calibrated stacker, the post-stacker correction B (9-BRANCH), routing, the LLM investigator (Sections 9–11), moderator briefs (Section 12) and the final evaluation (Section 13). All reported results come from this notebook. |
| `train.ipynb` | Companion notebook for the GPU work: review embeddings and nearest-neighbour similarity features, the DistilBERT text model and the paper's 1D CNN for D2. It reads feature tables saved by `pipeline.ipynb` and writes scores back to Drive. DistilBERT and the CNN are kept as comparisons only. |
| `initial_investigator_pipeline.ipynb` | The first LLM investigator design: a sequential tool-calling agent (up to six tool calls) that gave a fraud / genuine / insufficient-evidence verdict on routed cases. Kept as a record of the design the final pipeline replaced; not needed to reproduce the results. |
| `data/`, `artifacts/` (on Google Drive, linked below) | The original YelpZip release (Rayana and Akoglu, 2015; 608,598 restaurant reviews with Yelp filter labels) and the saved outputs of `train.ipynb` that `pipeline.ipynb` reads. |

## Reproducing the results (`pipeline.ipynb`)

The notebook is written for Google Colab with Google Drive. It runs on CPU; `train.ipynb` needs a T4 GPU for
DistilBERT and the embeddings.

### 1. Get the data and model artifacts

The data and the outputs of `train.ipynb` are too large for GitHub, so they are shared on Google Drive:

**[bt4012 shared folder](https://drive.google.com/drive/folders/1Hz5iiJ7ee6iaEpQK761FvCJTEYnO9wN5?usp=sharing)**

```
bt4012/
├── data/
│   ├── yelpzip/                      # original YelpZip release files (Section 1 input)
│   └── similarity_features.parquet   # embedding similarity features (train.ipynb T2)
└── artifacts/
    ├── neighbours.parquet            # embedding nearest neighbours (train.ipynb T2)
    ├── raw_scores/                   # d2_cnn and d3_distilbert scores + .meta.json (train.ipynb T3, T4)
    └── distilbert/full/              # d3_train_subsample.parquet (train.ipynb T3)
```

With these files in place, `pipeline.ipynb` runs on CPU and `train.ipynb` does not need to be run.

Copy the folder into your own Drive so the notebook can write its outputs next to it:

1. Open the link, select `data` and `artifacts`, and download them (or, if you have edit access, right-click the
   `bt4012` folder → **Organise → Add shortcut** → My Drive).
2. In your Drive, create `MyDrive/bt4012_yelp/` and upload `data/` and `artifacts/` into it, keeping the layout above.
   With a shortcut instead, the folder is `MyDrive/bt4012`.
3. In the `Config` cell (Section 0), set `project_root` to the folder you used, for example
   `Path("/content/drive/MyDrive/bt4012_yelp")` (the default) or `Path("/content/drive/MyDrive/bt4012")` for a
   shortcut. A shortcut needs edit access, because the notebook writes `outputs/`, `cache/` and `src/` there.

The notebook creates every other folder itself.

### 2. Add LLM keys (Sections 9–12 only)

The LLM sections call Qwen through NUS SoCLaaS, an OpenAI-compatible API. In Colab, open **Secrets** (key icon in the
left sidebar), add `SOCLAAS_API_KEY` and, optionally, `SOCLAAS_API_KEY_BACKUP`, and switch on notebook access. Sections
0–8 and the conventional part of 9-BRANCH run without keys.

### 3. Run top to bottom

1. Open `pipeline.ipynb` in Colab and choose **Runtime → Run all**. Section 0 installs dependencies, mounts Drive,
   sets seeds (`seed = 42`) and logs library versions to `versions.json`.
2. The `train.ipynb` outputs come from the shared folder. If one is missing, the notebook stops with a message naming
   the `train.ipynb` step that makes it (T2, T3 or T4). Run that step on a T4 runtime, then re-run `pipeline.ipynb`
   from the cell that stopped.
3. Section 1 creates `data/splits.parquet` once (70/15/15 by reviewer). Later runs load it and stop if the data no
   longer matches, rather than re-splitting.

### What to expect

- **Test labels stay locked** (`labelled()` / `unlock_test()`) until 9-BRANCH 3.4, which reads them once under an
  adoption rule fixed in advance. Section 13 re-scores the same frozen predictions.
- **Conventional results are deterministic** given the same splits and seed: detectors, stacker, B and the Section 13
  tables should match the reported figures.
- **LLM results depend on the response cache.** Every LLM response is cached on Drive by prompt hash, and the cache is
  not in this repository, so a fresh run makes new calls. The project's total LLM spend was about $4.20, and each
  section checks a spending cap before calling. Calls use temperature 0, but the model's outputs can still vary
  slightly, so LLM-based figures (pilot findings, brief faithfulness) may differ a little from the report.
- **Section 12's hand audit** reads a completed audit sheet (`audit_briefs_completed.csv`). On a fresh run, the
  notebook writes a blank sheet that must be scored by hand before 12.8b can report results.

### Data notes

YelpZip labels are Yelp's filter decisions, not confirmed fraud, so the models learn to reproduce the filter. D2, D4
and the review-date input use each account's full history in the dataset, so the results describe a retrospective
audit rather than screening at posting time.

## Reference

Rayana, S. and Akoglu, L. (2015). Collective opinion spam detection: bridging review networks and metadata. *KDD 2015*.
