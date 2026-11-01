# Natural Language Processing

## GROUP 12

| S.no | Team Member | Roll Number |
| --- | --- | --- |
| 1. | Kolasani Maithili | B230385CS |
| 2. | Ardhra Alaina Winston | B230198CS |
| 3. | Kola Pranathi | B230384CS |
| 4. | Samsani Siddesh | B220508CS |
| 5. | G Harivamshi | B220276CS |

## PROBLEM STATEMENT - 6

Investigate the effectiveness of SBERT-based sentence embeddings for predicting story points of later issues from the same project using chronological splitting.

### Text Representation - SBERT

Sentence embeddings represent an entire sentence or text passage as a numerical vector. SBERT produces sentence-level representations intended to capture semantic meaning.

### Within-project

Training and testing issues come from the same software project. This setting measures performance on issues from a project represented in the training data, but the model may learn project-specific patterns.

### Temporal splitting

Issues are sorted by creation date. Older issues are used for training, and the most recent issues for testing. Temporal validation must preserve chronological order.

## OBJECTIVES

1. Build a reproducible pipeline: TAWOS dump -> cleaned issues -> SBERT embeddings -> chronological splits.
2. Train the three-layer 1D CNN on the 384-d SBERT vectors, separately for each of the 39 TAWOS projects.
3. Select hyperparameters and epoch count per project from development data only, using the five chronological validation folds. Evaluate once on the held-out data from the newest 20% of each project.
4. Report MAE and MdAE for validation. For the final test, compare against trivial(median/mean) and ridge-regression baselines.

## DATASET DESCRIPTION

TAWOS (https://doi.org/10.5522/04/21308124): Jira issues from open-source agile projects

| Step | Rows |
| --- | --- |
| Raw issues in the SQL dump | 4,58,232 |
| Isuues with a numeric story point | 65,427 |
| After keeping 0 <= story points <= 100 | 65,286 |
| After dropping texts with fewer than 3 words | 65,015 |

- 39 projects; the largest is Lsstcorp Data Management (20,378 issues). No project fell below the 100-issue minimum.
- Zero (2,408) and fractional (3,311) labels are retained in the final data.
- Input text = cleaned title + cleaned description. Jira markup is removed, {code} / {noformat} contents and URLs are kept.
- 603 exact duplicate texts exist in the cleaned data.

## SETUP INSTRUCTIONS

- Developed in Google Colab.
- Python 3, PyTorch, sentence-transformers, scikit-learn, pandas, matplotlib.
- MariaDB/MySQL for data processing has been used.
- RUN:
    pip install -r requirements.txt

## REPRODUCING THE EXPERIMENT

1. Get the data. Download TAWOS.sql from the DOI above.
2. Run notebooks/01_preprocessing_embeddings_splits.ipynb top to bottom. It imports the dump, exports issue.csv/project.csv, applies the label and text rules, writes issue_clean.csv, builds and sanity-checks splits_temporal.json, and writes sbert_minilm_emb.npy + sbert_ids.csv. Paths at the top of each cell point to Google Drive; edit them if running elsewhere
3. Place the five output files in one folder.
4. Run notebooks/02_training_evaluation.ipynb.
  - Colab: it mounts Drive and reads MyDrive/TAWOS_Project6_Files.
  - Local: export TAWOS_DATA_DIR=./data TAWOS_RESULTS_DIR=./results.
  - Quick code check only (never for reported numbers): export P6_SMOKE=1.
  - The 4-configuration × 39-project × 5-fold CV took about 183 minutes in the recorded run; progress is checkpointed to cv_results_partial.csv so a disconnect can be resumed.
5. Outputs (CSVs, config.json, run_meta.json, figures) are written to RESULTS_DIR; copy them into results/.

Seeds: 42 (primary), 43, 44 (final-model sensitivity). cudnn.deterministic=True
