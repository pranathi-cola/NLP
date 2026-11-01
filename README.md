# SBERT Based Software Story Point Estimation 

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

### 1. Download the dataset

Download the TAWOS SQL dataset from:

https://doi.org/10.5522/04/21308124

The SQL dump is not included in this repository. Place the downloaded `TAWOS.sql` file in the root folder of your Google Drive.

### 2. Open Google Colab and install dependencies

Clone the repository:

```bash
!git clone https://github.com/pranathi-cola/NLP.git
%cd NLP
```

Install the required dependencies:

```bash
!pip install -r requirements.txt
```

### 3. Run data processing and SBERT embedding extraction

Open:

`notebooks/Data_Processing.ipynb`

Execute the notebook cells in order.

The notebook:

- Imports the TAWOS SQL dump into MariaDB.
- Exports issue and project information.
- Cleans issue descriptions and applies the story-point filtering policy.
- Creates chronological development and held-out test splits.
- Extracts SBERT sentence embeddings using `sentence-transformers/all-MiniLM-L6-v2`.

The preprocessing stage generates the following files:

```text
issue.csv
project.csv
issue_clean.csv
project_summary.csv
splits_temporal.json
sbert_minilm_emb.npy
sbert_ids.csv
```

Copy the five files required by the training notebook into the configured data directory:

```text
issue_clean.csv
project.csv
splits_temporal.json
sbert_minilm_emb.npy
sbert_ids.csv
```

The default Google Colab training data directory is:

`MyDrive/TAWOS_Project6_Files`

### 4. Run model training and evaluation

Open:

`notebooks/Model_Training.ipynb`

Execute the notebook cells in order.

The notebook:

- Loads cleaned issue data and SBERT embeddings.
- Performs five expanding-window chronological validation runs using six chronological blocks (one initial training block and five successive validation blocks).
- Selects hyperparameters and training epochs using development data.
- Trains a separate three-layer 1D CNN for each project.
- Evaluates the selected models on the held-out temporal test data.
- Compares CNN predictions against median, mean, and Ridge regression baselines.
- Generates evaluation metrics, CSV files, and plots.

**Note:** The validation procedure uses an initial training block followed by five chronological validation windows. It is not conventional shuffled five-fold cross-validation.

### 5. View the results

The generated outputs are saved in the configured results directory.

The repository's `results/` directory contains the recorded experimental results, including validation metrics, held-out test metrics, baseline comparisons, prediction files, and figures.

For the recorded experiment, the complete CNN cross-validation process took approximately 183 minutes on CPU. Execution time may vary depending on the hardware.

### 6. Reproducibility

The main random seed is 42. Additional final-model sensitivity experiments use seeds 43 and 44.

The model configuration, data-splitting strategy, hyperparameters, and experimental metadata are documented in:

- `configs/config.json`
- `results/run_meta.json`

## RESULTS

### Settings and metadata

| File | What's inside | 
| --- | --- |
| [Run Configuration](configs/config.json) | The settings used for the run: model name, seeds, learning-rate/dropout grid, batch size, epochs, CNN setup. |
| [Run Metadata](results/run_meta.json) | The same settings plus the config hash, device, the chosen config and epoch count for each project, and the epoch rule. |

### Five-Fold Validation results

| File | What's inside | 
| --- | --- |
| [All Validation runs](results/cv_results_all_configs.csv) | One row per project, fold and hyperparameter config (780 rows). Each row has validation MAE, MdAE, best epoch and training size. |
| [Selected-config validation runs](results/cv_results_selected_per_project.csv) | The same rows, but only for each project’s winning config. |
| [Best config per project](results/best_config_per_project.csv) | The chosen learning rate, dropout and final epoch count per project, with its mean validation MAE and MdAE. |
| [Validation summary per project](results/cv_per_project_summary.csv) | Mean and SD of validation MAE and MdAE across the five folds, per project. |
| [Validation summary per fold](results/cv_per_fold_macro_summary.csv) | Average MAE and MdAE for each fold number across all projects, plus mean training size. |

### Held-out test results

| File | What's inside | 
| --- | --- |
| [Test metrics per project](results/test_metrics_per_project.csv) | Final test MAE and MdAE per project, with the median and mean baselines and the duplicate-text check. |
| [Test metrics by seed](results/test_metrics_by_seed.csv) | Test MAE and MdAE per project for seeds 42, 43 and 44. |
| [Test predictions](results/test_predictions.csv) | Every test issue’s true and predicted story points (seed 42). |
| [CNN vs ridge vs median](results/cnn_vs_ridge_vs_median.csv) | Per project, test MAE and MdAE for the CNN, ridge regression and the median baseline side by side. |
| [Summary report](results/summary_report.csv) | The headline table: macro mean and SD for validation and test, plus the median baseline. |

### Figures

| File | What's inside | 
| --- | --- |
| [Validation error per fold](results/fig_cv_per_fold.png) | Macro MAE and MdAE for each validation fold. |
| [Test error per project](results/fig_test_mae_by_project.png) | Test error per project: CNN MAE and MdAE, median baseline and ridge.|
| [Validation vs test MAE](results/fig_val_vs_test_mae.png) | Validation MAE (mean ± SD) against test MAE per project. |
| [Predicted vs true story points](results/fig_scatter_test.png) | Predicted against true story points on the test set. |
| [Absolute error distribution](results/fig_abs_error_hist.png) | Histogram of absolute errors on the test set. |
