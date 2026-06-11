# Project 17: Emoji Prediction from Text

This project solves tweet emoji prediction on the TweetEval `emoji` benchmark. The work explores multiple text classification approaches, compares them against each other, and produces a series of reports that document the research, the modeling choices, and the score improvements.

## What We Built

The main pipeline trains and evaluates two baseline text classifiers:

- an MLP on TF-IDF bag-of-words features
- an LSTM on token sequences

From there, the project expands into stronger methods and research-driven follow-up runs:

- a GPU improvement run with stronger baselines and a RoBERTa-style model
- an ensemble and calibration run that combines research findings from online papers, notebooks, and model cards
- a public BERTweet emoji checkpoint evaluation

## Main Deliverables

- `notebooks/emoji_prediction_project17.ipynb`: notebook walkthrough of the workflow
- `src/train_and_evaluate.py`: main training and evaluation pipeline
- `src/improve_and_report.py`: stronger GPU-oriented comparison run
- `src/evaluate_bertweet_emoji.py`: public BERTweet emoji evaluation
- `src/ensemble_calibrate_report3.py`: ensemble tuning and report 3 generation
- `report/report-1.md`: first full technical report
- `report/report-2.md`: score improvement and stronger baseline report
- `report/report-3.md`: research-driven ensemble report
- `report/final-report.md`: consolidated final report

## Methods Compared

The project compares the following approaches:

- TF-IDF + MLP
- LSTM on tokenized text
- linear SVM with word and character TF-IDF features
- Twitter RoBERTa emoji model
- BERTweet emoji checkpoint
- ensemble calibration over the strongest candidates

## What We Evaluated

The project reports several metrics and artifacts depending on the run:

- top-1 accuracy
- top-3 accuracy
- macro-F1
- per-class metrics
- confusion matrices
- prediction CSVs and label files
- training histories
- score comparison plots

## How To Reproduce

```powershell
cd "D:\HARSHIT\NN PROJECT\project 17"
python -m pip install -r requirements.txt
$env:PYTHONIOENCODING="utf-8"
python src\train_and_evaluate.py
```

For the GPU improvement and research runs:

```powershell
cd "D:\HARSHIT\NN PROJECT\project 17"
python -m pip install -r requirements-gpu.txt
$env:PYTHONIOENCODING="utf-8"
python src\improve_and_report.py
python src\evaluate_bertweet_emoji.py
python src\ensemble_calibrate_report3.py
```

## Repository Structure

- `data/raw/`: TweetEval text and label files
- `models/`: saved model weights and preprocessing artifacts
- `results/`: metrics, plots, predictions, and score summaries
- `report/`: the written reports for each stage of the project
- `src/`: source code for training, evaluation, and reporting
- `notebooks/`: the notebook version of the workflow

## Dataset

The project uses TweetEval `emoji`, a SemEval-style benchmark with fixed train, validation, and test splits and 20 emoji labels.

## Notes On The Research Flow

The later reports were written after checking online notebooks, public discussions, model cards, and benchmark-style references. The goal was to identify which method was strongest for this task and why it worked better than the smaller baselines.

## Key Outcome

The project starts with two straightforward local models and ends with a more competitive research-driven pipeline that uses stronger features, stronger pretrained models, and ensemble ideas.
