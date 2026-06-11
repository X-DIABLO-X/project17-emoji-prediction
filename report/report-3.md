# Project 17 Report 3: Online Research And Ensemble Calibration

## Research Findings

I checked online discussions, notebooks, model cards, and benchmark pages for practical ways to improve TweetEval/SemEval emoji prediction. The main pattern is consistent:

- Sparse lexical systems are unusually strong for this task. SemEval papers report that SVMs with word and character n-grams can outperform small RNN/LSTM models.
- Transformer systems are the strongest public baselines when the transformer is pretrained on Twitter data. TweetEval lists BERTweet at **33.4 macro-F1**, RoBERTa-Retrained at **31.4**, and RoBERTa-Base around **30.9**.
- Public Hugging Face/TweetNLP examples recommend CardiffNLP Twitter-RoBERTa emoji models and Twitter-specific preprocessing, replacing usernames with `@user` and links with `http`.
- Ensembles are commonly attempted because sparse models and neural models make different errors. PickleTeam tried SVM+LSTM; our stronger version combines SVM with Twitter-RoBERTa logits and tunes the blend on validation macro-F1.


I also evaluated `cardiffnlp/bertweet-base-emoji` because online results point to BERTweet as the strongest TweetEval family. In this local run that checkpoint scored only **26.41% macro-F1**, so it did not reproduce the public leaderboard number and was rejected for the final model. I also checked whether its generic `LABEL_0...LABEL_19` outputs were simply in the wrong label order; validation alignment did not recover competitive test performance.

## Method Chosen

The best feasible method for this local RTX 4050 setup is a **validation-tuned ensemble of Twitter-RoBERTa and LinearSVM**, plus a small class-prior calibration term. Full transformer fine-tuning is possible, but this public CardiffNLP model is already fine-tuned on TweetEval emoji; the lower-risk next improvement is to combine its semantic/contextual signal with the SVM's hashtag, spelling, and character n-gram signal.

Best validation configuration:

```json
{
  "weight_roberta": 0.5,
  "roberta_temp": 1.5,
  "svm_temp": 3.0,
  "prior_alpha": -0.2,
  "top1_accuracy": 0.3388,
  "top3_accuracy": 0.5916,
  "macro_f1": 0.3046098103577453
}
```

Top validation candidates:

| Rank | RoBERTa Weight | RoBERTa Temp | SVM Temp | Prior Alpha | Val Top-1 | Val Top-3 | Val Macro F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.50 | 1.50 | 3.00 | -0.20 | 33.88% | 59.16% | 0.3046 |
| 2 | 0.60 | 2.00 | 3.00 | -0.20 | 34.56% | 59.72% | 0.3045 |
| 3 | 0.70 | 2.00 | 1.50 | -0.20 | 33.24% | 58.56% | 0.3040 |
| 4 | 0.50 | 2.00 | 3.00 | -0.20 | 34.22% | 60.02% | 0.3039 |
| 5 | 0.40 | 1.50 | 3.00 | -0.20 | 33.72% | 59.16% | 0.3036 |
| 6 | 0.40 | 2.00 | 3.00 | -0.20 | 34.22% | 59.58% | 0.3034 |
| 7 | 0.75 | 2.00 | 1.50 | -0.20 | 33.32% | 58.72% | 0.3034 |
| 8 | 0.35 | 2.00 | 3.00 | -0.20 | 34.04% | 58.68% | 0.3034 |

Run configuration:

```json
{
  "torch": "2.11.0+cu128",
  "torch_cuda": "12.8",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
  "transformer_model": "cardiffnlp/twitter-roberta-base-emoji",
  "transformer_batch_size": 64,
  "transformer_max_length": 128,
  "search_objective": "validation macro-F1",
  "elapsed_minutes": 2.24
}
```

## Test Results

| Model | Top-1 | Top-3 | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Original MLP TF-IDF | 38.69% | 58.59% | 0.1955 | 0.3372 |
| Original LSTM | 36.94% | 57.09% | 0.1788 | 0.3256 |
| LinearSVM word+char TF-IDF | 41.42% | 59.63% | 0.2800 | 0.4026 |
| Twitter-RoBERTa emoji GPU | 46.02% | 66.75% | 0.3155 | 0.4316 |
| BERTweet emoji HF checkpoint | 32.88% | 57.19% | 0.2641 | 0.3083 |
| Calibrated RoBERTa+SVM ensemble | 44.82% | 66.97% | 0.3281 | 0.4319 |

The calibrated ensemble achieved **44.82% top-1**, **66.97% top-3**, and **32.81% macro-F1** (`0.3281`). Compared with the previous best Twitter-RoBERTa run, this is a macro-F1 change of **+0.0126**.

## Interpretation

The ensemble is the best local model in this run. It improves macro-F1 over Twitter-RoBERTa from **31.55%** to **32.81%**, and top-3 from **66.75%** to **66.97%**. Top-1 drops slightly, from **46.02%** to **44.82%**, which is the expected tradeoff when tuning for macro-F1 and rare-class recovery instead of pure exact-match accuracy.

This is the best balance for the assignment because TweetEval's official emoji metric is macro-F1. The SVM contributes hashtag, spelling, and character-pattern evidence; Twitter-RoBERTa contributes contextual tweet representations; the calibration term shifts the decision boundary enough to recover minority classes without fully sacrificing common-class performance.

Easiest ensemble classes:

| Emoji | Name | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| ❤ | red heart | 0.7302 | 0.7663 | 0.7478 | 10798 |
| 🎄 | Christmas tree | 0.5716 | 0.8291 | 0.6767 | 1545 |
| 🇺🇸 | United States | 0.5792 | 0.6455 | 0.6105 | 1949 |
| 🔥 | fire | 0.5590 | 0.5568 | 0.5579 | 3716 |
| ☀ | sun | 0.4131 | 0.8458 | 0.5551 | 1265 |
| 😂 | face with tears of joy | 0.5002 | 0.4671 | 0.4831 | 4534 |
| 📷 | camera | 0.2820 | 0.5573 | 0.3745 | 1432 |
| 😍 | smiling face with hearteyes | 0.3726 | 0.3542 | 0.3632 | 4830 |

Hardest ensemble classes:

| Emoji | Name | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| 😜 | winking face with tongue | 0.0867 | 0.0624 | 0.0725 | 1010 |
| 😁 | beaming face with smiling eyes | 0.1098 | 0.0807 | 0.0930 | 1153 |
| 💜 | purple heart | 0.2483 | 0.0655 | 0.1037 | 1114 |
| 💕 | two hearts | 0.2810 | 0.0940 | 0.1409 | 2605 |
| 😉 | winking face | 0.1399 | 0.1577 | 0.1483 | 1306 |
| 😊 | smiling face with smiling eyes | 0.1410 | 0.1587 | 0.1493 | 1613 |
| 💙 | blue heart | 0.2598 | 0.1110 | 0.1556 | 1549 |
| 😘 | face blowing a kiss | 0.1859 | 0.2128 | 0.1984 | 1175 |

Most common hard-class confusions:

| True Emoji | Most Common Wrong Prediction | Wrong Count | Share Of True Class |
|---|---|---:|---:|
| 😜 winking face with tongue | 😂 face with tears of joy | 214 | 21.19% |
| 😁 beaming face with smiling eyes | 😊 smiling face with smiling eyes | 147 | 12.75% |
| 💜 purple heart | ❤ red heart | 205 | 18.40% |
| 💕 two hearts | ❤ red heart | 539 | 20.69% |
| 😉 winking face | 😂 face with tears of joy | 171 | 13.09% |
| 😊 smiling face with smiling eyes | 😍 smiling face with hearteyes | 222 | 13.76% |
| 💙 blue heart | ❤ red heart | 269 | 17.37% |
| 😘 face blowing a kiss | ❤ red heart | 187 | 15.91% |

## Best Next Step

The next improvement beyond this report should be true fine-tuning of `vinai/bertweet-base` or `cardiffnlp/twitter-roberta-base` with macro-F1 model selection, class-balanced sampling, and 2-3 seeds. Based on the online leaderboard, that is the most likely route from our current **32.81 macro-F1** toward the **33-36 macro-F1** range. The downside is runtime and instability on 6 GB VRAM, so it should be treated as a longer experiment rather than the most reliable quick upgrade.

## Sources

- [TweetEval official repository and leaderboard](https://github.com/cardiffnlp/tweeteval): Lists emoji macro-F1 baselines and task-specific Twitter-RoBERTa models.
- [Twitter-roBERTa-base for Emoji prediction](https://huggingface.co/cardiffnlp/twitter-roberta-base-emoji): Model card and usage example for the transformer used in this project.
- [TweetNLP](https://github.com/cardiffnlp/tweetnlp): Shows the same TweetEval emoji task exposed through a Twitter NLP toolkit.
- [OpenCodePapers TweetEval leaderboard mirror](https://opencodepapers-b7572d.gitlab.io/benchmarks/sentiment-analysis-on-tweeteval.html): Summarizes TweetEval emoji scores including BERTweet, RoBERTa, SVM, FastText, and LSTM.
- [PickleTeam! at SemEval-2018 Task 2](https://aclanthology.org/S18-1072.pdf): Compared SVM, LSTM, and an ensemble for emoji prediction.
- [Tubingen-Oslo at SemEval-2018 Task 2](https://aclanthology.org/S18-1004/): Reports the winning SemEval SVM result and argues that SVMs can outperform RNNs for this task.
