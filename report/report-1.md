# Project 17 Report: Emoji Prediction From Text

## Executive Summary

This project predicts the most likely emoji for a tweet using the TweetEval `emoji` benchmark. The required comparison is between an MLP over bag-of-words features and an LSTM over token sequences. The best model in this local run is **MLP TF-IDF BoW**, with **38.69% top-1 accuracy** and **58.59% top-3 accuracy** on the official TweetEval test split.

| Model | Top-1 Accuracy | Top-3 Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| MLP TF-IDF BoW | 38.69% | 58.59% | 0.1955 | 0.3372 |
| LSTM | 36.94% | 57.09% | 0.1788 | 0.3256 |

Top-3 accuracy is important for this task because emoji usage is inherently ambiguous: a tweet about love, celebration, friends, or a photo can plausibly map to several visually different but semantically related emojis. Top-1 measures exact prediction; top-3 measures whether the model can place the correct emoji among its most plausible alternatives.

## Problem And Dataset

The task is a 20-class tweet classification problem: remove or ignore the emoji label and predict which emoji best matches the text. TweetEval's emoji subset is based on SemEval-2018 Task 2 and provides fixed train, validation, and test files. This run verified the expected split sizes: 45,000 training examples, 5,000 validation examples, and 50,000 test examples.

Training label distribution:

| Emoji | Name | Train Count | Train Share |
|---|---|---:|---:|
| ❤ | red heart | 9204 | 20.45% |
| 😍 | smiling face with hearteyes | 4901 | 10.89% |
| 😂 | face with tears of joy | 4713 | 10.47% |
| 💕 | two hearts | 2043 | 4.54% |
| 🔥 | fire | 2146 | 4.77% |
| 😊 | smiling face with smiling eyes | 2132 | 4.74% |
| 😎 | smiling face with sunglasses | 2078 | 4.62% |
| ✨ | sparkles | 2345 | 5.21% |
| 💙 | blue heart | 1287 | 2.86% |
| 😘 | face blowing a kiss | 1391 | 3.09% |
| 📷 | camera | 1982 | 4.40% |
| 🇺🇸 | United States | 946 | 2.10% |
| ☀ | sun | 1246 | 2.77% |
| 💜 | purple heart | 980 | 2.18% |
| 😉 | winking face | 1224 | 2.72% |
| 💯 | hundred points | 934 | 2.08% |
| 😁 | beaming face with smiling eyes | 1350 | 3.00% |
| 🎄 | Christmas tree | 1397 | 3.10% |
| 📸 | camera with flash | 1510 | 3.36% |
| 😜 | winking face with tongue | 1191 | 2.65% |

The distribution is imbalanced. Frequent emojis such as hearts, laughter, and camera-related labels provide many examples, while rarer or more context-specific emojis have fewer and noisier cues. This makes macro-F1 lower than weighted-F1 and makes per-class analysis more informative than a single accuracy number.

## Research Context And Method Choice

The SemEval task and follow-up systems show that emoji prediction from tweets behaves like short-text classification with strong lexical cues. Several SemEval participants found that sparse lexical models such as TF-IDF plus linear SVM were difficult for RNNs to beat on this dataset, especially without large external pretraining. PickleTeam used both SVM and LSTM signals, which supports the core comparison in this project: sparse bag-of-words evidence versus sequence modeling. Public demo projects and notebooks often use LSTM-style embedding models, while benchmark systems show that strong sparse baselines remain hard to beat on this exact task. DeepMoji and later transformer/TweetNLP-style models show that large-scale emoji pretraining can learn richer emotional representations, but those approaches are heavier than the requested MLP-vs-LSTM assignment and reduce interpretability.

For this project, the MLP with TF-IDF bag-of-words is the strongest transparent baseline: it directly captures hashtags, words, repeated social phrases, and short n-grams that are common in tweets. The LSTM is included because it can model token order and local composition, but a small CPU-trained LSTM from scratch has less prior knowledge than TF-IDF and can struggle with rare words, hashtags, and class imbalance.

## Implementation

- Dataset download: raw TweetEval files from CardiffNLP GitHub.
- Preprocessing: URL and user-mention normalization, lowercase tokenization, hashtag retention.
- MLP input: TF-IDF word unigram/bigram features, capped vocabulary, sparse mini-batches converted to dense tensors per batch.
- LSTM input: train-only vocabulary, padded token ID sequences, learned embeddings, final hidden state classifier.
- Evaluation: top-1 accuracy, top-3 accuracy, macro-F1, weighted-F1, per-class precision/recall/F1, confusion matrix, and prediction CSVs.

Run configuration:

```json
{
  "seed": 42,
  "device": "cpu",
  "quick": false,
  "epochs_mlp": 4,
  "epochs_lstm": 4,
  "batch_size_mlp": 256,
  "batch_size_lstm": 256,
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "dropout": 0.35,
  "min_df": 2,
  "max_features": 20000,
  "mlp_hidden_dim": 256,
  "lstm_vocab_size": 20000,
  "max_len": 48,
  "embedding_dim": 128,
  "lstm_hidden_dim": 128,
  "lstm_layers": 1
}
```

## Results And Interpretation

The MLP top-1 score is **38.69%** and its top-3 score is **58.59%**. The LSTM top-1 score is **36.94%** and its top-3 score is **57.09%**.

When the MLP wins, the reason is usually that tweet emoji prediction contains many lexical shortcuts: `love`, `happy`, `birthday`, `christmas`, `photo`, `beach`, `sun`, `fire`, and hashtag forms often point directly toward a small group of emojis. TF-IDF also handles rare but high-value tokens well. When the LSTM wins or narrows the gap, it is usually because order and local phrase composition matter, such as distinguishing affectionate messages from laughter or sarcasm.

## Easiest Emojis

Using the best model's per-class F1/recall, these labels were easiest:

| Emoji | Name | F1 | Recall | Support | Indicative training tokens |
|---|---|---:|---:|---:|---|
| ❤ | red heart | 0.8043 | 0.8546 | 10798 | my, love, !, the, with, you |
| 🎄 | Christmas tree | 0.5727 | 0.6809 | 1545 | christmas, merry, tree, #christmas, eve, #christmastree |
| ☀ | sun | 0.4286 | 0.2941 | 1265 | sun, sunshine, sunny, morning, beach, summer |
| 🔥 | fire | 0.4088 | 0.3907 | 3716 | lit, fire, user, #fire, hot, was |
| 😂 | face with tears of joy | 0.3689 | 0.5664 | 4534 | lol, user, the, when, !, lmao |
| 📷 | camera | 0.3202 | 0.6250 | 1432 | user, by, the, rider, los, of |

These are easier when the text has distinctive words, seasonal vocabulary, platform conventions, or narrow contexts. For example, `christmas`, `merry`, and holiday terms strongly identify the Christmas tree; `photo`, `camera`, and location/photo-sharing language help camera labels; heart emojis often align with repeated affection words.

## Hardest Emojis

Using the best model's per-class F1/recall, these labels were hardest:

| Emoji | Name | F1 | Recall | Support | Indicative training tokens |
|---|---|---:|---:|---:|---|
| 📸 | camera with flash | 0.0000 | 0.0000 | 2417 | user, by, photo, shoot, credit, the |
| 😉 | winking face | 0.0000 | 0.0000 | 1306 | the, to, #mividaesunatombola, !, #livefromvegas, you |
| 💯 | hundred points | 0.0000 | 0.0000 | 1244 | the, paisley, female, user, real, keeping |
| 😁 | beaming face with smiling eyes | 0.0000 | 0.0000 | 1153 | !, the, in, to, user, smile |
| 💜 | purple heart | 0.0000 | 0.0000 | 1114 | purple, #purplerain, prince, #prince, my, lavender |
| 😜 | winking face with tongue | 0.0000 | 0.0000 | 1010 | !, the, to, in, ?, and |

These are harder when the emoji meaning overlaps with other classes or the text lacks explicit lexical clues. Hearts confuse with each other because red, blue, purple, and two-hearts often accompany similar affection text. Winking and playful face emojis can be interchangeable in casual tweets. Some labels are also affected by class imbalance: fewer examples give the model less evidence and lower recall.

Most common wrong predictions for the hardest labels:

| True Emoji | Most Common Wrong Prediction | Wrong Count | Share Of That True Class |
|---|---|---:|---:|
| 📸 camera with flash | 📷 camera | 1462 | 60.49% |
| 😉 winking face | 😂 face with tears of joy | 476 | 36.45% |
| 💯 hundred points | 😂 face with tears of joy | 438 | 35.21% |
| 😁 beaming face with smiling eyes | 😂 face with tears of joy | 368 | 31.92% |
| 💜 purple heart | 😍 smiling face with hearteyes | 394 | 35.37% |
| 😜 winking face with tongue | 😂 face with tears of joy | 408 | 40.40% |

## Methods We Could Have Tried

- Linear SVM or logistic regression on TF-IDF: likely very competitive for this benchmark and often stronger than small neural models.
- Character n-grams: useful for hashtags, slang, spelling variation, and elongated words.
- Class weighting or focal loss: could improve rare emoji recall but may reduce top-1 accuracy on frequent labels.
- Bidirectional LSTM with attention: closer to strong SemEval neural systems, especially with pretrained Twitter embeddings.
- DeepMoji-style pretraining: powerful but requires huge emoji-labeled corpora and is outside the assignment's lightweight local scope.
- BERTweet/TweetNLP transformer fine-tuning: likely strongest modern option, but it changes the comparison from MLP-vs-LSTM to pretrained transformer fine-tuning.

## Sources

- [SemEval-2018 Task 2: Multilingual Emoji Prediction](https://aclanthology.org/S18-1003/): Original shared task defining emoji prediction from tweets.
- [TweetEval benchmark repository](https://github.com/cardiffnlp/tweeteval): Provides unified fixed splits and raw files used in this project.
- [TweetEval paper](https://aclanthology.org/2020.findings-emnlp.148/): Documents TweetEval as a standardized Twitter classification benchmark.
- [Tübingen-Oslo at SemEval-2018 Task 2](https://coltekin.net/cagri/papers/coltekin2018semeval.pdf): Reports strong SVM performance and discusses RNN difficulty on emoji prediction.
- [PickleTeam! at SemEval-2018 Task 2](https://aclanthology.org/S18-1072.pdf): Combines SVM and LSTM predictions, motivating comparison between sparse and sequence models.
- [DeepMoji](https://github.com/bfelbo/deepmoji): Large-scale emoji-pretrained representation model trained on 1.2B tweets.
- [NTUA-SLP at SemEval-2018 Task 2](https://arxiv.org/abs/1804.06657): Attention LSTM system that ranked highly using pretrained Twitter embeddings.
- [NLP-text2emoji public project](https://github.com/joonasrooben/NLP-text2emoji): Example public implementation using bidirectional LSTMs and word/character embeddings.

## Reproducibility Notes

Run the pipeline from the project root with:

```powershell
$env:PYTHONIOENCODING="utf-8"
python src\train_and_evaluate.py
```

Saved artifacts include `results/metrics_summary.json`, `results/per_class_*.csv`, `results/predictions_*.csv`, confusion matrix PNGs, and model checkpoints in `models/`.
