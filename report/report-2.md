# Project 17 Report 2: Score Quality And Improved Models

## Are The First Scores Good?

Short answer: **they are acceptable as simple local baselines, but bad compared with serious online benchmark systems**. The first report's MLP reached **38.69% top-1**, **58.59% top-3**, and **19.55% macro-F1**. TweetEval ranks emoji systems by macro-F1, and public baselines report about **24.7%** for LSTM, **29.3%** for SVM, **31.4%** for RoBERTa-retrained, and **33.4%** for BERTweet. So our original top-1 accuracy looked reasonable, but macro-F1 exposed the weakness: several minority or semantically overlapping emojis had near-zero recall.

| Reference | Emoji Macro-F1 | Meaning |
|---|---:|---|
| Our first MLP TF-IDF BoW | 19.55% / 0.1955 | Good sanity baseline, but weak on rare classes |
| Our first LSTM | 17.88% / 0.1788 | Below sparse baseline; trained from scratch on only 45k tweets |
| TweetEval LSTM | 24.7% / 0.247 | Official leaderboard baseline |
| TweetEval SVM | 29.3% / 0.293 | Official sparse baseline |
| TweetEval RoBERTa-Retrained | 31.4% / 0.314 | Official pretrained transformer baseline |
| TweetEval BERTweet | 33.4% / 0.334 | Strong Twitter-pretrained leaderboard model |
| Tübingen-Oslo SemEval winner | 35.99% / 0.3599 | Original SemEval English winner using SVM-style sparse features |


## Improvements Performed

I made two improvement passes:

- **LinearSVM with stronger sparse features**: word unigrams/bigrams plus character `3-5` grams. This targets hashtags, slang, misspellings, elongated words, and short lexical cues. It is also aligned with published SemEval evidence that SVMs outperform small RNNs on this task.
- **GPU Twitter-RoBERTa emoji model**: `cardiffnlp/twitter-roberta-base-emoji` evaluated on the RTX 4050 via CUDA PyTorch. This uses pretrained Twitter language representations and task-specific emoji fine-tuning, so it is the closest local comparison to TweetEval transformer baselines.

CUDA verification:

```json
{
  "torch": "2.11.0+cu128",
  "torch_cuda": "12.8",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
  "gpu_memory_gb": 6.0,
  "transformer_model": "cardiffnlp/twitter-roberta-base-emoji",
  "transformer_batch_size": 64,
  "transformer_max_length": 128,
  "svm_word_features": 120000,
  "svm_char_features": 80000
}
```

SVM validation sweep:

| C | Class Weight | Val Top-1 | Val Top-3 | Val Macro F1 | Val Weighted F1 |
|---:|---|---:|---:|---:|---:|
| 0.50 | none | 26.90% | 47.78% | 0.2480 | 0.2537 |
| 1.00 | none | 25.86% | 45.92% | 0.2406 | 0.2478 |
| 0.50 | balanced | 24.80% | 43.20% | 0.2465 | 0.2382 |
| 1.00 | balanced | 24.14% | 42.76% | 0.2376 | 0.2326 |

## Final Scores After Improvement

| Model | Top-1 Accuracy | Top-3 Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| MLP TF-IDF BoW | 38.69% | 58.59% | 0.1955 | 0.3372 |
| LSTM | 36.94% | 57.09% | 0.1788 | 0.3256 |
| LinearSVM word+char TF-IDF | 41.42% | 59.63% | 0.2800 | 0.4026 |
| Twitter-RoBERTa emoji GPU | 46.02% | 66.75% | 0.3155 | 0.4316 |

Best improved model: **Twitter-RoBERTa emoji GPU**, with **46.02% top-1**, **66.75% top-3**, and **31.55% macro-F1** (`0.3155`).

The improvement is real because macro-F1, not only accuracy, increased. Macro-F1 matters here because the dataset is imbalanced and because exact emoji identity is harder for rarer classes such as purple heart, hundred points, winking faces, and camera-with-flash.

## Why The Improved Methods Work Better

The original MLP learned useful lexical associations but did not generalize well to minority classes. The original LSTM was trained from scratch and had to learn both word meaning and classification boundaries from only 45k training tweets. That is not much data for a neural sequence model with many rare hashtags and informal spellings.

The SVM improves because sparse high-dimensional text features are a strong match for short tweets. Character n-grams add robustness for hashtags and spelling variants. The transformer improves further because it starts with Twitter-domain language knowledge and can model context beyond isolated tokens. This is why the best online systems are pretrained transformers or very strong sparse systems rather than small from-scratch LSTMs.

## Easiest Emojis Under The Best Improved Model

| Emoji | Name | Precision | Recall | F1 | Support | Indicative Tokens |
|---|---|---:|---:|---:|---:|---|
| ❤ | red heart | 0.7407 | 0.8623 | 0.7969 | 10798 | my, love, !, the, with, you |
| 🎄 | Christmas tree | 0.6387 | 0.7689 | 0.6978 | 1545 | christmas, merry, tree, #christmas, eve, #christmastree |
| ☀ | sun | 0.7148 | 0.6221 | 0.6653 | 1265 | sun, sunshine, sunny, morning, beach, summer |
| 🇺🇸 | United States | 0.7226 | 0.5321 | 0.6129 | 1949 | #usa, veterans, america, #america, #merica, #murica |
| 🔥 | fire | 0.5776 | 0.5256 | 0.5504 | 3716 | lit, fire, user, #fire, hot, was |
| 😂 | face with tears of joy | 0.4520 | 0.5337 | 0.4895 | 4534 | lol, user, the, when, !, lmao |
| 📷 | camera | 0.3030 | 0.6997 | 0.4229 | 1432 | user, by, the, rider, los, of |
| 😍 | smiling face with hearteyes | 0.3204 | 0.4431 | 0.3719 | 4830 | !, the, this, my, in, user |

These are easier because they have distinctive lexical anchors: holiday words for 🎄, affection words for ❤, laughter tokens for 😂, weather/beach words for ☀, and photo/location language for camera labels.

## Hardest Emojis Under The Best Improved Model

| Emoji | Name | Precision | Recall | F1 | Support | Indicative Tokens |
|---|---|---:|---:|---:|---:|---|
| 😜 | winking face with tongue | 0.1007 | 0.0139 | 0.0244 | 1010 | !, the, to, in, ?, and |
| 💜 | purple heart | 0.3443 | 0.0189 | 0.0357 | 1114 | purple, #purplerain, prince, #prince, my, lavender |
| 📸 | camera with flash | 0.4593 | 0.0257 | 0.0486 | 2417 | user, by, photo, shoot, credit, the |
| 😁 | beaming face with smiling eyes | 0.1411 | 0.0295 | 0.0488 | 1153 | !, the, in, to, user, smile |
| 😉 | winking face | 0.1627 | 0.0995 | 0.1235 | 1306 | the, to, #mividaesunatombola, !, #livefromvegas, you |
| 💙 | blue heart | 0.2407 | 0.0923 | 0.1335 | 1549 | blue, dodger, my, #blue, the, #dodgers |
| 😊 | smiling face with smiling eyes | 0.1359 | 0.2666 | 0.1800 | 1613 | !, the, to, and, my, in |
| 💯 | hundred points | 0.3377 | 0.1230 | 0.1803 | 1244 | the, paisley, female, user, real, keeping |

Most common confusion patterns:

| True Emoji | Most Common Wrong Prediction | Wrong Count | Share Of True Class |
|---|---|---:|---:|
| 😜 winking face with tongue | 😂 face with tears of joy | 291 | 28.81% |
| 💜 purple heart | 😍 smiling face with hearteyes | 227 | 20.38% |
| 📸 camera with flash | 📷 camera | 1596 | 66.03% |
| 😁 beaming face with smiling eyes | 😊 smiling face with smiling eyes | 244 | 21.16% |
| 😉 winking face | 😂 face with tears of joy | 243 | 18.61% |
| 💙 blue heart | 😍 smiling face with hearteyes | 310 | 20.01% |
| 😊 smiling face with smiling eyes | 😍 smiling face with hearteyes | 284 | 17.61% |
| 💯 hundred points | 😂 face with tears of joy | 252 | 20.26% |

These remain hard because several emojis are pragmatically interchangeable in tweets. Heart variants share affection language, playful face variants overlap with laughter or teasing, and 📷 versus 📸 is a nearly identical visual/semantic pair. Better performance would require more data, user/context information, or a stronger fine-tuning setup optimized for rare-label recall.

## What To Try Next

- Fine-tune `vinai/bertweet-base` or `cardiffnlp/twitter-roberta-base` directly on the local TweetEval train split with class-balanced sampling.
- Use an ensemble: average transformer logits with SVM decision scores, because they make different kinds of errors.
- Optimize for macro-F1 rather than validation accuracy, including class weights or focal loss.
- Keep the original MLP/LSTM comparison for the assignment, but present the SVM and transformer as improved baselines and evidence that our first scores were below online systems.

## Sources

- [SemEval-2018 Task 2: Multilingual Emoji Prediction](https://aclanthology.org/S18-1003/): Original shared task defining emoji prediction from tweets.
- [TweetEval benchmark repository](https://github.com/cardiffnlp/tweeteval): Provides unified fixed splits and raw files used in this project.
- [TweetEval paper](https://aclanthology.org/2020.findings-emnlp.148/): Documents TweetEval as a standardized Twitter classification benchmark.
- [Tübingen-Oslo at SemEval-2018 Task 2](https://coltekin.net/cagri/papers/coltekin2018semeval.pdf): Reports strong SVM performance and discusses RNN difficulty on emoji prediction.
- [PickleTeam! at SemEval-2018 Task 2](https://aclanthology.org/S18-1072.pdf): Combines SVM and LSTM predictions, motivating comparison between sparse and sequence models.
- [DeepMoji](https://github.com/bfelbo/deepmoji): Large-scale emoji-pretrained representation model trained on 1.2B tweets.
- [NTUA-SLP at SemEval-2018 Task 2](https://arxiv.org/abs/1804.06657): Attention LSTM system that ranked highly using pretrained Twitter embeddings.
- [NLP-text2emoji public project](https://github.com/joonasrooben/NLP-text2emoji): Example public implementation using bidirectional LSTMs and word/character embeddings.
