# Project 17 Final Detailed Report: Emoji Prediction From Text

## 1. Executive Summary

This project studied **emoji prediction from tweet text**. Given a tweet with the original emoji removed, the task is to predict the most likely emoji from 20 TweetEval/SemEval emoji classes.

The assignment specifically required:

- MLP on bag-of-words features.
- LSTM on tweet token sequences.
- Top-1 and top-3 accuracy.
- Analysis of easiest and hardest emojis.
- Research into online methods, notebooks, discussions, and stronger alternatives.
- A final method with technical justification.

The final best local method was:

**Calibrated Twitter-RoBERTa + LinearSVM ensemble**

Final test performance:

| Model | Top-1 Accuracy | Top-3 Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| MLP TF-IDF BoW | 38.69% | 58.59% | 0.1955 | 0.3372 |
| LSTM | 36.94% | 57.09% | 0.1788 | 0.3256 |
| LinearSVM word+char TF-IDF | 41.42% | 59.63% | 0.2800 | 0.4026 |
| Twitter-RoBERTa emoji GPU | 46.02% | 66.75% | 0.3155 | 0.4316 |
| BERTweet emoji HF checkpoint | 32.88% | 57.19% | 0.2641 | 0.3083 |
| Calibrated RoBERTa+SVM ensemble | 44.82% | 66.97% | 0.3281 | 0.4319 |

The final ensemble had slightly lower top-1 accuracy than Twitter-RoBERTa alone, but it had the best **macro-F1**, which is the important official metric for TweetEval/SemEval emoji prediction because it treats all emoji classes equally. This matters because the dataset is highly imbalanced: common emojis such as red heart and laughter dominate the data, while rarer emojis are much harder to recover.

There was no k-fold cross-validation in this project. TweetEval provides fixed train/validation/test splits. The phrase "first fold" is therefore interpreted here as the **fixed validation split used for model selection**. The final ensemble was selected on that validation split and then evaluated once on the official test split.

## 2. Dataset And Evaluation Setup

We used the public TweetEval `emoji` benchmark, derived from SemEval-2018 Task 2.

Verified split sizes:

| Split | Examples |
|---|---:|
| Train | 45,000 |
| Validation | 5,000 |
| Test | 50,000 |
| Labels | 20 emoji classes |

Each example is:

```text
x_i = tweet text
y_i = emoji label in {0, 1, ..., 19}
```

The model produces a score vector:

```text
s_i = [s_i1, s_i2, ..., s_i20]
```

The predicted class is:

```text
y_hat_i = argmax_c s_ic
```

### Metrics

**Top-1 accuracy**

Top-1 accuracy is the fraction of examples where the highest-scoring emoji is the true emoji:

```text
Top1 = (1 / N) * sum_i 1[argmax_c s_ic = y_i]
```

**Top-3 accuracy**

Top-3 accuracy checks whether the true emoji appears among the three highest-scoring classes:

```text
Top3 = (1 / N) * sum_i 1[y_i in Top3(s_i)]
```

Top-3 is meaningful because emoji usage is ambiguous. A happy affectionate tweet may plausibly be red heart, smiling face with heart-eyes, two hearts, or face blowing a kiss.

**Macro-F1**

For each emoji class `c`:

```text
Precision_c = TP_c / (TP_c + FP_c)
Recall_c    = TP_c / (TP_c + FN_c)
F1_c        = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)
```

Macro-F1 averages F1 across all classes equally:

```text
MacroF1 = (1 / C) * sum_c F1_c
```

This is stricter than accuracy because rare emojis matter as much as common emojis. A model can get decent top-1 accuracy by predicting common emojis, but macro-F1 exposes whether it fails on minority classes.

## 3. Online Research Findings

The research showed four important patterns.

First, TweetEval and SemEval use macro-F1 as the key emoji metric. Online leaderboard numbers report roughly:

| Online Reference Model | Emoji Macro-F1 |
|---|---:|
| LSTM | 24.7 |
| SVM | 29.3 |
| RoBERTa-Base | 30.9 |
| RoBERTa-Retrained | 31.4 |
| BERTweet | 33.4 |
| SemEval English winning SVM-style system | 35.99 |

Second, sparse linear models are unusually strong for this task. SemEval papers repeatedly reported that SVMs with word and character n-grams outperformed from-scratch RNN/LSTM models. This happens because emoji prediction from tweets often depends on short lexical signals: hashtags, slang, spelling variants, seasonal words, repeated punctuation, and short phrases.

Third, pretrained Twitter transformers are the strongest public neural approach. Twitter-RoBERTa and BERTweet are pretrained on huge Twitter corpora, so they already understand social media spelling, hashtags, usernames, URLs, and informal sentiment cues.

Fourth, ensembling is a reasonable next step. Sparse models and transformers make different errors. Sparse models capture exact lexical surface forms. Transformers capture context and semantic meaning. Combining them can improve macro-F1 if the blend is tuned on validation data.

Sources used:

- TweetEval official repository and leaderboard: https://github.com/cardiffnlp/tweeteval
- Twitter-RoBERTa emoji model: https://huggingface.co/cardiffnlp/twitter-roberta-base-emoji
- BERTweet emoji checkpoint: https://huggingface.co/cardiffnlp/bertweet-base-emoji
- TweetNLP toolkit: https://github.com/cardiffnlp/tweetnlp
- OpenCodePapers TweetEval leaderboard mirror: https://opencodepapers-b7572d.gitlab.io/benchmarks/sentiment-analysis-on-tweeteval.html
- PickleTeam SemEval paper: https://aclanthology.org/S18-1072.pdf
- Tubingen-Oslo SemEval paper: https://aclanthology.org/S18-1004/

## 4. Method 1: MLP On Bag-Of-Words / TF-IDF

### Concept

The first required baseline was an MLP over bag-of-words features. Since raw bag-of-words counts are noisy, we used TF-IDF word features.

The tweet is converted into a sparse vector:

```text
x in R^V
```

where `V` is the vocabulary size. Each coordinate corresponds to a word or n-gram feature.

The TF-IDF value for term `t` in document `d` is:

```text
tfidf(t, d) = tf(t, d) * idf(t)
```

with:

```text
idf(t) = log((N + 1) / (df(t) + 1)) + 1
```

where:

- `N` is the number of training tweets.
- `df(t)` is the number of tweets containing term `t`.

Common words receive lower weight, while distinctive words and hashtags receive higher weight.

### MLP Mathematics

The MLP computes:

```text
h1 = ReLU(W1 x + b1)
h2 = ReLU(W2 h1 + b2)
s  = W3 h2 + b3
p  = softmax(s)
```

The softmax converts logits into class probabilities:

```text
p_c = exp(s_c) / sum_j exp(s_j)
```

Training minimizes cross-entropy:

```text
L = -sum_i log p_i[y_i]
```

### Result

| Model | Top-1 | Top-3 | Macro F1 |
|---|---:|---:|---:|
| MLP TF-IDF BoW | 38.69% | 58.59% | 0.1955 |

### Why It Worked Partly

The MLP learned common lexical patterns:

- `love`, `my`, `you` -> heart emojis.
- `lol`, `lmao`, laughter phrases -> tears of joy.
- `christmas`, `merry`, `tree` -> Christmas tree.
- `sun`, `sunny`, `beach` -> sun.

### Why It Was Not Enough

The MLP had poor macro-F1 because it failed many rare classes. It used sparse TF-IDF features but then passed them through dense neural layers. With only 45k training tweets and 20 imbalanced classes, it leaned toward common patterns and did not recover rare emoji boundaries well.

The model got reasonable top-1 accuracy but weak macro-F1, meaning it was not fair across all emoji classes.

## 5. Method 2: LSTM From Scratch

### Concept

The second required baseline was an LSTM over tweet token sequences. Unlike bag-of-words, an LSTM sees tokens in order:

```text
tweet = [w1, w2, ..., wT]
```

Each token is mapped to an embedding:

```text
e_t = Embedding(w_t)
```

### LSTM Mathematics

At each time step, the LSTM updates gates:

```text
i_t = sigmoid(W_i e_t + U_i h_{t-1} + b_i)
f_t = sigmoid(W_f e_t + U_f h_{t-1} + b_f)
o_t = sigmoid(W_o e_t + U_o h_{t-1} + b_o)
g_t = tanh(W_g e_t + U_g h_{t-1} + b_g)
```

The memory cell is:

```text
c_t = f_t * c_{t-1} + i_t * g_t
```

The hidden state is:

```text
h_t = o_t * tanh(c_t)
```

The final hidden state is classified:

```text
s = W h_T + b
p = softmax(s)
```

### Result

| Model | Top-1 | Top-3 | Macro F1 |
|---|---:|---:|---:|
| LSTM | 36.94% | 57.09% | 0.1788 |

### Why It Underperformed

The LSTM was trained from scratch. That means it had to learn:

- word meanings,
- hashtag behavior,
- slang,
- emoji associations,
- rare class boundaries,
- and sequence composition,

all from only 45k training examples.

That is not enough for strong neural sequence learning on noisy tweets. The LSTM also had higher model complexity than linear sparse methods, but less useful prior knowledge than a pretrained transformer.

This matches online SemEval findings: from-scratch RNN/LSTM models often underperform SVMs on this emoji task.

## 6. Method 3: LinearSVM With Word And Character TF-IDF

### Concept

After checking online systems, we improved the sparse baseline using a LinearSVM with:

- word unigrams and bigrams,
- character 3-5 grams,
- TF-IDF weighting.

The final feature vector was:

```text
phi(x) = [word_tfidf(x), char_tfidf(x)]
```

Character n-grams are important for tweets because they capture:

- hashtags,
- spelling variants,
- elongated words,
- fragments inside usernames or places,
- punctuation patterns,
- informal expressions.

Examples:

```text
#christmastree -> chr, hri, ris, ..., tree
loooove        -> loo, ooo, oov, ove
sunshine       -> sun, uns, nsh, ...
```

### SVM Mathematics

For each class `c`, the one-vs-rest LinearSVM learns a linear decision function:

```text
s_c(x) = w_c^T phi(x) + b_c
```

The predicted class is:

```text
y_hat = argmax_c s_c(x)
```

The hinge-loss objective is:

```text
min_w  (1/2) ||w||^2 + C * sum_i max(0, 1 - y_i * (w^T phi(x_i) + b))
```

For multiclass classification, this is applied in one-vs-rest form.

### Result

| Model | Top-1 | Top-3 | Macro F1 |
|---|---:|---:|---:|
| LinearSVM word+char TF-IDF | 41.42% | 59.63% | 0.2800 |

### Why It Improved The Score

The SVM is well suited to short text because the decision boundary is linear in a very high-dimensional lexical space. Tweets often contain strong direct clues:

- `merry`, `christmas`, `tree` -> Christmas tree.
- `sun`, `beach`, `summer` -> sun.
- `lit`, `fire`, `hot` -> fire.
- `lol`, `lmao` -> tears of joy.

The SVM improved macro-F1 from 0.1955 to 0.2800. That is a large improvement because it recovered more rare-class signal than the MLP/LSTM.

## 7. Method 4: Twitter-RoBERTa Emoji Model On GPU

### Concept

Next we used `cardiffnlp/twitter-roberta-base-emoji`, a RoBERTa-base model pretrained on Twitter data and fine-tuned for TweetEval emoji prediction.

This was run on the RTX 4050 using:

```text
torch 2.11.0+cu128
CUDA 12.8
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
VRAM: 6 GB
```

### Transformer Mathematics

Transformers represent each token with contextual self-attention.

Given token representations:

```text
X = [x_1, x_2, ..., x_T]
```

the model computes:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

Self-attention is:

```text
Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V
```

This lets every token attend to every other token. For tweet emoji prediction, this means the model can learn interactions such as:

- affection words plus person mentions,
- sarcasm-like laughter patterns,
- holiday context,
- photo/location wording,
- sentiment-bearing phrases.

The classification head uses the final contextual representation and outputs:

```text
s = W h_cls + b
p = softmax(s)
```

### Result

| Model | Top-1 | Top-3 | Macro F1 |
|---|---:|---:|---:|
| Twitter-RoBERTa emoji GPU | 46.02% | 66.75% | 0.3155 |

### Why It Improved The Score

Twitter-RoBERTa improved because it brought external knowledge from large Twitter pretraining. Unlike the LSTM, it did not learn tweet language from scratch. It already had representations for:

- hashtags,
- URLs,
- mentions,
- informal grammar,
- social media sentiment,
- slang,
- common tweet structures.

It became the best single model and was much closer to online transformer baselines.

## 8. Method 5: BERTweet Emoji Hugging Face Checkpoint

### Why We Tried It

Online leaderboard research showed BERTweet as a very strong TweetEval model, around 33.4 macro-F1. Therefore, we evaluated `cardiffnlp/bertweet-base-emoji`.

### Result

| Model | Top-1 | Top-3 | Macro F1 |
|---|---:|---:|---:|
| BERTweet emoji HF checkpoint | 32.88% | 57.19% | 0.2641 |

### Why It Was Rejected

This checkpoint did not reproduce the expected leaderboard-level performance. We also checked whether the generic output labels `LABEL_0 ... LABEL_19` were simply in the wrong order. Validation alignment did not recover competitive test performance.

So the model was useful as a research check, but it was not selected as the final method.

## 9. Final Method: Calibrated Twitter-RoBERTa + SVM Ensemble

### Concept

The final successful method combined:

1. Twitter-RoBERTa contextual logits.
2. LinearSVM sparse lexical decision scores.
3. Score normalization.
4. Temperature scaling.
5. A small class-prior calibration term.
6. Validation search optimized for macro-F1.

This worked because the two models make different kinds of predictions:

- RoBERTa understands semantic context.
- SVM captures exact lexical and character-level clues.

The final model used both.

### Score Normalization

The two models output scores on different scales:

- RoBERTa outputs neural logits.
- SVM outputs signed margin scores.

So we standardized each model's validation/test scores class-wise:

```text
z_c = (s_c - mean_c) / std_c
```

where `mean_c` and `std_c` were estimated from validation scores.

This prevents one model from dominating only because its raw scores have larger magnitude.

### Temperature Scaling

Temperature controls confidence:

```text
z'_c = z_c / T
```

If `T > 1`, the distribution becomes softer. If `T < 1`, it becomes sharper.

The final validation-selected temperatures were:

```text
RoBERTa temperature = 1.5
SVM temperature     = 3.0
```

This softened both models, especially the SVM, so the ensemble could combine ranking information instead of blindly trusting overconfident margins.

### Class-Prior Calibration

The train distribution is imbalanced. Common emojis can dominate the model. To adjust class behavior, we used a prior term.

Let:

```text
pi_c = training frequency of class c
```

The prior bias is:

```text
b_c = -log(pi_c)
```

This gives rarer classes a larger positive bias. In the final formula we used a coefficient:

```text
alpha = -0.2
```

The negative value means the validation split preferred a small correction back toward common-class confidence after score normalization and blending. This is important: calibration was not guessed; it was selected by validation macro-F1.

### Final Ensemble Formula

For class `c`, the final score was:

```text
S_c(x) =
  w * (R_c(x) / T_R)
  + (1 - w) * (V_c(x) / T_S)
  + alpha * b_c
```

where:

- `R_c(x)` = normalized Twitter-RoBERTa score for class `c`.
- `V_c(x)` = normalized SVM score for class `c`.
- `w` = RoBERTa ensemble weight.
- `T_R` = RoBERTa temperature.
- `T_S` = SVM temperature.
- `alpha` = class-prior coefficient.
- `b_c = -log(pi_c)` = class-prior bias.

The final prediction is:

```text
y_hat = argmax_c S_c(x)
```

### Best Validation Configuration

The selected validation configuration was:

```json
{
  "weight_roberta": 0.5,
  "roberta_temp": 1.5,
  "svm_temp": 3.0,
  "prior_alpha": -0.2,
  "validation_top1_accuracy": 0.3388,
  "validation_top3_accuracy": 0.5916,
  "validation_macro_f1": 0.3046
}
```

### Final Test Result

| Model | Top-1 | Top-3 | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Calibrated RoBERTa+SVM ensemble | 44.82% | 66.97% | 0.3281 | 0.4319 |

### Why This Was The Final Winning Method

The ensemble was the best method by macro-F1:

```text
Macro-F1 improvement over MLP:          +0.1325
Macro-F1 improvement over LSTM:         +0.1493
Macro-F1 improvement over SVM:          +0.0480
Macro-F1 improvement over RoBERTa:      +0.0126
Macro-F1 improvement over BERTweet HF:  +0.0640
```

It did not maximize top-1 accuracy. Twitter-RoBERTa alone had higher top-1:

```text
Twitter-RoBERTa top-1: 46.02%
Ensemble top-1:        44.82%
```

But the ensemble improved:

```text
Twitter-RoBERTa macro-F1: 0.3155
Ensemble macro-F1:        0.3281

Twitter-RoBERTa top-3:    66.75%
Ensemble top-3:           66.97%
```

Because TweetEval/SemEval focuses on macro-F1, the ensemble is the final successful method.

## 10. Easiest Emojis And Why

The final ensemble's easiest classes were:

| Emoji | Name | F1 | Why Easier |
|---|---|---:|---|
| ❤ | red heart | 0.7478 | Very frequent; strong affection words such as love, my, you |
| 🎄 | Christmas tree | 0.6767 | Highly distinctive seasonal words: christmas, merry, tree |
| 🇺🇸 | United States | 0.6105 | Location and patriotic text cues |
| 🔥 | fire | 0.5579 | Words like fire, lit, hot; strong slang association |
| ☀ | sun | 0.5551 | Weather/beach/summer vocabulary |
| 😂 | face with tears of joy | 0.4831 | Laughter tokens such as lol, lmao, funny |

These classes are easier because their tweets often contain explicit lexical anchors. The model does not need deep reasoning when a phrase like "merry christmas" or "beautiful sunny beach day" appears.

## 11. Hardest Emojis And Why

The final ensemble's hardest classes were:

| Emoji | Name | F1 | Why Harder |
|---|---|---:|---|
| 😜 | winking face with tongue | 0.0725 | Playful face overlaps with laughter and winking |
| 😁 | beaming face with smiling eyes | 0.0930 | Similar to other happy/smiling emojis |
| 💜 | purple heart | 0.1037 | Heart variants share similar affection language |
| 💕 | two hearts | 0.1409 | Confused with red heart and heart-eyes |
| 😉 | winking face | 0.1483 | Used in broad playful contexts |
| 😊 | smiling face with smiling eyes | 0.1493 | Generic positive smile; weak unique lexical cues |
| 💙 | blue heart | 0.1556 | Heart color often not inferable from text |
| 😘 | face blowing a kiss | 0.1984 | Affection overlaps with hearts |

The hardest classes are difficult because emoji choice is often personal, stylistic, and underdetermined by text. For example:

```text
love you so much
```

could reasonably map to:

```text
❤, 😍, 💕, 😘, 💜, 💙
```

This is not always a model failure. Sometimes the text genuinely does not contain enough information to identify the exact emoji variant.

## 12. Important Confusion Patterns

The final ensemble's hard-class confusions show the problem clearly:

| True Emoji | Most Common Wrong Prediction | Interpretation |
|---|---|---|
| 😜 winking face with tongue | 😂 tears of joy | Playful text overlaps with laughter |
| 😁 beaming face | 😊 smiling face | Smile variants are semantically close |
| 💜 purple heart | ❤ red heart | Heart color is rarely explicit |
| 💕 two hearts | ❤ red heart | Affection language overlaps |
| 😉 winking face | 😂 tears of joy | Teasing and laughter overlap |
| 😊 smiling face | 😍 heart-eyes | Positive affection overlap |
| 💙 blue heart | ❤ red heart | Heart color ambiguity |
| 😘 kiss face | ❤ red heart | Affection/kiss/heart overlap |

This explains why macro-F1 is hard to push much higher. The model must distinguish not just sentiment, but exact emoji style.

## 13. Why We Did Not Stop At The Required MLP vs LSTM

The assignment required MLP versus LSTM, and we completed that. However, the research showed both were below strong online systems.

The MLP and LSTM answered the assignment question:

- MLP BoW performed better than LSTM.
- This agreed with online SemEval observations that sparse lexical methods can beat RNNs on this task.

But the final project needed a stronger working model and technical justification. Therefore we extended the work:

1. LinearSVM to test the strongest classical text-classification direction.
2. Twitter-RoBERTa to test pretrained transformer direction.
3. BERTweet checkpoint to test the leaderboard-motivated model family.
4. Calibrated ensemble to combine complementary strengths.

This progression was evidence-driven rather than arbitrary.

## 14. Final Conclusion

The final successful method was the **Calibrated Twitter-RoBERTa + LinearSVM ensemble**.

It succeeded because:

- Twitter-RoBERTa contributed contextual semantic understanding from large Twitter pretraining.
- LinearSVM contributed precise lexical, hashtag, spelling, and character n-gram evidence.
- Score normalization made the model scores comparable.
- Temperature scaling softened overconfident outputs.
- Prior calibration adjusted the class distribution effect.
- Validation macro-F1 tuning selected the blend objectively.

The final result:

```text
Top-1 Accuracy: 44.82%
Top-3 Accuracy: 66.97%
Macro-F1:       0.3281
Weighted-F1:    0.4319
```

This is close to the public TweetEval BERTweet macro-F1 of 0.334 and is much stronger than the original required MLP/LSTM baselines.

The final recommendation is:

- Use the **Calibrated RoBERTa+SVM ensemble** when the goal is best macro-F1 and balanced emoji prediction.
- Use **Twitter-RoBERTa alone** when the goal is maximum top-1 accuracy and simpler deployment.
- Keep **MLP and LSTM** in the report as required educational baselines.
- Consider full fine-tuning of `vinai/bertweet-base` or `cardiffnlp/twitter-roberta-base` with class-balanced sampling and multiple random seeds as the next major improvement.

## 15. Reproducibility

Main project files:

- `src/train_and_evaluate.py`
- `src/improve_and_report.py`
- `src/evaluate_bertweet_emoji.py`
- `src/ensemble_calibrate_report3.py`

Main result folders:

- `results/`
- `results/improved/`
- `results/bertweet/`
- `results/ensemble/`

Main reports:

- `report/report-1.md`
- `report/report-2.md`
- `report/report-3.md`
- `report/final-report.md`

Re-run commands:

```powershell
cd "D:\HARSHIT\NN PROJECT\project 17"
$env:PYTHONIOENCODING="utf-8"
python src\train_and_evaluate.py
python src\improve_and_report.py
python src\evaluate_bertweet_emoji.py
python src\ensemble_calibrate_report3.py
```
