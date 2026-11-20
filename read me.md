# Toxic Comment Detection Using a Bidirectional LSTM

**NLP Capstone Project Report**

**Task:** Multi-label text classification
**Model:** Bidirectional LSTM (BiLSTM) trained in PyTorch
**Interface:** Streamlit web app

---

## 1. Introduction

The internet runs on comments. Most are harmless, but a small share contain
insults, threats or hate speech. On a busy website, human moderators cannot read
everything, so an automatic filter is useful — not to replace moderators, but to
point them to the comments that need attention.

This project builds such a filter. It reads a comment and decides, for six
different kinds of toxicity, whether that comment belongs to each one.

The six categories are:

| Category | Simple meaning |
|---|---|
| toxic | Generally rude or harmful |
| severe_toxic | Extremely abusive |
| obscene | Swearing or vulgar language |
| threat | Threatening to harm someone |
| insult | Attacking a person directly |
| identity_hate | Hate based on religion, race, gender, etc. |

**Key point:** this is a *multi-label* problem, not a normal "pick one option"
problem. One comment can be `toxic` **and** `insult` **and** `obscene` at the same
time. Some comments have no labels at all.

---

## 2. Why this problem is hard

1. **Most comments are clean.** About 90% of the dataset has no toxicity label.
   A lazy model that always says "not toxic" would be 90% accurate and completely
   useless. So accuracy alone is a bad way to judge this project.
2. **Rare categories.** `threat` appears in only 0.3% of comments. There are very
   few examples for the model to learn from.
3. **Context matters.** Sarcasm, jokes, quotes, and words used between friends can
   look toxic to a machine.
4. **Similar categories overlap.** "Obscene" and "insult" often appear in the same
   comment, so the model has to learn the difference, not just detect bad words.

---

## 3. Dataset

The project uses the **Jigsaw Toxic Comment Classification** dataset — a public
collection of Wikipedia talk-page comments, each labelled by human reviewers.

| Item | Value |
|---|---|
| Total comments | 159,571 |
| Comments with at least one toxic label | 16,225 (about 10%) |
| Comments with no toxic label | 143,346 (about 90%) |
| Number of labels | 6 |

How often each label appears:

| Category | Positive comments | Share of dataset |
|---|---|---|
| toxic | 15,294 | 9.58% |
| obscene | 8,449 | 5.30% |
| insult | 7,877 | 4.94% |
| severe_toxic | 1,595 | 1.00% |
| identity_hate | 1,405 | 0.88% |
| threat | 478 | 0.30% |

The gap between `toxic` (9.58%) and `threat` (0.30%) is more than 30 times. This
imbalance shapes almost every design decision that follows.

---

## 4. Step 1 — Cleaning the text

A neural network cannot read words; it needs numbers. Before converting words to
numbers, we tidy the text so that the same sentence always looks the same:

- Make everything lowercase ("BAD" and "bad" become the same word).
- Replace web links with the token `urltoken`.
- Replace email addresses with the token `emailtoken`.
- Remove symbols and punctuation, but keep letters, numbers and apostrophes.
- Collapse extra spaces.

**Why replace links and emails instead of deleting them?** If we simply delete
them, the sentence can quietly change meaning. Replacing them keeps a placeholder
in the sentence so the model knows "something was here".

**Important:** the exact same cleaning function is used for training, testing and
new comments typed into the app. If the cleaning differed, the model would be
confused by inputs that look different from what it learned on.

---

## 5. Step 2 — Turning words into numbers

Each word is given an ID number:

- The most common word gets a low number, rare words get higher numbers.
- `0` is reserved for **padding** (empty space).
- `1` is reserved for **unknown words**.
- The vocabulary is capped at the **50,000** most common words.

Every comment is cut or padded to exactly **200 tokens**. Padding exists because
the model processes comments in batches, and all items in a batch must be the same
length.

The vocabulary is built **only from the training comments**. This is done on
purpose: if we let the model peek at test data during preparation, the final
scores would be too optimistic. This mistake is called *data leakage*.

**Also stored:** the real length of each comment. This lets the model ignore the
padding instead of treating empty space as real words.

---

## 6. Step 3 — Splitting the data

| Subset | Share | Comments | Purpose |
|---|---|---|---|
| Training | 70% | 111,700 | The model learns from these |
| Validation | 15% | 23,935 | Tuning and choosing thresholds |
| Test | 15% | 23,936 | Final exam — used once, at the end |

The split is **stratified**, which means the rare labels are spread fairly across
all three sets. Without this, `threat` might have almost no examples in the test
set and the final score would be meaningless.

The test set is kept locked away until the very end. It is never used to train the
model or to choose settings, because a score on data the model has already seen is
not a real score.

---

## 7. Step 4 — The model

The model is a **Bidirectional LSTM**. In plain terms:

| Layer | What it does |
|---|---|
| Embedding | Turns each word ID into a list of 100 numbers that capture its meaning |
| BiLSTM (2 layers) | Reads the sentence left-to-right **and** right-to-left, 128 units per direction |
| Dropout (0.35) | Randomly switches off some connections during training to prevent memorising |
| Output layer | Produces 6 numbers — one score per toxicity category |

**Why bidirectional?** In "you are not an idiot", the word "not" appears before
"idiot". Reading only left-to-right, the model may already have decided the
comment is insulting. Reading both directions gives it context from the whole
sentence.

**Why an LSTM and not a simple network?** A plain network treats a sentence as a
bag of unconnected words. An LSTM keeps a running memory, so word order and
grammar matter.

**Output:** six separate scores, each passed through a *sigmoid* function to give a
value between 0 and 1. A score of 0.91 for `insult` means "the model is fairly
confident this is an insult". These scores are independent, which is exactly what
multi-label classification needs.

---

## 8. Step 5 — Training

| Setting | Value |
|---|---|
| Optimiser | AdamW (learning rate 0.001, weight decay 0.0001) |
| Loss function | Binary cross-entropy with logits |
| Batch size | 128 |
| Maximum epochs | 15 |
| Early stopping patience | 3 epochs (no improvement) |
| Dropout | 0.35 |
| Training time | about 11 minutes on a GPU |

### Handling the imbalance

During training, mistakes on rare categories were made more expensive. The model
was told: *"Missing a `threat` is much worse than missing a harmless comment."*
This was done with a per-label **positive class weight**:

| Category | Weight used |
|---|---|
| toxic | 9.43 |
| severe_toxic | 20.00 |
| obscene | 17.88 |
| threat | 20.00 |
| insult | 19.26 |
| identity_hate | 20.00 |

Weights were capped at 20 to stop training from becoming unstable (the raw values
for `threat` would have been over 300).

### How the best model was chosen

After each epoch, the model was scored on the validation set using **macro-F1**, a
measure that gives every category equal importance — including the rare ones.
Whichever epoch scored best was saved.

### Training progress

| Epoch | Train loss | Val loss | Val macro-F1 |
|---|---|---|---|
| 1 | 0.4622 | 0.3197 | 0.4116 |
| 2 | 0.2956 | 0.2841 | 0.4786 |
| 3 | 0.2490 | 0.2677 | 0.4654 |
| 4 | 0.2167 | 0.2455 | 0.4876 |
| 5 | 0.1953 | 0.2366 | 0.5209 |
| 6 | 0.1783 | 0.2653 | 0.5205 |
| 7 | 0.1651 | 0.2825 | 0.5173 |
| 8 | 0.1533 | 0.2680 | 0.5544 |
| 9 | 0.1416 | 0.2725 | 0.5615 |
| 10 | 0.1336 | 0.2500 | 0.5719 |
| 11 | 0.1263 | 0.3151 | 0.5751 |
| 12 | 0.1191 | 0.3384 | 0.5746 |
| 13 | 0.1147 | 0.3205 | 0.5869 |
| 14 | 0.1102 | 0.3118 | 0.5831 |
| 15 | 0.1067 | 0.3535 | 0.5921 |

**Reading this table honestly:** training loss fell steadily from 0.46 to 0.11,
while validation loss flattened around 0.25 and then rose. That gap is the usual
sign of **overfitting** — the model is getting better at the training comments but
no longer improving on unseen ones. The final epoch happened to give the best
validation macro-F1, so the best checkpoint is epoch 15, but the margin is thin and
there is little room left to improve with this setup.

---

## 9. Step 6 — Choosing a decision threshold

Each category outputs a probability. We still need a cut-off: above this line =
"detected", below = "not detected". The default is 0.5, but 0.5 is rarely the best
choice when categories are unbalanced.

For every category, a range of thresholds was tested on the **validation** set, and
the one with the best F1-score was kept:

| Category | Chosen threshold |
|---|---|
| toxic | 0.88 |
| severe_toxic | 0.86 |
| obscene | 0.90 |
| threat | 0.78 |
| insult | 0.88 |
| identity_hate | 0.88 |

These are deliberately high. The model is only willing to shout "toxic" when it is
fairly sure. This keeps false alarms down, at the cost of missing some borderline
cases.

These thresholds are tuned for *this* dataset. They are not universal rules about
what counts as toxicity.

---

## 10. Results

All numbers below come from the test set of 23,936 comments, which the model had
never seen during training or tuning.

### 10.1 Overall results

| Metric | Score |
|---|---|
| Micro precision | 0.7203 |
| Micro recall | 0.7983 |
| Micro F1 | **0.7573** |
| Macro precision | 0.5914 |
| Macro recall | 0.7221 |
| Macro F1 | **0.6421** |
| Weighted F1 | 0.7653 |
| Subset accuracy | 0.9176 |

### 10.2 Results by category

| Category | Precision | Recall | F1-score | Avg. precision | Test examples |
|---|---|---|---|---|---|
| toxic | 0.8418 | 0.7768 | **0.8080** | 0.8910 | 2,294 |
| obscene | 0.7696 | 0.8595 | **0.8121** | 0.8888 | 1,267 |
| insult | 0.6781 | 0.8376 | **0.7494** | 0.8155 | 1,182 |
| severe_toxic | 0.3843 | 0.7573 | **0.5099** | 0.4862 | 239 |
| identity_hate | 0.4291 | 0.5877 | **0.4960** | 0.4377 | 211 |
| threat | 0.4458 | 0.5139 | **0.4774** | 0.4043 | 72 |

---

## 11. What the results actually mean

In simple words, using the `toxic` category as an example:

- **Precision 0.84** — when the model says "toxic", it is right about 84% of the
  time. Roughly 1 in 6 flags is a false alarm.
- **Recall 0.78** — of all the genuinely toxic comments, it catches about 78%. It
  misses roughly 1 in 5.
- **F1 0.81** — a single balanced score combining the two.

**The strong categories:** `obscene`, `toxic` and `insult` all score F1 above 0.74.
These categories have thousands of training examples and contain fairly obvious
signals, so the model learns them well.

**The weak categories:** `threat` (F1 0.48) and `identity_hate` (F1 0.50) are the
hard cases. Their precision is low — under 0.45 — which means that when the model
flags one, more than half the time it is a false alarm. Recall is also only around
0.5, so it misses about half the real cases. The cause is simple: only 334 `threat`
examples existed in training. That is not enough data for a model of this size.

**About subset accuracy (0.9176):** this looks impressive but is misleading. It
means "all six labels are exactly right at once". Because about 90% of comments are
completely clean, and the model correctly calls most of them clean, this number is
high without the model being good at spotting toxicity. **Micro F1 (0.757) and
Macro F1 (0.642) are the honest headline numbers.**

**About the precision/recall trade-off:** the high thresholds push recall up for
rare categories, which is why `severe_toxic` catches 76% of cases but is wrong more
than half the time it raises an alarm. For moderation, this is often the right
choice — a human reviewer can dismiss a false alarm in one second, but a missed
threat can be serious. The thresholds could be raised to reduce noise, or lowered
to catch more, depending on how the tool is used.

---

## 12. The web application

The trained model is wrapped in a **Streamlit** app (`app.py`), so the project can
be shown to someone without touching Python code.

How it works:

1. `model_utils.py` loads the saved weights, vocabulary, label order and thresholds.
2. The user types a comment into a text box.
3. The comment goes through the same cleaning pipeline used in training.
4. The model returns six probabilities.
5. Each probability is compared with its own threshold.
6. The app shows a red or green result plus a bar for every category.

**Safety check on load:** the app verifies that the vocabulary size, label order,
and thresholds in the saved files all match the checkpoint. If they do not, it
stops with a clear error instead of producing nonsense predictions.

The app has three tabs:

| Tab | Contents |
|---|---|
| Live Prediction | Type a comment, see the six-category breakdown |
| Model Performance | Overall and per-category test metrics |
| Category Analysis | Bar charts comparing categories |

Example behaviour:

| Comment | Result |
|---|---|
| "Have a nice day! I hope everything goes well." | No toxicity label detected |
| "You are an idiot and nobody likes you." | Potentially toxic |
| "I disagree with your opinion, but I respect your point of view." | No toxicity label detected |

The last example matters: a simple banned-word filter would flag it because of the
word "disagree" or because of the negative tone. This model reads the whole
sentence and correctly treats disagreement as civil.

---

## 13. Files in the project

| File | Purpose |
|---|---|
| `Toxic_Comment_Detection_NLP_Capstone_Presentation_Ready.ipynb` | Full project notebook: data, training, evaluation |
| `app.py` | Streamlit user interface |
| `model_utils.py` | Loading, cleaning, encoding and prediction logic |
| `model_output/bilstm_toxic_comment_model.pt` | Trained model weights (about 22 MB) |
| `model_output/vocabulary.json` | The 50,000-word vocabulary |
| `model_output/model_config.json` | Architecture settings and thresholds |
| `model_output/test_metrics.csv` | Saved test scores used by the app |

To run the app:

```
streamlit run app.py
```

---

## 14. Limitations

1. **One dataset, one style.** The model learned from Wikipedia talk pages. Reddit,
   gaming chat, Twitter and comment sections on news sites use different slang,
   emoji and sarcasm. Performance there would probably be lower.
2. **Sarcasm and context are missed.** "Great job, genius" may be read literally.
   Quotes and reclaimed language are also difficult.
3. **Rare categories are unreliable.** `threat` and `identity_hate` produce too many
   false alarms to be trusted without human review.
4. **No conversation history.** Each comment is judged alone, with no knowledge of
   who replied to whom.
5. **Word-level only.** The model has no understanding of meaning beyond word
   patterns. It cannot reason about intent.
6. **Scores are not probabilities.** A value of 0.9 does not mean "90% chance this
   is toxic" in a strict statistical sense. It is a model score compared against a
   tuned cut-off.
7. **Training stopped at the limit.** Validation macro-F1 was still slowly rising at
   epoch 15, so more epochs or stronger regularisation might improve results — but
   validation loss was already rising, so gains would likely be small.
8. **Bias in the data.** Human labels in the original dataset may themselves
   contain bias, and the model inherits that.

---

## 15. Future improvements

1. **Use a transformer.** Models such as DistilBERT or RoBERTa understand context
   far better than an LSTM and usually reach much higher F1 on this exact dataset.
2. **Compare against a simple baseline.** A TF-IDF + Logistic Regression model is
   fast and surprisingly strong. A baseline makes the BiLSTM result meaningful by
   comparison. (Code for this exists in the notebook but is switched off.)
3. **Add explainability.** Highlight which words pushed the prediction up, so a
   moderator can see *why* a comment was flagged.
4. **Batch mode.** Allow uploading a CSV of comments and downloading the results.
5. **Multilingual support.** Train on non-English data or use a multilingual model.
6. **Collect feedback.** Let moderators mark predictions as right or wrong, then
   use that data to retrain.
7. **Human-in-the-loop design.** Treat the model as a triage tool that sorts
   comments by risk, not as an automatic judge.

---

## 16. Conclusion

This project built a complete toxicity detection pipeline, end to end: cleaning
text, building a vocabulary from training data only, splitting data carefully,
training a Bidirectional LSTM, correcting for class imbalance, tuning per-category
thresholds on validation data, and reporting honest final scores on a held-out test
set — then packaging the result into a usable web app.

The model reaches a **micro F1 of 0.757** and a **macro F1 of 0.642**. It handles
common categories like `obscene`, `toxic` and `insult` well, and struggles with
`threat` and `identity_hate` because those categories have very few examples.

The most valuable lesson from this project is not the final F1 score. It is that
**accuracy is a misleading metric for imbalanced problems**, and that **careful data
splitting and validation-based tuning matter as much as the model architecture**.
A simple LSTM, trained properly and evaluated honestly, produces a genuinely useful
moderation aid — one that should support human moderators rather than replace them.

## Appendix — Glossary in plain English {: .pagebreak }

| Term | Meaning |
|---|---|
| **Multi-label** | One item can have several correct answers at once |
| **Label** | One of the six toxicity categories |
| **Token** | One word (or word piece) after the text has been split up |
| **Vocabulary** | The fixed list of words the model knows |
| **Embedding** | A list of numbers that represents a word's meaning |
| **LSTM** | A network that reads a sequence in order and remembers earlier words |
| **Bidirectional** | Reading the sentence in both directions |
| **Dropout** | Randomly ignoring some connections during training to avoid memorising |
| **Epoch** | One full pass through all training comments |
| **Batch** | A small group of comments processed together |
| **Loss** | A number showing how wrong the model currently is (lower is better) |
| **Overfitting** | Learning training data by heart instead of learning general patterns |
| **Class imbalance** | Some categories appear far more often than others |
| **Precision** | Of the comments I flagged, how many were really toxic? |
| **Recall** | Of the toxic comments that exist, how many did I catch? |
| **F1-score** | A balance between precision and recall |
| **Micro F1** | F1 calculated over all label decisions combined |
| **Macro F1** | Average of each category's F1, treating all categories equally |
| **Threshold** | The cut-off score above which a category counts as detected |
| **Validation set** | Data used to tune settings |
| **Test set** | Data used only for the final honest score |
| **Data leakage** | Accidentally letting test data influence training |
| **Stratified split** | Splitting so that rare labels stay balanced in every subset |
