# Sentiment Analyzer

Classifies product reviews as positive, negative or neutral, using TF-IDF
features and logistic regression.

**Tools:** Python, Pandas, Scikit-learn, regex, Matplotlib, Seaborn

## Dataset
Kaggle "Women's E-Commerce Clothing Reviews": 23,486 rows, of which 22,641
have review text. There is no sentiment column, so I made the labels from star
ratings: 1-2 stars = negative, 3 stars = neutral, 4-5 stars = positive.

| Class | Reviews | Share |
|---|---|---|
| Positive | 17,448 | 77.1% |
| Neutral | 2,823 | 12.5% |
| Negative | 2,370 | 10.5% |

The data file is not included in this repo; download it from Kaggle and upload
it to the notebook.

## What I did
- Checked the text for noise: 0 URLs, 0 emojis and 0 ALL-CAPS words, but
  repeated letters ("sooo") in 300 reviews (1.3%)
- Wrote a regex cleaning function: lowercase, remove links and emojis, shrink
  repeated letters, remove punctuation and digits, and expand contractions
  ("didn't" becomes "did not") so negations survive tokenization
- Used a stratified 80/20 split (18,112 train, 4,529 test)
- Converted text to TF-IDF features inside a Scikit-learn Pipeline, so the
  vocabulary is learned from training data only
- Compared Naive Bayes and logistic regression with 5-fold stratified
  cross-validation on the training data, judged on accuracy and macro-F1
  (the average F1 of the three classes)
- Tested four text setups, chose the best by cross-validation, and used the
  test set for the final score
- Did error analysis on the test set: confusion matrix, most confident
  mistakes, top words per class, and 20 random errors read by hand

## Cross-validation (training data)
| Model | Accuracy | Macro-F1 |
|---|---|---|
| Always predict "positive" | 0.771 | 0.290 |
| Naive Bayes | 0.776 | 0.327 |
| Naive Bayes, no class prior | 0.800 | 0.558 |
| Logistic regression | 0.816 | 0.554 |
| Logistic regression, class weights | 0.764 | 0.593 |

Four text setups, all with class-weighted logistic regression:

| Setup | Accuracy | Macro-F1 |
|---|---|---|
| A: scikit-learn stop words, single words | 0.764 | 0.593 |
| B: keep negation words, single words | 0.767 | 0.596 |
| C: keep negation words, single words + word pairs | 0.794 | 0.613 |
| **D: no stop-word removal, single words + word pairs** | **0.806** | **0.635** |

Naive Bayes dropped to a macro-F1 of 0.30 to 0.35 once word pairs were added,
so I used logistic regression.

## Results (held-out test set, 4,529 reviews)
Final model: setup D with class-weighted logistic regression.
**Accuracy 0.806, macro-F1 0.638.**

| Class | Precision | Recall | F1 | Reviews |
|---|---|---|---|---|
| Negative | 0.52 | 0.59 | 0.56 | 474 |
| Neutral | 0.38 | 0.52 | 0.44 | 565 |
| Positive | 0.95 | 0.88 | 0.92 | 3,490 |

Confusion matrix (rows are the real class):

| | Predicted negative | Predicted neutral | Predicted positive |
|---|---|---|---|
| Real negative | 281 | 164 | 29 |
| Real neutral | 152 | 294 | 119 |
| Real positive | 104 | 311 | 3,075 |

The accuracy looks only a little better than the "always positive" baseline
(0.771), because 77% of reviews are positive. Macro-F1 (0.638 versus 0.290) shows
the real gain.

## Error analysis
- 879 of 4,529 test reviews were wrong (19.4%). The most common mistake was a
  real positive review predicted as neutral (311).
- Reviews containing a negation word (not, no, never, nothing) had a 24.7%
  error rate, versus 12.5% for reviews without one. This does not prove that
  negation causes the errors, because negation words are also more common in
  negative and neutral reviews, which are harder classes.
- Top neutral features include "however", "but" and "unfortunately": the
  model treats contrast words as a sign of a mixed review.
- I read 20 random errors. About 14 were mixed reviews that praise and
  complain in the same text, 2 had a star rating that disagrees with the
  words (for example, a 4-5 star review that begins "I was so disappointed"),
  2 involved negation phrases, and 2 were plain model misses. None looked
  clearly sarcastic.
- Several of the most confident mistakes are 3-star reviews that read as
  fully positive, so some of the errors come from the labels, not the model.

## Challenges and fixes
- **Negation was being deleted:** scikit-learn's standard English stop-word
  list contains "not", "no" and "never", so "not good" turns into "good".
  I expanded contractions in the cleaning step, kept the negation words, and
  added word pairs (bigrams). Cross-validated macro-F1 went from 0.593 to
  0.596 (keeping negations) to 0.613 (adding word pairs), and 0.635 when I
  stopped removing stop words altogether. On 14 hand-written test sentences
  (negation, mixed feelings, sarcasm), setups A and B got 5 right and setup C
  got 7. That set is too small for a percentage and setup D was not tested
  on it. Neither model fixed "I would never buy this again" or "This is not
  bad at all".
- **Imbalanced classes:** 77% of reviews are positive, and negative (10.5%) and
  neutral (12.5%) are small. Class weights raised cross-validated macro-F1
  from 0.554 to 0.593 for logistic regression, and lowered accuracy from 0.816
  to 0.764. Neutral stayed the hardest class (F1 0.44).
- **Messy text:** the data had far less noise than I expected (no URLs or
  emojis), so the cleaning mostly mattered for contractions and repeated
  letters.

## Limitations
- Labels come from star ratings, and a 3-star review is not always neutral
  text. Some star ratings disagree with the words.
- Neutral reviews are often mixed, so a single label per review loses
  information.
- One domain (women's clothing), with a lot of size and fit talk. The model
  may not work on other product types.
- The 14 hand-written test sentences are a diagnostic, not a score.
- Model selection note: I first ran the test set on setup C (accuracy 0.798,
  macro-F1 0.621) by mistake, because I had left it as the default. Setup D
  had the best cross-validation score, so I then ran the test set on D, which
  gave the scores above. Both runs used the same test set.

## Run it
1. Download the dataset from Kaggle.
2. Open `sentiment_analyzer.ipynb` in Google Colab.
3. Upload the CSV and run all cells.
