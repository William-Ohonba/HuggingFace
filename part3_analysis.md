# Part 3: Clinical Summarization

## Setup

Default pipeline: `pipeline("summarization")` loads **`sshleifer/distilbart-cnn-12-6`**,
a distilled BART fine-tuned on the **CNN/DailyMail news** dataset. It has never
seen a clinical note during training — this is already a model–task mismatch,
the same class of problem raised in Part 1.

## Code

See `part3.py`.

## Raw Output

With `max_length=40` (as specified in the PDF):

```
 EKG documented as ST elevation in II, III, aVF; repeat note says
 "no acute ST changes" Troponin pending . Wife says he nearly passed out in
```

(Note: the pipeline also emitted a warning that `min_length=56 > max_length=40`,
so generation stopped at the cap and the final sentence was cut off mid-phrase:
*"nearly passed out in"*.)

## Critical Angle

### Hallucinated details
The summary is largely extractive, so it does not invent new symptoms or diagnoses.
However, it **collapses two contradictory EKG readings into a single flat statement**
("ST elevation in II, III, aVF; repeat note says 'no acute ST changes'") without
any hedging. A downstream reader who trusted the summary could easily mis-attribute
the two readings to the same interpretation, which is a subtle form of
fabricated coherence — the model manufactures a narrative where the source
explicitly documents disagreement.

### Omitted critical facts
The 40-token cap forces the model to drop almost everything clinically decisive:

- **Chief complaint entirely missing** — no mention of chest pressure radiating
  to the left arm, nausea, or diaphoresis (the classic inferior-MI triad that
  matches the ST elevation in II/III/aVF).
- **All chart discrepancies lost** — age (67 vs 64), onset (2 hours vs yesterday),
  medication adherence (listed on lisinopril/metformin vs stopped 3 months ago),
  and allergies (NKDA vs penicillin rash). In triage these contradictions are
  the single most important signal about data quality.
- **Diabetes denial vs A1c 9.1** omitted — a red flag for undertreated diabetes
  in a patient presenting with possible ACS.
- **Pending troponin** is mentioned, but with no framing that it is the
  outstanding item driving disposition.

### Overconfidence
- The summary is declarative throughout. There is no "the record is internally
  inconsistent" or "two EKG readings disagree" — the model presents conflicting
  clinical statements with the same confident tone it would use for a news
  headline, which is exactly the wrong register for a chart review.
- The output **truncates mid-sentence** ("nearly passed out in") yet the
  pipeline returns it with no indication that the result is incomplete.
  In a real workflow, a clinician scanning a dashboard could miss that this
  is a cut-off fragment, not a finished thought.
- The underlying model was trained to summarize news stories, where omitting
  or smoothing over contradictions is usually fine. It carries that prior into
  a setting where contradictions are the story.

## Takeaway

A generic news-trained summarizer applied to a multi-author clinical note
produces output that is fluent, extractive, and dangerously lossy: it silences
the chart's internal disagreements, drops the presenting complaint, and can
truncate mid-phrase without any uncertainty signal. This is a concrete example
of the model–task alignment problem the activity is designed to surface.
