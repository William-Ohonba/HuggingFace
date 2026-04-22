# ============================================================
# Clinical NLP in 45 Minutes — In-Class Activity
# ============================================================

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# ============================================================
# PART 1: Clinical Text Classification (Domain Mismatch)
# ============================================================
print("=" * 60)
print("PART 1: Sentiment Classifier on Clinical Notes")
print("=" * 60)

classifier = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

texts = [
    "Patient reports chest pain and shortness of breath.",
    "Routine follow-up visit for blood pressure check.",
    "Mild headache for two days, no other symptoms."
]

for t in texts:
    print(t, "->", classifier(t))

# ------------------------------------------------------------
# Part 1 Answers
# ------------------------------------------------------------
# Q: What labels does the model return?
# A: POSITIVE and NEGATIVE — SST-2 sentiment labels from movie
#    review training data.
#
# Q: Are those labels a good fit for clinical triage?
# A: No. "Positive" and "Negative" have no clinical meaning.
#    A chest pain note being labeled "NEGATIVE" tells you nothing
#    about urgency — it just sounds linguistically negative.
#
# Q: What was this model trained on?
# A: The Stanford Sentiment Treebank (SST-2), a dataset of movie
#    reviews labeled for positive/negative sentiment.
#
# Q: Would this be safe to use in a hospital?
# A: Absolutely not. It mislabels high-acuity cases, the labels
#    are semantically meaningless for clinical decisions, and
#    relying on it could cause dangerous triage errors.
# ------------------------------------------------------------


# ============================================================
# PART 2: Domain-Appropriate Model (BioClinicalBERT Triage)
# ============================================================
print("\n" + "=" * 60)
print("PART 2: BioClinicalBERT Triage Model")
print("=" * 60)

model_id = "VolodymyrPugachov/BioClinicalBERT-Triage"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

triage_classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None
)

for t in texts:
    print(t, "->", triage_classifier(t))

# ------------------------------------------------------------
# Part 2 Answers
# ------------------------------------------------------------
# Q: How do predictions change relative to Part 1?
# A: Instead of POSITIVE/NEGATIVE, outputs are triage-relevant
#    labels (e.g., urgency levels). Chest pain should score higher
#    urgency; routine follow-up should score lower.
#
# Q: Which label set is more meaningful for a healthcare task?
# A: BioClinicalBERT's labels, because they map to real triage
#    decision-making rather than sentiment polarity.
#
# Q: How was this model trained and what does it mean?
# A: Fine-tuned on clinical text (MIMIC-III notes) starting from
#    BioClinicalBERT — a base model pre-trained on PubMed abstracts
#    and clinical notes. Fine-tuning on triage examples teaches it
#    to understand clinical severity rather than general language.
# ------------------------------------------------------------


# ============================================================
# PART 3: Clinical Summarization
# ============================================================
print("\n" + "=" * 60)
print("PART 3: Clinical Summarization")
print("=" * 60)

summarizer = pipeline("summarization")

note = """
ED triage: 67-year-old male with HTN and type 2 diabetes presents with
chest pressure radiating to left arm for 2 hours after mowing lawn.
Reports nausea and diaphoresis. Home meds listed as lisinopril and metformin.
Allergies: NKDA.
Nursing addendum 15 min later: patient states he is 64, not 67. Says pain
started yesterday, not today, and now denies radiation, nausea, or sweating.
States he stopped taking all meds 3 months ago. Allergy listed as penicillin
causes rash.
Resident note: wife says he nearly passed out in driveway and was clutching
his chest. Patient denies diabetes but prior chart shows A1c 9.1 last month.
Initial EKG documented as ST elevation in II, III, aVF; repeat note says
"no acute ST changes." Troponin pending.
"""

summary = summarizer(note, max_length=40)
print("Summary:", summary)

# ------------------------------------------------------------
# Part 3 Critical Analysis
# ------------------------------------------------------------
# HALLUCINATED DETAILS:
#   The model may state a confirmed diagnosis (e.g., STEMI) when
#   none exists — troponin is pending and EKG readings contradict
#   each other. Any definitive clinical conclusion is a hallucination.
#
# OMITTED CRITICAL FACTS:
#   - Medication discontinuation 3 months ago (major safety flag)
#   - Penicillin allergy correction (was listed as NKDA initially)
#   - Conflicting EKG readings (ST elevation vs. no acute changes)
#   These are all safety-critical but likely dropped in a short summary.
#
# OVERCONFIDENCE:
#   The summary will state facts flatly with no hedging, despite the
#   note being full of contradictions. A safe clinical tool would flag
#   inconsistencies; this model will silently pick one version.
# ------------------------------------------------------------


# ============================================================
# PART 4: "Break the Model" Challenge
# ============================================================
print("\n" + "=" * 60)
print("PART 4: Adversarial Inputs")
print("=" * 60)

adversarial_inputs = [
    # Negation — model may ignore "denies" and flag as high urgency anyway
    "Patient denies chest pain, but has been clutching left arm all morning.",

    # Contradictory vitals vs. self-report
    "Patient reports no pain whatsoever. BP 80/40, GCS 3, unresponsive.",

    # Rare/incidental finding with positive framing
    "Patient feels great. Incidental finding of bilateral adrenal masses on CT.",

    # Double negation
    "There is no absence of concerning symptoms at this time.",

    # Dangerous situation framed positively
    "Patient is very calm and relaxed. O2 sat 72%. Breathing independently.",
]

print("\n--- Adversarial inputs through Part 1 SST-2 classifier ---")
sentiment_classifier = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
for inp in adversarial_inputs:
    print(inp)
    print("  ->", sentiment_classifier(inp))
    print()

print("\n--- Adversarial inputs through Part 2 triage classifier ---")
for inp in adversarial_inputs:
    print(inp)
    print("  ->", triage_classifier(inp))
    print()

# ------------------------------------------------------------
# Part 4 Notes
# ------------------------------------------------------------
# KEY FAILURE MODES TO LOOK FOR:
#
# 1. Negation blindness: "denies chest pain" may still trigger high
#    urgency because the model sees "chest pain" as a token, not its
#    negated meaning.
#
# 2. Vital sign ignorance: BP 80/40 and O2 sat 72% are immediately
#    life-threatening, but models trained on symptom text may miss
#    numerical severity entirely.
#
# 3. Framing bias: positive language ("feels great", "calm") can
#    suppress urgency scores even when objective data is critical.
#
# 4. Double negation: "no absence of" = present, but NLP models
#    often fail to resolve stacked negations correctly.
# ------------------------------------------------------------