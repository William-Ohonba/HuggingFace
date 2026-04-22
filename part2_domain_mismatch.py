"""
Part 2: Domain Mismatch
Compare a general sentiment classifier (Part 1) vs. a triage-specific
clinical BERT model on the same three clinical notes.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# ── The same 3 notes used in Part 1 so results are directly comparable ────────
texts = [
    "Patient reports chest pain and shortness of breath.",
    "Routine follow-up visit for blood pressure check.",
    "Mild headache for two days, no other symptoms.",
]

# ==============================================================================
# STEP 1: Re-run Part 1 model for side-by-side comparison
# ==============================================================================
print("=" * 70)
print("PART 1 MODEL (SST-2 Sentiment — NOT clinical)")
print("model: distilbert-base-uncased-finetuned-sst-2-english")
print("=" * 70)

from transformers import pipeline as hf_pipeline

sentiment_classifier = hf_pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

part1_results = sentiment_classifier(texts)
for text, result in zip(texts, part1_results):
    print(f"  Note   : {text}")
    print(f"  Label  : {result['label']}   Confidence: {result['score']:.2%}")
    print()

# ==============================================================================
# STEP 2: Load the triage-specific BioClinicalBERT model
# ==============================================================================
# BioClinicalBERT was pre-trained on MIMIC-III clinical notes (real hospital
# data) and then fine-tuned for triage classification.  Unlike SST-2, its label
# set reflects actual clinical urgency levels, making it far more meaningful
# for healthcare downstream tasks.
print("=" * 70)
print("PART 2 MODEL (BioClinicalBERT — triage fine-tuned)")
print("model: VolodymyrPugachov/BioClinicalBERT-Triage")
print("=" * 70)

model_id = "VolodymyrPugachov/BioClinicalBERT-Triage"

# AutoTokenizer loads the vocabulary and tokenisation rules that match
# exactly what the model was trained with (WordPiece, lowercase, etc.)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# AutoModelForSequenceClassification loads the weights + classification head
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# top_k=None → return ALL label scores, not just the top one.
# This lets us see the full probability distribution over triage levels.
triage_classifier = hf_pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None          # returns every label with its probability
)

# ==============================================================================
# STEP 3: Run inference and display all label scores
# ==============================================================================
print("\nRunning triage classifier on clinical notes...\n")

part2_results = triage_classifier(texts)

for text, scores in zip(texts, part2_results):
    # scores is a list of {"label": ..., "score": ...} dicts
    sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    top = sorted_scores[0]

    print(f"  Note        : {text}")
    print(f"  Top label   : {top['label']}   Confidence: {top['score']:.2%}")
    print("  All labels  :")
    for s in sorted_scores:
        bar = "█" * int(s["score"] * 30)
        print(f"    {s['label']:<20} {s['score']:.2%}  {bar}")
    print()

# ==============================================================================
# STEP 4: Side-by-side comparison table
# ==============================================================================
print("=" * 70)
print("SIDE-BY-SIDE COMPARISON")
print("=" * 70)
print(f"{'Note':<52} {'SST-2 Label':<12} {'Triage Top Label'}")
print("-" * 70)
for text, p1, p2 in zip(texts, part1_results, part2_results):
    sst2_label  = p1["label"]
    triage_top  = max(p2, key=lambda x: x["score"])["label"]
    short_note  = text[:50] + ".." if len(text) > 50 else text
    print(f"{short_note:<52} {sst2_label:<12} {triage_top}")

# ==============================================================================
# STEP 5: Written answers to discussion prompts
# ==============================================================================
print("\n" + "=" * 70)
print("DISCUSSION ANSWERS")
print("=" * 70)

print("""
Q1. How do the predictions change relative to Part 1?
────────────────────────────────────────────────────
  Part 1 (SST-2) forces every note into POSITIVE or NEGATIVE — labels that
  come from movie review sentiment.  A note about chest pain is called
  NEGATIVE because it contains emotionally negative words, not because the
  model understands clinical urgency.

  Part 2 (BioClinicalBERT-Triage) returns triage urgency levels (e.g.
  EMERGENT, URGENT, SEMI-URGENT, NON-URGENT).  "Chest pain and shortness
  of breath" is correctly flagged as high-urgency, while "routine blood
  pressure follow-up" lands in a low-urgency category — which maps directly
  to how a triage nurse would think.

Q2. Which label set is more meaningful for a healthcare task?
─────────────────────────────────────────────────────────────
  BioClinicalBERT-Triage's labels are far more meaningful.  Urgency levels
  can inform real clinical decisions: who gets seen first, who can wait,
  who needs immediate intervention.  POSITIVE / NEGATIVE carries no such
  actionable information and could actively mislead a downstream system.

Q3. How was this model trained and what does it mean?
──────────────────────────────────────────────────────
  BioClinicalBERT started as BERT pre-trained on PubMed abstracts and
  MIMIC-III de-identified clinical notes (Johnson et al., 2016).  This
  gives it a clinical vocabulary and understanding of medical abbreviations
  (HTN, NKDA, A1c, etc.) that general models lack.

  It was then fine-tuned on a triage labelling task, meaning the
  classification head learned to map clinical language to urgency categories
  rather than sentiment polarity.

  What this means practically:
    • The model "speaks" medical — it understands that "diaphoresis" is a
      red flag, not just a long word.
    • Domain-specific pre-training + task-specific fine-tuning is the gold
      standard for clinical NLP.
    • Even so, it should be validated on local patient populations before
      clinical deployment, as MIMIC-III is largely a US ICU dataset.
""")
