from transformers import pipeline

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

result = summarizer(note, max_length=40)
print(result)
