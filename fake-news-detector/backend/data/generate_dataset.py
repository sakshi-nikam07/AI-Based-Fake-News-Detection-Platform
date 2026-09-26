"""
generate_dataset.py
--------------------
Builds a labeled dataset of REAL vs FAKE news text samples for training
the fake-news classifier. Since this environment has no internet access
to download a public dataset (e.g. Kaggle's Fake/Real News dataset),
this script generates a realistic, pattern-rich synthetic corpus:

- REAL samples: written in a measured, factual, attributed journalistic
  style (agencies, officials, data, dates, sources).
- FAKE samples: written in a sensational, clickbait, conspiratorial
  style (ALL CAPS claims, excessive punctuation, unverified claims,
  emotional manipulation, miracle cures, secret plots).

This is a DEMO dataset meant to teach the classifier the *stylistic and
linguistic* signals commonly associated with misinformation. For a
production system, replace this with a vetted, human-labeled dataset
(e.g. LIAR, FakeNewsNet, ISOT Fake/Real News) and retrain.

Run: python generate_dataset.py
Produces: dataset.csv (columns: text, label)  label: 1 = FAKE, 0 = REAL
"""

import csv
import random

random.seed(42)

# ---------------------------------------------------------------------------
# Building blocks for REAL news (factual, sourced, measured tone)
# ---------------------------------------------------------------------------
real_subjects = [
    "the Reserve Bank of India", "the Ministry of Health", "the World Health Organization",
    "the United Nations", "NASA", "the European Central Bank", "the Supreme Court",
    "the state government", "the Election Commission", "local municipal authorities",
    "the finance ministry", "researchers at a leading university", "the central bank",
    "the Department of Agriculture", "city officials", "the national statistics bureau",
    "the education ministry", "the transport authority", "the environment ministry",
    "the health department"
]
real_actions = [
    "announced a revised policy on {topic} following a review meeting held on Tuesday",
    "released quarterly data showing a {pct}% change in {topic}",
    "confirmed that new guidelines on {topic} will take effect next month",
    "reported that {topic} figures remained stable compared to the previous quarter",
    "published a study indicating measurable progress in {topic} over the past year",
    "held a press briefing to clarify recent changes to {topic} regulations",
    "signed an agreement aimed at improving {topic} in the coming fiscal year",
    "presented a report to lawmakers detailing the current state of {topic}",
    "clarified that claims circulating about {topic} are not accurate, citing official records",
    "said in a statement that {topic} initiatives are progressing as scheduled"
]
real_topics = [
    "public infrastructure spending", "vaccination coverage", "interest rates",
    "employment", "agricultural exports", "renewable energy capacity",
    "urban water supply", "school enrollment", "road safety", "air quality",
    "digital literacy programs", "small business lending", "healthcare access",
    "climate adaptation funding", "cybersecurity standards", "public transportation"
]

def make_real():
    subj = random.choice(real_subjects)
    action = random.choice(real_actions).format(
        topic=random.choice(real_topics), pct=random.randint(1, 15)
    )
    tail = random.choice([
        "Officials said further details would be shared in a follow-up briefing.",
        "The report is based on data collected over the last six months.",
        "A spokesperson confirmed the information was verified through official channels.",
        "The findings were reviewed by independent analysts before publication.",
        "No immediate changes are expected beyond what was outlined in the statement.",
        "The agency said it would continue monitoring the situation and issue updates as needed.",
    ])
    return f"{subj.capitalize()} {action}. {tail}"

# ---------------------------------------------------------------------------
# Building blocks for FAKE news (sensational, unverified, manipulative tone)
# ---------------------------------------------------------------------------
fake_openers = [
    "SHOCKING: You won't believe what {subject} is hiding about {topic}!!!",
    "BREAKING: Secret documents PROVE {subject} lied about {topic}",
    "Doctors HATE this one trick that {subject} doesn't want you to know about {topic}",
    "EXPOSED: The truth about {topic} that {subject} tried to bury",
    "This ONE simple fact about {topic} will change everything you thought you knew!!",
    "Insiders reveal {subject} is secretly controlling {topic} for hidden profits",
    "WAKE UP! {subject} has been lying to you about {topic} for years",
    "Leaked video allegedly shows {subject} admitting the truth about {topic}",
    "They don't want you to see this: {subject} caught covering up {topic}",
    "Miracle discovery about {topic} that {subject} is desperately trying to suppress",
]
fake_subjects = [
    "the government", "Big Pharma", "the mainstream media", "scientists",
    "a secret organization", "world leaders", "tech billionaires",
    "the health ministry", "banks", "a shadowy global elite"
]
fake_topics = [
    "vaccines", "5G towers", "your bank account", "the food supply",
    "the water you drink", "climate change data", "the upcoming election",
    "a new virus", "your phone's camera", "the moon landing footage",
    "cancer cures", "the stock market crash"
]
fake_tails = [
    "Share this before it gets DELETED!!!",
    "Wake up, sheeple — the truth is finally out!",
    "Experts are too scared to talk about this openly.",
    "This changes EVERYTHING and nobody is reporting it.",
    "The mainstream media refuses to cover this story.",
    "Forward this to everyone you know before it's too late!",
    "One insider risked everything to leak this information.",
    "This is the story they don't want going viral.",
]

def make_fake():
    subject = random.choice(fake_subjects)
    topic = random.choice(fake_topics)
    opener = random.choice(fake_openers).format(subject=subject, topic=topic)
    tail = random.choice(fake_tails)
    return f"{opener} {tail}"

# ---------------------------------------------------------------------------
# A smaller set of hand-written, high-signal examples (anchors)
# ---------------------------------------------------------------------------
handwritten_real = [
    "The central bank kept interest rates unchanged at its policy meeting on Thursday, citing stable inflation figures.",
    "The health ministry confirmed 1,204 new cases in its daily bulletin, in line with the previous week's trend.",
    "Parliament passed the amended labor bill after a extended debate, with the vote recorded at 214 to 187.",
    "The university's engineering department published peer-reviewed findings on battery efficiency in a scientific journal.",
    "City officials opened the newly built metro line after two years of construction, with the first trains running Monday.",
    "The company reported quarterly earnings in line with analyst expectations, according to a filing with regulators.",
    "Weather authorities issued a moderate rainfall advisory for the coastal districts through the weekend.",
    "The election commission released the final voter turnout figures, confirming a 63.4 percent participation rate.",
]
handwritten_fake = [
    "Government secretly adds mind-control chemicals to tap water, whistleblower claims in viral post!!!",
    "Scientists confirm the earth stopped rotating for three seconds last night, mainstream media silent!!",
    "Billionaire admits on hidden camera that vaccines were designed to shrink the population by 2030.",
    "Leaked memo shows tech companies plan to replace all humans with robots by next year, insiders say.",
    "Doctors shocked as man cures cancer overnight with kitchen spice big pharma has hidden for decades.",
    "Secret society caught rigging global weather patterns to control food prices, exclusive report reveals.",
    "Bank accounts to be frozen nationwide next week according to anonymous insider leak circulating online.",
    "Aliens spotted helping world leaders write new laws, government refuses to comment on the footage.",
]

# ---------------------------------------------------------------------------
# Assemble dataset
# ---------------------------------------------------------------------------
rows = []
for _ in range(320):
    rows.append((make_real(), 0))
for _ in range(320):
    rows.append((make_fake(), 1))
for t in handwritten_real:
    rows.append((t, 0))
for t in handwritten_fake:
    rows.append((t, 1))

random.shuffle(rows)

with open("dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(rows)

print(f"Generated dataset.csv with {len(rows)} rows "
      f"({sum(1 for _, l in rows if l == 0)} real, {sum(1 for _, l in rows if l == 1)} fake).")
