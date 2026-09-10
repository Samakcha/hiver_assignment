# American Airlines AI Support Agent

An AI customer-support agent built for the Hiver SDE Intern take-home assignment using the Customer Support on Twitter dataset.

The system takes an incoming customer message and:

1. Classifies it into one of 15 support intents.
2. Retrieves historically similar American Airlines customer-support interactions.
3. Selects the most relevant historical examples.
4. Generates a grounded draft response based on those examples.
5. Decides whether the request should be auto-handled or escalated, with a reason.

The system is designed for support guidance and response drafting. It does not perform real-time booking, refund, baggage, or account actions.

---

## 1. Dataset and Scope

The project uses the **Customer Support on Twitter** dataset from Kaggle.

I selected **AmericanAir** because it provides enough customer-support interactions while covering a broad range of airline support scenarios.

After reconstructing customer → AmericanAir response pairs using the dataset's response pointers, the project contains approximately **36,500 historical customer-response pairs**.

The scope is limited to support scenarios represented in the historical dataset.

---

## 2. Intent Taxonomy

The final taxonomy contains 15 intents:

| Intent | Description |
|---|---|
| `flight_disruption` | Cancellation, delay, missed connection, or active disruption |
| `baggage` | Baggage problems or baggage-related questions |
| `booking_change` | Changing, cancelling, or modifying a booking |
| `seat` | Seat selection or seat assignment |
| `loyalty_upgrade` | Upgrades, loyalty status, miles, or priority benefits |
| `fees_charges` | Fees, charges, or unexpected costs |
| `refund_compensation` | Refund, voucher, or compensation requests |
| `flight_information` | Flight/status information without an active disruption |
| `airport_gate_staff` | Airport, gate, or staff-related issues |
| `onboard_aircraft` | Aircraft or onboard-experience issues |
| `contact_support` | Attempts to contact customer support |
| `check_in` | Check-in problems |
| `follow_up` | Follow-up on an existing support case |
| `non_support` | Not a customer-support request |
| `other_unclear` | Insufficient information or unclear request |

The primary-intent rule is to classify the customer's **main actionable goal**. Emotional tone does not determine the intent. When multiple issues are present, the primary actionable request is selected.

---

## 3. System Design

### Intent Classification

The incoming message is classified by an LLM using the frozen 15-intent taxonomy and explicit intent-boundary rules.

### Historical Retrieval

Historical customer-response pairs are indexed using TF-IDF with unigram and bigram features.

For each incoming message:

1. TF-IDF retrieves candidate historical interactions using cosine similarity.
2. An LLM selects the most relevant examples from those candidates.
3. The selected examples are passed to the response generator.

This combines inexpensive lexical retrieval with LLM-based relevance selection.

### Reply Generation

The response generator receives:

- the customer message
- the selected historical customer messages
- the corresponding historical American Airlines responses

It is instructed to synthesize a concise response rather than copy historical responses and to avoid unsupported claims.

### Escalation

The current escalation policy escalates:

- `refund_compensation`
- `fees_charges`
- `flight_disruption`
- `baggage`
- `airport_gate_staff`
- `other_unclear`

These categories may require case-specific handling that cannot safely be resolved from historical guidance alone.

The agent returns:

```text
intent
reply
decision
reason
```

---

## 4. Evaluation

A 200-example Golden Set was created using stratified sampling by customer-message length.

The examples were manually reviewed against the frozen taxonomy. An LLM was used only to provide initial annotation suggestions; the final labels were manually reviewed and corrected.

### Final Intent Classification

| Metric | Score |
|---|---:|
| Accuracy | 63.5% |
| Macro F1 | 62.8% |
| Weighted F1 | 63.6% |

The final LLM evaluation was performed on the 200-example Golden Set without using its labels to train the classifier.

### Baselines

Two classical TF-IDF classification baselines were implemented:

| Model | Macro F1 |
|---|---:|
| TF-IDF + Logistic Regression | 19.8% |
| TF-IDF + Linear SVM | 26.0% |
| LLM classifier | 62.8% |

The Logistic Regression and Linear SVM numbers are development baseline results from a stratified 75/25 split of the Golden Set. The LLM result is the final evaluation on the full 200-example Golden Set, so these should not be interpreted as a perfectly matched benchmark.

---

## 5. Reply Quality

Generated replies were evaluated using an LLM-as-judge on a 20-example sample.

Each response was scored from 1–5 on relevance, helpfulness, grounding, and clarity.

| Criterion | Score |
|---|---:|
| Relevance | 3.50 |
| Helpfulness | 2.90 |
| Grounding | 4.35 |
| Clarity | 4.75 |
| **Overall** | **3.88 / 5** |

The strongest aspects were grounding and clarity. Helpfulness was weaker because historical Twitter responses often do not contain enough information to fully resolve a customer's situation.

An independently labeled 30-example sample achieved **70% exact human agreement** with the reference labels.

### LLM-Judge / Human Agreement

The reply-quality judge was compared against an independently human-rated subset of 7 replies.

| Criterion | Exact Agreement |
|---|---:|
| Relevance | 42.9% |
| Helpfulness | 14.3% |
| Grounding | 57.1% |
| Clarity | 71.4% |
| **Overall** | **46.4%** |

Overall exact agreement was **46.4% across 28 criterion-level ratings** (7 replies × 4 criteria).

Agreement was strongest for clarity and grounding, while helpfulness showed the largest disagreement. This suggests that reply-quality judgments involving practical usefulness are more subjective and should be interpreted cautiously given the small sample size.

---

## 6. Failure Analysis — Top 5 Failure Modes

The final evaluation contained 73 misclassified examples out of 200. The largest errors were concentrated in a few recurring patterns.

### 1. Context-dependent follow-ups

**Example 1**

> `@AmericanAir Thanks! Will DM you`

**Actual:** `follow_up`  
**Predicted:** `non_support`

**Example 2**

> `@AmericanAir Of course no response. Because you don't care`

**Actual:** `follow_up`  
**Predicted:** `non_support`

**Hypothesis:** Follow-up messages often depend on the previous conversation. Short acknowledgements, complaints about missing responses, and references to an existing case contain little standalone information, making them easy to classify as generic or non-support messages.

**Potential improvement:** Include conversation history when available and explicitly detect references to previous contact, responses, DMs, or unresolved cases.

---

### 2. Ambiguous flight disruptions

**Example**

> `@AmericanAir Yep, meant 5608. This is what 4.5 hours in an airport on Thanksgiving does to a person.`

**Actual:** `flight_disruption`  
**Predicted:** `other_unclear`

**Example**

> `@AmericanAir So - not likely that AA would book a pax with that tight a connection, but the replacement F/A is ok?`

**Actual:** `flight_disruption`  
**Predicted:** `other_unclear`

**Hypothesis:** Some disruption messages describe the consequences or context of a disrupted journey without explicitly using words such as "cancelled" or "delayed."

**Potential improvement:** Use conversation context and add more examples of indirect disruption language to the classifier's demonstrations.

---

### 3. Loyalty and benefit ambiguity

**Example**

> `@AmericanAir Thanks! Doesn't look like I am keeping that status this year :(`

**Actual:** `loyalty_upgrade`  
**Predicted:** `non_support`

**Example**

> `@AmericanAir I’m not a million miler and am just a small fish in the frequent flyer pool, but I do try to stay loyal to AA.`

**Actual:** `loyalty_upgrade`  
**Predicted:** `non_support`

**Hypothesis:** Loyalty-related messages can look conversational rather than like explicit support requests, especially when they mention status or loyalty indirectly.

**Potential improvement:** Strengthen the loyalty/status definition with representative examples and distinguish loyalty-related statements from general conversation.

---

### 4. Very low-information messages

**Example**

> `@AmericanAir https://t.co/KQS0EKWRff`

**Actual:** `other_unclear`  
**Predicted:** `non_support`

**Example**

> `@AmericanAir WA 🌎`

**Actual:** `other_unclear`  
**Predicted:** `non_support`

**Hypothesis:** Extremely short messages provide insufficient semantic information for reliable intent classification. The model tends to default to the more common `non_support` category.

**Potential improvement:** Add an explicit low-information/insufficient-context detection step before intent classification and route these messages to `other_unclear`.

---

### 5. Flight-related non-support chatter

**Example**

> `@AmericanAir thanks for the quick response! Hopefully it's up in the air soon! 👍`

**Actual:** `non_support`  
**Predicted:** `flight_disruption`

**Example**

> `@AmericanAir You don't need to check the status the flight been delayed and I been known fuck off`

**Actual:** `non_support`  
**Predicted:** `flight_disruption`

**Hypothesis:** Messages containing flight-related words such as "flight" or "delayed" can appear to be active disruption requests even when the customer is simply commenting, acknowledging a response, or making a non-actionable statement.

**Potential improvement:** Classify based on the customer's actionable goal rather than individual flight-related keywords, with stronger examples distinguishing support requests from conversational messages.

---

## 7. Misleading Headline Number

**Weighted F1 of 63.6% can be misleading if presented alone.**

The intent distribution is imbalanced, with common categories such as `non_support` and `follow_up` occurring more frequently than several smaller support categories.

Weighted F1 therefore gives more influence to frequent intents.

For this reason, **Macro F1 of 62.8% is the more informative headline metric**, because every intent contributes equally.

The remaining gap between performance on common and smaller intents is also an important limitation of the current system.

---

## 8. Golden Set and Human Agreement

The Golden Set contains 200 examples sampled approximately equally across four message-length groups:

- Short: 51
- Medium: 49
- Long: 50
- Very long: 50

The final labels were manually reviewed using the frozen taxonomy and primary-intent rule.

For the human-agreement check, 30 examples were independently labeled without viewing the reference labels.

**Result: 70% exact agreement**

This provides evidence that the taxonomy is reasonably understandable, while also showing that ambiguous support messages remain a significant source of error.

---

## 9. Reproducibility

### Requirements

- Python 3.10+
- OpenAI API key
- Customer Support on Twitter dataset

### Setup

```bash
git clone <repository-url>
cd hiver_sde_assignment

python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Start Jupyter:

```bash
jupyter notebook
```

Run the notebooks in order:

1. `01_dataset_audit.ipynb` — dataset analysis and AmericanAir selection
2. `02_baseline.ipynb` — classification baselines
3. `03_evaluation.ipynb` — final evaluation, reply-quality evaluation, human agreement, and failure analysis

The main implementation is:

```text
src/agent.py
```

The raw dataset is excluded from Git because of its size. The processed AmericanAir customer-response pairs and evaluation artifacts are included in the repository.

---

## 10. Repository Structure

```text
hiver_sde_assignment/
├── data/
│   ├── americanair_intent_discovery.csv
│   └── processed/
│       └── americanair_customer_pairs.csv
│
├── evaluation/
│   ├── golden_set_annotation.csv
│   ├── baseline_results.csv
│   ├── final_llm_results.csv
│   ├── evaluation_predictions.csv
│   ├── confusion_pairs.csv
│   ├── reply_quality_judge.csv
│   ├── human_agreement_judge.csv
│   ├── human_reply_quality_10.csv
│   ├── human_agreement_independent.csv
│   ├── human_agreement_results.csv
│   ├── escalation_results.csv
│   └── final_summary.csv
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_evaluation.ipynb
│
├── src/
│   └── agent.py
│
├── decision_log.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 11. Decision Log

`decision_log.md` contains the major project decisions covering:

- Brand selection
- Dataset direction
- Taxonomy design
- Primary-intent rule
- Golden Set construction
- Sampling strategy
- Classification baselines
- LLM classification
- Historical retrieval
- Example selection
- Response generation
- Escalation policy
- Evaluation methodology
- Failure analysis

---

## 12. Limitations

The system has several important limitations:

- Historical Twitter responses are not guaranteed to represent current American Airlines policies.
- The agent cannot perform real-time account, booking, baggage, or refund actions.
- Follow-up messages are difficult to classify without conversation history.
- TF-IDF retrieval can miss semantically similar messages with different wording.
- LLM-generated responses are only as useful as the historical evidence retrieved.
- Escalation is currently policy-based rather than learned from historical escalation outcomes.
- The reply-quality judge uses a small evaluation sample and should be treated as directional evidence rather than a definitive quality score.

---

## 13. Example

### Input

```text
My flight was cancelled. What should I do?
```

### Agent Output

```text
Intent: flight_disruption

Reply:
We're sorry your flight was cancelled. If you need help rebooking, please contact us and we can take a closer look at your reservation. If you already have your record locator, send it to us in a DM and we'll help from there.

Decision: escalate

Reason:
flight_disruption may require case-specific handling.
```