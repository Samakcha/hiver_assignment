from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


client = OpenAI()


INTENTS = [
    "flight_disruption",
    "baggage",
    "booking_change",
    "seat",
    "loyalty_upgrade",
    "fees_charges",
    "refund_compensation",
    "flight_information",
    "airport_gate_staff",
    "onboard_aircraft",
    "contact_support",
    "check_in",
    "follow_up",
    "non_support",
    "other_unclear",
]


ESCALATION_INTENTS = {
    "refund_compensation",
    "fees_charges",
    "flight_disruption",
    "baggage",
    "airport_gate_staff",
    "other_unclear",
}


def classify_intent(customer_message):
    prompt = f"""
Classify the customer message into exactly one of these intents:

{INTENTS}

Rules:
- flight_disruption: active cancellation, delay, missed connection, or other disruption
- flight_information: asking for flight/status information without an active problem
- baggage: baggage problems or baggage questions
- booking_change: changing, cancelling, or modifying a booking
- seat: seat assignment or seat selection
- loyalty_upgrade: upgrades, loyalty status, miles, or priority benefits
- fees_charges: fees, charges, or unexpected costs
- refund_compensation: requesting a refund, voucher, or compensation
- airport_gate_staff: airport, gate, or staff issues
- onboard_aircraft: issues with the aircraft or onboard experience
- contact_support: trying to reach/contact customer support
- check_in: check-in problems
- follow_up: following up on an existing support case
- non_support: not a customer support request
- other_unclear: unclear or insufficient information

Customer message:
{customer_message}

Return only the intent name.
"""

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt
    )

    intent = response.output_text.strip()

    if intent not in INTENTS:
        return "other_unclear"

    return intent


def retrieve_similar_messages(
    customer_message,
    historical_pairs,
    retriever_vectorizer,
    historical_vectors,
    k=3
):
    query_vector = retriever_vectorizer.transform([customer_message])
    scores = cosine_similarity(
        query_vector,
        historical_vectors
    ).flatten()

    top_indices = scores.argsort()[-k:][::-1]
    top_indices = [i for i in top_indices if scores[i] >= 0.20]
    results = historical_pairs.iloc[top_indices].copy()
    results["similarity"] = scores[top_indices]

    # Reset index so LLM-selected example numbers match DataFrame rows
    results = results.reset_index(drop=True)

    return results


def generate_reply(customer_message, context):
    prompt = f"""
You are an American Airlines customer support agent.

Write a concise, helpful response to the customer.

Use the historical examples as evidence for how American Airlines
handled similar cases.

Synthesize the examples rather than copying one response.
Only state information supported by the historical examples.

If the examples do not provide enough information to resolve the issue,
recommend contacting American Airlines support.

Customer message:
{customer_message}

Historical examples:
{context}

Return only the customer-facing reply.
"""

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt
    )

    return response.output_text.strip()


def decide_escalation(intent):
    if intent in ESCALATION_INTENTS:
        return {
            "decision": "escalate",
            "reason": f"{intent} may require case-specific handling."
        }

    return {
        "decision": "auto_handle",
        "reason": "The request can be answered using historical support guidance."
    }

def select_relevant_examples(customer_message, intent, candidates, k=3):
    examples = ""

    for i, (_, row) in enumerate(candidates.iterrows()):
        examples += f"""
Example {i}:
Customer: {row['customer_message']}
AmericanAir response: {row['brand_response']}
"""

    prompt = f"""
You are selecting historical customer-support examples for an American Airlines support agent.

Customer message:
{customer_message}

Predicted intent:
{intent}

From the examples below, select the {k} examples that are most relevant to the customer's issue.

Prefer examples with the same intent and similar problem.
Do not select examples just because they share individual words.

{examples}

Return only 3 numbers between 0 and 9, separated by commas.
"""

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt
    )

    return response.output_text.strip()


def run_agent(
    customer_message,
    historical_pairs,
    retriever_vectorizer,
    historical_vectors
):
    # 1. Classify intent
    intent = classify_intent(customer_message)

    # 2. Retrieve candidate examples
    candidates = retrieve_similar_messages(
        customer_message,
        historical_pairs,
        retriever_vectorizer,
        historical_vectors,
        k=10
    )

    # 3. Select the most relevant examples
    selected = select_relevant_examples(
        customer_message,
        intent,
        candidates,
        k=3
    )

    # 4. Convert selected IDs into rows
    selected_ids = [
        int(x.strip())
        for x in selected.split(",")
        if x.strip().isdigit()
    ]

    selected_ids = [
        i for i in selected_ids
        if 0 <= i < len(candidates)
    ]

    if len(selected_ids) < 3:
        selected_ids = list(range(min(3, len(candidates))))

    selected_rows = candidates.iloc[selected_ids[:3]]

    # 5. Build context
    context = "\n\n".join(
        f"Customer: {row['customer_message']}\n"
        f"AmericanAir response: {row['brand_response']}"
        for _, row in selected_rows.iterrows()
    )

    # 6. Generate reply
    reply = generate_reply(
        customer_message,
        context
    )

    # 7. Decide escalation
    decision = decide_escalation(intent)

    return {
        "intent": intent,
        "reply": reply,
        "decision": decision["decision"],
        "reason": decision["reason"]
    }