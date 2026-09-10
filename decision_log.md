# Decision Log

## 1. Brand selection
Selected AmericanAir because it has sufficient customer-support volume while keeping the project scope manageable.

## 2. Dataset direction
Used inbound customer tweets and matched them to AmericanAir outbound responses using the dataset's response pointers.

## 3. Intent taxonomy
Created a 15-intent taxonomy based on recurring support themes in the AmericanAir data.

## 4. Primary-intent rule
When a message contains multiple issues, classify it by the customer's primary actionable request.

## 5. Golden Set size
Created a 200-example Golden Set for final held-out evaluation.

## 6. Golden Set sampling
Used stratified sampling by message length to avoid evaluating only short or long messages.

## 7. Baseline 1
Implemented TF-IDF with Logistic Regression as a simple text-classification baseline.

## 8. Baseline 2
Implemented TF-IDF with Linear SVM because it is a strong lightweight baseline for sparse text classification.

## 9. LLM classifier
Used GPT-5.4-mini for intent classification after comparing its expected quality/cost trade-off with smaller models.

## 10. Retrieval approach
Used TF-IDF similarity over historical AmericanAir customer messages to retrieve similar support cases.

## 11. Example selection
Added an LLM selection step to choose the most relevant historical examples from the retrieved candidates.

## 12. Reply generation
Generated replies using the selected historical customer-response examples as grounding evidence.

## 13. Escalation policy
Escalated disruption, baggage, fee, refund, airport/staff, and unclear cases because they can require case-specific handling.

## 14. Evaluation
Evaluated intent classification on a held-out 200-example Golden Set and evaluated reply quality separately using an LLM judge.

## 15. Failure analysis
Inspected misclassified examples and identified five recurring failure modes, including context-dependent follow-ups and ambiguous disruption messages.