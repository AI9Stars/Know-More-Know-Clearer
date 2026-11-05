"""Every prompt used by the pipeline, in one place.

    ``SAMPLING_PROMPT``        behavioral profiling
    ``REGION_ASSIGNMENT``      meta-cognitive region assignment
    ``SEARCH_QUERY``           cognitive tagging and query generation
    ``QA_GENERATION[...]``     region-specific QA generation

This module is the only place prompt text appears. Nothing here imports from
the stages, so prompts can be read, diffed, or replaced without touching
pipeline logic.
"""

from __future__ import annotations

from .regions import Region

# --------------------------------------------------------------------------
# Stage 1 -- behavioral profiling
#
# Deliberately plain, so the profile reflects the model's inherent knowledge
# rather than its instruction-following.
# --------------------------------------------------------------------------

SAMPLING_PROMPT = "Please provide the user with the answer directly. Question: {question} \nAnswer:"

#: Responses drawn per query to build the behavioral profile.
DEFAULT_K = 16


# --------------------------------------------------------------------------
# Stage 2 -- meta-cognitive region assignment
# --------------------------------------------------------------------------

#: Reference accuracy boundaries quoted inside the prompt. Also used by the
#: deterministic fallback when the model output is unusable.
ACCURACY_HIGH = 0.80
ACCURACY_LOW = 0.20

#: Allowed deviation from the boundaries when answer patterns justify it.
TOLERANCE = 0.10

REGION_ASSIGNMENT = """
Role. You are an expert in diagnosing a language model's knowledge status. Your goal is to assign a precise and unambiguous region label based on behavioral evidence.

You are given:
    - A question and its gold answer
    - Multiple sampled responses with accuracy and uncertainty
    - Aggregated statistics: mean accuracy and mean uncertainty

### Task Overview
1. Analyze Behavioral Patterns
    - Consistency: Are answers semantically stable across samples, or randomly scattered?
    - Confidence vs. Accuracy: Does low uncertainty align with correct answers, or indicate confident but incorrect responses?

2. Assign Knowledge Region
Follow a performance-oriented principle. Use accuracy and uncertainty as reference signals, but let the observed answer pattern determine the final decision.

    - Knowledge Mastered
        - Answers are semantically stable with consistently low uncertainty
        - Reference: Mean accuracy >= 0.80
        - A tolerance of 0.10 is allowed if reasoning remains coherent and consistently correct

    - Knowledge Missing
        - Answers appear as scattered guesses or high-uncertainty responses
        - Reference: Mean accuracy <= 0.20
        - A tolerance of 0.10 is allowed if behavior resembles random guessing

    - Knowledge Confused
        - Answers fluctuate due to ambiguity or incomplete reasoning
        - Reference: Mean accuracy between 0.20 and 0.80
        - Mixed partial success and failure indicates fragmented knowledge

3. Provide Explanation
Write a concise 2-4 sentence rationale describing the model's typical response behavior and the reason for the assigned region.

### Input
Question: {question}
Gold Answer: {gold_answer}
Mean Accuracy: {mean_accuracy}
Mean Uncertainty: {mean_uncertainty}
Samples (one per line, JSON): {samples_json}

### Output Format (JSON only)
{{
    "Knowledge_Status": "Knowledge Mastered | Knowledge Confused | Knowledge Missing",
    "Explanation": "2-4 sentence diagnostic rationale"
}}
""".strip()


# --------------------------------------------------------------------------
# Stage 3 -- cognitive tagging and search-query generation
# --------------------------------------------------------------------------

SEARCH_QUERY = """
Role. You are a meta-cognitive analyst specializing in diagnosing and repairing knowledge states in large language models. Your task is to interpret an upstream diagnosis (mastered, confused, or missing) and produce a single targeted intervention search query.

### Objectives
    1. Extract Cognitive Tags Identify key entities or concepts from the case that reveal what knowledge is absent, unstable, or already stable. These extracted elements are called [Cognitive Tags].

    2. Formulate the Search Query Produce ONE high-precision search query anchored on the extracted [Cognitive Tags] to repair cognitive distortion or expand the stable knowledge boundary.

### Global Instructions
1. Align with Knowledge Status

    - Knowledge Missing Strategy: The model lacks a reliable internal representation and fails consistently. First extract [Cognitive Tags] by identifying the essential prerequisite entities or concepts required to reach the gold answer. Then construct a clear, definition- or concept-oriented search query anchored on these tags to retrieve the missing background knowledge needed for reconstruction.

    - Knowledge Confused Strategy: The model produces inconsistent answers due to fragmented internal representations. First extract [Cognitive Tags] by identifying uncertain slots and the key entities or relations that must be clarified. Then construct a disambiguation-focused search query anchored on these tags to retrieve the missing links needed to stabilize the reasoning chain.

    - Knowledge Mastered Strategy: The model demonstrates stable reasoning supported by a coherent internal representation. First extract [Cognitive Tags] by identifying the core entities or concepts underlying correct behavior. Then construct an expansion-oriented search query anchored on these tags to retrieve closely related information that extends the knowledge boundary while preserving the established structure.

2. Search Query Requirements

    - The query must be retrieval-ready and written as a natural search instruction
    - The query must be grounded in the case and anchored on the extracted [Cognitive Tags]
    - Do not repeat the original question verbatim; reformulate it strategically

3. Output Constraint
You must output exactly one JSON object. No explanation, no markdown, no extra text.

### Input
Question: {question}
Gold Answer: {gold_answer}
Knowledge Status: {knowledge_status}
Diagnosis Explanation: {explanation}

### Output Format (JSON only)
{{
    "Search_Query": "..."
}}
""".strip()


# --------------------------------------------------------------------------
# Stage 5 -- region-specific QA generation
#
# Rule 4 of all three templates forbids reasoning steps, so generated pairs are
# (question, answer) only.
# --------------------------------------------------------------------------

_QA_MASTERED = """
Role. You are generating QA pairs for the Knowledge Mastered region, where the model already demonstrates stable understanding.

### Task
Using the retrieved passages, expand around the same core fact implied by the query. Construct new QA pairs that explore related attributes, comparisons, relations, or broader background knowledge connected to this fact. Each QA pair should contain a new question grounded in the same core fact and a concise final answer (one short sentence, no explanation). Generate 1-2 QA pairs.

### Rules
    1. All questions must revolve around the same core fact implied by the query.
    2. Questions should extend the knowledge boundary in a controlled and relevant direction.
    3. The answer must be concise (at most one short sentence, <= 20 words).
    4. Do not include reasoning steps or explanations anywhere.

### Input
Query: {query}
Retrieved Passages: {retrieved_passages}

### Output Format (JSON only)
{{
    "qa_pairs": [
      {{"question": "...", "answer": "..."}}
    ]
}}
""".strip()

_QA_CONFUSED = """
Role. You are generating QA pairs for the Knowledge Confused region, where the model shows unstable or ambiguous understanding.

### Task
Keep the original query scenario, but use the retrieved passages to repair underspecified or ambiguous parts. Construct QA pairs that clarify entities, relations, time constraints, or reasoning steps while staying close to the core factual target. Each QA pair should include a clearer and more specific question and a concise final answer (one short sentence, no explanation). Prefer generating 1 clarified main question plus 1-3 decomposed sub-questions.

### Rules
    1. Preserve the original task setting, but make implicit details explicit.
    2. Questions should resolve ambiguity or decompose complex reasoning into simpler parts.
    3. The answer must be concise (at most one short sentence, <= 20 words).
    4. Do not include reasoning steps or explanations anywhere.

### Input
Query: {query}
Retrieved Passages: {retrieved_passages}

### Output Format (JSON only)
{{
    "qa_pairs": [
      {{"question": "...", "answer": "..."}}
    ]
}}
""".strip()

_QA_MISSING = """
Role. You are generating training QA pairs for the Knowledge Missing region, where the model lacks foundational knowledge.

### Task
Using the retrieved passages as the primary source, construct QA pairs that target missing definitions or basic factual knowledge related to the query. Each QA pair should consist of a short, clear factual question and a concise final answer (one short sentence, no explanation). Focus on simple, core concepts that can be directly derived from the provided information. Generate 2-4 QA pairs.

### Rules
    1. Every question must be answerable using only the retrieved passages.
    2. Questions should be definition-oriented or basic fact-oriented.
    3. The answer must be concise (at most one short sentence, <= 20 words).
    4. Do not include reasoning steps or explanations anywhere.

### Input
Search Query: {query}
Retrieved Passages: {retrieved_passages}

### Output Format (JSON only)
{{
    "qa_pairs": [
      {{"question": "...", "answer": "..."}}
    ]
}}
""".strip()

QA_GENERATION = {
    Region.MASTERED: _QA_MASTERED,
    Region.CONFUSED: _QA_CONFUSED,
    Region.MISSING: _QA_MISSING,
}

#: How many QA pairs each region's prompt asks for. Used only for logging.
EXPECTED_PAIRS = {
    Region.MASTERED: (1, 2),
    Region.CONFUSED: (2, 4),
    Region.MISSING: (2, 4),
}
