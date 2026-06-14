# Query Expansion Prompt — Single Record

You are an evaluation dataset generator. Your task is to take a seed query and produce ONE complete evaluation record (not 10). Each record must be a fully annotated test case for a semantic parsing system.

You must respond with valid JSON only — no explanations, no markdown, no code fences.

## Input seed

```json
{seed_json}
```

## Topic & Domain Abbreviations

Use these abbreviations for the `id` field:

| Abbreviation | Full ID |
|---|---|
| NLG | natural_logic |
| MATH | mathematical_calculation |
| INFO | information_search |
| PSYCH | human_psychology |
| MIX | mixed |
| BANK | banking_finance |
| TRAV | travel |
| HOME | home_auto |
| FOOD | food_dining |
| HLTH | healthcare |
| COMM | communication |
| UTIL | info_utilities |
| ENTR | entertainment |

## Relation Type Taxonomy

### Formal types (5)

**SYMMETRIC**: R(x,y) ⇔ R(y,x) — direction doesn't change truth.
Examples: equals, is_sibling_of, is_married_to, is_the_same_as.

**INVERSE**: R(x,y) ⇒ R⁻¹(y,x) with R ≠ R⁻¹ — has a distinct inverse.
Examples: is_parent_of ↔ is_child_of, wrote ↔ was_written_by.

**ASYMMETRIC**: R(x,y) ⇏ R(y,x) — fixed direction, no valid inverse.
Examples: causes, loves, owns, drives. **This is the default.**

**ORDERING**: R(x,y) ⇒ ¬R(y,x), transitive — defines an order.
Examples: greater_than, before, after, older_than.

**FUNCTIONAL**: ∀x ∃!y : R(x,y) — one-to-one mapping with reverse lookup.
Examples: is_capital_of, has_ssn, is_located_at.

### Operational modes (3) — how the Logic Transformer handles each type

| Operational mode | Types included | Logic Transformer action |
|---|---|---|
| **direct** | ASYMMETRIC, ORDERING, FUNCTIONAL | Keep triple as-is. AMR subject is the real subject. |
| **inverse_normalize** | INVERSE | Swap S↔O and replace R with R⁻¹ to obtain the canonical active-voice direction. |
| **bidirectional** | SYMMETRIC | Emit both (S,R,O) and (O,R,S) so the VSA creates a non-directional binding. |

## CLINC150 Intent List

{intent_list_formatted}

Use "oos": true with expected_intents=[] for OOS variants.

## This call's specification

{specification}

## ID for this record

{assigned_id}

## Output record schema

Each variant must be a complete record:

```json
{
  "id": "str — {TOPIC_ABBREV}-{DOMAIN_ABBREV}-{NNN}",
  "query": "str — the actual user utterance",
  "summary": "str — a one-sentence clarification of what the user wants",
  "topic": "str — one of: natural_logic, mathematical_calculation, information_search, human_psychology, mixed",
  "domain_cluster": "str — one of the 8 domains",
  "vague": "bool — always false for these single-turn evaluation records",
  "k": "int — 1 to 5",
  "oos": "bool",
  "expected_intents": "list[str] — from CLINC150; [] if oos=true",
  "expected_oos_label": "str|null — descriptive label for OOS category, null for IND",
  "expected_grounding_anchors": "list[{'term': str, 'type': 'entity'|'concept'}]",
  "expected_unrefined_triples": "list[{'triple': [S,R,O], 'relation_type': str}]",
  "expected_refined_triples": "list[{'triple': [S,R,O], 'relation_type': str}]",
  "relation_types": "dict[str, {'formal_type': str, 'inferred_by': str, 'operational_mode': str}]",
  "expected_nesting_graph": "list[{'parent': str, 'child': str, 'relation': str}] — REQUIRED non-empty if k>1; empty [] if k=1",
  "expected_association_triggers": "list[str] — empty [] for single-turn queries",
  "meta": {
    "reversal_pair_id": "str|null",
    "reversal_type": "str|null",
    "is_reversed_version": "bool",
    "equivalent_to": "str|null"
  }
}
```

## Field-level guidelines

### Topic and domain
- Use the `topic` and `domain_cluster` values from the seed record — do not change them

### expected_intents
- Must be from the CLINC150 list above
- Length must equal k
- Sort alphabetically

### expected_oos_label
- Must be null when oos=false
- When oos=true, provide a short descriptive label (2-5 words) explaining why the query is out-of-scope
- Examples: "hypothetical_time_travel", "third_party_app_question", "non_english_mix"
- Labels should be snake_case, descriptive of the OOS category

### expected_grounding_anchors
- Extract named entities (people, places, dates, numbers, specific terms) as "entity"
- Extract key concepts (abstract ideas, qualities, categories) as "concept"
- Include all critical details a user might expect the system to capture
- 2-6 anchors per query typically

### expected_unrefined_triples
- Capture the raw Subject-Relation-Object as they naturally appear in text
- Relations should be verbs or prepositions from the original text
- Each triple must have a relation_type from the 5-type taxonomy
- 1-5 triples per query typically

### expected_refined_triples
- Refined triples use canonical forms:
  - Normalize pronouns: "I", "my", "me" → "user"
  - Normalize entities: "my account" → "user_account", "that restaurant" → "restaurant"
  - Use consistent, normalized predicate names (lemma form): "cause" not "causes"/"caused"
  - Apply canonical directions: for INVERSE relations, use the active voice form

### relation_types
- A dict mapping each relation string to an object with formal_type, inferred_by, operational_mode
- Every relation in unrefined_triples AND refined_triples must have an entry
- inferred_by must be one of: amr_arg0_arg1, amr_passive, preposition_directional, lexical_mutuality, lexical_equality, lexical_possession, lexical_kinship_symmetric, lexical_kinship_asymmetric, lexical_causal, lexical_location, lexical_comparative, unique_identification, fallback

### expected_nesting_graph (REQUIRED when k > 1)
- REQUIRED when k > 1: you MUST populate this with at least one entry
- When k=1: MUST be empty []
- Each entry represents a dependency between two intents
- relation can be: "parallel", "prerequisite", "subgoal", "alternative"

### expected_association_triggers
- Only populate if the query contains continuity markers
- Leave empty [] for single-turn queries

### Reversal pairs
- When generating a reversal pair variant:
  - Both variants must have the same reversal_pair_id
  - Variant A (first): is_reversed_version=false
  - Variant B (second): is_reversed_version=true, equivalent_to=Variant A's ID
  - reversal_type must match: "inverse_voice", "inverse_relation", "clause_order", or "symmetric_direction"
  - Both variants must have the same triples after refinement (isomorphic)

## Output format

Respond with a single JSON object (NOT wrapped in a "records" array):

```json
{
  "id": "{assigned_id}",
  ...
}
```
