from __future__ import annotations

from typing import Any, Dict, List, Literal, NotRequired, TypedDict

IntentOOSStatus = Literal["IND", "OOS", "UNKNOWN"]


class IntentLayer(TypedDict, total=False):
    """
    - Stores the MIDLM-selected multi-intent labels (K + top-K)
    - Stores the OpenMax IND/OOS decision
    """
    oos_ind_status: IntentOOSStatus
    k: int
    selected_intents: List[str]
    confidence: NotRequired[float]


class NestingLayer(TypedDict, total=False):
    # Placeholder for later: a dependency graph / nesting structure.
    nesting_graph: NotRequired[List[Dict[str, Any]]]


class MetaReasoningLayer(TypedDict, total=False):
    # Placeholder for later: continuity pointers / placeholders.
    association_triggers: NotRequired[List[Dict[str, Any]]]


class GroundingAnchor(TypedDict):
    term: str
    score: float
    type: NotRequired[Literal["entity", "concept", "untyped"]]


class StateLayer(TypedDict, total=False):
    # Placeholder for later: logic triples (S, R, O). What VSA will eventually use
    triples: NotRequired[List[Dict[str, Any]]] 
    # What AMR outputs initially (saved for KeyBERT/Logic Transformer)
    unrefined_triples: NotRequired[List[Dict[str, Any]]]
    # Verified anchors from KeyBERT
    grounding_anchors: NotRequired[List[GroundingAnchor]]


class CognitiveAnalysisObject(TypedDict):
    intent: IntentLayer
    nesting: NestingLayer
    meta_reasoning: MetaReasoningLayer
    state: StateLayer
    meta: Dict[str, Any]