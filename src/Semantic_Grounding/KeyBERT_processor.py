from typing import Any, Dict
from keybert import KeyBERT

# Initialize globally to keep the model in memory.
# all-MiniLM-L6-v2 is highly optimized for English sentence embeddings.
kw_model = KeyBERT("all-MiniLM-L6-v2")


def extract_and_ground(cao: Dict[str, Any], text: str) -> Dict[str, Any]:
    """
    Extracts semantic keywords from the query to act as 'Ground Truth' anchors.
    These anchors will be used later to prune and validate the AMR graph.
    """
    print(f"[KeyBERT] Extracting anchors for text: {text[:80]}...")

    try:
        # Extract keywords with diversity (MMR) and n-gram range 1-3 for full names/concepts
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            use_mmr=True,
            diversity=0.6,
        )

        # Format into the GroundingAnchor structure
        anchors = []
        for item in keywords:
            term = item[0]
            score_val = item[1]
            score = float(score_val) if isinstance(score_val, (int, float)) else 0.0
            anchors.append({"term": term, "score": round(score, 4), "type": "untyped"})

        # Ensure state layer exists and inject anchors
        cao.setdefault("state", {})
        cao["state"]["grounding_anchors"] = anchors

        cao.setdefault("meta", {})
        cao["meta"]["keybert_ok"] = True

        print(
            f"[KeyBERT] Extracted {len(anchors)} anchors: {[a['term'] for a in anchors]}"
        )

    except Exception as e:
        print(f"[KeyBERT] Extraction failed: {e}")
        cao.setdefault("state", {})
        cao["state"]["grounding_anchors"] = []
        cao.setdefault("meta", {})
        cao["meta"]["keybert_ok"] = False
        cao["meta"]["keybert_error"] = str(e)

    return cao
