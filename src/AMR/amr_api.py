# Tesis/src/AMR/amr_api.py
import traceback
from typing import Any, Dict, List, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from transition_amr_parser.parse import AMRParser

app = FastAPI(title="AMR Parsing Service (Python 3.8)")

parser = None
model_loaded = False

@app.on_event("startup")
async def startup_event():
    global parser, model_loaded
    print("Loading AMR model... This may take a moment.")
    try:
        parser = AMRParser.from_pretrained("AMR2-structbart-L")
        model_loaded = True
        print("AMR Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load AMR model: {e}")
        traceback.print_exc()
        model_loaded = False

class ParseRequest(BaseModel):
    text: str

@app.get("/health")
def health_check():
    return JSONResponse({
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
    })

def sanitize_value(value: str) -> str:
    """Removes quotes and PropBank sense suffixes (e.g., 'want-01' -> 'want')."""
    clean = value.strip('"')
    # Optional: Remove sense numbers if you want pure concepts for VSA
    if "-" in clean and clean[-2:].isdigit():
        clean = clean.rsplit("-", 1)[0]
    return clean

def get_node_type(node_id: str, concept: str) -> str:
    """Categorizes nodes to assist KeyBERT and VSA layers."""
    concept_lower = concept.lower()
    if concept.startswith('"') and concept.endswith('"'):
        return "literal"
    if concept_lower in ["amr-unknown", "this", "that", "it", "previous"]:
        return "placeholder"
    return "concept"

def extract_initial_triples(nodes: Dict[str, str], edges: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
    """
    Finalized AMR Cleaning Procedure:
    1. Collapses name nodes (op1, op2...).
    2. Normalizes inverse relations (-of).
    3. Sanitizes literals and marks node types.
    4. Identifies placeholders for MetaReasoning.
    """
    triples = []
    name_map = {}
    
    # 1. Collapse Name Nodes
    # Finds nodes linked to :op edges to create full entity strings
    for src, rel, tgt in edges:
        if rel.startswith(":op") and nodes.get(src) == "name":
            # Strip quotes immediately for the name map
            val = nodes.get(tgt, tgt).strip('"')
            name_map.setdefault(src, []).append((rel, val))
    
    # Join parts (op1, op2...) in order
    resolved_names = {}
    for name_node, parts in name_map.items():
        parts.sort(key=lambda x: x[0]) # Ensure op1 comes before op2
        resolved_names[name_node] = " ".join([p[1] for p in parts])

    # 2. Map structural 'name' nodes back to their parents
    # e.g., (p / person :name (n / name :op1 "John")) -> p is now "John"
    parent_entity_map = {}
    for src, rel, tgt in edges:
        if rel == ":name" and tgt in resolved_names:
            parent_entity_map[src] = resolved_names[tgt]

    # 3. Process Edges with Normalization and Sanitization
    for src, rel, tgt in edges:
        # Skip purely structural/internal edges already processed
        if rel.startswith(":op") or rel == ":name":
            continue
            
        predicate = rel.lstrip(':')
        subject_id, object_id = src, tgt

        # A. Inverse Normalization (-of)
        # Flip the direction so Actor -> Action is consistent
        if predicate.endswith("-of"):
            predicate = predicate.replace("-of", "")
            subject_id, object_id = tgt, src

        # B. Resolve Values (Name > Parent Map > Node Concept)
        s_raw = parent_entity_map.get(subject_id) or nodes.get(subject_id, subject_id)
        o_raw = parent_entity_map.get(object_id) or nodes.get(object_id, object_id)

        # C. Sanitize and Type Marking
        s_value = sanitize_value(str(s_raw))
        o_value = sanitize_value(str(o_raw))
        
        # D. Store enriched triple
        triples.append({
            "subject": {
                "id": subject_id,
                "value": s_value,
                "type": get_node_type(subject_id, str(s_raw))
            },
            "predicate": predicate,
            "object": {
                "id": object_id,
                "value": o_value,
                "type": get_node_type(object_id, str(o_raw))
            }
        })
        
    return triples

@app.post("/parse")
def parse_text(request: ParseRequest):
    if not model_loaded or parser is None:
        raise HTTPException(status_code=503, detail="AMR model not loaded")

    try:
        tokens, positions = parser.tokenize(request.text)
        annotations, machines = parser.parse_sentence(tokens)
        amr = machines.get_amr()

        # Extract raw components safely
        penman_str = amr.to_penman() if amr else ""
        nodes = amr.nodes if amr else {}
        edges = amr.edges if amr else []
        root = amr.root if amr else None
        
        raw_alignments = getattr(amr, "alignments", {})
        formatted_alignments = [{"node": k, "token_index": v} for k, v in raw_alignments.items()] if isinstance(raw_alignments, dict) else raw_alignments

        unrefined_triples = extract_initial_triples(nodes, edges)

        return {
            "amr_graph": penman_str,
            "ast": {
                "nodes": nodes,
                "edges": edges,
                "root": root,
                "alignments": formatted_alignments,
                "tokens": tokens
            },
            "unrefined_triples": unrefined_triples
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Parse error: {str(e)}")
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)