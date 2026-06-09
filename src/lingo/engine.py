"""
Engine Lingo — Real implementation.

Receives the fully-populated CognitiveAnalysisObject (CAO) from the pipeline
and generates a natural-language response by interpreting all CAO layers:
  - IntentLayer  → what the user wants (intents, OOS)
  - StateLayer   → facts and beliefs (refined triples, grounding anchors)
  - NestingLayer → dependency graph (future)
  - MetaReasoningLayer → continuity pointers (future)

Designed to be extended with skill selectors and knowledge-base accessors.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class Message(TypedDict):
    role: str  # "user" | "assistant" | "system"
    content: str


def process_query(
    query: str,
    history: List[Message] | None = None,
    config: Dict[str, Any] | None = None,
    cao: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Engine Lingo — core response generator.

    Consumes the CAO to build a structured natural-language reply.
    Falls back to the stub behaviour when *cao* is not provided.
    """
    if history is None:
        history = []
    if config is None:
        config = {}
    if cao is None:
        return _stub_response(query, history, config)

    # ── extract layers ────────────────────────────────────────────────
    intent_layer: dict = cao.get("intent", {})
    nesting_layer: dict = cao.get("nesting", {})
    meta_layer: dict = cao.get("meta", {})
    state_layer: dict = cao.get("state", {})

    oos_status: str = intent_layer.get("oos_ind_status", "UNKNOWN")
    selected_intents: list = intent_layer.get("selected_intents", [])
    k: int = intent_layer.get("k", 0)

    original_query: str = meta_layer.get("original_query", query)
    completed_query: str = meta_layer.get("completed_query", query)
    summary: str = meta_layer.get("summary", "")
    summary_thought: str = meta_layer.get("summary_thought", "")

    triples: list = state_layer.get("triples", [])
    anchors: list = state_layer.get("grounding_anchors", [])

    amr_ok: bool = meta_layer.get("amr_ok", False)
    midlm_ok: bool = meta_layer.get("midlm_ok", False)
    keybert_ok: bool = meta_layer.get("keybert_ok", False)
    logic_ok: bool = meta_layer.get("logic_transformer_ok", False)

    amr_graph_str: str = meta_layer.get("amr_graph", "")

    # ── build response parts ──────────────────────────────────────────
    parts: list[str] = []

    # 1. Welcome / acknowledgement
    parts.append(f"**Consulta resuelta:** {completed_query}")

    # 2. Intent summary
    if midlm_ok and k > 0 and selected_intents:
        intents_str = ", ".join(f"`{i}`" for i in selected_intents)
        parts.append(
            f"- **Intenciones detectadas ({k}):** {intents_str}"
        )
    elif midlm_ok and oos_status == "OOS":
        parts.append(
            "- **Intenciones:** La consulta no corresponde a ningun "
            "dominio conocido (OOS). Las intenciones detectadas se "
            "descartan."
        )
    else:
        parts.append(
            "- **Intenciones:** No se detectaron intenciones "
            "CLINC150 conocidas."
        )

    # 3. OOS gate
    if oos_status == "IND":
        parts.append(
            "- **Gate IND/OOS:** La consulta esta dentro del dominio "
            "conocido (IND)."
        )
    elif oos_status == "OOS":
        parts.append(
            "- **Gate IND/OOS:** La consulta esta fuera del dominio "
            "conocido (OOS)."
        )
    else:
        parts.append(
            "- **Gate IND/OOS:** No se pudo determinar el dominio "
            "(UNKNOWN)."
        )

    # 4. AMR / semantic structure
    if amr_ok:
        n_triples_raw = len(state_layer.get("unrefined_triples", []))
        parts.append(
            f"- **Estructura semantica (AMR):** Se extrajeron "
            f"{n_triples_raw} tripletes no refinados del grafo."
        )
    else:
        parts.append(
            "- **Estructura semantica (AMR):** No disponible "
            "(servidor no accesible o error)."
        )

    # 5. KeyBERT grounding
    if keybert_ok and anchors:
        top_anchors = [
            f'"{a["term"]}" (score: {a["score"]:.3f})'
            for a in anchors[:5]
        ]
        parts.append(
            f"- **Anclajes semanticos:** {', '.join(top_anchors)}"
        )
    elif keybert_ok:
        parts.append(
            "- **Anclajes semanticos:** Extraidos, pero ninguno "
            "supero el umbral de confianza."
        )
    else:
        err = meta_layer.get("keybert_error", "desconocido")
        parts.append(
            f"- **Anclajes semanticos:** Error en la extraccion "
            f"({err})."
        )

    # 6. Logic Transformer / refined triples
    if logic_ok and triples:
        parts.append(
            f"- **Tripletas refinadas (Logica):** {len(triples)} "
            f"tripletas listas para razonamiento simbolico."
        )
        # Show first few triples
        for t in triples[:4]:
            parts.append(
                f"  · `({t.get('subject', '?')}, "
                f"{t.get('predicate', '?')}, "
                f"{t.get('object', '?')})` "
                f"[{t.get('tarski_type', '?')}]"
            )
    elif logic_ok:
        parts.append(
            "- **Tripletas refinadas:** No se generaron tripletas "
            "(posiblemente por falta de anclajes)."
        )
    else:
        parts.append(
            "- **Tripletas refinadas:** El transformador logico no "
            "se ejecuto correctamente."
        )

    # 7. Summary thought (JDV reasoning)
    if summary_thought:
        parts.append(
            f"- **Razonamiento del Juez de Vaguedad:**\n"
            f"  {summary_thought}"
        )

    # 8. Nesting (placeholder)
    nesting_graph = nesting_layer.get("nesting_graph", [])
    if nesting_graph:
        parts.append(
            f"- **Intenciones anidadas:** {len(nesting_graph)} "
            f"relaciones de dependencia detectadas."
        )

    # ── assemble ──────────────────────────────────────────────────────
    body = "\n\n".join(parts)

    footer = (
        "\n\n---\n"
        "*Este analisis fue generado por el motor Lingo (NS-CII) "
        "a partir del Cognitive Analysis Object (CAO).*"
    )

    answer = body + footer

    return {
        "content": answer,
        "meta": {
            "engine": "lingo_real",
            "used_layers": {
                "intent": True,
                "state": True,
                "nesting": bool(nesting_graph),
                "meta_reasoning": False,
            },
            "pipeline_status": {
                "midlm_ok": midlm_ok,
                "amr_ok": amr_ok,
                "keybert_ok": keybert_ok,
                "logic_ok": logic_ok,
            },
            "intent_count": k,
            "triple_count": len(triples),
            "anchor_count": len(anchors),
        },
    }


# ── fallback stub (kept for backward compatibility) ───────────────────


def _stub_response(
    query: str,
    history: List[Message],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Original stub response — used when no CAO is supplied."""
    simulated_answer = (
        "Esta es una respuesta simulada del motor Lingo.\n\n"
        "Resumen de la consulta actual:\n"
        f"- Query: {query}\n"
        f"- Mensajes previos en la conversacion: {len(history)}\n"
        "\n"
        "Cuando el motor real este implementado, aqui se invocara el "
        "pipeline de deteccion de intencion, seleccion de skills y acceso "
        "a las bases de informacion configuradas."
    )
    return {
        "content": simulated_answer,
        "meta": {
            "engine": "lingo_stub",
            "used_config": config,
            "history_length": len(history),
        },
    }
