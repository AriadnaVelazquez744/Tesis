from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class Message(TypedDict):
    role: str  # "user" | "assistant" | "system"
    content: str


def process_query(
    query: str,
    history: List[Message] | None = None,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Stub del motor Lingo.

    Recibe la consulta del usuario junto con el historial de conversación y
    una configuración opcional (p.ej. bases de información y skills activas),
    y devuelve una respuesta simulada lista para ser reemplazada por el
    pipeline real.
    """
    if history is None:
        history = []

    if config is None:
        config = {}

    simulated_answer = (
        "Esta es una respuesta simulada del motor Lingo.\n\n"
        "Resumen de la consulta actual:\n"
        f"- Query: {query}\n"
        f"- Mensajes previos en la conversación: {len(history)}\n"
        "\n"
        "Cuando el motor real esté implementado, aquí se invocará el "
        "pipeline de detección de intención, selección de skills y acceso "
        "a las bases de información configuradas."
    )

    return {
        "content": simulated_answer,
        "meta": {
            "used_config": config,
            "history_length": len(history),
            "engine": "lingo_stub",
        },
    }

