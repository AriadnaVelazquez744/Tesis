from __future__ import annotations

import json

from .cao_types import CognitiveAnalysisObject


def format_cao_as_markdown(cao: CognitiveAnalysisObject) -> str:
    """
    Formats the CAO as a JSON code block so the current Streamlit UI can display it.
    """

    payload = json.dumps(cao, ensure_ascii=False, indent=2)
    return f"```json\n{payload}\n```"

