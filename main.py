import uuid

import streamlit as st

from src.Vagueness_Judge.runtime import default_clarification_state
from src.lingo.pipeline import run_main_pipeline


def _init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "config" not in st.session_state:
        st.session_state.config = {
            "kb_sources": ["default"],
            "skills": ["core"],
            "session_id": st.session_state.session_id,
        }
    if "clarification_state" not in st.session_state:
        st.session_state.clarification_state = default_clarification_state()


def main() -> None:
    st.set_page_config(
        page_title="Interfaz de tesis - Lingo",
        page_icon="💬",
        layout="centered",
    )

    _init_session_state()

    st.title("Interfaz de interacción de tesis")
    st.caption(
        "Prototipo minimalista para interactuar con el motor Lingo "
        "y sus bases de información."
    )

    with st.container():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_query = st.chat_input("Escribe tu consulta…")

    if user_query:
        st.session_state.messages.append(
            {"role": "user", "content": user_query}
        )
        with st.chat_message("user"):
            st.markdown(user_query)

        response = run_main_pipeline(
            user_text=user_query,
            history=st.session_state.messages,
            config=st.session_state.config,
            clarification_state=st.session_state.clarification_state,
        )
        st.session_state.clarification_state = response.get(
            "clarification_state",
            default_clarification_state(),
        )

        assistant_content = response.get("content", "")
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_content}
        )

        with st.chat_message("assistant"):
            st.markdown(assistant_content)

            meta = response.get("meta", {})
            vagueness_raw = meta.get("vagueness_raw")
            if vagueness_raw:
                with st.expander("🔍 JDV Debug"):
                    st.code(vagueness_raw, language="json", line_numbers=True)


if __name__ == "__main__":
    main()
