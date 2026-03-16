import streamlit as st

from src.lingo.engine import process_query


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "config" not in st.session_state:
        st.session_state.config = {
            "kb_sources": ["default"],
            "skills": ["core"],
        }


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

        response = process_query(
            query=user_query,
            history=st.session_state.messages,
            config=st.session_state.config,
        )

        assistant_content = response.get("content", "")
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_content}
        )

        with st.chat_message("assistant"):
            st.markdown(assistant_content)


if __name__ == "__main__":
    main()

