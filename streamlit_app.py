import streamlit as st
from audio_recorder_streamlit import audio_recorder
import uuid

from utils import tts, stt
from localization.locales import LOCALES


chat_history = []

language = st.radio(
    "Мова/Language",
    options=["uk", "en"],
    index=0,
    horizontal=True,
)


st.session_state["language"] = language

INITIAL_MESSAGE = [
    {  # Initial message from the assistant
        "id": uuid.uuid4().hex,
        "role": "assistant",
        "content": LOCALES[language]["hello_assistant"],
    },
]


# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state["messages"] = []

if "history" not in st.session_state:
    st.session_state["history"] = []

if "language" not in st.session_state:
    st.session_state["language"] = language


def build_sidebar():
    with open(f"localization/sidebar_{language}.md", "r") as sidebar_file:
        sidebar_content = sidebar_file.read()

    st.sidebar.markdown(sidebar_content)

    # Add a reset button
    if st.sidebar.button(LOCALES[language]["reset_chat"]):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.session_state["messages"] = INITIAL_MESSAGE
        st.session_state["history"] = []


def build_chat():
    # Prompt for user input and save
    if prompt := st.chat_input():
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "id": uuid.uuid4().hex}
        )

    if uploaded_file := st.file_uploader(
        LOCALES[language]["choose_audiofile"], type="wav"
    ):
        st.session_state.messages.append(
            {
                "role": "user",
                "content": "*audio file*",
                "audio": uploaded_file.getvalue(),
                "id": uuid.uuid4().hex,
            }
        )

    if audio_bytes := audio_recorder(LOCALES[language]["record_audio"]):
        text = stt(audio_bytes, st.session_state["language"])
        st.session_state.messages.append(
            {
                "role": "user",
                "content": text,
                "audio": audio_bytes,
                "id": uuid.uuid4().hex,
            }
        )

    for message in st.session_state.messages:
        msg_component = st.chat_message(message["role"])
        msg_component.write(message["content"])
        if "audio" in message:
            msg_component.audio(message["audio"], format="audio/wav")
        else:
            btn = msg_component.button(
                LOCALES[language]["synthesize"], key=message["id"]
            )
            if btn:
                message["audio"] = tts(message["content"], st.session_state["language"])
                st.experimental_rerun()


build_sidebar()
if "messages" not in st.session_state.keys() or len(st.session_state["messages"]) == 0:
    if st.button(LOCALES[language]["start_chat"]):
        st.session_state["messages"] = INITIAL_MESSAGE
        st.experimental_rerun()
else:
    build_chat()
