import streamlit as st
from audio_recorder_streamlit import audio_recorder
import uuid
from core.agent import init_agent, init_chromadb, init_content_embeddings, init_qna_retrieval
from langchain.memory import RedisChatMessageHistory, StreamlitChatMessageHistory
from core.llm_wrapers import LLMChatHandler

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
st.caption(f"Session: {st.session_state.get('session_id', '')}")

INITIAL_MESSAGE = [
    {  # Initial message from the assistant
        "id": uuid.uuid4().hex,
        "role": "assistant",
        "content": LOCALES[language]["hello_assistant"],
    },
]

@st.cache_resource
def init_cache():
    cached_embedder, chroma_emb_client = init_chromadb()
    context_retriever = init_content_embeddings(cached_embedder, chroma_emb_client)
    cached_conversational_rqa, llm = init_qna_retrieval(context_retriever, cached_embedder, chroma_emb_client)
    agent = init_agent(cached_conversational_rqa, llm)
    return agent

AGENT = init_cache()

# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state["messages"] = []

if "history" not in st.session_state:
    st.session_state["history"] = []

if "language" not in st.session_state:
    st.session_state["language"] = language

def get_llm_client(session_id):                
    chat_history = RedisChatMessageHistory(session_id=session_id, url=f"redis://localhost:6379/2")

    chat_handler = LLMChatHandler(AGENT, chat_history)
    return chat_handler

def append_message(text, audio=None):
    msg_obj = {"role": "user", "content": text, "id": uuid.uuid4().hex}
    if audio:
        msg_obj["audio"] = audio

    st.session_state.messages.append(msg_obj)
    
    response = get_llm_client(st.session_state["session_id"]).send_message(text)

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "id": uuid.uuid4().hex}
    )

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
        chat_history.clear()
        st.experimental_rerun()


def build_chat():
    if prompt := st.chat_input():
        append_message(prompt)

    # if uploaded_file := st.file_uploader(
    #     LOCALES[language]["choose_audiofile"], type="wav"
    # ):
    #     st.session_state.messages.append(
    #         {
    #             "role": "user",
    #             "content": "*audio file*",
    #             "audio": uploaded_file.getvalue(),
    #             "id": uuid.uuid4().hex,
    #         }
    #     )

    if audio_bytes := audio_recorder(LOCALES[language]["record_audio"]):
        text = stt(audio_bytes, st.session_state["language"])
        append_message(text, audio_bytes)

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
        st.session_state["session_id"] = uuid.uuid4().hex
        st.experimental_rerun()
else:
    build_chat()
