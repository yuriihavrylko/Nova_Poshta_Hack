import glob
import hashlib
import inspect
import os

import chromadb
import langchain
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain, RetrievalQA, create_tagging_chain_pydantic
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import RedisChatMessageHistory, ConversationBufferWindowMemory
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.storage import LocalFileStore, RedisStore
from langchain.tools import StructuredTool
from langchain.vectorstores import Chroma
import redis
import spacy

from core.llm_wrapers import *
from core.tool_functions import *
from core.utils import *
import hashlib


REDIS_HOST = "localhost"
CHROMA_HOST = "localhost"
CHROMA_PERSIST_DIRECTORY = "/chroma"

EMBEDDING_MODEL = "text-embedding-ada-002"
KNOWLEDGE_BASE_DIR = "./knowledge_base"

RETRIEVER_COLLECTION_SETTINGS = {
    "info": [{"name": "bm25", "k": 1, "score_threshold": 0.35}, {"name": "semantic", "k": 3, "score_threshold": 0.35}],
    "links": [{"name": "semantic", "k": 1}]
}

CREATE_DATABASE = False

def init_chromadb():
    chroma_emb_client = chromadb.HttpClient(host=CHROMA_HOST, port=8000)
    chroma_emb_client._settings.is_persistent = True
    chroma_emb_client._settings.persist_directory=CHROMA_PERSIST_DIRECTORY

    redis_emb_client = redis.Redis(host=REDIS_HOST, port=6379, db=0)
    redis_emb_store = RedisStore(client=redis_emb_client, namespace=EMBEDDING_MODEL)

    cached_embedder = CachedEmbeddings.from_bytes_store(OpenAIEmbeddings(model=EMBEDDING_MODEL), redis_emb_store, namespace=EMBEDDING_MODEL)

    if CREATE_DATABASE:
        collection_names = create_knowledge_vectordb(KNOWLEDGE_BASE_DIR, cached_embedder)
        print(collection_names)
    return cached_embedder, chroma_emb_client

def init_content_embeddings(cached_embedder, chroma_emb_client):
    spacy_nlp = spacy.load("uk_core_news_sm")

    retrievers = []
    for collection_name, collection_config in RETRIEVER_COLLECTION_SETTINGS.items():
        collection_retrievers = []

        for retriever_info in collection_config:
            if retriever_info["name"] == "bm25":
                collection_texts = load_texts(os.path.join(KNOWLEDGE_BASE_DIR, collection_name))
                bm25 = BM25Retriever.from_texts(collection_texts, preprocess_func=lambda x: [token.lemma_ for token in spacy_nlp(x)], **retriever_info)
                collection_retrievers.append(bm25)
            elif retriever_info["name"] == "semantic":
                collection_db = Chroma(embedding_function=cached_embedder, collection_name=collection_name,
                                    client=chroma_emb_client, persist_directory=CHROMA_PERSIST_DIRECTORY)
                semantic_retriever = collection_db.as_retriever(search_type="similarity", search_kwargs=retriever_info)
                collection_retrievers.append(semantic_retriever)

        if len(collection_retrievers) > 1:
            retrievers.append(EnsembleRetriever(retrievers=collection_retrievers))
        else:
            retrievers.append(collection_retrievers[0])

    context_retriever = MergerRetriever(retrievers=retrievers) if len(retrievers) > 1 else retrievers[0]
    return context_retriever

def init_qna_retrieval(context_retriever, cached_embedder, chroma_emb_client):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", verbose=True)

    rqa_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            template=("You are an AI assistant who answers customer questions about the services and processes "
                    "of the postal company Nova Poshta. Use the following pieces of context to answer the question. "
                    "Answer only in Ukrainian, regardless of the question language.\n\nCONTEXT:\n{context}\n\n"
                    "USER QUESTION: {question}\n\n"
                    "If the question is not related to the context, tell to contact the support. If the answer is not "
                    "contained in the context, tell to contact support. Don't make up the answer. If the question is not "
                    "related to the postal services or it doesn't make sense, tell that you can't answer it.\n\n"
                    "ANSWER IN UKRAINIAN:'")
        )]
    )
    rqa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=context_retriever, return_source_documents=True,
                                            chain_type_kwargs={"prompt": rqa_prompt_template})


    condense_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            template=("You are an AI assistant who answers customer questions about the services and processes "
                    "of the postal company Nova Poshta. Given the following conversation and a follow user up input, "
                    "rephrase it to be a standalone question."
                    "\n\nLast Messages:\n{last_messages}\n\nHuman Follow Up Input: {question}\n\n"
                    "If the follow up user input is not related to the last messages, return it as it is.\n"
                    "REPHRASED QUESTION IN UKRAINIAN:")
        )]
    )
    condense_chain = LLMChain(llm=llm, prompt=condense_prompt_template)


    chroma_questions_db = Chroma(embedding_function=cached_embedder, collection_name="questionss",
                                client=chroma_emb_client, persist_directory=CHROMA_PERSIST_DIRECTORY)
    redis_qa_client = redis.Redis(host=REDIS_HOST, port=6379, db=3)
    rqa_cache = CompletionCache(chroma_questions_db, redis_qa_client)
    cached_conversational_rqa = CachedConversationalRQA(condense_chain, rqa_chain, rqa_cache)

    return cached_conversational_rqa, llm

def init_agent(cached_conversational_rqa, llm):
    tools = [
        Tool(
            name="package_info",
            func=get_package_info,
            args_schema=Package,
            description="Useful for when you need to get tracking details and other information about the package",
        ),
        StructuredTool.from_function(
            func=calculate_delivery_cost,
            args_schema=Delivery,
            description="Useful for when you need to estimate the delivery cost"
        ),
        StructuredTool.from_function(
            func=estimate_delivery_date,
            args_schema=DeliveryDetails,
            description="Useful for when you need to estimate package delivery date",
        ),
        Tool(
            name="question_answering",
            func=lambda question: cached_conversational_rqa(question, []),
            args_schema=Question,
            description="Useful for answering any type of questions, always use it if user asks a question",
            return_direct=True
        )
    ]

    agent_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            template=("You are an AI assistant of the postal company Nova Poshta, which performs basic operations: "
                    "tracking parcels, calculating service costs, and informing about delivery terms. "
                    "You can also answer questions about the services and processes of the company."
                    "If the question is not related to the Nova Poshta or it doesn't make sense, tell that you can't answer it.\n{chat_messages}")
        ),
        HumanMessagePromptTemplate.from_template(
            template="{input}"
        ),
        SystemMessagePromptTemplate.from_template(
            template="Do not answer the questions that are not related to the postal, logistics, delivery, courier and related services and processes."
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, llm_prompt=agent_prompt_template, verbose=False)
    agent.agent.prompt = agent_prompt_template
    return agent