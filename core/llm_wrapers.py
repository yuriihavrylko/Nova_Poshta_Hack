from langchain.embeddings import CacheBackedEmbeddings
from langchain.schema import get_buffer_string


class CachedEmbeddings(CacheBackedEmbeddings):
    def embed_query(self, text):
        return self.embed_documents([text])[0]


class CompletionCache:
    def __init__(self, chroma_db, redis_client, score_threshold=0.15):
        self.redis_client = redis_client
        self.chroma_db = chroma_db
        self.score_threshold = score_threshold

    def get(self, prompt):
        chroma_response = self.chroma_db.similarity_search_with_score(
            prompt, k=1)
        if chroma_response:
            document, score = chroma_response[0]
            if score < self.score_threshold:
                return self.redis_client.get(document.page_content).decode()

    def set(self, prompt, completion):
        self.chroma_db.add_texts(
            [prompt], ids=[hashlib.sha256(prompt.encode()).hexdigest()])
        self.redis_client.set(prompt, completion)


class CachedConversationalRQA:
    def __init__(self, condense_chain, rqa_chain, rqa_cache, k=2,
                 condense_output_key="text", rqa_output_key="result"):
        self.condense_chain = condense_chain
        self.rqa_chain = rqa_chain
        self.cache = rqa_cache
        self.k = 2
        self.condense_output_key = condense_output_key
        self.rqa_output_key = rqa_output_key

    def __call__(self, question, chat_messages):
        cached_completion = self.cache.get(question)
        if cached_completion:
            return cached_completion

        last_messages = chat_messages[-self.k * 2:] if self.k > 0 else []
        if last_messages:
            last_messages_str = get_buffer_string(last_messages)
            question = self.condense_chain(
                {"question": question, "last_messages": last_messages_str})[self.condense_output_key]
            rephrased_cache_completion = self.cache.get(question)
            if rephrased_cache_completion:
                return rephrased_cache_completion

        completion = self.rqa_chain(question)[self.rqa_output_key]
        self.cache.set(question, completion)
        return completion


class LLMChatHandler:
    def __init__(self, agent, chat_history, k=4):
        self.agent = agent
        self.chat_history = chat_history
        self.k = k

    def send_message(self, message):
        chat_messages = self.chat_history.messages[-self.k *
                                                   2:] if self.k > 0 else []
        agent_output = self.agent.run(
            {"input": message, "chat_messages": get_buffer_string(chat_messages)})

        self.chat_history.add_user_message(message)
        self.chat_history.add_ai_message(agent_output)

        return agent_output
