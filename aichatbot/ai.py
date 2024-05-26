import bs4
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.inmemory import InMemoryVectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage

# Initialize the session history store
store = {}

def create_retriever(url, chunk_size=1000, chunk_overlap=200):
    # Load web page content
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    
    # Create in-memory vector store and retriever
    inmemory_vectorstore = InMemoryVectorStore.from_documents(
        documents=splits, 
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    )
    
    inmemory_retriever = inmemory_vectorstore.as_retriever()
    
    return inmemory_retriever

def initialize_conversational_rag_chain(inmemory_retriever):
    # Define the LLM
    llm = Ollama(model="phi3", temperature=0.3, top_k=5)

    # Define the contextualization system prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    # Create the contextualize question prompt template
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
    )

    # Create the history aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, inmemory_retriever, contextualize_q_prompt
    )

    # Define the question answering system prompt
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create the question answer chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create the RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Define the function to get session history
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # Create the conversational RAG chain with message history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain

def send_query(conversational_rag_chain, query, session_id):
    # Invoke the conversational RAG chain with the given query and session ID
    result = conversational_rag_chain.invoke(
        {"input": query},
        config={
            "configurable": {"session_id": session_id}
        }
    )
    # Return the answer from the result
    return result["answer"]

# Function to retrieve the chat history for a session
def get_chat_history(store, session_id):
    if session_id in store:
        return store[session_id].messages
    else:
        return []