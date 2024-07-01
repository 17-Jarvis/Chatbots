import time
import os
import re
import glob
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

class MessageHistoryChain:
    def __init__(self, retriever, llm, prompt, memory):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.memory = memory

    def invoke(self, inputs, response_placeholder):
        query = inputs["question"]
        context_documents = self.retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in context_documents])
        
        # Extract URLs from context
        urls = [doc.metadata.get('url', '') for doc in context_documents if 'url' in doc.metadata]
        urls_str = "\n".join(list(set(urls)))
        
        # Format the chat history
        chat_history = "\n".join(
            [f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" for msg in self.memory.chat_memory.messages]
        )

        context += "\n\n"+urls_str
        
        # Format the prompt input
        prompt_input = self.prompt.format(context=context, question=query, chat_history=chat_history)
        
        # Simulate streaming by breaking down the response
        response_parts = self.llm([HumanMessage(content=prompt_input)]).content.split("\n\n")
        response_list = []
        
        for part in response_parts:
            response_list.append(part.strip())
            response_placeholder.markdown("\n\n".join(response_list))  # Append URLs to the response
            time.sleep(1.5)
        
        response = "\n\n".join(response_list)
        self.memory.chat_memory.add_ai_message(AIMessage(content=response))
        return response

# Streamlit UI for querying
st.title("Sports GPT")

@st.cache_data(show_spinner=False)
def load_data(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    data = []
    url_pattern = r'https?://(?:www\.)?[\w-]+\.[\w.-]+(?:/\S*)?'
    for file_path in csv_files:
        
        if os.path.exists(file_path):
            
            try:
                loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
                documents = loader.load()
                for doc in documents:
                    # Add URLs to metadata
                    match = re.search(url_pattern, doc.page_content.split(",")[-1].strip())  # Assuming URL is the last column
                    if match:
                        doc.metadata['url'] = match.group(0)
                    print(doc.metadata['url'])
                data.extend(documents)
            except Exception as e:
                # st.write(f"Error loading {file_path}: {e}")
                continue
        else:
            st.write(f"File {file_path} does not exist.")
    return data

@st.cache_data(show_spinner=False)
def split_data(_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return text_splitter.split_documents(_data)

@st.cache_resource(show_spinner=False)
def create_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

@st.cache_resource(show_spinner=False)
def create_vector_db(_text_chunks, _embeddings):
    return Chroma.from_documents(
        documents=_text_chunks,
        embedding=_embeddings,
        collection_name="local-rag"
    )

@st.cache_resource(show_spinner=False)
def setup_llm():
    local_model = "mistral"
    return ChatOllama(model=local_model)

# Load and process data
st.write("Loading and processing data...")
folder_path = "./data"  # Replace with your actual folder path
data = load_data(folder_path)
if data:
    text_chunks = split_data(data)
    st.write(f"Data split into {len(text_chunks)} chunks.")

    # Initialize embeddings and vector store
    st.write("Initializing embeddings and vector store...")
    embeddings = create_embeddings()
    vector_db = create_vector_db(text_chunks, embeddings)

    # Set up the LLM and retriever
    st.write("Setting up LLM and retriever...")
    llm = setup_llm()

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are a highly knowledgeable conversational AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve the most relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines. Maintain a professional and friendly tone.
        Original question: {question}"""
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )

    # RAG prompt
#     template = """
#     Based on the following context and chat history, answer the user's question accurately and comprehensively:
# 
#     {context}
# 
#     If the context doesn't provide enough information, use your knowledge to answer related to the question  accurately.
#     But give more importance to the context having url while answering the question using your knowlegde.
#     If user's questions are too generalised ask follow up questions to understand the user queries before answering their queries
#     Question: {question}
#     Chat History: {chat_history}
# 
#     Ensure your response is:
#     1. Clear, detailed, and helpful.
#     2. Includes relevant product URLs from the context when suggesting a product.
#     3. If a URL is not provided in the context, do not mention url is not provided; simply skip it.
#     4.Don't provide url which is not present  in the context and don't repeat the same url in the response again and again.
#       For example, if a product is mentioned in your response, append its URL from the context at the end of the response. Do not create URLs; use only those provided in the context.
#     5. If user's questions are too generalised ask follow up questions to understand the user queries before answering their queries.
# """
    template = """
        You are an conversational AI assistant specializing in breifly answering the question based on the context. Your primary goal is to provide helpful, accurate, and personalized information to users based on the given context.
        Given the following context and chat history, address the user's question:
        {context}

        Question: {question}
        Chat History: {chat_history}

        Important Instructions:
        1. If the user's query is general and not directly related to specific sports equipment or activities, ask ONE broad, non-personal follow-up question to clarify their needs.

        2. Your follow-up question should aim to understand what specific information or recommendations the user is seeking related to sports or outdoor activities.

        3. Provide detailed response for lis  ted products.Do not make assumptions about the user's interests or needs based on limited information.

        4. When listing products:
           - Include the full product name
           - Make the product name itself a hyperlink using markdown syntax: [Product Name](URL)
           - Do not include a separate "Link" text
           - Provide a Detailed description about the product 

        6. Don't provide url which is not present  in the context and don't repeat the same url in the response again and again.
#           For example, if a product is mentioned in your response, append its URL from the context at the end of the response. Do not create URLs; use only those provided in the context

        7. If the context doesn't provide enough information for a specific query, use general knowledge to provide a brief, relevant response related to context. Start the response with the friendly tone and end with "Let me know if you have any specific queries or  need more information. I'm here to help!"

        Remember, your goal is to clarify the user's needs and provide relevant, easy-to-access information about products when appropriate.
"""

    prompt = ChatPromptTemplate.from_template(template)

    # Define a function to get or create session history
    def get_session_history(session_id):
        if "history" not in st.session_state:
            st.session_state.history = {}
        if session_id not in st.session_state.history:
            st.session_state.history[session_id] = ConversationBufferMemory()
        return st.session_state.history[session_id]

    # Define the chain with message history
    chain = MessageHistoryChain(retriever, llm, prompt, get_session_history("session"))

    # Streamlit UI for querying
    st.write("Ready to accept queries.")
    query_input = st.text_input("Enter your query:")

    # Button to submit query
    if st.button("Enter"):
        if query_input:
            memory = get_session_history("session")
            memory.chat_memory.add_user_message(HumanMessage(content=query_input))
            
            # Placeholder for the response
            response_placeholder = st.empty()
            result = chain.invoke({"question": query_input}, response_placeholder)
            
            # Append to session history
            st.session_state.history["session"].chat_memory.add_ai_message(AIMessage(content=result))

    # Display query history
    if "history" in st.session_state and "session" in st.session_state.history:
        st.write("*Query History:*")
        history = st.session_state.history["session"].chat_memory.messages
        for message in history:
            role = "User" if isinstance(message, HumanMessage) else "AI"
            st.write(f"*{role}:* {message.content}")
            st.write("---")
else:
    st.write("No data loaded. Please check the CSV files in the specified folder path.")



