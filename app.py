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

# Initialize session state attributes
if "user_info" not in st.session_state:
    st.session_state.user_info = {}
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = []
if "model_questions" not in st.session_state:
    st.session_state.model_questions = ["Welcome"]
#if "messages" not in st.session_state:
 #   st.session_state.messages = []

class MessageHistoryChain:
    def __init__(self, retriever, llm, prompt, memory, dataset_features):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.memory = memory
        self.dataset_features = dataset_features
        self.required_info = ['primary use', 'age', 'gender', 'height', 'weight'] + dataset_features
        self.questions_asked = 0
        self.max_questions = 6
        self.last_question = None

    def invoke(self, inputs, response_placeholder):
        query = inputs["user_answer"]
        last_question = self.last_question
    
        print(f"Received query: {query}")
        print(f"Current user info before update: {st.session_state.user_info}")

        if "conversation_started" not in st.session_state:
            st.session_state.conversation_started = False

        if not st.session_state.conversation_started:
            st.session_state.conversation_started = True
            response = self.start_query()
        else:
            # Update user info with the previous question as the key
            if last_question:
                self.update_user_info(query, last_question)

            next_action = self.ask_next_question(query)
            if next_action == "READY_TO_RECOMMEND" or self.questions_asked >= self.max_questions:
                response = self.get_recommendations(query)
            else:
                response = next_action
            self.last_question = response if next_action != "READY_TO_RECOMMEND" else None

        #print(f"Current user info after update: {st.session_state.user_info}")
        #print(f"Generated response: {response}")
        #print(f"Last question updated to: {self.last_question}")

        # Simulate streaming
        response_parts = response.split("\n\n")
        for part in response_parts:
            response_placeholder.markdown(part.strip())
            time.sleep(0.5)

        self.memory.chat_memory.add_user_message(HumanMessage(content=query))
        self.memory.chat_memory.add_ai_message(AIMessage(content=response))
        return response

    def get_initial_message(self):
        return """Hello! I'm DecaBot, your sports equipment assistant. I'll ask you some questions to help find the best products for you. """
    
    def start_query(self):
        return "What is the sport or product you're looking for?"

    def update_user_info(self, query, question):
        if "user_info" not in st.session_state:
            st.session_state.user_info = {}
        st.session_state.user_info[question] = query
        print(f"Updated user info: {st.session_state.user_info}")

    def has_all_required_info(self):
        # Check if all required information is present in the values of user_info
        return all(any(info.lower() in value.lower() for value in st.session_state.user_info.values()) for info in self.required_info)

    def ask_next_question(self, query):
        self.questions_asked += 1
        chat_history = "\n".join(
            [f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" 
             for msg in self.memory.chat_memory.messages]
        )

        context_prompt = f"""
        You are DecaBot, an AI assistant specializing in sports equipment recommendations. Your task is to ask the next relevant question to gather information about the user's needs.

        Current user information: {st.session_state.user_info}
        Required information: {self.required_info}
        Questions asked so far: {self.questions_asked}
        Maximum questions to ask: {self.max_questions}

        Conversation history:
        {chat_history}

        User's latest response: {query}

        Instructions:
        1. Start with asking about primary use (sport or product), then age, gender, height, weight, etc. Ask one question at a time to make it interactive.
        2. Analyze the current user information and the conversation history.
        3. Identify the next piece of required information that hasn't been collected yet.
        4. Don't ask questions which  more technical about the product, ask more generalize questions which can be  more effective to find the user's needs.
        5. Sounds like a human asking a question not like robot , be friendly and polite.
        6. Do not ask for information that has already been provided.
        7. If you've asked {self.max_questions} questions or more, instead of asking another question, respond with: "READY_TO_RECOMMEND"

        Provide only the next question, without any additional explanation or context.
        """

        response = self.llm([HumanMessage(content=context_prompt)]).content
        return response

    def get_recommendations(self,query,user_info):
        chat_history = "\n".join(
            [f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" 
            for msg in self.memory.chat_memory.messages]
        )

        context_documents = self.retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in context_documents])
        urls = [doc.metadata.get('url', '') for doc in context_documents if 'url' in doc.metadata]
        urls_str = "\n".join(list(set(urls)))
        context += "\n\n" + urls_str

        recommendation_prompt = f"""
        You are DecaBot, an AI assistant specializing in sports equipment recommendations. Your task is to provide personalized product recommendations based on the user's information.

        Context: {context}

        Chat History: {chat_history}

        User Information: {user_info}

        User's latest query: {query}

        Instructions:
        1. Analyze the user's information and the available product data.
        2. Provide 2-3 product recommendations that best match the user's needs and preferences.
        3. For each recommendation, include:
        a. Product name as a hyperlink using markdown syntax: [Product Name](URL)
        b. A brief description of why this product is suitable for the user
        4. Only use URLs from the provided context.
        5. Ensure that you only recommend products that are present in the dataset and match the user's information.
        6. Be friendly and conversational in your tone.
        7. Summarize the user's preferences and requirements before providing recommendations.

        Provide your recommendations in a concise and easy-to-read format.
        """

        response = self.llm([HumanMessage(content=recommendation_prompt)]).content
        return response

# Streamlit UI for querying
st.title("DecaBot - Your Sports Equipment Assistant")

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
                data.extend(documents)
            except Exception as e:
                st.write(f"Error loading {file_path}: {e}")
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

def extract_dataset_features(data):
    features = set()
    for doc in data:
        features.update(doc.page_content.split(','))
    return list(features - {'age', 'gender', 'height', 'weight', 'primary use'})

def get_session_history(session_id):
    if "history" not in st.session_state:
        st.session_state.history = {}
    if session_id not in st.session_state.history:
        st.session_state.history[session_id] = ConversationBufferMemory(return_messages=True)
    return st.session_state.history[session_id]

def questions_answers_dict_mapping(model_questions,user_inputs):
    print(len(user_inputs))
    print(len(model_questions))
    qa_mapping = {model_questions[i]: user_inputs[i] for i in range(len(user_inputs))}
    return qa_mapping


# Load and process data
st.write("Loading and processing data...")
folder_path = "./data"  # Replace with your actual folder path
data = load_data(folder_path)
if data:
    text_chunks = split_data(data)
    st.write(f"Data split into {len(text_chunks)} chunks.")

    # Extract dataset features
    dataset_features = extract_dataset_features(data)

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
    template = """
    You are DecaBot, an AI assistant specializing in sports equipment recommendations. Based on the given context, chat history, and user information, provide helpful and accurate responses.

    Context: {context}

    Question: {question}
    Chat History: {chat_history}

    Instructions:
    1. Provide detailed and personalized responses based on the user's information and query.
    2. When recommending products, include the full product name as a hyperlink using markdown syntax: [Product Name](URL).
    3. Give a brief description of why each recommended product is suitable for the user.
    4. Only use URLs provided in the context.
    5. If more information is needed, ask relevant follow-up questions.
    6. Keep your responses concise, friendly, and focused on the user's needs.

    Remember, your goal is to provide helpful and personalized information about sports equipment based on the user's profile and the available product data.
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Define the chain with message history
    chain = MessageHistoryChain(retriever, llm, prompt, get_session_history("session"), dataset_features)

    # Streamlit UI for querying
    st.write("Ready to chat!")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Send welcome message
        welcome_message = chain.get_initial_message()
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_prompt := st.chat_input("Your response"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        if len(st.session_state.model_questions)>7:
            user_info_dict = questions_answers_dict_mapping(st.session_state.model_questions, st.session_state.user_inputs)
            st.session_state.messages.append({"role": "assistant", "content": "The number of questions has exceeded 6. The questions and answers have been processed."})
            recommend= chain.get_recommendations(st.session_state.model_questions,user_info_dict)
            st.write(recommend)
        else:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                st.session_state.user_inputs.append(user_prompt)  # Store user input in session state
                print(st.session_state.user_inputs)
                print(f"Invoking chain with prompt: {user_prompt}")
                
                response = chain.invoke({"user_answer": user_prompt}, response_placeholder)
                st.session_state.model_questions.append(response)  # Store model questions in session state
                print(st.session_state.model_questions)
                st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear the input after sending
    st.session_state.input = ""

else:
    st.write("No data loaded. Please check the CSV files in the specified folder path.")
