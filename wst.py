import time
import os
import re
import glob
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from PIL import Image
import imagehash

app = Flask(__name__)
socketio = SocketIO(app)

class SocketIOCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        socketio.emit('new_token', {'token': token})

class MessageHistoryChain:
    def __init__(self, retriever, llm, prompt, memory):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.memory = memory

    def invoke(self, inputs):
        query = inputs["question"]
        context_documents = self.retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in context_documents])
        
        urls = [doc.metadata.get('url', '') for doc in context_documents if 'url' in doc.metadata]
        urls_str = "\n".join(list(set(urls)))
        
        chat_history = "\n".join(
            [f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" for msg in self.memory.chat_memory.messages]
        )

        context += "\n\n" + urls_str
        
        prompt_input = self.prompt.format(context=context, question=query, chat_history=chat_history)
        
        response = self.llm([HumanMessage(content=prompt_input)]).content
        main_response = response.split('\n\n')[-1]
        self.memory.chat_memory.add_ai_message(AIMessage(content=main_response))
        return main_response

def generate_images_metadata(image_path):
    filename = os.path.basename(image_path)
    name, _ = os.path.splitext(filename)

    with Image.open(image_path) as img:
        hash = str(imagehash.average_hash(img))
    description = f"Image of {name.replace('_', ' ')}"
    return f"{description}\nFilename: {filename}\nHash: {hash}"

def createe_metadata_files(image_folder):
    for image_file in glob.glob(os.path.join(image_folder, "*")):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            metadata_file = image_file.rsplit('.', 1)[0] + '.txt'
            if not os.path.exists(metadata_file):
                metadata = generate_images_metadata(image_file)
                with open(metadata_file, 'w') as f:
                    f.write(metadata)
                print(f"Created metadata file for {image_file}")

def index_images(image_folder):
    image_index = []
    for image_file in glob.glob(os.path.join(image_folder, "*")):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            metadata_file = image_file.rsplit('.', 1)[0] + '.txt'
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = f.read().strip()
            else:
                metadata = generate_images_metadata(image_file)
            image_index.append({
                'content': metadata,
                'metadata': {'path': image_file}
            })
    return image_index

def create_image_vector_store(image_index, embeddings):
    texts = [item['content'] for item in image_index]
    metadatas = [item['metadata'] for item in image_index]
    return Chroma.from_texts(texts, embeddings, metadatas=metadatas)

def search_images(query, image_vector_store, top_k=2):
    results = image_vector_store.similarity_search(query, k=top_k)
    return [doc.metadata['path'] for doc in results]

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
                    match = re.search(url_pattern, doc.page_content.split(",")[-1].strip())
                    if match:
                        doc.metadata['url'] = match.group(0)
                    print(doc.metadata['url'])
                data.extend(documents)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        else:
            print(f"File {file_path} does not exist.")
    return data

def split_data(_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return text_splitter.split_documents(_data)

def create_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

def create_vector_db(_text_chunks, _embeddings):
    return Chroma.from_documents(
        documents=_text_chunks,
        embedding=_embeddings,
        collection_name="local-rag"
    )

def setup_llm():
    local_model = "mistral"
    return ChatOllama(model=local_model, callbacks=[SocketIOCallbackHandler()])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

@socketio.on('send_message')
def handle_message(data):
    query = data['message']
    memory.chat_memory.add_user_message(HumanMessage(content=query))
    result = chain.invoke({"question": query})
    
    # Image retrieval logic
    relevant_images = search_images(query, image_vector_store)
    
    image_urls = [f"/images/{subimage_folder}/{os.path.basename(img)}" for img in relevant_images]
    
    emit('receive_message', {'message': result, 'images': image_urls})

if __name__ == '__main__':
    print("Loading and processing data...")
    folder_path = "/home/system2/Documents/projects/declathon/data"  # Replace with your actual folder path
    data = load_data(folder_path)

    if data:
        text_chunks = split_data(data)
        print(f"Data split into {len(text_chunks)} chunks.")

        print("Initializing embeddings and vector store...")
        embeddings = create_embeddings()
        vector_db = create_vector_db(text_chunks, embeddings)

        print("Setting up LLM and retriever...")
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

        template = """
            You are an conversational AI assistant specializing in briefly answering the question based on the context. Your primary goal is to provide helpful, accurate, and personalized information to users based on the given context.
            Given the following context and chat history, address the user's question:
            {context}

            Question: {question}
            Chat History: {chat_history}

            Important Instructions:
            1. If the user's query is general and not directly related to specific sports equipment or activities, ask ONE broad, non-personal follow-up question to clarify their needs.

            2. Your follow-up question should aim to understand what specific information or recommendations the user is seeking related to sports or outdoor activities.

            3. Provide detailed response for listed products. Do not make assumptions about the user's interests or needs based on limited information.

            4. When listing products:
               - Include the full product name
               - Make the product name itself a hyperlink using markdown syntax: [Product Name](URL)
               - Do not include a separate "Link" text
               - Provide a Detailed description about the product 

            5. Don't provide URLs which are not present in the context and don't repeat the same URL in the response again and again.
               For example, if a product is mentioned in your response, append its URL from the context at the end of the response. Do not create URLs; use only those provided in the context.

            6. If the context doesn't provide enough information for a specific query, use general knowledge to provide a brief, relevant response related to context. Start the response with a friendly tone and end with "Let me know if you have any specific queries or need more information. I'm here to help!"

            Remember, your goal is to clarify the user's needs and provide relevant, easy-to-access information about products when appropriate.
        """

        prompt = ChatPromptTemplate.from_template(template)

        memory = ConversationBufferMemory()
        chain = MessageHistoryChain(retriever, llm, prompt, memory)

        print("Creating metadata files...")
        subimage_folder = 'accesories_images'
        image_folder = os.path.join('images', subimage_folder)
        createe_metadata_files(image_folder)

        print("Indexing images...")
        image_index = index_images(image_folder)

        print("Creating image vector store...")
        image_vector_store = create_image_vector_store(image_index, embeddings)

        print("Starting the Flask app...")
        socketio.run(app, debug=True)

    else:
        print("No data loaded. Please check the CSV files in the specified folder path.")
