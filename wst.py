import os
import re
import glob
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from PIL import Image
import imagehash
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
import pandas as pd

app = Flask(__name__)
socketio = SocketIO(app)

TEMPLATE = """
You are DecaBot, a friendly and empathetic sports assistant AI. Your task is to provide helpful information about sports and sports products in a conversational manner.
Given the following context and chat history, address the user's question:

Context: {context}
Question: {question}
Chat History: {chat_history}

Instructions:
1. Always start by acknowledging the user's specific request or question.
2. If the query is not related to sports or sports products, provide a general answer using your knowledge, maintaining a friendly tone.
3. If the query is related to sports or sports products:
   - Answer using your knowledge and ONLY the information from the provided context.
   - If the exact request can't be met (e.g., specific price range, feature), politely explain why and offer the closest alternatives.
   - ONLY suggest products that are explicitly mentioned in the context.
   - When mentioning products, ALWAYS include the exact URL provided in the context. DO NOT create or infer URLs.
   - If asked to compare products, use a clear structure for easy reading.
4. Always maintain a conversational tone and remember previous interactions.
5. If no relevant products are found in the context, inform the user politely and offer to help with general sports information instead.
6. End your response with an open-ended question or an offer for further assistance to keep the conversation going.

Remember to be helpful, accurate, personalized, and empathetic in your responses, while strictly adhering to the information in the context for product suggestions and URLs.
"""

class SocketIOCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        socketio.emit('new_token', {'token': token})

class ConversationChain:
    def __init__(self, retriever, llm, prompt, memory):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.memory = memory

    def invoke(self, query):
        chat_history = self.get_chat_history()
        is_sports_related = self.is_sports_request(query, chat_history)
        
        if is_sports_related:
            context_documents = self.retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in context_documents])

            urls = [doc.metadata.get('url', '') for doc in context_documents if 'url' in doc.metadata]
            urls_str = "\n".join(list(set(urls)))
            context += "\n\n"+urls_str

            if "compare" in query.lower():
                response = self.compare_products(query,context)
            else:
                response = self.answer_sports_query(query,context)
        else:
            context = ""
            response = self.answer_general_query(query)

        self.memory.chat_memory.add_user_message(HumanMessage(content=query))
        self.memory.chat_memory.add_ai_message(AIMessage(content=response))
        return response, is_sports_related
    
    def answer_sports_query(self, query, context):
        prompt_input = f"""
        You are DecaBot, a friendly and empathetic sports assistant. Address the following query based ONLY on the given context. Do not invent or suggest any products or information not explicitly stated in the context.

        Context: {context}

        Query: {query}

        Chat History: {self.get_chat_history()}

        Instructions:
        1. Start with a warm, empathetic acknowledgment of the user's specific request.
        2. If the exact request can't be met, express understanding and apologize in a natural, conversational manner.
        3. ONLY mention products that are EXPLICITLY listed in the context. DO NOT create or invent any products.
        4. ONLY use URLs that are EXPLICITLY provided in the context. NEVER create or invent URLs.
        5. Suggest some products provided in the context even it is not met the request. Introduce these suggestions in a helpful, encouraging way.
        6. If no relevant products are found in the context, inform the user and offer to help with general sports information instead.
        7. Maintain a friendly, conversational tone throughout, as if speaking to a friend.
        8. If a comparison between two products is needed, DO NOT compare them here. Instead, output the text "COMPARE_PRODUCTS" followed by the names of the two products to compare.
        9. NEVER invent or assume any information not present in the context.

        Your response:
    """
        response = self.llm([HumanMessage(content=prompt_input)]).content
        if "COMPARE_PRODUCTS" in response:
            _, products = response.split("COMPARE_PRODUCTS", 1)
            products = products.strip().split(",")
            if len(products) == 2:
                comparison = self.compare_products(f"Compare {products[0]} and {products[1]}", context)
                response = response.replace(f"COMPARE_PRODUCTS{products[0]},{products[1]}", comparison)
            else:
                # If we can't extract two product names, we'll use the whole response as a query
                comparison = self.compare_products(response, context)
                response = comparison
        return response
    
    def answer_general_query(self,query):
        prompt_input = f"""
        Answer the following question using your general knowledge. Do not mention or suggest any specific products.
        
        Question: {query}
        Chat History: {self.get_chat_history()}
        """
        response = self.llm([HumanMessage(content=prompt_input)]).content
        return response
    
    def compare_products(self,query,context):
        response_schemas = [
            ResponseSchema(name="product1", description="Name of the first product"),
            ResponseSchema(name="product2", description="Name of the second product"),
            ResponseSchema(name="price1", description="Price of the first product"),
            ResponseSchema(name="price2", description="Price of the second product"),
            ResponseSchema(name="features1", description="Key features of the first product"),
            ResponseSchema(name="features2", description="Key features of the second product"),
            ResponseSchema(name="url1", description="URL of the first product"),
            ResponseSchema(name="url2", description="URL of the second product"),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        compare_prompt = PromptTemplate(
            template="""
            Compare two sports products based on the following context and query. 
            Only use products that are explicitly mentioned in the context.
            Provide the actual URLs for the products as found in the context. Do not create or infer URLs.

            Context: {context}
            Query: {query}

            {format_instructions}

            Comparison:
            """,
            input_variables=["context", "query"],
            partial_variables={"format_instructions": format_instructions}
        )
        compare_input = compare_prompt.format(context=context, query=query)
        comparison_result = self.llm([HumanMessage(content=compare_input)]).content

        try:
            parsed_comparison = output_parser.parse(comparison_result)
            
            # Create a pandas DataFrame for the comparison
            df = pd.DataFrame({
                "": ["Name", "Price", "Key Features", "URL"],
                parsed_comparison["product1"]: [
                    parsed_comparison["product1"],
                    parsed_comparison["price1"],
                    parsed_comparison["features1"],
                    parsed_comparison["url1"]
                ],
                parsed_comparison["product2"]: [
                    parsed_comparison["product2"],
                    parsed_comparison["price2"],
                    parsed_comparison["features2"],
                    parsed_comparison["url2"]
                ]
            })
            
            # Convert DataFrame to HTML table
            table_md= df.to_markdown(index=False)
            
            response = f"Here's a comparison of {parsed_comparison['product1']} and {parsed_comparison['product2']}:\n\n{table_md}"
            
        except Exception as e:
            response = f"I apologize, but I couldn't generate a proper comparison. Here's what I found:\n\n{comparison_result}"

        return response



    def is_sports_request(self, query, chat_history):
        sports_check_prompt = f"""
        Is this query about sports, sports products, athletic activities, fitness, sports equipment, comparison between the products,asking for suggestion of sports products with respect to the cost or features?
        Respond with only 'True' or 'False'.

        Chat History: {chat_history}
        Query: {query}
        """
        response = self.llm([HumanMessage(content=sports_check_prompt)]).content.strip().lower()
        return response == 'true'

    def get_chat_history(self):
        return "\n".join([f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" 
                          for msg in self.memory.chat_memory.messages])

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

def welcome_user(llm):
    welcome_prompt = """
    You are DecaBot, a sports assistant AI. Your task is to welcome the user 
    to the conversation. Introduce yourself briefly and mention that you're here 
    to help with sports-related questions and product recommendations. Keep it 
    friendly and concise.
    """
    response = llm([HumanMessage(content=welcome_prompt)]).content
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

@socketio.on('connect')
def handle_connect():
    emit('receive_message', {'message': welcome_message, 'images': []})

@socketio.on('send_message')
def handle_message(data):
    query = data['message']
    result, is_sports_related = chain.invoke(query)
    image_urls = []
    if is_sports_related:
        relevant_images = search_images(query, image_vector_store, top_k=1)
        image_urls = [f"/images/{os.path.relpath(img, 'images')}" for img in relevant_images]
    
    emit('receive_message', {'message': result, 'images': image_urls})

# Image handling functions (simplified)
def index_images(image_folder):
    image_index = []
    for image_file in glob.glob(os.path.join(image_folder, "*")):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_index.append({
                'content': f"Image of {os.path.basename(image_file)}",
                'metadata': {'path': image_file}
            })
    return image_index

def index_all_images(main_folder):
    image_index = []
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            image_index.extend(index_images(subfolder_path))
    return image_index

def create_image_vector_store(image_index, embeddings):
    texts = [item['content'] for item in image_index]
    metadatas = [item['metadata'] for item in image_index]
    return Chroma.from_texts(texts, embeddings, metadatas=metadatas)

def search_images(query, image_vector_store, top_k=1):
    results = image_vector_store.similarity_search(query, k=top_k)
    return [doc.metadata['path'] for doc in results]

if __name__ == '__main__':
    try:
        print("Loading and processing data...")
        folder_path = "./data"  # Replace with your actual folder path
        data = load_data(folder_path)

        if data:
            text_chunks = split_data(data)
            print(f"Data split into {len(text_chunks)} chunks.")

            print("Initializing embeddings and vector store...")
            embeddings = create_embeddings()
            vector_db = create_vector_db(text_chunks, embeddings)

            print("Setting up LLM and retriever...")
            llm = setup_llm()

            welcome_message = welcome_user(llm)

            prompt = ChatPromptTemplate.from_template(TEMPLATE)

            memory = ConversationBufferMemory()
            chain = ConversationChain(vector_db.as_retriever(), llm, prompt, memory)

            main_folder = 'images'
            all_image_index =index_all_images(main_folder)
            
            print("Creating image vector store")
            image_vector_store = create_image_vector_store(all_image_index, embeddings)

            print("Starting the Flask app...")
            socketio.run(app, debug=True)
            
    except Exception as e:
        print(f"An error occurred during setup: {e}")
