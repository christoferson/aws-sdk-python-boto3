import os

from langchain.llms import Bedrock

from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import Qdrant

import qdrant_client
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    embeddings_model_id = 'amazon.titan-embed-text-v1'
    query = "EC2とはなんですか？"
    document_path='embedding/files/ec2_faq.txt'
    vector_db_collection_name='ec2_faq'
    vector_db_path='embedding/vectordb/qdrant'
    model_id = 'anthropic.claude-v1'
    model_kwargs = { "temperature": 0.0 }

    vectordb = demo_embeddings_create_vector_db(bedrock_runtime, embeddings_model_id, document_path, vector_db_collection_name, vector_db_path) # Run this once to initialize vector db

def demo_embeddings_create_vector_db(bedrock_runtime, 
        model_id='amazon.titan-embed-text-v1',
        document_path='embedding/files/ec2_faq.txt',
        vector_db_collection_name='ec2_faq',
        vector_db_path='embedding/vectordb/qdrant'):

    print("Call demo_embeddings_create_vector_db")

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DOCUMENT_PATH = os.path.join(ROOT_DIR, document_path)
    print("DOCUMENT_PATH: " + DOCUMENT_PATH)
    VECTOR_DB_PATH = os.path.join(ROOT_DIR, vector_db_path)
    print("VECTORDB_PATH: " + VECTOR_DB_PATH)

    # 2. Load Data
    loader = UnstructuredHTMLLoader(file_path=DOCUMENT_PATH, mode='single')
    data = loader.load()

    # 3. Chunk the Loaded Data - Set Chunk Size to conform to Titan Embeddings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=(1024), chunk_overlap=100,
        separators=["\n\n", "\n"]
    )
    splited_data = text_splitter.split_documents(data)
    num_docs = len(splited_data)
    print(num_docs)

    # 4. Create Amazon Titan Embeddings Object
    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = model_id
    )

    # 5. Create VectorStore using Qdrant database
    db = Qdrant.from_documents(
        documents = splited_data, 
        embedding = embeddings, 
        path = VECTOR_DB_PATH,
        collection_name = vector_db_collection_name, 
        distance_func = 'Dot')
    
    print(f"Vector Database Initialized. {VECTOR_DB_PATH}")

    return db