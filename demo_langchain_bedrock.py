import os

from langchain.llms import Bedrock
from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage

from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import Qdrant

import qdrant_client
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ValueError: Error raised by bedrock service: An error occurred (AccessDeniedException) when calling the InvokeModel operation: Your account is not authorized to invoke this API operation.
# NotImplementedError: Provider ai21 model does not support chat.

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    #model_id = "ai21.j2-mid-v1"
    model_id = "anthropic.claude-v1"
    model_kwargs = { "temperature": 0.0 }
    prompt = "What is the diameter of Earth?"

    #demo_invoke_model_predict(bedrock_runtime, model_id, model_kwargs, prompt)

    #demo_invoke_model_chat(bedrock_runtime, model_id, model_kwargs, prompt)

    # Embeddings Test
    embeddings_model_id = 'amazon.titan-embed-text-v1'
    query = "EC2とはなんですか？"
    document_path='embedding/files/ec2_faq.txt'
    vector_db_collection_name='ec2_faq'
    vector_db_path='embedding/vectordb/qdrant'
    model_id = 'anthropic.claude-v1'
    model_kwargs = { "temperature": 0.0 }

    #vectordb = demo_embeddings_create_vector_db(bedrock_runtime, embeddings_model_id, document_path, vector_db_collection_name, vector_db_path) # Run this once to initialize vector db
    vectordb = demo_embeddings_load_vector_db(bedrock_runtime, embeddings_model_id, vector_db_collection_name, vector_db_path)
    demo_embeddings_peek_vector_db(vectordb, query)
    demo_embeddings_invoke_model_chat(bedrock_runtime, model_id, vectordb, model_kwargs, query)

def demo_invoke_model_predict(bedrock_runtime, model_id, model_kwargs, prompt):

    llm = Bedrock(
        client = bedrock_runtime,
        model_id = model_id,
        model_kwargs = model_kwargs,
    )

    output = llm.predict(prompt)

    print(output)

def demo_invoke_model_chat(bedrock_runtime, model_id, model_kwargs, prompt):

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = model_id,
        model_kwargs = model_kwargs,
    )

    result = llm([
        HumanMessage(content = prompt)
    ])

    print(result.content)

def demo_embeddings_create_vector_db(bedrock_runtime, 
                                     model_id='amazon.titan-embed-text-v1',
                                     document_path='embedding/files/ec2_faq.txt',
                                     vector_db_collection_name='ec2_faq',
                                     vector_db_path='embedding/vectordb/qdrant'):

    print("Call demo_embeddings_create_vector_db")

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DOCUMENT_PATH = os.path.join(ROOT_DIR, document_path) #DOCUMENT_PATH = os.path.join(ROOT_DIR, 'embedding/files/ec2_faq.html')
    print("DOCUMENT_PATH: " + DOCUMENT_PATH)
    VECTOR_DB_PATH = os.path.join(ROOT_DIR, vector_db_path)
    print("VECTORDB_PATH: " + VECTOR_DB_PATH)

    # 2. Load Data
    loader = UnstructuredHTMLLoader(file_path=DOCUMENT_PATH, mode='single')
    data = loader.load()

    # 3. Chunk the Loaded Data - Set Chunk Size to conform to Titan Embeddings
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=(512*0.6), chunk_overlap=512*0.1)
    #text_splitter = RecursiveCharacterTextSplitter(
    #    chunk_size=(512*0.6), chunk_overlap=512*0.1, #chunk_size=4000, chunk_overlap=100
    #    separators=["\n\n", "\n"] #separators=["\n\n", "\r\n", "\n"]
    #)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=(1024), chunk_overlap=100, #chunk_size=4000, chunk_overlap=100
        separators=["\n\n", "\n"] #separators=["\n\n", "\r\n", "\n"]
    )
    splited_data = text_splitter.split_documents(data)
    num_docs = len(splited_data)
    print(num_docs)

    # 4. Create Amazon Titan Embeddings Object
    embeddings = BedrockEmbeddings(
        client=bedrock_runtime,
        model_id=model_id
    )

    # 5. Create VectorStore using Qdrant database
    db = Qdrant.from_documents(
        documents=splited_data, 
        embedding=embeddings, 
        path=VECTOR_DB_PATH,
        collection_name=vector_db_collection_name, 
        distance_func='Dot')
    
    print(f"Vector Database Initialized. {VECTOR_DB_PATH}")

    return db

    
def demo_embeddings_load_vector_db(bedrock_runtime, model_id='amazon.titan-embed-text-v1', vector_db_collection_name='ec2_faq', vector_db_path='embedding/vectordb/qdrant'):

    print("Call demo_embeddings_load_vector_db")

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    VECTOR_DB_PATH = os.path.join(ROOT_DIR, vector_db_path)
    print("VECTOR_DB_PATH: " + VECTOR_DB_PATH)

    client = qdrant_client.QdrantClient(
        path=VECTOR_DB_PATH, prefer_grpc=True
    )

    embeddings = BedrockEmbeddings(
        client=bedrock_runtime,
        model_id=model_id
    )

    db = Qdrant(
        client=client, collection_name=vector_db_collection_name, 
        embeddings=embeddings
    )

    print(f"Loaded Vector Database. {VECTOR_DB_PATH}")

    return db

def demo_embeddings_peek_vector_db(vectordb, query):

    print("Call demo_embeddings_peek_vector_db")

    #docs = vectordb.similarity_search(query=query, k=10)
    #for i in docs:
    #    print({"content": i.page_content[0:20], "metadata": i.metadata})

    docs = vectordb.similarity_search_with_score(query=query, k=10)
    for i in docs:
        doc, score = i
        print({"score": score, "content": doc.page_content[0:20], "metadata": doc.metadata} )

def demo_embeddings_invoke_model_chat(bedrock_runtime, model_id, vectordb, model_kwargs, query):

    print("Call demo_embeddings_invoke_model_chat")

    # 1. Create Prompt Template for Claude

    prompt_template = '''Human: 
    Text: {context}

    Question: {question}

    Answer the question based on the text provided. If the text doesn't contain the answer, reply that the answer is not available.

    Assistant:
    '''

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=['context', 'question']
    )

    # 2. Instantiate Claude Bedrock

    llm = Bedrock(
        client = bedrock_runtime,
        model_id = model_id,
        model_kwargs = model_kwargs,
        verbose=True
    )

    # 3. Create RetrievalQA

    chain_type_kwargs = {"prompt": PROMPT}

    qa = RetrievalQA.from_chain_type(llm = llm, 
                                 chain_type = "stuff", 
                                 retriever = vectordb.as_retriever(), 
                                 chain_type_kwargs = chain_type_kwargs, 
                                 return_source_documents = True)
    
    # 4. Invoke

    answer = qa(query)

    #print(answer)
    print()
    print(f"Query: {answer['query']}")
    print(f"Answer: {answer['result']}")
    print("Source Documents: ")
    for source_document in answer["source_documents"]:
        print(f"- {source_document.page_content[0:70]}") #print(f"{source_document.page_content[0:50]} {source_document.metadata}")
    