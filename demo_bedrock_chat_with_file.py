import boto3
import json
import config
import numpy as np
import cmn_utils
import logging
import base64

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# https://aws.amazon.com/jp/blogs/machine-learning/knowledge-bases-in-amazon-bedrock-now-simplifies-asking-questions-on-a-single-document/
# https://aws.amazon.com/jp/blogs/machine-learning/knowledge-bases-in-amazon-bedrock-now-simplifies-asking-questions-on-a-single-document/
# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/retrieve_and_generate.html
# The supported file formats are PDF, MD (Markdown), TXT, DOCX, HTML, CSV, XLS, and XLSX. 
# Make that the file size does not exceed 10 MB and contains no more than 20,000 tokens. 


def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")
    bedrock_agent_runtime = session.client('bedrock-agent-runtime', region_name="us-east-1")


    file_base64 = None
    with open("data/movies.csv", "rb") as file:
        file_base64 = base64.b64encode(file.read())
    print(len(file_base64))

    model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
    uploaded_file_type = "text/plain"
    uploaded_file_bytes = file_base64
    uploaded_file_name = "movies.csv"
    prompt = """
    What is the highest grossing film?
    """
    session_id = demo_stream_invoke_model_anthropic_claude_v3(bedrock_agent_runtime, model_id, prompt,
        uploaded_file_type, uploaded_file_bytes, uploaded_file_name, current_session_id=None)
    prompt = "When was that movie aired?"
    session_id = demo_stream_invoke_model_anthropic_claude_v3(bedrock_agent_runtime, model_id, prompt,
        uploaded_file_type, uploaded_file_bytes, uploaded_file_name, current_session_id=session_id)

def demo_stream_invoke_model_anthropic_claude_v3(bedrock_agent_runtime, model_id, input_text,
        uploaded_file_type,
        uploaded_file_bytes,
        uploaded_file_name, current_session_id) -> str:
    
    session_id = None

    try:

        params = {
                "input": {"text": input_text},
                "retrieveAndGenerateConfiguration": {
                    "type": "EXTERNAL_SOURCES",
                    "externalSourcesConfiguration": {
                        "modelArn": model_id,
                        "sources": [
                            {
                                "sourceType": "BYTE_CONTENT",
                                "byteContent": {
                                    "contentType": uploaded_file_type,
                                    "data": uploaded_file_bytes,
                                    "identifier": uploaded_file_name,
                                },
                            }
                        ],
                    },
                },
            }
        
        if current_session_id != None:
            params["sessionId"] = current_session_id
        
        #body = json.dumps(params)
        #print(body)
        response = bedrock_agent_runtime.retrieve_and_generate(**params)

        print(response["output"]["text"])
        #print(response["citations"])

        session_id = response["sessionId"]
        print(session_id)

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))

    return session_id