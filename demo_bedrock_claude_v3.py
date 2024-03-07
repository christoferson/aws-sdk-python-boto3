import boto3
import json
import config
import numpy as np
import cmn_utils
import logging

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    #demo_list_foundation_models(bedrock)

    #demo_invoke_model(bedrock_runtime, "ai21.j2-mid-v1", "What is the diameter of Earth?")

    #demo_invoke_model_anthropic_claude(bedrock_runtime)

    #demo_stream_invoke_model_anthropic_claude(bedrock_runtime)

    model_id = "amazon.titan-embed-text-v1"

    #demo_embedding_calculate_with_cosine_similarity(bedrock_runtime, model_id)
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
    model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
    #model_id = "anthropic.claude-v2"
    demo_invoke_model_anthropic_claude_v3(bedrock_runtime, model_id)


def demo_list_foundation_models(bedrock):

    #>>> import pprint
    #>>> pprint.pprint(bedrock.list_foundation_models()["modelSummaries"])

    response = bedrock.list_foundation_models()

    print(response)

def demo_invoke_model(bedrock_runtime, model_id, prompt):

    request = {
        "prompt": prompt,
        "temperature": 0.0
    }

    response = bedrock_runtime.invoke_model(modelId = model_id, body = json.dumps(request))

    response_body_json = json.loads(response["body"].read())

    print(response_body_json["completions"][0]["data"]["text"])

def demo_invoke_model_anthropic_claude(bedrock_runtime, model_id = "anthropic.claude-v1"):

    print("Call demo_invoke_model_anthropic_claude")

    prompt="""\n\nHuman: What is the diameter of the earth?
        Assistant:
    """

    request = {
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 0.5,
        "top_k": 300,
        "max_tokens_to_sample": 2048,
        "stop_sequences": []
        }

    response = bedrock_runtime.invoke_model(modelId = model_id, body = json.dumps(request))

    response_body_json = json.loads(response["body"].read())

    print(f"Answer: {response_body_json['completion']}")

def demo_embedding_calculate(bedrock_runtime, model_id, prompt):

    request = {
        "inputText": prompt
    }

    response = bedrock_runtime.invoke_model(modelId = model_id, body = json.dumps(request).encode('UTF-8'))

    response_body_json = json.loads(response["body"].read())

    embedding = response_body_json["embedding"]

    print(response_body_json.get("inputTextTokenCount"))
    #print(embedding)

    return embedding

def demo_embedding_calculate_with_cosine_similarity(bedrock_runtime, model_id):

    val1 = demo_embedding_calculate(bedrock_runtime, model_id, "埋め込みの実験のためのサンプルテキストです")
    val2 = demo_embedding_calculate(bedrock_runtime, model_id, "ぺぺろんちに食べたい")
    val3 = demo_embedding_calculate(bedrock_runtime, model_id, "This is an example text for testing embeddings.")

    print(cmn_utils.cosine_similarity(val1, val2))
    print(cmn_utils.cosine_similarity(val2, val3))
    print(cmn_utils.cosine_similarity(val1, val3))


def demo_stream_invoke_model_anthropic_claude(bedrock_runtime, model_id = "anthropic.claude-v2"):

    print("Call demo_stream_invoke_model_anthropic_claude")

    prompt="""\n\nHuman: What is the diameter of the earth?
        Assistant:
    """

    request = {
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 0.5,
        "top_k": 300,
        "max_tokens_to_sample": 2048,
        "stop_sequences": []
        }

    response = bedrock_runtime.invoke_model_with_response_stream(modelId = model_id, body = json.dumps(request))

    stream = response["body"]
    if stream:
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                print(json.loads(chunk.get("bytes").decode()))


#---

def generate_message(bedrock_runtime, model_id, system_prompt, messages, max_tokens):

    body=json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages
        }  
    )  

    
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get('body').read())
   
    return response_body


def demo_invoke_model_anthropic_claude_v3(bedrock_runtime, model_id):
    """
    Entrypoint for Anthropic Claude message example.
    """

    try:

        system_prompt = "You are a helpful assistant."
        max_tokens = 1000

        # Prompt with user turn only.
        user_message =  {"role": "user", "content": "Give me a trivia about pluto"}
        messages = [user_message]

        response = generate_message(bedrock_runtime, model_id, system_prompt, messages, max_tokens)
        print("User turn only.")
        print(json.dumps(response, indent=4))

        # Prompt with both user turn and prefilled assistant response.
        #Anthropic Claude continues by using the prefilled assistant text.
        assistant_message =  {"role": "assistant", "content": "<emoji>"}
        messages = [user_message, assistant_message]
        #response = generate_message(bedrock_runtime, model_id,system_prompt, messages, max_tokens)
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": messages
            }  
        )  

        response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
        response_body = json.loads(response.get('body').read())
        response = response_body
   
        print("User turn and prefilled assistant response.")
        print(json.dumps(response, indent=4))

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))
