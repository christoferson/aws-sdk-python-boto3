import boto3
import json
import config
import numpy as np
import cmn_utils

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    #demo_list_foundation_models(bedrock)

    #demo_invoke_model(bedrock_runtime, "ai21.j2-mid-v1", "What is the diameter of Earth?")

    model_id = "amazon.titan-embed-text-v1"

    val1 = demo_embedding_calculate(bedrock_runtime, model_id, "埋め込みの実験のためのサンプルテキストです")
    val2 = demo_embedding_calculate(bedrock_runtime, model_id, "ぺぺろんちに食べたい")
    val3 = demo_embedding_calculate(bedrock_runtime, model_id, "This is an example text for testing embeddings.")

    print(cosine_similarity(val1, val2))
    print(cosine_similarity(val2, val3))
    print(cosine_similarity(val1, val3))

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

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))