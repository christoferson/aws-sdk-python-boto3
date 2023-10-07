import boto3
import json
import config

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    #demo_list_foundation_models(bedrock)

    demo_invoke_model(bedrock_runtime, "ai21.j2-mid-v1", "What is the diameter of Earth?")

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
