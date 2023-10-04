import boto3
import json
import config

def run_demo(session):

    bedrock = session.client('bedrock')

    kwargs = {
        "modelId": "",
        "contentType": "application/json",
        "accept": "*/*",
        "body": ""
    }

    response = bedrock.list_foundation_models()

    print(response)