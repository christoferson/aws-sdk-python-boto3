import boto3
import json
import config
import numpy as np
import cmn_utils
import uuid

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    bedrock_agent_runtime = session.client('bedrock-agent-runtime', region_name="us-east-1")

    demo_invoke_bedrock_agent_runtime(bedrock_agent_runtime)


def demo_invoke_bedrock_agent_runtime(bedrock_agent_runtime, prompt="What is the first US ammendment?"):

    print("Call demo_invoke_bedrock_agent_runtime")

    agent_id = config.bedrock_agent["agent_id"]
    agent_alias_id = config.bedrock_agent["agent_alias_id"]

    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/invoke_agent.html
    response = bedrock_agent_runtime.invoke_agent(
        agentId=agent_id, 
        agentAliasId=agent_alias_id, 
        sessionId=str(uuid.uuid4()), 
        inputText=prompt, 
        enableTrace=True, endSession=False)
    
    print(f"Answer: {response}")

    text = ""
    event_stream = response['completion']
    for event in event_stream:        
        if 'chunk' in event:
            text += event['chunk']['bytes'].decode("utf-8")
            print(text)

