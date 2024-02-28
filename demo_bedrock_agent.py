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

    prompt = ""

    demo_invoke_bedrock_agent_runtime(bedrock_agent_runtime, prompt)


def demo_invoke_bedrock_agent_runtime(bedrock_agent_runtime, prompt="What is the first US ammendment?"):

    print("Call demo_invoke_bedrock_agent_runtime")

    agent_id = config.bedrock_agent["agent_id"]
    agent_alias_id = config.bedrock_agent["agent_alias_id"]

    print(f"agent_id={agent_id} agent_alias_id={agent_alias_id}")

    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/invoke_agent.html
    response = bedrock_agent_runtime.invoke_agent(
        agentId=agent_id, 
        agentAliasId=agent_alias_id, # Use TSTALIASID as the agentAliasId to invoke the draft version of your agent.
        sessionId=str(uuid.uuid4()),  # you continue an existing session with the agent if the value you set for the idle session timeout hasn't been exceeded.
        inputText=prompt, 
        enableTrace=False, 
        endSession=False  # true to end the session with the agent.
    )
    
    print(f"Answer: {response}")

    text = ""
    event_stream = response['completion']
    for event in event_stream:        
        print(f"event={event}")
        if 'chunk' in event:
            text += event['chunk']['bytes'].decode("utf-8")
            print(event)
        #else:
            #print(event)
    print(text)
    print("End")
