import boto3
import json
import config
import uuid
import traceback
import logging

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    bedrock_agent_runtime = session.client('bedrock-agent-runtime', region_name="us-east-1")

    session_id = str(uuid.uuid4())

    # Does not answer anything outside KB or ActionGroup
    prompt = "foo"
    
    demo_invoke_bedrock_agent_runtime(bedrock_agent_runtime, prompt, session_id)

    demo_close_bedrock_agent_session(bedrock_agent_runtime, session_id)

        

def demo_invoke_bedrock_agent_runtime(bedrock_agent_runtime, prompt, session_id):

    print(f"Call demo_invoke_bedrock_agent_runtime session_id={session_id} prompt={prompt}")

    agent_id = config.bedrock_agent["agent_id"]
    agent_alias_id = config.bedrock_agent["agent_alias_id"]

    print(f"agent_id={agent_id} agent_alias_id={agent_alias_id}")

    try:

        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/invoke_agent.html
        response = bedrock_agent_runtime.invoke_agent(
            agentId=agent_id, 
            agentAliasId=agent_alias_id, # Use TSTALIASID as the agentAliasId to invoke the draft version of your agent.
            sessionId=session_id,  # you continue an existing session with the agent if the value you set for the idle session timeout hasn't been exceeded.
            inputText=prompt, 
            enableTrace=True, 
            endSession=False  # true to end the session with the agent.
        )
        
        print(f"Answer: {response}")

        answer = ""
        sources = []
        sources_text = ""
        generated_text = ""
        event_stream = response['completion']
        for event in event_stream:        
            print(f"type={type(event)} event={event}")
            if 'chunk' in event:
                chunk = event['chunk']
                answer += chunk['bytes'].decode("utf-8")
                print(event)
                if 'attribution' in chunk: # If a knowledge base was queried, an attribution object with a list of citations is returned.
                    attribution = chunk['attribution']
                    for citation in attribution['citations']:
                        # generatedResponsePart object contains the text generated by the model based on the information from the text in the retrievedReferences
                        generated_text = citation['generatedResponsePart']['textResponsePart']['text']
                        # retrievedReferences object contains the exact text in the chunk relevant to the query alongside the S3 location of the data source
                        for reference in citation['retrievedReferences']: 
                            reference_text = reference['content']['text']
                            sources_text += reference_text
                            location = reference['location']
                            if location['type'] == 's3':
                                sources.append(location['s3Location']['uri'])
        
        print("*********************************************************")
        print(f"Prompt: {prompt}")
        print("*********************************************************")
        print(f"Answer: {answer}")
        print("*********************************************************")
        print(f"Sources: {sources}")
        print("*********************************************************")
        print(f"Sources Text: {sources_text}")
        print("*********************************************************")
        print(f"Generated Text: {generated_text}")
        print("*********************************************************")

    except Exception as e:
        logging.error(traceback.format_exc())

    print("End")



def demo_close_bedrock_agent_session(bedrock_agent_runtime, session_id):

    print(f"Call demo_close_bedrock_agent_session session_id={session_id}")

    agent_id = config.bedrock_agent["agent_id"]
    agent_alias_id = config.bedrock_agent["agent_alias_id"]

    print(f"agent_id={agent_id} agent_alias_id={agent_alias_id}")

    try:

        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/invoke_agent.html
        response = bedrock_agent_runtime.invoke_agent(
            agentId=agent_id, 
            agentAliasId=agent_alias_id, # Use TSTALIASID as the agentAliasId to invoke the draft version of your agent.
            sessionId=session_id,  # you continue an existing session with the agent if the value you set for the idle session timeout hasn't been exceeded.
            inputText="-", 
            #enableTrace=False, 
            endSession=True  # true to end the session with the agent.
        )

    except Exception as e:
        logging.error(traceback.format_exc())

    print("End")
