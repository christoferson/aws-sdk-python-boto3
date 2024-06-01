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

    model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
    demo_stream_invoke_model_anthropic_claude_v3(bedrock_runtime, model_id, "Give me a trivia about pluto")

def demo_stream_invoke_model_anthropic_claude_v3(bedrock_runtime, model_id, input_text):

    try:

        system_prompt = "You are a helpful assistant."
        max_tokens = 1000

        # Prompt with user turn only.
        user_message =  {"role": "user", "content": input_text}
        messages = [user_message]

        #response = generate_message(bedrock_runtime, model_id, system_prompt, messages, max_tokens)
        #print("User turn only.")
        #print(json.dumps(response, indent=4))

        # Prompt with both user turn and prefilled assistant response.
        #Anthropic Claude continues by using the prefilled assistant text.
        #assistant_message =  {"role": "assistant", "content": "<emoji>"}
        #messages = [user_message, assistant_message]
        #response = generate_message(bedrock_runtime, model_id,system_prompt, messages, max_tokens)
        body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user", # Valid values are user and assistant. 
                        "content": [
                            {"type": "text", "text": input_text}, #Valid values are image and text. 
                            #{"type": "image", 
                                #"source": {
                                    #"type": "base64",
                                    #"media_type": "image/jpeg", 
                                    #"data": encoded_string.decode('utf-8')
                                #}
                            #}
                        ]
                    }
                ],
                "temperature": 0.1,
                "top_p": 0.999,
                "top_k": 0, #100,000,000
                #"stop_sequences": [""]
            })
        print(body)
        response = bedrock_runtime.invoke_model_with_response_stream(body=body, modelId=model_id)
        #response_body = json.loads(response.get('body').read())
        #response = response_body
        print(response)
   

        for event in response.get("body"):
            chunk = json.loads(event["chunk"]["bytes"])

            #message_start {'type': 'message_start', 'message': {'id': 'msg_01FfyBMnL4zddyMP5YNkYjhi', 'type': 'message', 'role': 'assistant', 'content': [], 'model': 'claude-3-sonnet-28k-20240229', 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 21, 'output_tokens': 1}}}

            if chunk['type'] == 'message_start':
                print(f"\Input Tokens: {chunk['message']['usage']['input_tokens']}")

            elif chunk['type'] == 'message_delta':
                print(f"\nStop reason: {chunk['delta']['stop_reason']}")
                print(f"Stop sequence: {chunk['delta']['stop_sequence']}")

                #print(f"Input tokens: {chunk['usage']['input_tokens']}")
                print(f"Output tokens: {chunk['usage']['output_tokens']}")

            elif chunk['type'] == 'content_block_delta':
                if chunk['delta']['type'] == 'text_delta':
                    print(chunk['delta']['text'], end="")

            # message_stop {'type': 'message_stop', 'amazon-bedrock-invocationMetrics': {'inputTokenCount': 21, 'outputTokenCount': 212, 'invocationLatency': 4296, 'firstByteLatency': 742}}
            elif chunk['type'] == 'message_stop':
                #if chunk['delta']['type'] == 'text_delta':
                metrics = chunk['amazon-bedrock-invocationMetrics']
                print(f"inputTokenCount: {metrics['inputTokenCount']}")
                print(f"outputTokenCount: {metrics['outputTokenCount']}")
                print(f"invocationLatency: {metrics['invocationLatency']}")
                print(f"firstByteLatency: {metrics['firstByteLatency']}")

            
            elif chunk['type'] == 'content_block_start':
                pass

            else:
                print(f"\n{chunk['type']} {chunk}\n")

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))
