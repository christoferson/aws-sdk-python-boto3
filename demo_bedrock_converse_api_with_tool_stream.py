# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shows how to use tools with the Converse API and the Cohere Command R model.
"""

import logging
import boto3
import json
import copy

import demo_bedrock_converse_api_with_tool_lib
from demo_bedrock_converse_api_with_tool_lib import CalculatorBedrockConverseTool


from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class StationNotFoundError(Exception):
    """Raised when a radio station isn't found."""
    pass


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_demo(session):

    
    #bedrock = session.client('bedrock')

    
    #print(f"********************** definition={tool.definition}")

    #demo_tool_stream(session, "What is the most popular song on WZPZ?")
    #demo_tool_stream(session, "What is 87 + 5?")
    #demo_tool_stream(session, "What is 87.7383298383+5.378475943548?")
    demo_tool_stream(session, "What is 3.14 raised to the power of 0.12345?")
    #demo_tool_stream(session, "What is the diameter of the sun?")


def demo_tool_stream(session, input_text):


    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    calculator_tool = CalculatorBedrockConverseTool()

    tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": "top_song",
                    "description": "Get the most popular song played on a radio station.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "sign": {
                                    "type": "string",
                                    "description": "The call sign for the radio station for which you want the most popular song. Example calls signs are WZPZ, and WKRP."
                                }
                            },
                            "required": [
                                "sign"
                            ]
                        }
                    }
                }
            },
            
            calculator_tool.definition
            
        ]
    }
    

    try:
        print(f"Question: {input_text}")

        message_user = { "role": "user", "content": [{"text": input_text}] }

        messages = [message_user]

        tool_invocation = generate_text(bedrock_runtime, model_id, tool_config, messages)
        print(tool_invocation)

        if tool_invocation['tool_name'] != None:

            tool_name = tool_invocation['tool_name']
            tool_args_json = json.loads(tool_invocation['tool_arguments'])

            tool_request_message = {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": tool_invocation['tool_use_id'],
                            "name": tool_invocation['tool_name'],
                            "input": json.loads(tool_invocation['tool_arguments'])
                        }
                    }
                ]
            }

            if tool_name == "top_song":
                song = ""
                artist = ""
                if tool_name == "top_song":
                    song, artist = get_top_song(tool_args_json['sign'])

                tool_result_message = {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                    "toolUseId": tool_invocation['tool_use_id'],
                                    "content": [{"json": {"song": song, "artist": artist}}]
                                    #"status": 'error',
                                }

                        }
                    ]
                }
            if calculator_tool.matches(tool_name):
                expr_result = calculator_tool.invoke(tool_args_json['expression'])

                tool_result_message = {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                    "toolUseId": tool_invocation['tool_use_id'],
                                    "content": [{"json": {"expr_result": expr_result}}]
                                    #"status": 'error',
                                }

                        }
                    ]
                }

            messages = [message_user, tool_request_message, tool_result_message]
            tool_invocation = generate_text(bedrock_runtime, model_id, tool_config, messages)
            print(tool_invocation)

    except ClientError as err:
        message = err.response['Error']['Message']
        logger.error("A client error occurred: %s", message)
        print(f"A client error occured: {message}")

    else:
        print(
            f"Finished generating text with model {model_id}.")


def generate_text(bedrock_client, model_id, tool_config, messages):
    """Generates text using the supplied Amazon Bedrock model. If necessary,
    the function handles tool use requests and sends the result to the model.
    Args:
        bedrock_client: The Boto3 Bedrock runtime client.
        model_id (str): The Amazon Bedrock model ID.
        tool_config (dict): The tool configuration.
        input_text (str): The input text.
    Returns:
        Nothing.
    """

    tool_invocation = {
        "tool_name": None
    }

    logger.info("Generating text with model %s", model_id)

    system_prompts = [{"text" : """You are a question answering bot. Do not use any of the provided tools if you have the answer readily available.
                       """}]
    inference_config = {"temperature": 0.0}
    additional_model_fields = {"top_k": 1.0}

    response = bedrock_client.converse_stream(
        modelId=model_id,
        messages=messages,
        toolConfig=tool_config,
        system=system_prompts,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields
    )

    stream = response.get('stream')
    tool_input = ""
    if stream:
        for event in stream:

            if 'messageStart' in event:
                print(f"\nRole: {event['messageStart']['role']}")

            if 'contentBlockStart' in event:
                content_block_start = event['contentBlockStart']
                print(content_block_start)
                if 'start' in content_block_start:
                    content_block_start_start = content_block_start['start']
                    if 'toolUse' in content_block_start_start:
                        content_block_tool_use = content_block_start_start['toolUse']
                        tool_use_id = content_block_tool_use['toolUseId']
                        tool_use_name = content_block_tool_use['name']
                        print(f"tool_use_id={tool_use_id} tool_use_name={tool_use_name}")
                        tool_invocation['tool_name'] = tool_use_name
                        tool_invocation['tool_use_id'] = tool_use_id

            if 'contentBlockDelta' in event:
                content_delta = event['contentBlockDelta']['delta']
                if 'text' in content_delta:
                    print(content_delta['text'], end="")
                if 'toolUse' in content_delta:
                    content_delta_tool_input = content_delta['toolUse']['input']
                    tool_input += content_delta_tool_input
                    

            if 'messageStop' in event:
                message_stop_reason = event['messageStop']['stopReason']
                print(f"\nStop reason: {message_stop_reason}")
                if "tool_use" == message_stop_reason:
                    tool_input_json = json.loads(tool_input)
                    print(tool_input_json) #{'sign': 'WZPZ'}
                    tool_invocation['tool_arguments'] = tool_input
                    pass
                else:
                    #'end_turn'|'max_tokens'|'stop_sequence'|'content_filtered'
                    pass

            if 'metadata' in event:
                metadata = event['metadata']
                if 'usage' in metadata:
                    print("\nToken usage")
                    print(f"Input tokens: {metadata['usage']['inputTokens']}")
                    print(
                        f":Output tokens: {metadata['usage']['outputTokens']}")
                    print(f":Total tokens: {metadata['usage']['totalTokens']}")
                if 'metrics' in event['metadata']:
                    print(
                        f"Latency: {metadata['metrics']['latencyMs']} milliseconds")

    return tool_invocation




def get_top_song(call_sign):
    """Returns the most popular song for the requested station.
    Args:
        call_sign (str): The call sign for the station for which you want
        the most popular song.

    Returns:
        response (json): The most popular song and artist.
    """

    song = ""
    artist = ""
    if call_sign == 'WZPZ':
        song = "Elemental Hotel"
        artist = "8 Storey Hike"

    else:
        raise StationNotFoundError(f"Station {call_sign} not found.")

    return song, artist
