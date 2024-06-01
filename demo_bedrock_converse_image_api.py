import logging
import boto3

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    system_text = "You are an economist with access to lots of data."
    input_text = "Write an article about impact of high inflation to GDP of a country."

    try:

        response = generate_conversation(bedrock_runtime, model_id, system_text, input_text)

        output_message = response['output']['message']

        print(f"Role: {output_message['role']}")

        for content in output_message['content']:
            print(f"Text: {content['text']}")

        token_usage = response['usage']
        print(f"Input tokens:  {token_usage['inputTokens']}")
        print(f"Output tokens:  {token_usage['outputTokens']}")
        print(f"Total tokens:  {token_usage['totalTokens']}")
        print(f"Stop reason: {response['stopReason']}")

    except ClientError as err:
        message = err.response['Error']['Message']
        logger.error("A client error occurred: %s", message)
        print(f"A client error occured: {message}")

    else:
        print(f"Finished generating text with model {model_id}.")
        
def generate_conversation(bedrock_runtime,
                     model_id,
                     system_text,
                     input_text):
    """
    Sends a message to a model.
    Args:
        bedrock_client: The Boto3 Bedrock runtime client.
        model_id (str): The model ID to use.
        system_text (JSON) : The system prompt.
        input text : The input message.

    Returns:
        response (JSON): The conversation that the model generated.

    """

    logger.info("Generating message with model %s", model_id)

    # Message to send.
    message = {
        "role": "user",
        "content": [{"text": input_text}]
    }
    messages = [message]
    system_prompts = [{"text" : system_text}]

    temperature = 0.5
    top_k = 200

    inference_config = {"temperature": temperature}
    additional_model_fields = {"top_k": top_k}

    response = bedrock_runtime.converse(
        modelId=model_id,
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields
    )

    return response



