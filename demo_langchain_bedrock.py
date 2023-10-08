from langchain.llms import Bedrock
from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage

# ValueError: Error raised by bedrock service: An error occurred (AccessDeniedException) when calling the InvokeModel operation: Your account is not authorized to invoke this API operation.
# NotImplementedError: Provider ai21 model does not support chat.

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    #model_id = "ai21.j2-mid-v1"
    model_id = "anthropic.claude-v1"
    model_kwargs = { "temperature": 0.0 }
    prompt = "What is the diameter of Earth?"

    #demo_invoke_model_predict(bedrock_runtime, model_id, model_kwargs, prompt)

    demo_invoke_model_chat(bedrock_runtime, model_id, model_kwargs, prompt)

def demo_invoke_model_predict(bedrock_runtime, model_id, model_kwargs, prompt):

    llm = Bedrock(
        client = bedrock_runtime,
        model_id = model_id,
        model_kwargs = model_kwargs,
    )

    output = llm.predict(prompt)

    print(output)

def demo_invoke_model_chat(bedrock_runtime, model_id, model_kwargs, prompt):

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = model_id,
        model_kwargs = model_kwargs,
    )

    result = llm([
        HumanMessage(content = prompt)
    ])

    print(result.content)