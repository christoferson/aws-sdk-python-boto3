from langchain.llms import Bedrock

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "ai21.j2-mid-v1"

    #llm = Bedrock(
        #credentials_profile_name = "profile",
        #model_id = model_id
    #)

    llm = Bedrock(
        client = bedrock_runtime,
        model_id = model_id,
        model_kwargs = {"temperature": 0.0},
    )

    demo_invoke_model_predict(llm, model_id, "What is the diameter of Earth?")

def demo_invoke_model_predict(llm, model_id, prompt):

    output = llm.predict(prompt)
    print(output)