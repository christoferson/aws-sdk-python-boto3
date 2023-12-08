# aws-sdk-python-boto3
AWS SDK Python Boto3

[quickstart](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html)
[documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

###### Install the latest Boto3 release via PIP:
pip install boto3
pip install boto3 --upgrade
pip show boto3

###### Install LangChain
pip install langchain

pip install unstructured

pip install qdrant-client

#pip install pypdf==3.14.0

[image-lib](https://note.nkmk.me/en/python-pillow-basic/)
pip install Pillow

###### Output Dependencies
pip freeze > requirements.txt


###### SDK vs LangChain Model Invocation Code

SDK

```(python)

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    request = {
        "prompt": prompt,
        "temperature": 0.0
    }

    response = bedrock_runtime.invoke_model(modelId = model_id, body = json.dumps(request))

    response_body_json = json.loads(response["body"].read())

    print(response_body_json["completions"][0]["data"]["text"])
```

LangChain

```(python)

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    llm = Bedrock(
        client = bedrock_runtime,
        model_id = model_id,
        model_kwargs = {"temperature": 0.0},
    )

    output = llm.predict(prompt)

    print(output)

```


### Links

- [invoke_model](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/invoke_model.html)

- [blip2-git](https://github.com/aws-samples/amazon-sagemaker-genai-content-moderation/blob/main/blip2-sagemaker.ipynb)

- [blip2](https://aws.amazon.com/jp/blogs/machine-learning/build-a-generative-ai-based-content-moderation-solution-on-amazon-sagemaker-jumpstart/)

- https://github.com/AUTOMATIC1111/stable-diffusion-webui

- https://stability.ai/news/stability-ai-makes-its-stable-diffusion-models-available-on-amazons-new-bedrock-service


### Links - Titan Image Generation

- https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html