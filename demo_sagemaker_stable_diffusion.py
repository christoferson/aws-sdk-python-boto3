from PIL import Image
import io
import base64
import json
import boto3
from typing import Union, Tuple
import os
import datetime

import config_stable_diffusion

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "stability.stable-diffusion-xl"
    model_id = "stability.stable-diffusion-xl-v0"

    iconfig = config_stable_diffusion.shoe_1A
    #demo_sd_generate_image(bedrock_runtime, model_id, iconfig["text"], iconfig["negative"], iconfig["style"], iconfig["scale"])
    demo_sd_generate_image(session)


####################

endpoint_name = 'jumpstart-dft-stabilityai-stable-diffusion-xl-base-1-0'

def query_endpoint_with_json_payload(session, payload):

    print(f"Call query_endpoint_with_json_payload payload={payload}")

    client = session.client('runtime.sagemaker')

    
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_IMG_FILENAME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_IMG_PATH = os.path.join(ROOT_DIR, "sagemaker/stable-diffusion/out/{}{}".format(OUTPUT_IMG_FILENAME, ".png"))
    print("OUTPUT_IMG_PATH: " + OUTPUT_IMG_PATH)

    response = client.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', 
                                      Body=json.dumps(payload).encode('utf-8'), Accept='application/json')
    
    response_dict = json.loads(response['Body'].read())

    generated_image = response_dict['generated_image']

    #with Image.open(io.BytesIO(base64.b64decode(model_response))) as image:
        #display(image)


    response_image = base64_to_image(generated_image)

    response_image.save(OUTPUT_IMG_PATH)

    return generated_image



def demo_sd_generate_image(session):

    print(f"Call demo_sd_generate_image")



    payload = {
        "text_prompts":[{"text": "jaguar in the Amazon rainforest"}],
        "width": 1024,
        "height": 1024,
        "sampler": "DPMPP2MSampler",
        "cfg_scale": 7.0,
        "steps": 50,
        "seed": 133,
        "use_refiner": True,
        "refiner_steps": 40,
        "refiner_strength": 0.2
    }

    query_endpoint_with_json_payload(session, payload=payload)

    print("END")





### Utilities

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_image(base64_str):
    return Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
