from PIL import Image
import io
import base64
import json
import os
import datetime
import config
import random

import config_stable_diffusion
import demo_sagemaker_stable_diffusion_lib

# SageMaker Studio
# SageMaker JumpStart
# SageMaker XL 1.0 (OpenSource)
# Deploy , Do not select Run In Notebook
# Endpoint wil be ready in 20-30 minutes

def run_demo(session):

    sagemaker_region_name = config.sagemaker["region_name"]
    sagemaker_endpoint_name = config.sagemaker["endpoint_name"]

    sagemaker_runtime = session.client('runtime.sagemaker')
    sagemaker = session.client('sagemaker', region_name=sagemaker_region_name)

    print(f"sagemaker_endpoint_name={sagemaker_endpoint_name}")
    demo_sagemaker_sd_generate_image(sagemaker_runtime, sagemaker_endpoint_name)


####################

def cmn_sagemaker_sd_generate_image(sagemaker_runtime, endpoint_name, payload):

    print(f"Call query_endpoint_with_json_payload payload={payload}")

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_IMG_FILENAME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_IMG_PATH = os.path.join(ROOT_DIR, "sagemaker/stable-diffusion/out/{}{}".format(OUTPUT_IMG_FILENAME, ".png"))
    print("OUTPUT_IMG_PATH: " + OUTPUT_IMG_PATH)

    response = sagemaker_runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', 
                                      Body=json.dumps(payload).encode('utf-8'), Accept='application/json')
    
    response_dict = json.loads(response['Body'].read())

    generated_image_base64 = response_dict['generated_image']

    response_image = demo_sagemaker_stable_diffusion_lib.base64_to_image(generated_image_base64)

    response_image.save(OUTPUT_IMG_PATH)

    with open("{}.json".format(OUTPUT_IMG_PATH), "w") as f:
        json.dump(payload, f, ensure_ascii = False)

    return generated_image_base64

def demo_sagemaker_sd_generate_image(session, endpoint_name):

    print(f"Call demo_sagemaker_sd_generate_image")

    iconfig = config_stable_diffusion.shoe_1A
    #text = "jaguar in the Amazon rainforest"
    text = iconfig["text"]
    negative_prompts = iconfig["negative"]

    # 1024x1024
    payload = {
        "text_prompts":[{"text": text, "weight": 1}],
        "width": 1152,
        "height": 896,
        "sampler": "DPMPP2MSampler",
        "cfg_scale": 7.0,
        "steps": 50,
        "seed": random.randint(0, 4294967295), #133,
        "use_refiner": True,
        "refiner_steps": 40,
        "refiner_strength": 0.2,
        #"style_preset": "origami",
        "negative": negative_prompts
    }

    cmn_sagemaker_sd_generate_image(session, endpoint_name, payload)

    print("END")


### Utilities

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_image(base64_str):
    return Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
