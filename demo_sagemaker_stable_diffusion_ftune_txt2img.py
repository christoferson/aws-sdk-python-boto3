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



def run_demo(session):

    sagemaker_region_name = config.sagemaker["region_name"]
    sagemaker_endpoint_name = config.sagemaker["endpoint_finetune_name"]

    sagemaker_runtime = session.client('runtime.sagemaker')
    sagemaker = session.client('sagemaker', region_name=sagemaker_region_name)

    print(f"sagemaker_endpoint_name={sagemaker_endpoint_name}")
    demo_sagemaker_sd_generate_text_2_image(sagemaker_runtime, sagemaker_endpoint_name)


####################

def cmn_sagemaker_sd_generate_image(sagemaker_runtime, endpoint_name, payload):

    print(f"Call cmn_sagemaker_sd_generate_image payload={payload}")

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_IMG_FILENAME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_IMG_PATH = os.path.join(ROOT_DIR, "sagemaker/stable-diffusion-ft/out/{}{}".format(OUTPUT_IMG_FILENAME, ".png"))
    print("OUTPUT_IMG_PATH: " + OUTPUT_IMG_PATH)

    response = sagemaker_runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', 
                                      Body=json.dumps(payload).encode('utf-8'), Accept='application/json;jpeg')
    
    generated_images, prompt = parse_response_multiple_images(response)
    #for img in generated_images:
    #    display_image(img, prompt)

    #response_dict = json.loads(response['Body'].read())
    #print(response_dict)

    decoded_images = [decode_base64_image(image) for image in generated_images]

    for idx, decoded_image in enumerate(decoded_images):
        decoded_image.save(OUTPUT_IMG_PATH)

    with open("{}.json".format(OUTPUT_IMG_PATH), "w") as f:
        json.dump(payload, f, ensure_ascii = False)

    return None

def demo_sagemaker_sd_generate_text_2_image(session, endpoint_name):

    print(f"Call demo_sagemaker_sd_generate_text_2_image")

    #iconfig = config_stable_diffusion.shoe_1A
    iconfig = config_stable_diffusion.sagemaker_stable_diffusion_v21_riobugger
    text = iconfig["text"]
    negative_prompts = iconfig["negative"]

    payload = {
        "prompt": text,
        "width": 800, #736, #1152, \u0027OutOfMemoryError\u0027
        "height": 576, #896,
        "seed": random.randint(0, 4294967295), #133,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "negative_prompt": iconfig["negative_text"],
    }

    
    cmn_sagemaker_sd_generate_image(session, endpoint_name, payload)

    print("END")


### Utilities
    
def decode_base64_image(image_string):
  base64_image = base64.b64decode(image_string)
  buffer = io.BytesIO(base64_image)
  return Image.open(buffer)

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_image(base64_str):
    return Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))

import matplotlib.pyplot as plt
import numpy as np

def parse_response(query_response):
    response_dict = json.loads(query_response['Body'].read())
    return response_dict['generated_image'], response_dict['prompt']

def display_image(img, prmpt):
    plt.figure(figsize=(12,12))
    plt.imshow(np.array(img))
    plt.axis('off')
    plt.title(prmpt)
    plt.show()



def parse_response_multiple_images(query_response):
    response_dict = json.loads(query_response['Body'].read())
    return response_dict['generated_images'], response_dict['prompt']   