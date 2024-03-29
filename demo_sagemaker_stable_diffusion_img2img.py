from PIL import Image
import json
import os
import datetime
import config
import random

import config_stable_diffusion
import demo_sagemaker_stable_diffusion_lib

# Code to invoke Stable Diffusion XL 1.0 OpenSource Sagemaker Endpoint

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

    reference_image = "jaguar.png"

    print(f"sagemaker_endpoint_name={sagemaker_endpoint_name}")
    demo_sagemaker_sd_generate_image_2_image(sagemaker_runtime, sagemaker_endpoint_name, reference_image)


####################

def cmn_sagemaker_sd_generate_image_2_image(sagemaker_runtime, endpoint_name, payload, reference_image_filename):

    print(f"Call cmn_sagemaker_sd_generate_image_2_image payload={payload}")

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_IMG_FILENAME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_IMG_PATH = os.path.join(ROOT_DIR, "sagemaker/stable-diffusion/out/{}{}".format(OUTPUT_IMG_FILENAME, ".png"))
    print("OUTPUT_IMG_PATH: " + OUTPUT_IMG_PATH)

    #
    file_extension = ".png"
    file_name, file_extension = os.path.splitext(reference_image_filename)
    INPUT_IMG_PATH = os.path.join(ROOT_DIR, "stable-diffusion/v1/in/{}".format(reference_image_filename))
    print("INPUT_IMG_PATH: " + INPUT_IMG_PATH)

    print(f"Loading Reference Image ... {INPUT_IMG_PATH}")
    size = 1024
    input_image_b64 = demo_sagemaker_stable_diffusion_lib.image_to_base64(Image.open(INPUT_IMG_PATH).resize((size, size)))

    payload["init_image"] = input_image_b64 #"init_image": input_image_b64

    response = sagemaker_runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', 
                                      Body=json.dumps(payload).encode('utf-8'), Accept='application/json')
    
    response_dict = json.loads(response['Body'].read())

    generated_image_base64 = response_dict['generated_image']

    response_image = demo_sagemaker_stable_diffusion_lib.base64_to_image(generated_image_base64)

    response_image.save(OUTPUT_IMG_PATH)

    with open("{}.json".format(OUTPUT_IMG_PATH), "w") as f:
        json.dump(payload, f, ensure_ascii = False)

    return generated_image_base64



def demo_sagemaker_sd_generate_image_2_image(session, endpoint_name, reference_image_filename):

    print(f"Call demo_sagemaker_sd_generate_image_2_image. reference_imgage_filename={reference_image_filename}")

    iconfig = config_stable_diffusion.sagemaker_stable_diffusion_1_os
    text = iconfig["text"]
    negative_prompts = iconfig["negative"]

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

    cmn_sagemaker_sd_generate_image_2_image(session, endpoint_name, payload, reference_image_filename)

    print("END")

