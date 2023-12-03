import boto3
import json
import config
import os

import base64
import io
import datetime
import random
from PIL import Image

import config_amz_titan_image


def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    # Amazon Titan Image Generator 1.0 - Bedrock

    model_id = "amazon.titan-image-generator-v1:0"
    
    iconfig = config_amz_titan_image.shoe_2B
    demo_titan_v1_generate_image(bedrock_runtime, model_id, iconfig["text"].strip(), iconfig["negative_text"].strip(), iconfig["quality"], iconfig["style"], iconfig["scale"])


####################

def demo_titan_v1_generate_image(bedrock_runtime, model_id, 
                                 prompt : str, 
                                 negative_prompts : str, 
                                 quality : str ="standard",
                                 style_preset="comic-book", 
                                 cfg_scale : float = 8.0):

    print(f"Call demo_titan_v1_generate_image | style_preset={style_preset} | cfg_scale={cfg_scale}")

    print(f"PROMPT: {prompt}")
    print(f"NEG_PROMPT: {negative_prompts}")

    ####

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_extension = ".png"
    OUTPUT_IMG_PATH = os.path.join(ROOT_DIR, "titan-image/out/{}{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), file_extension))
    print("OUTPUT_IMG_PATH: " + OUTPUT_IMG_PATH)

    seed = random.randint(0, 214783647)
    steps = 30 #50
    start_schedule = 0.6
    change_prompt = prompt
    size = 1024

    # 
    config = {
        "filename": OUTPUT_IMG_PATH,
        "seed": seed,
        "change_prompt": change_prompt,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "size": size,
        "quality": quality,
        "negative_prompts": negative_prompts
    }

    # 
    body = json.dumps(
        {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": config["change_prompt"],
                "negativeText": negative_prompts,
            },
            "imageGenerationConfig": {
                "cfgScale": config["cfg_scale"], #8, #Range: 1.0 (exclusive) to 10.0
                "seed": config["seed"], #Range: 0 to 214783647
                "quality": quality, #Options: standard/premium
                "width": 1024,
                "height": 1024,
                "numberOfImages": 1 #Range: 1 to 5
            }
        }
    )

    print(json.dumps(body, indent=2))

    # 
    print("Generating Image ...")
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)

    # 
    response_body = json.loads(response.get("body").read())
    print(response_body.keys())
    response_image = base64_to_image(response_body["images"][0])
    #response_image = base64_to_image(response_body["images"][0].get("base64"))

    # 
    response_image.save(OUTPUT_IMG_PATH)
    # 
    with open("{}.json".format(OUTPUT_IMG_PATH), "w") as f:
        json.dump(config, f, ensure_ascii = False)

    print("Complete")


### Utilities

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_image(base64_str):
    return Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
