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
    
    iconfig = config_amz_titan_image.shoe_2A
    demo_titan_v1_generate_image(bedrock_runtime, model_id, iconfig["text"], iconfig["negative"], iconfig["style"], iconfig["scale"])


####################

####################

def demo_titan_v1_generate_image(bedrock_runtime, model_id, prompt, negative_prompts, style_preset="comic-book", cfg_scale = 10):

    print(f"Call demo_titan_v1_generate_image | style_preset={style_preset} | cfg_scale={cfg_scale}")

    print(f"PROMPT: {prompt}")
    print(f"NEG_PROMPT: {negative_prompts}")

    ####

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_extension = ".png"
    OUTPUT_IMG_PATH = os.path.join(ROOT_DIR, "stable-diffusion/out/{}{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), file_extension))
    print("OUTPUT_IMG_PATH: " + OUTPUT_IMG_PATH)

    seed = random.randint(0, 4294967295)
    steps = 30 #50
    #cfg_scale = cfg_scale
    start_schedule = 0.6
    change_prompt = prompt
    #negative_prompts = negative_prompts
    #style_preset = style_preset
    size = 1024

    # 
    config = {
        "filename": OUTPUT_IMG_PATH,
        "seed": seed,
        "change_prompt": change_prompt,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "start_schedule": start_schedule,
        "style_preset": style_preset,
        "size": size,
        "negative_prompts": negative_prompts
    }

    # 
    body = json.dumps(
        {
            "text_prompts": (
                [{"text": config["change_prompt"]}]
                #[{"text": config["change_prompt"], "weight": 1.0}]
                #+ [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
            ),
            "cfg_scale": config["cfg_scale"],
            #"clip_guidance_preset"
            #"height": "1024",
            #"width": "1024",
            "seed": config["seed"],
            #"start_schedule": config["start_schedule"],
            "steps": config["steps"],
            #"style_preset": config["style_preset"]
        }
    )

    print(body)

    #print(body)

    # 
    print("Generating Image ...")
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)

    # 
    response_body = json.loads(response.get("body").read())
    response_image = base64_to_image(response_body["artifacts"][0].get("base64"))

    # 
    response_image.save(OUTPUT_IMG_PATH)
    # 
    with open("{}.json".format(OUTPUT_IMG_PATH), "w") as f:
        json.dump(config, f, ensure_ascii = False)

    print("Complete")


def demo_sd_generate_image_multi_prompt(bedrock_runtime, model_id, prompt, negative_prompts, style_preset="comic-book", cfg_scale = 10):

    print(f"Call demo_sd_generate_image | style_preset={style_preset} | cfg_scale={cfg_scale}")

    print(f"PROMPT: {prompt}")
    print(f"NEG_PROMPT: {negative_prompts}")

    ####

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_extension = ".png"
    OUTPUT_IMG_PATH = os.path.join(ROOT_DIR, "stable-diffusion/out/{}{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), file_extension))
    print("OUTPUT_IMG_PATH: " + OUTPUT_IMG_PATH)

    seed = random.randint(0, 8000000)
    steps = 75 #50
    #cfg_scale = cfg_scale
    start_schedule = 0.6
    change_prompt = prompt
    #negative_prompts = negative_prompts
    #style_preset = style_preset
    size = 1024

    # 
    config = {
        "filename": OUTPUT_IMG_PATH,
        "seed": seed,
        "change_prompt": change_prompt,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "start_schedule": start_schedule,
        "style_preset": style_preset,
        "size": size,
        "negative_prompts": negative_prompts
    }

    # 
    body = json.dumps(
        {
            "text_prompts": (
                [{"text": config["change_prompt"], "weight": 0.5}]
                + [{"text": "Casio GShock themed Sneakers", "weight": 0.5}]
                + [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
            ),
            "cfg_scale": config["cfg_scale"],
            "seed": config["seed"],
            #"start_schedule": config["start_schedule"],
            "steps": config["steps"],
            "style_preset": config["style_preset"]
        }
    )

    print(body)

    #print(body)

    # 
    print("Generating Image ...")
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)

    # 
    response_body = json.loads(response.get("body").read())
    response_image = base64_to_image(response_body["artifacts"][0].get("base64"))

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
