import boto3
import json
import config
import os

import base64
import io
import datetime
import random
from PIL import Image
import config_stable_diffusion


def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "stability.stable-diffusion-xl"
    model_id = "stability.stable-diffusion-xl-v0"

    iconfig = config_stable_diffusion.shoe_1A
    demo_sd_generate_image(bedrock_runtime, model_id, iconfig["text"], iconfig["negative"], iconfig["style"], iconfig["scale"])


####################

def demo_sd_generate_image(bedrock_runtime, model_id, prompt, negative_prompts, style_preset="comic-book", cfg_scale = 10):

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
                [{"text": config["change_prompt"], "weight": 1.0}]
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


####################

def demo_sd_generate_image_with_reference(bedrock_runtime, model_id, reference_imgage_filename, prompt, negative_prompts, style_preset="comic-book"):

    print(f"Call demo_sd_generate_image_with_reference | style_preset={style_preset}")

    ####

    print(f"PROMPT: {prompt}")
    print(f"NEG_PROMPT: {negative_prompts}")

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_name, file_extension = os.path.splitext(reference_imgage_filename)
    INPUT_IMG_PATH = os.path.join(ROOT_DIR, "stable-diffusion/in/{}".format(reference_imgage_filename))
    print("INPUT_IMG_PATH: " + INPUT_IMG_PATH)
    OUTPUT_IMG_FILENAME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_IMG_PATH = os.path.join(ROOT_DIR, "stable-diffusion/out/{}{}".format(OUTPUT_IMG_FILENAME, file_extension))
    print("OUTPUT_IMG_PATH: " + OUTPUT_IMG_PATH)

    seed = random.randint(0, 1000)
    steps = 50
    cfg_scale = 30
    start_schedule = 0.6
    change_prompt = prompt
    negative_prompts = negative_prompts
    style_preset = style_preset
    size = 512

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
        "input_image": INPUT_IMG_PATH,
    }

    # 
    print(f"Loading Reference Image ... {INPUT_IMG_PATH}")
    input_image_b64 = image_to_base64(Image.open(INPUT_IMG_PATH).resize((size, size)))
    body = json.dumps(
        {
            "text_prompts": [{"text": config["change_prompt"], "weight": 1.0}],
            "cfg_scale": config["cfg_scale"],
            "seed": config["seed"],
            "start_schedule": config["start_schedule"],
            "steps": config["steps"],
            "style_preset": config["style_preset"],
            "init_image": input_image_b64,
        }
    )

    #print(body)

    # 
    print("Generating Image ...")
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)

    # 
    response_body = json.loads(response.get("body").read())

    print(f"Saving Image ... {OUTPUT_IMG_PATH}")
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
