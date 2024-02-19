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
import config_stable_diffusion_xl10

# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-1-0-image-image-mask.html

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "stability.stable-diffusion-xl"
    model_id = "stability.stable-diffusion-xl-v0"

    # Stable Diffusion XL 1.0 - Bedrock

    model_id = "stability.stable-diffusion-xl-v1"
    iconfig = config_stable_diffusion_xl10.shoe_2D_v3 #shoe_2D #shoe_2D
    #demo_sd_generate_text_to_image_xl_v1(bedrock_runtime, model_id, iconfig["text"].strip(), iconfig["negative"], iconfig["style"], iconfig["scale"])

    #iconfig = config_stable_diffusion_xl10.prompt_me_0_male #shoe_2D
    #reference_image = "input.png"
    #reference_image = "me.jpg"
    #reference_image = "DSC_2175.jpg"
    #reference_image = "IMG_1123.jpeg"
    #demo_sd_generate_text_to_image_xl_v1(bedrock_runtime, model_id, iconfig["text"].strip(), iconfig["negative"], iconfig["style"], iconfig["scale"]);

    #shoe_2D_IMG_IMG, shoe_2D_IMG_IMG_PURPLE, shoe_2D_IMG_IMG_ORANGE
    iconfig = config_stable_diffusion_xl10.shoe_2D_IMG_IMG_ORANGE
    reference_image = "1011B703_401_SB_FR_GLB.png"
    demo_sd_generate_image_to_image_xl_v1(bedrock_runtime, model_id, reference_image, iconfig["text"].strip(), iconfig["negative"], iconfig["style"], iconfig["scale"], iconfig["image_strength"])


    #demo_sd_generate_image_to_image_inpaint_xl_v1(bedrock_runtime, model_id, reference_image, "Add Sun", iconfig["negative"], iconfig["style"], iconfig["scale"])

####################

# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-1-0-text-image.html
def demo_sd_generate_text_to_image_xl_v1(bedrock_runtime, model_id, prompt, negative_prompts, style_preset="comic-book", cfg_scale = 10):

    print(f"Call demo_sd_generate_text_to_image_xl_v1 | style_preset={style_preset} | cfg_scale={cfg_scale}")

    print(f"PROMPT: {prompt}")
    print(f"NEG_PROMPT: {negative_prompts}")

    ####

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_extension = ".png"
    OUTPUT_IMG_PATH = os.path.join(ROOT_DIR, "stable-diffusion/v1/out/{}{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), file_extension))
    print("OUTPUT_IMG_PATH: " + OUTPUT_IMG_PATH)

    seed = random.randint(0, 4294967295)
    steps = 50 #150 #30 #50
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
                #[{"text": config["change_prompt"]}]
                #[{"text": config["change_prompt"], "weight": 1.0}]
                #+ [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
                [{"text": config["change_prompt"], "weight": 1.0}]
                + [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
            ),
            "cfg_scale": config["cfg_scale"], # Determines how much the final image portrays the prompt. Use a lower number to increase randomness in the generation. 0-35,7
            #"clip_guidance_preset"
            #"height": "1024",
            #"width": "1024",
            "seed": config["seed"], # The seed determines the initial noise setting.0-4294967295,0
            #"start_schedule": config["start_schedule"],
            "steps": config["steps"], # Generation step determines how many times the image is sampled. 10-50,50
            "style_preset": config["style_preset"],
            "samples": 1,
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

def demo_sd_generate_image_to_image_xl_v1(bedrock_runtime, model_id, reference_imgage_filename, prompt, negative_prompts, style_preset="comic-book", cfg_scale = 10, image_strength= 0.65):

    print(f"Call demo_sd_generate_image_to_image_xl_v1 | style_preset={style_preset} | cfg_scale={cfg_scale} | reference_imgage_filename={reference_imgage_filename} | image_strength={image_strength}")

    print(f"PROMPT: {prompt}")
    print(f"NEG_PROMPT: {negative_prompts}")

    ####

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_extension = ".png"

    file_name, file_extension = os.path.splitext(reference_imgage_filename)
    INPUT_IMG_PATH = os.path.join(ROOT_DIR, "stable-diffusion/v1/in/{}".format(reference_imgage_filename))
    print("INPUT_IMG_PATH: " + INPUT_IMG_PATH)

    OUTPUT_IMG_PATH = os.path.join(ROOT_DIR, "stable-diffusion/v1/out/{}{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), file_extension))
    print("OUTPUT_IMG_PATH: " + OUTPUT_IMG_PATH)

    seed = random.randint(0, 4294967295)
    steps = 50 #150 #30 #50
    #cfg_scale = cfg_scale
    start_schedule = 0.6
    change_prompt = prompt
    #negative_prompts = negative_prompts
    #style_preset = style_preset
    size = 1024

    print(f"Loading Reference Image ... {INPUT_IMG_PATH}")
    input_image_b64 = image_to_base64(Image.open(INPUT_IMG_PATH).resize((size, size)))

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
        "negative_prompts": negative_prompts,
        "input_image": INPUT_IMG_PATH,
        "init_image_mode": "IMAGE_STRENGTH",
        "image_strength": image_strength, #0.95,
    }

    # 
    body = json.dumps(
        {
            "text_prompts": (
                #[{"text": config["change_prompt"]}]
                #[{"text": config["change_prompt"], "weight": 1.0}]
                #+ [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]

                [{"text": config["change_prompt"], "weight": 1.0}]
                + [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
            ),
            "init_image_mode": config["init_image_mode"],
            "image_strength": config["image_strength"],
            "cfg_scale": config["cfg_scale"],
            #"clip_guidance_preset"
            #"height": "1024",
            #"width": "1024",
            "seed": config["seed"],
            #"start_schedule": config["start_schedule"],
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
    response_image = base64_to_image(response_body["artifacts"][0].get("base64"))

    # 
    response_image.save(OUTPUT_IMG_PATH)
    # 
    with open("{}.json".format(OUTPUT_IMG_PATH), "w") as f:
        json.dump(config, f, ensure_ascii = False)

    print("Complete")



####################

def demo_sd_generate_image_to_image_inpaint_xl_v1(bedrock_runtime, model_id, reference_imgage_filename, prompt, negative_prompts, style_preset="comic-book", cfg_scale = 10):

    print(f"Call demo_sd_generate_image_to_image_inpaint_xl_v1 | style_preset={style_preset} | cfg_scale={cfg_scale} | reference_imgage_filename={reference_imgage_filename}")

    print(f"PROMPT: {prompt}")
    print(f"NEG_PROMPT: {negative_prompts}")

    ####

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_extension = ".png"

    file_name, file_extension = os.path.splitext(reference_imgage_filename)
    INPUT_IMG_PATH = os.path.join(ROOT_DIR, "stable-diffusion/v1/in/{}".format(reference_imgage_filename))
    print("INPUT_IMG_PATH: " + INPUT_IMG_PATH)

    INPUT_IMG_MASK_PATH = os.path.join(ROOT_DIR, "stable-diffusion/v1/in/{}".format("black_mask.png"))
    print("INPUT_IMG_MASK_PATH: " + INPUT_IMG_MASK_PATH)

    OUTPUT_IMG_PATH = os.path.join(ROOT_DIR, "stable-diffusion/v1/out/{}{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), file_extension))
    print("OUTPUT_IMG_PATH: " + OUTPUT_IMG_PATH)

    seed = random.randint(0, 4294967295)
    steps = 50 #150 #30 #50
    #cfg_scale = cfg_scale
    start_schedule = 0.6
    change_prompt = prompt
    #negative_prompts = negative_prompts
    #style_preset = style_preset
    size = 1024

    print(f"Loading Reference Image ... {INPUT_IMG_PATH}")
    input_image_b64 = image_to_base64(Image.open(INPUT_IMG_PATH).resize((size, size)))

    print(f"Loading Mask Image ... {INPUT_IMG_MASK_PATH}")
    input_image_mask_b64 = image_to_base64(Image.open(INPUT_IMG_MASK_PATH).resize((size, size)))

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
        "negative_prompts": negative_prompts,
        "input_image": INPUT_IMG_PATH
    }

    # 
    body = json.dumps(
        {
            "text_prompts": (
                #[{"text": config["change_prompt"]}]
                #[{"text": config["change_prompt"], "weight": 1.0}]
                #+ [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]

                [{"text": config["change_prompt"], "weight": 1.0}]
                + [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
            ),
            "cfg_scale": config["cfg_scale"],
            #"clip_guidance_preset"
            #"height": "1024",
            #"width": "1024",
            "seed": config["seed"],
            #"start_schedule": config["start_schedule"],
            "steps": config["steps"],
            #"style_preset": config["style_preset"]
            "init_image": input_image_b64,
            "mask_source": "MASK_IMAGE_BLACK",
            "mask_image": input_image_mask_b64,
        }
    )

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
