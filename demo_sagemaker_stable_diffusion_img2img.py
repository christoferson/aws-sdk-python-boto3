from PIL import Image
import io
import base64
import json
import os
import datetime
import config
import random

import config_stable_diffusion

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

    #cmn_cleanup_delete_endpoints(sagemaker)

    reference_image = "input.png"

    print(f"sagemaker_endpoint_name={sagemaker_endpoint_name}")
    #iconfig = config_stable_diffusion.shoe_1A
    demo_sagemaker_sd_generate_image_2_image(sagemaker_runtime, sagemaker_endpoint_name, reference_image)


####################

def cmn_list_models(sagemaker):
    print(f"cmn_list_models")
    #result = sagemaker.list_endpoints(StatusEquals='InService') # StatusEquals='OutOfService'|'Creating'|'Updating'|'SystemUpdating'|'RollingBack'|'InService'|'Deleting'|'Failed'
    result = sagemaker.list_models(MaxResults = 3, NameContains="stable-d") # StatusEquals='OutOfService'|'Creating'|'Updating'|'SystemUpdating'|'RollingBack'|'InService'|'Deleting'|'Failed'
    print(result)
    for Model in result['Models']:
        print(Model)
    
def cmn_create_deploy_endpoints(sagemaker):
    print(f"cmn_create_deploy_endpoints")
    print("TODO")


def cmn_cleanup_delete_endpoints(sagemaker):
    print(f"cmn_cleanup_delete_endpoints")
    #result = sagemaker.list_endpoints(StatusEquals='InService') # StatusEquals='OutOfService'|'Creating'|'Updating'|'SystemUpdating'|'RollingBack'|'InService'|'Deleting'|'Failed'
    result = sagemaker.list_endpoints() # StatusEquals='OutOfService'|'Creating'|'Updating'|'SystemUpdating'|'RollingBack'|'InService'|'Deleting'|'Failed'
    print(result)
    for Endpoint in result['Endpoints']:
        endpoint_name = Endpoint['EndpointName']
        print(f"Deleting Endpoint: {endpoint_name}")
        result = sagemaker.delete_endpoint(EndpointName=endpoint_name)
        print(result)
        print()
    print("Cleanup End")

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

    response_image = base64_to_image(generated_image_base64)

    response_image.save(OUTPUT_IMG_PATH)

    with open("{}.json".format(OUTPUT_IMG_PATH), "w") as f:
        json.dump(payload, f, ensure_ascii = False)

    return generated_image_base64



def demo_sagemaker_sd_generate_image_2_image(session, endpoint_name, reference_imgage_filename):

    print(f"Call demo_sagemaker_sd_generate_image_2_image")

    iconfig = config_stable_diffusion.shoe_1A
    #text = "jaguar in the Amazon rainforest"
    text = iconfig["text"]
    negative_prompts = iconfig["negative"]

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_extension = ".png"

    file_name, file_extension = os.path.splitext(reference_imgage_filename)
    INPUT_IMG_PATH = os.path.join(ROOT_DIR, "stable-diffusion/v1/in/{}".format(reference_imgage_filename))
    print("INPUT_IMG_PATH: " + INPUT_IMG_PATH)

    print(f"Loading Reference Image ... {INPUT_IMG_PATH}")
    size = 1024
    input_image_b64 = image_to_base64(Image.open(INPUT_IMG_PATH).resize((size, size)))

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
        "negative": negative_prompts,
        "init_image": input_image_b64
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
