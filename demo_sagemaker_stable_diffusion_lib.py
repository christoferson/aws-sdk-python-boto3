import io
import base64
from PIL import Image

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

def cmn_list_models(sagemaker):
    print(f"cmn_list_models")
    #result = sagemaker.list_endpoints(StatusEquals='InService') # StatusEquals='OutOfService'|'Creating'|'Updating'|'SystemUpdating'|'RollingBack'|'InService'|'Deleting'|'Failed'
    result = sagemaker.list_models(MaxResults = 3, NameContains="stable-d") # StatusEquals='OutOfService'|'Creating'|'Updating'|'SystemUpdating'|'RollingBack'|'InService'|'Deleting'|'Failed'
    print(result)
    for Model in result['Models']:
        print(Model)

### Image Utilities

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_image(base64_str):
    return Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
