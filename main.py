import os
import boto3
import config
import demo_s3
import demo_kendra
import demo_dynamodb
import demo_sts
import demo_rekognition
import demo_iam
import demo_bedrock

import demo_langchain_bedrock
import demo_langchain_bedrock_embeddings

import demo_bedrock_stable_diffusion
import demo_bedrock_amz_titan_image
import demo_sagemaker_stable_diffusion
import demo_sagemaker_stable_diffusion_txt2img
import demo_sagemaker_stable_diffusion_img2img
import demo_sagemaker_stable_diffusion_ftune_txt2img

import demo_bedrock_stable_diffusion_xl_v1
import demo_bedrock_agent
import demo_comprehend

import demo_sagemaker

import demo_opensearch

#os.environ['AWS_DEFAULT_REGION'] = 'eu-west-1'

print(f"Python Version: {boto3.__version__}")

session = boto3.Session(
    aws_access_key_id=config.aws["aws_access_key_id"],
    aws_secret_access_key=config.aws["aws_secret_access_key"],
    region_name=config.aws["region_name"],
)

#demo_s3.run_demo(session)
#demo_kendra.run_demo(session)
#demo_dynamodb.run_demo(session)
#demo_sts.run_demo(session)
#demo_rekognition.run_demo(session)
#demo_iam.run_demo(session)
#demo_bedrock.run_demo(session)

#demo_langchain_bedrock.run_demo(session)
#demo_langchain_bedrock_embeddings.run_demo(session)

# STABLE DIFFUSION
#demo_bedrock_stable_diffusion.run_demo(session)
#demo_sagemaker_stable_diffusion.run_demo(session)
#demo_sagemaker_stable_diffusion_txt2img.run_demo(session)
#demo_sagemaker_stable_diffusion_ftune_txt2img.run_demo(session)
#demo_sagemaker_stable_diffusion_img2img.run_demo(session)
demo_bedrock_stable_diffusion_xl_v1.run_demo(session)

#demo_bedrock_agent.run_demo(session)



# Bedrock Image Generator
#demo_bedrock_amz_titan_image.run_demo(session)

#demo_sagemaker.run_demo(session)

#demo_opensearch.run_demo(session)

#demo_comprehend.run_demo(session)

print("End")