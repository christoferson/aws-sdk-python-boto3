import os
import boto3
import config
import demo_s3
import demo_kendra
import demo_dynamodb
import demo_sts
import demo_rekognition
import demo_iam

#os.environ['AWS_DEFAULT_REGION'] = 'eu-west-1'

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
demo_iam.run_demo(session)

print("End")