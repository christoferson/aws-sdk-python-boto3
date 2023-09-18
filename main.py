import os
import boto3
import demo_s3

session = boto3.Session(
    aws_access_key_id='foo',
    aws_secret_access_key='bar',
    region_name='us-east-1',
)

demo_s3.run_demo(session)

print("End")