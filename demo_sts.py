import boto3
import config

def run_demo(session):

    sts = session.client("sts")

    response = sts.get_caller_identity()

    print(response['Account'])