import boto3
import config

def run_demo(session):

    iam = session.client('iam')

    response = iam.list_server_certificates()

    print(response)

    print(response["ServerCertificateMetadataList"])