import boto3
import config

def run_demo(session):

    rekognition = session.client("rekognition")

    run_demo_detect_labels(rekognition)


def run_demo_detect_labels(rekognition):

    bucket = config.rekognition['bucket']
    key = config.rekognition['object2']

    response = rekognition.detect_labels(Image={'S3Object':{'Bucket': bucket, 'Name': key }})

    print('Detected labels for ' + key)
    for label in response['Labels']:
        print (label['Name'] + ' : ' + str(label['Confidence']))
