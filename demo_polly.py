import boto3
import config

def run_demo(session):

    rekognition = session.client("rekognition")

    #run_demo_detect_labels(rekognition)

    run_demo_text_in_image(rekognition)

    #run_demo_detect_moderation_labels(rekognition)


def run_demo_detect_labels(rekognition):

    bucket = config.rekognition['bucket']
    key = config.rekognition['object5']

    response = rekognition.detect_labels(
        Image={'S3Object':{'Bucket': bucket, 'Name': key }},
        MinConfidence=55
    )

    print('Detected labels for ' + key)
    for label in response['Labels']:
        text_name = label['Name']
        text_confidence = str(round(float(label['Confidence']), 2))
        print (f'{text_name} {text_confidence}')

def run_demo_text_in_image(rekognition):

    bucket = config.rekognition['bucket']
    key = config.rekognition['object5']

    response = rekognition.detect_text(
        Image={'S3Object':{'Bucket': bucket, 'Name': key }},
        Filters={'WordFilter': {'MinConfidence': 80}}
    )

    text_detections = response['TextDetections']

    print(f'Detected {len(text_detections)} labels for {key}')
    for label in text_detections:
        text_id = str(label['Id'])
        text_value = label['DetectedText']
        text_confidence = str(round(float(label['Confidence']), 2))
        text_type = label['Type']

        print (f'{text_id} {text_value} {text_confidence} {text_type}')


def run_demo_detect_moderation_labels(rekognition):

    bucket = config.rekognition['bucket']
    key = config.rekognition['object5']

    response = rekognition.detect_moderation_labels(
        Image={'S3Object':{'Bucket': bucket, 'Name': key }},
        MinConfidence=55
    )

    moderation_labels = response['ModerationLabels']

    print(f'Detected {len(moderation_labels)} ModerationLabels for {key}')
    for label in moderation_labels:
        text_name = label['Name']
        text_confidence = str(round(float(label['Confidence']), 2))
        print (f'{text_name} {text_confidence}')