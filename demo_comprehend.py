import boto3
import config
import json

def run_demo(session):

    comprehend = session.client("comprehend")

    text = """
    Hello Zhang Wei, I am John. Your AnyCompany Financial Services, LLC credit card account 1111-0000-1111-0008 has a minimum payment of $24.53 that is due by July 31st. Based on your autopay settings, we will withdraw your payment on the due date from your bank account number XXXXXX1111 with the routing number XXXXX0000. 
    Customer feedback for Sunshine Spa, 123 Main St, Anywhere. Send comments to Alice at sunspa@mail.com. 
    I enjoyed visiting the spa. It was very comfortable but it was also very expensive. The amenities were ok but the service made the spa a great experience.
    """

    run_demo_detect_entities(comprehend, text)


def run_demo_detect_entities(comprehend, text):

    response = comprehend.detect_entities(
        Text=text, LanguageCode="en"
    )

    print(json.dumps(response, sort_keys=True, indent=4))

    for label in response['Entities']:
        text_name = label['Text']
        text_confidence = str(round(float(label['Score']), 2))
        text_type = label['Type']
        text_offset_start = label['BeginOffset']
        text_offset_end = label['EndOffset']
        print (f'{text_name} {text_confidence} {text_type}')