import boto3
import config
from botocore.exceptions import ClientError

def run_demo(session):

    dynamodb = session.resource("dynamodb")

    run_demo_table_info(dynamodb)



def run_demo_table_info(dynamodb):

    print("*** run_demo_table_info\n")

    table_name = config.dynamodb["table_name"]

    try:

        table = dynamodb.Table(table_name)

        print("creation_date_time: " + str(table.creation_date_time))

    except ClientError as err:
        if err.response['Error']['Code'] == 'ResourceNotFoundException':
            exists = False
        else:
            print("Couldn't check for existence of %s. Here's why: %s: %s", table_name,
                err.response['Error']['Code'], err.response['Error']['Message'])
            raise
    else:
        print("\n")
