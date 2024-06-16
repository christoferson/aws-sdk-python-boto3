from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
import botocore
import time
import json
import config

#https://docs.aws.amazon.com/ja_jp/opensearch-service/latest/developerguide/serverless-sdk.html

def run_demo(session):

    client = session.client('opensearchserverless')
    service = 'aoss'
    region = 'us-east-1'
    credentials = session.get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

    collection_url = config.opensearch_hr.get("collection_url")
    index_name = config.opensearch_hr.get("index_name")
    search_data(collection_url, awsauth, index_name)


def createCollection(client):
    """Creates a collection"""
    try:
        response = client.create_collection(
            name='tv-sitcoms',
            type='SEARCH'
        )
        return(response)
    except botocore.exceptions.ClientError as error:
        if error.response['Error']['Code'] == 'ConflictException':
            print('[ConflictException] A collection with this name already exists. Try another name.')
        else:
            raise error

def waitForCollectionCreation(client):
    """Waits for the collection to become active"""
    response = client.batch_get_collection(
        names=['tv-sitcoms'])
    # Periodically check collection status
    while (response['collectionDetails'][0]['status']) == 'CREATING':
        print('Creating collection...')
        time.sleep(30)
        response = client.batch_get_collection(
            names=['tv-sitcoms'])
    print('\nCollection successfully created:')
    print(response["collectionDetails"])
    # Extract the collection endpoint from the response
    host = (response['collectionDetails'][0]['collectionEndpoint'])
    final_host = host.replace("https://", "")
    indexData(final_host)


def indexData(host, awsauth):
    """Create an index and add some sample data"""
    # Build the OpenSearch client
    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300
    )
    # It can take up to a minute for data access rules to be enforced
    time.sleep(45)

    # Create index
    response = client.indices.create('sitcoms-eighties')
    print('\nCreating index:')
    print(response)

    # Add a document to the index.
    response = client.index(
        index='sitcoms-eighties',
        body={
            'title': 'Seinfeld',
            'creator': 'Larry David',
            'year': 1989
        },
        id='1',
    )
    print('\nDocument added:')
    print(response)



def search_data(host, awsauth, index_name):
    """Create an index and add some sample data"""
    # Build the OpenSearch client
    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300
    )


    query ={
        "query":{
            "wildcard":{
                "text": {
                    "value": "*Ferry*",
                    "case_insensitive": "false"
                }
            }
        }
    }
    result = client.search(index=index_name, body=json.dumps(query))
    print(result)

    result_json = json.dumps(result, indent=2)
    print(result_json)

    if "hits" in result:
        hits = result["hits"]
        if "hits" in hits:
            source = hits["hits"][0]["_source"]
            if "embeddings" in source:
                source["embeddings"] = []
                result_json = json.dumps(result, indent=2)
                print(result_json) #"embeddings"
    


#def main():
#    createEncryptionPolicy(client)
#    createNetworkPolicy(client)
#    createAccessPolicy(client)
#    createCollection(client)
#    waitForCollectionCreation(client)

