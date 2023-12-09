import boto3
import config
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

def run_demo(session):

    host = config.opensearch["domain_url"]
    region = config.opensearch["region"]
    credentials = session.get_credentials()
    awsauth = AWSV4SignerAuth(credentials, region, service="es")

    search = OpenSearch(
        hosts = [{'host': host, 'port': 443}],
        http_auth = awsauth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        timeout=300
    )

    demo_opensearch_basic(search)


def demo_opensearch_basic(search):

    print("Run demo_opensearch_basic")

    document = {
        "title": "Moneyball",
        "director": "Bennett Miller",
        "year": "2011"
    }

    # Try Put Index
    search.index(index="movies", id="5", body=document)

    # Try Search Index
    print(search.get(index="movies", id="5"))  

    print("end")