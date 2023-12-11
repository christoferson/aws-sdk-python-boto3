import boto3
import config
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import csv
import json

def run_demo(session):

    host = config.opensearch["domain_url"] # Domain Url without https://
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

    index_name = "movies";
    query = {
        "query": {
            "match":{"Film":"Twilight"}
        }
    }
    query2 = {
        "query": {
            "match":{"Lead Studio":"Warner"}
        },
        "sort": {
            "Year": {
                "order": "desc"
            }
        }
    }
    
    query3 = {
        "query": {
            "match":{"Lead Studio":"Warner"},
            "range":{"Audience score %": { "gte" : 55, "lte" : 70}}
        },
        "sort": {
            "Audience score %": {
                "order": "desc"
            }
        }
    }

    query4 = {
        "query": {
            "bool" : {
                "must" : [
                    { "match":{"Lead Studio":"Warner"}, },
                    { "range":{"Audience score %": { "gte" : 55, "lte" : 70}} }
                ]
            }
        },
        "sort": {
            "Audience score %": {
                "order": "desc"
            }
        }
    }

    #demo_opensearch_basic(search, index_name)
    #demo_opensearch_populate(search, index_name)
    demo_opensearch_search(search, index_name, query4)

def demo_opensearch_basic(search, index_name):

    print(f"Run demo_opensearch_basic index_name={index_name}")

    document = {
        "title": "Moneyball",
        "director": "Bennett Miller",
        "year": "2011"
    }

    # Try Put Index
    search.index(index=index_name, id="5", body=document)

    # Try Search Index
    print(search.get(index=index_name, id="5"))  

    search.indices.delete(index=index_name)

    print("end")


index_settings = {
  "mappings": {
    "properties": {
      "Film":{
        "type": "text",
        "analyzer": "kuromoji"
      },
      "Genre":{
        "type": "text",
        "analyzer": "kuromoji"
      },
      "Lead Studio":{
        "type": "text",
        "analyzer": "kuromoji"
      },
      "Year":{
        "type": "integer"
      },
      "Audience score %":{
        "type": "float"
      },
      "Rotten Tomatoes %":{
        "type": "float"
      },
      "Profitability":{
        "type": "double"
      },
      "Worldwide Gross":{
        "type": "text"
      },
    }
  }
}

def demo_opensearch_populate(search, index_name):

    print(f"Run demo_opensearch_populate index_name={index_name}")

    search.indices.delete(index=index_name)

    if not search.indices.exists(index=index_name):
        search.indices.create(index=index_name,body=index_settings)
        #search.indices.create(index=index_name)

    id = 1
    with open("data/movies.csv","r",encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for data in reader:
            search.index(
                index = index_name,
                body = data,
                id = id
            )
            id += 1

    print("end")


def demo_opensearch_search(search, index_name, query):

    print(f"Run demo_opensearch_search index_name={index_name}")

    if not search.indices.exists(index=index_name):
        print(f"Index {index_name} does not exist.")
        return

    result = search.search(index=index_name, body=json.dumps(query))
    #print(result)
    print(f"Hit Count: {result['hits']['total']['value']}")
    for idx, hit in enumerate(result['hits']['hits']):
        print(f"{idx+1} {hit['_source']}")

    print("end")