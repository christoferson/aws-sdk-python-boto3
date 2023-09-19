
import boto3
import pprint
import config

#Facetable: Indicates that the field can be used to create search facets, a count of results for each value in the field. The default is false .
#Searchable: Determines whether the field is used in the search. If the Searchable field is true, you can use relevance tuning to manually tune how Amazon Kendra weights the field in the search. The default is true for string fields and false for number and date fields.
#Displayable: Determines whether the field is returned in the query response. The default is true.
#Sortable: Determines whether the field can be used to sort the results of a query. If you specify sorting on a field that does not have Sortable set to true, Amazon Kendra returns an exception. The default is false.

#Relevance Tuning
#In order to allow an attribute to be used to boost a document you need to mark it as searchable.

# Access Control
# TODO: Add ACLs to the S3 folders

def run_demo(session):

    kendra = session.client("kendra")

    index_id = config.kendra['indexid']
    query = "What service has 11 nines of durability?"

    response = kendra.query(QueryText = query, IndexId = index_id,
        AttributeFilter = {
            'AndAllFilters': 
                [ 
                    {"EqualsTo": {"Key": "_category","Value": {"StringValue": "Best Practices"}}},
                ]
        },
        SortingConfiguration=
        {
            'DocumentAttributeKey': '_created_at',
            'SortOrder': 'ASC'
        }
    )

    #pp = pprint.PrettyPrinter(indent=1)
    #pp.pprint(response)
    print(str(response["TotalNumberOfResults"]) + "\n")
    print(str(response["FacetResults"]) + "\n")

    print("\nSearch results for query: " + query + "\n")        

    for query_result in response["ResultItems"]:

        print("-------------------")
        print("Type: " + str(query_result["Type"]))
            
        if query_result["Type"]=="ANSWER" or query_result["Type"]=="QUESTION_ANSWER":
            answer_text = query_result["DocumentExcerpt"]["Text"]
            print(answer_text)

        if query_result["Type"]=="DOCUMENT":
            if "DocumentTitle" in query_result:
                document_title = query_result["DocumentTitle"]["Text"]
                print("Title: " + document_title)
            document_text = query_result["DocumentExcerpt"]["Text"]
            print(document_text)

        print("------------------\n\n")