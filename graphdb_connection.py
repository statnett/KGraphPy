"""Shows how connect to a GraphDB client, with examples.

It requires user and password handling, which should be done via a key vault or encrypted environment variables. 

"""

import cim_plugin
import os
from rdflib import URIRef, Graph, Variable
from rdflib.contrib.graphdb.client import GraphDBClient
from cim_plugin.processor import CIMProcessor
from cim_plugin.utilities import load_graphs_from_cimxml

USER: str = os.getenv("GRAPHDB_USER", "christha")
PASSWORD: str = os.getenv("GRAPHDB_PASSWORD", "wrong_password")
url: str = "https://rndpsvc.statnett.no/graphdb/"

def get_token() -> str:
    """Token makes authentication easier."""
    with GraphDBClient(url) as client:
        user = client.login(USER, PASSWORD)

        print(f"Logged in as: {user.username}")
        # print(f"Authorities: {user.authorities}")
        token = user.token
        return token

def get_graph() -> CIMProcessor:
    """A quick helper to load a lot of data."""
    file="../Nordic44/instances/Grid/cimxml/Nordic44-HV_EQ.xml"
    ds = load_graphs_from_cimxml([file])
    return ds[0]

def upload_to_graphdb_with_text(client: GraphDBClient, repo_name: str) -> None:
    """Example of how to upload data using text/turtle."""
    data = """
        PREFIX ex: <http://example.org/>
        ex:Monday a ex:Day .
    """
    repo = client.repositories.get(repo_name)
    repo.upload(data=data, content_type="text/turtle")

def update_graphdb_with_sparql(client: GraphDBClient, repo_name: str) -> None:
    """Example of how to update the repository using SPARQL Update."""
    query = "INSERT DATA { <http://example.org/Sunday> <http://example.org/isin> <http://example.org/TheWeekend> }"
    repo = client.repositories.get(repo_name)
    repo.update(query)

def add_named_graph(client: GraphDBClient, repo_name: str, graph: Graph, identifier: URIRef|str) -> None:
    """Example of how to add a named graph."""
    repo = client.repositories.get(repo_name)
    repo.graphs.add(graph_name=identifier, data=graph)
    # To add namespaces:
    for prefix, ns in graph.namespace_manager.store.namespaces():
        repo.namespaces.set(prefix, ns)

def query_graphdb(client: GraphDBClient, repo_name: str) -> None:
    """Example of how to query the repository."""
    query = "SELECT * WHERE { <http://example.org/Thursday> ?p ?o } LIMIT 10"
    repo = client.repositories.get(repo_name)
    
    result = repo.query(query)
    
    if result.type == "ASK":
        print(result.askAnswer)
    elif result.type in ("CONSTRUCT", "DESCRIBE"):
        print(result.graph)
    else:
        for row in result.bindings:
            print(f"Predicate: {row[Variable('p')]}, Object: {row[Variable('o')]}")
        # or you can print the entire result as a list of bindings: 
        # print(result.bindings)


def main() -> None:
    token = get_token()
    with GraphDBClient(url, auth=token) as token_client:
        repo = token_client.repositories.get("test_graph_ch")

        print(repo.size())
        print(repo.graph_names())
        print(repo.namespaces.get("cim"))
        ds = repo.get(subj=URIRef("http://example.org/Thursday"))   # How to get triples from any graph in the repo
        print(list(ds))
        query_graphdb(token_client, "test_graph_ch")
        


if __name__ == "__main__":
    main()