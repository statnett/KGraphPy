"""Sorting JSON-LD for serialization."""

import json
from rdflib import URIRef, Literal, Namespace
from urllib.request import urlopen, Request
from typing import Optional, Any

DEFAULT_CONTEXT_LINK = "https://raw.githack.com/Sveino/Inst4CIM-KG/develop/rdf-improved/cim-context-new.jsonld"

def load_json_from_url(url):
    # Pretend to be a browser — some corporate proxies require this
    req = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urlopen(req) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        data = response.read().decode(charset)
        return json.loads(data)


def extract_datatype_map(context):
    dt_map = {}

    context = context.get("@context", context)  # Handle case where context is wrapped in @context
    prefixes = {k: v for k, v in context.items() if isinstance(v, str)}
    
    for term, definition in context.items():
        if isinstance(definition, dict) and "@type" in definition:
            # Expand prefix:local into full URI
            if ":" in term:
                prefix, local = term.split(":", 1)
                if prefix in prefixes:
                    predicate_uri = prefixes[prefix] + local
                else:
                    continue
            else:
                continue

            # Expand datatype prefix
            dtype = definition["@type"]
            if ":" in dtype:
                dprefix, dlocal = dtype.split(":", 1)
                if dprefix in prefixes:
                    dtype_uri = prefixes[dprefix] + dlocal
                else:
                    dtype_uri = dtype
            else:
                dtype_uri = dtype

            dt_map[predicate_uri] = dtype_uri

    return dt_map


def enrich_graph_datatypes(graph, dt_map):
    triples_to_add = []
    triples_to_remove = []
    
    for s, p, o in graph:
        p_str = str(p)
        
        if p_str in dt_map and isinstance(o, Literal):
            expected_dt = URIRef(dt_map[p_str])

            # Case 1: literal has no datatype
            if o.datatype is None:
                triples_to_remove.append((s, p, o))
                triples_to_add.append((s, p, Literal(o.value, datatype=expected_dt)))

            # Case 2: literal has wrong datatype
            elif o.datatype != expected_dt:
                print(o, o.datatype, "should be", expected_dt)
                triples_to_remove.append((s, p, o))
                triples_to_add.append((s, p, Literal(o.value, datatype=expected_dt)))

    for t in triples_to_remove:
        graph.remove(t)
    for t in triples_to_add:
        graph.add(t)


def sort_subjects(nodes: list[dict[str, Any]], priority_subject: Optional[URIRef|str] = None) -> list[dict[str, Any]]:
    """Sort a list of JSON-LD nodes by @id (i.e. triple subject), with an optional priority subject first.
    
    The rest of the nodes are sorted alphanumerically by their @id. 

    Parameters:
        nodes (list[dict[str, Any]]): List of JSON-LD node objects to sort.
        priority_subject (URIRef|str, optional): If provided, the node with this @id will be placed first.
        
    Returns:
        list[dict[str, Any]]: The sorted list of node objects.
    """
    priority_id = str(priority_subject) if priority_subject else None

    if priority_id is None:
        return sorted(nodes, key=lambda n: n.get("@id", ""))

    return sorted(
        nodes,
        key=lambda n: (0 if n.get("@id") == priority_id else 1, n.get("@id", ""))
    )

def sort_predicates(node: dict[str, Any]) -> dict[str, Any]:
    """Sort the predicates of a JSON-LD node with @id and @type first and then the rest in alphabetical order.
    
    Other @ keys are sorted after @id and @type, but before non-@ keys, in alphabetical order.

    Parameters:
        node (dict[str, Any]): A JSON-LD node represented as a dictionary.

    Returns:
        dict[str, Any]: A new dictionary with the same key-value pairs as the input node but with keys sorted in the specified order.
    """
    def sort_key(k: str) -> tuple[int, str]:
        if k == "@id":
            return (0, "")
        if k == "@type":
            return (1, "")
        if k.startswith("@"):
            return (2, k)
        return (3, k)

    sorted_keys = sorted(node.keys(), key=sort_key) # Make a sorted list of the keys based on the defined order.
    return {k: node[k] for k in sorted_keys}    # Sort the dictionary according to the list

def reorder_jsonld(raw_jsonld: str, priority_subject: Optional[URIRef|str] = None) -> str:
    data = json.loads(raw_jsonld)

    # Determine where nodes live
    if isinstance(data, dict) and "@graph" in data:
        nodes = data["@graph"]
        container = data
        key = "@graph"
    elif isinstance(data, list):
        nodes = data
        container = None
        key = None
    else:
        return json.dumps(data, indent=2, ensure_ascii=False)

    # 1. Sort subjects
    try:
        nodes_sorted = sort_subjects(nodes, priority_subject)
        print(nodes_sorted)
    # 2. Sort predicates inside each subject
        nodes_sorted = [sort_predicates(node) for node in nodes_sorted]
        print(nodes_sorted)
    except AttributeError:  # If nodes have unexpected structure return unsorted
        return json.dumps(data, indent=2, ensure_ascii=False)

    # Put nodes back
    if container is not None:
        container[key] = nodes_sorted
        result = container
    else:
        result = nodes_sorted

    print(result)
    return json.dumps(result, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("JSON-LD sorting utility.")
    