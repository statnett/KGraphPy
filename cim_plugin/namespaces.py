"""The namespaces used which are not default to rdflib."""

from rdflib import Namespace, Node, URIRef, Graph
from rdflib.namespace import NamespaceManager


MD = Namespace("http://iec.ch/TC57/61970-552/ModelDescription/1#") 

CIM = Namespace("https://cim.ucaiug.io/ns#")

EU = Namespace("https://cim.ucaiug.io/ns/eu#")

MODEL = Namespace("https://model4powersystem.no/")



def collect_specific_namespaces(triples: list[tuple[Node, Node, Node]], namespace_manager: NamespaceManager) -> dict[str, URIRef]:
    """Collect namespaces used in the a list of triples.

    Parameters:
        triples (list[tuple[Node, Node, Node]]): The list of triples. Can be extracted from a Graph object.
        namespace_manager [NamespaceManager]: The namespace object to collect the namespaces from.

    Returns:
        dict[str, URIRef]: The namespaces as prefix: namespace.
    """
    used_uris = set()
    for s, p, o in triples:
        for term in (s, p, o):
            if isinstance(term, URIRef):
                used_uris.add(str(term))

    # Sort namespaces longest-first to ensure specific ones match first. Prevents overlapping namespaces from being lost.
    known = sorted(namespace_manager.namespaces(), key=lambda item: len(str(item[1])), reverse=True)
    
    used = {}

    for uri in used_uris:
        for prefix, ns_uri in known:
            ns_str = str(ns_uri)
            if uri.startswith(ns_str):
                used[prefix] = ns_uri
                break
    
    return used


def update_namespace_in_triples(graph: Graph, old_namespace: str, new_namespace: str) -> None:
    """Update an old namespace with a new namespace for every triple in a graph.
    
    Parameters:
        graph (Graph): The graph where namespaces are to be replaced.
        old_namespace (str): The namespace to be replaced.
        new_namespace (str): The namespace to replace the old with.

    Raises:
        ValueError: If old_namespace is an empty string. 
                    Prevents the new namespace being inserted between every character.
    """
    if not old_namespace:
        raise ValueError("old_namespace cannot be an empty string")
    
    to_add = [] 
    to_remove = [] 
    
    for s, p, o in graph: 
        new_s = URIRef(str(s).replace(old_namespace, new_namespace)) if isinstance(s, URIRef) and str(s).startswith(old_namespace) else s 
        new_p = URIRef(str(p).replace(old_namespace, new_namespace)) if isinstance(p, URIRef) and str(p).startswith(old_namespace) else p 
        new_o = URIRef(str(o).replace(old_namespace, new_namespace)) if isinstance(o, URIRef) and str(o).startswith(old_namespace) else o 
        
        if (new_s, new_p, new_o) != (s, p, o): 
            to_remove.append((s, p, o)) 
            to_add.append((new_s, new_p, new_o)) 
            
    for triple in to_remove: 
        graph.remove(triple) 
        
    for triple in to_add: 
        graph.add(triple)

if __name__ == "__main__":
    print("Namespaces for cim graphs.")