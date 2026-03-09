"""The namespaces used which are not default to rdflib."""

from rdflib import Namespace, Node, URIRef
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
