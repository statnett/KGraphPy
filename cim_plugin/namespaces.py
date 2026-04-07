"""The namespaces used which are not default to rdflib."""

from rdflib import Namespace, Node, URIRef, Graph
from rdflib.namespace import NamespaceManager, DefinedNamespace, DCAT

# Default namespaces
CIM = Namespace("https://cim.ucaiug.io/ns#")    # Sometimes CGMES_CIM is used instead

EU = Namespace("https://cim.ucaiug.io/ns/eu#")  # Sometimes CGMES_EU is used instead

MODEL = Namespace("https://model4powersystem.no/")

RDFG = Namespace("http://www.w3.org/2004/03/trix/rdfg-1/")

JSONLD = Namespace("https://www.w3.org/ns/json-ld#")

EUMD = Namespace("https://cim4.eu/ns/Metadata-European#")   # Sometimes MD is used instead

# CGMES exception namespaces
CGMES_CIM = Namespace("http://iec.ch/TC57/CIM100#")

CGMES_EU = Namespace("http://iec.ch/TC57/CIM100-EuropeanExtension/1/0#")


# Custom namespaces for this project
class DCAT_CIM(DCAT):   # This is exactly the same as dcat, but with a few extra cim specific properties added.
    version: URIRef  # The version indicator (name or identifier) of a resource. Info taken from https://www.w3.org/TR/vocab-dcat/#Property:resource_version
    isVersionOf: URIRef  # This property is intended for relating a non-versioned or abstract resource to several versioned resources, e.g., snapshots. Info taken from https://eepublicdownloads.entsoe.eu/clean-documents/CIM_documents/Grid_Model_CIM/MetadataDatasetDistributionSpecification_v2-4-0.pdf.


class _MDModelNamespace:
    def __init__(self, ns):
        self._ns = ns

    def __getattr__(self, name):
        # Produces URIs like Model.profile, Model.description, etc.
        return URIRef(self._ns[f"Model.{name}"])


class MD(DefinedNamespace):
    _NS = Namespace("http://iec.ch/TC57/61970-552/ModelDescription/1#")

    FullModel = _NS["FullModel"]
    Model = _MDModelNamespace(_NS)

    _extras = [ # None of these literals have datatype and are therefore string
        "Model.profile", # URIRef or Literal uri? Possibly could be either.
        "Model.description", # Literal
        "Model.version", # Literal integer
        "Model.created", # Literal datetime
        "Model.scenarioTime", # Literal datetime
        "Model.modelingAuthoritySet", # Literal uri or URIRef?
        "Model.DependentOn", # URIRef
        "Model.Supersedes"  # Unsure. Not used in any of the sample files.
    ]

def collect_specific_namespaces(triples: list[tuple[Node, Node, Node]], namespace_manager: NamespaceManager) -> dict[str, URIRef]:
    """Collect namespaces used in a list of triples.

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