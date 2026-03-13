"""Parser for CIM/XML files."""

from rdflib.parser import Parser, InputSource
from rdflib.plugins.parsers.rdfxml import RDFXMLParser
from rdflib import URIRef, Namespace, Graph
from linkml_runtime.utils.schemaview import SchemaView
from linkml_runtime.linkml_model.meta import TypeDefinition 
import yaml
import logging
from typing import Optional
from cim_plugin.utilities import extract_uuid
from cim_plugin.namespaces import update_namespace_in_triples
import io
import contextlib

logger = logging.getLogger('cimxml_logger')


class CIMXMLParser(Parser):
    name = "cimxml"
    format = "cimxml"

    def __init__(self) -> None:
        super().__init__()
        logger.info("CIMXMLParser loaded")

    def parse(self, source: InputSource, sink: Graph, **kwargs) -> None:
        logger.info("CIMXMLParser.parse called")
        rdfxml = RDFXMLParser()     # Parsing data as if it was RDF/XML format
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):   
            rdfxml.parse(source, sink, **kwargs)
        
        fix_qualifier_for_all_uuids(sink)   # Fix rdf:ID errors created by the RDFXMLParser and remove _ and #_


def _clean_uri(uri: URIRef, uri_map: dict[str, URIRef]) -> URIRef:
    """Clean uri uuid to a urn:uuid: format.
    
    Parameters:
        uri (URIRef): The uri with the uuid to clean.
        uri_map: A map to keep track of which uuid has been clean (for caching).

    Returns:
        URIRef: The uuid with urn:uuid: qualifier.
    """
    uri_str = str(uri)
    uuid_val = extract_uuid(uri_str)
    if not uuid_val:
        return uri

    if uri_str not in uri_map:
        uri_map[uri_str] = URIRef(f"urn:uuid:{uuid_val}")

    return uri_map[uri_str]

def fix_qualifier_for_all_uuids(graph: Graph) -> None:
    """Fix the qualifier for all uuids in graph to urn:uuid: format. 
    
        Ex. http://example.com#_00000000-0000-4000-8000-000000000001 is fixed to 
        urn:uuid:00000000-0000-4000-8000-000000000001.

    Parameters:
        graph (Graph): The Graph to be modified.
    """
    uri_map = {}

    for s, p, o in list(graph):
        new_s = _clean_uri(s, uri_map) if isinstance(s, URIRef) else s
        new_o = _clean_uri(o, uri_map) if isinstance(o, URIRef) else o

        if (new_s, p, new_o) != (s, p, o):
            graph.remove((s, p, o))
            graph.add((new_s, p, new_o))



# No longer used anywhere. Remove?
def _get_current_namespace_from_model(schemaview: SchemaView, prefix: str) -> Optional[str]:
    """Get namespace for a given prefix from a linkML SchemaView.
    
    Parameters:
        schemaview (Schemaview): Target for collection of namespace.
        prefix (str): The prefix of the namespace.
    
    Raises:
        ValueError: - If SchemaView or SchemaView.schema is not found.
                    - If SchemaView contains the attribute "namespaces" (indicating that it is outdated).

    Returns:
        str: The namespace of the given prefix.
        None: If the prefix is not found in the schema.
    """
    if not schemaview or not schemaview.schema:
        raise ValueError("Schemaview not found or schemaview is missing schema.")

    schema = schemaview.schema

    # The attribute "namespaces" is deprecated. The presence of it indicates an outdated schemaview.
    # As an outdated schemaview may also have other issues, the function raises an error to fail the entire process.
    if hasattr(schema, "namespaces"):
        raise ValueError("The attribute 'namespaces' found in schema. This schemaview is outdated.")

    prefixes = getattr(schema, "prefixes", None)
    if isinstance(prefixes, dict):
        p = prefixes.get(prefix)
        if p and hasattr(p, "prefix_reference"):
            return p.prefix_reference

    return None


def _get_current_namespace_from_graph(graph: Graph, prefix: str) -> Optional[str]:
    """Get namespace uri for a given prefix.
    
    Parameters:
        graph (Graph): Target for collection of namespace.
        prefix (str): The prefix of the namespace.

    Returns:
        str: The namespace of the given prefix.
        None: If the prefix is not found in the graph.
    """
    for pfx, ns in graph.namespace_manager.namespaces():
        if pfx == prefix:
            return str(ns)
    return None


# No longer in use. Remove?
def update_namespace_in_model(schemaview: SchemaView, prefix: str, new_namespace: str) -> None:
    """Update namespace in linkML SchemaView for a given prefix.
    
    Parameters:
        schemaview (SchemaView): The schemaview to update.
        prefix (str): The prefix to be given new namespace.
        new_namespace (str): The new namespace.

    Raises:
        ValueError: If schemaview is not found or missing schema.
    """
    if not schemaview or not schemaview.schema:
        raise ValueError("Schemaview not found or schemaview is missing schema.")

    schema = schemaview.schema

    prefixes = getattr(schema, "prefixes", None)
    if isinstance(prefixes, dict):
        p = prefixes.get(prefix)
        if p and hasattr(p, "prefix_reference"):
            p.prefix_reference = new_namespace

    schemaview.__init__(schema)

# No longer in use. Remove?
def ensure_correct_namespace_graph(graph: Graph, prefix: str, correct_namespace: str) -> None:
    """Ensure that graph has correct namespace for given prefix, and correct if not.
    
    Parameters:
        graph (Graph): The graph to check/correct.
        prefix (str): The prefix to check the namespace for.
        correct_namespace (str): The namespace the prefix should have.

    Raises:
        ValueError: - If correct_namespace is an empty string.
                    - If the prefix is not found in the graph.
    """
    stripped_namespace = correct_namespace.strip()

    if not stripped_namespace:
        raise ValueError("Namespace cannot be an empty string.")
    
    current = _get_current_namespace_from_graph(graph, prefix)

    if current is None:
        return
        # raise ValueError(f"No namespace is called by this prefix: '{prefix}'.")

    if current == stripped_namespace:
        logger.info(f"Graph has correct namespace for {prefix}.")
        return

    logger.info(f"Wrong namespace detected for {prefix} in graph. Correcting to {stripped_namespace}.")
    
    graph.bind(prefix, Namespace(stripped_namespace), replace=True)
    update_namespace_in_triples(graph, current, stripped_namespace)

# No longer in use. Remove?
def inject_integer_type(schemaview: SchemaView) -> None: 
    """Inject integer into types in the SchemaView of a linkML file.

    Parameters:
        schemaview (SchemaView): The schemaview where the integer is to be injected.

    Raises:
        ValueError: If schemaview.schema is None or schemaview.schema.types is not a dictionary.
    
    """
    if schemaview.schema is None:
        raise ValueError("No schema found for schemaview")

    types = schemaview.schema.types

    if isinstance(types, dict):
        if "integer" in types:
            return 

        t = TypeDefinition( name="integer", base="int", uri="http://www.w3.org/2001/XMLSchema#integer" )     
        types["integer"] = t 
        schemaview.set_modified()

# Not in use anymore. Might be usefull later.
def find_slots_with_range(schema_path: str, datatype: str) -> set[str]:
    """Find all slot names in a linkML whose attribute definition has a given range datatype.

    Parameters:
        schema_path (str): Path to the raw LinkML YAML file.
        datatype (str): The datatype to search for (e.g. "integer").

    Raises:
        ValueError: If an attribute definition is not a dict.

    Returns:
        List[str]: A list of slot names that use the given datatype.
    """
    with open(schema_path) as f:
        raw = yaml.safe_load(f)

    matching_slots = set()

    for cls_name, cls in raw.get("classes", {}).items():
        if not cls:
            continue

        attrs = cls.get("attributes") or {}
        for slot_name, slot_def in attrs.items():
            if not isinstance(slot_def, dict):
                raise ValueError(
                    f"{slot_name} in class {cls_name} has unexpected structure. "
                    "Attributes must be dictionaries."
                )

            if slot_def.get("range") == datatype:
                matching_slots.add(slot_name)

    return matching_slots


if __name__ == "__main__":
    print("cimxml plugin for rdflib")
