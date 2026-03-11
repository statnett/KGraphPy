"""Parser for CIM/XML files."""

from rdflib.parser import Parser, InputSource
from rdflib.plugins.parsers.rdfxml import RDFXMLParser
from rdflib import URIRef, Literal, Namespace, Graph
from rdflib.namespace import XSD
from linkml_runtime.utils.schemaview import SchemaView, SlotDefinition
from linkml_runtime.linkml_model.meta import TypeDefinition 
import yaml
import logging
from typing import Optional, cast   #, Any
from cim_plugin.exceptions import LiteralCastingError
from cim_plugin.utilities import extract_uuid
from cim_plugin.namespaces import update_namespace_in_triples
from cim_plugin.enriching import _build_slot_index, create_typed_literal, resolve_datatype_from_slot
import io
import contextlib

logger = logging.getLogger('cimxml_logger')


class CIMXMLParser(Parser):
    name = "cimxml"
    format = "cimxml"

    def __init__(self, schema_path: str|None=None) -> None:
        super().__init__()
        self.schema_path: str|None = schema_path
        self.schemaview: SchemaView|None = None
        self.slot_index: dict|None = None
        # self.class_index: dict|None = None
        logger.info("CIMXMLParser loaded")

    def parse(self, source: InputSource, sink: Graph, **kwargs) -> None:
        logger.info("CIMXMLParser.parse called")
        rdfxml = RDFXMLParser()     # Parsing data as if it was RDF/XML format
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):   
            rdfxml.parse(source, sink, **kwargs)
        
        # self.normalize_rdf_ids(sink)     # Fix rdf:ID errors created by the RDFXMLParser and remove _ and #_
        fix_qualifier_for_all_uuids(sink)

        if "schema_path" in kwargs:
            self.schema_path = kwargs["schema_path"]
        if self.schema_path and self.schemaview is None:    # Load model from linkML file
            self.schemaview = SchemaView(self.schema_path)
            # self.ensure_correct_namespace_model(prefix="cim", correct_namespace=CIM)  # Ensures that the linkML has correct namespace for the cim prefix
            # self.ensure_correct_namespace_model(prefix="eu", correct_namespace=EU)  # Ensures that the linkML has correct namespace for the eu prefix
            # self.patch_missing_datatypes_in_model() # If linkML does not contain all necessary types, it is fixed here
            self.slot_index = _build_slot_index(self.schemaview)    # Build index for more effective retrieval of datatypes
            # self.enrich_literal_datatypes(sink)    # Add datatypes from model
            # self.post_process(sink)
        # else:
        #     logger.error("Cannot perform post processing without the model. Data parsed as RDF/XML.")
        
    # def post_process(self, graph: Graph) -> None:
    #     logger.info("Running post-process")
    #     self.normalize_rdf_ids(graph)     # Fix rdf:ID errors created by the RDFXMLParser and remove _ and #_
    #     # ensure_correct_namespace_graph(graph, prefix="cim", correct_namespace=CIM)  # Ensures that data has correct namespace for the cim prefix
    #     # ensure_correct_namespace_graph(graph, prefix="eu", correct_namespace=EU)    # Ensures that data has correct namespace for the eu prefix
    #     self.enrich_literal_datatypes(graph)    # Add datatypes from model


    def ensure_correct_namespace_model(self, prefix: str, correct_namespace: str):
        """Ensure that the given prefix has correct namespace in schemaview and update it if not.
        
        Parameters:
            prefix (str): The prefix to check for correct namespace.
            new_namespace (str): The correct namespace.

        Raises:
            ValueError: - If schemaview is not found.
                        - If given prefix is not found in schemaview.
        """
        if not self.schemaview:
            raise ValueError("Schemaview not found")

        current = _get_current_namespace_from_model(self.schemaview, prefix)
        
        if current is None:
            raise ValueError(f"Prefix {prefix} not found in schemaview")

        if current == correct_namespace:
            logger.info(f"Model has correct namespace for {prefix}.")
            return

        logger.info(f"Wrong namespace detected for {prefix} in model. Correcting to {correct_namespace}.")
        update_namespace_in_model(self.schemaview, prefix, correct_namespace)
        

    def patch_missing_datatypes_in_model(self) -> None:
        """Patch a linkML schemaview with a missing datatype.

        Currently patches integer.

        Raises:
            TypeError: If schemaview.schema.types is not a dict.
        """
        if self.schema_path and self.schemaview and self.schemaview.schema:

            types = self.schemaview.schema.types
            if not isinstance(types, dict):
                raise TypeError(f"Expected types to be dict, got {type(types)}")
            
            if "integer" in types:
                logger.info("Integer is present in SchemaView.")
                return
        
            try:
                t = TypeDefinition( name="integer", base="int", uri="http://www.w3.org/2001/XMLSchema#integer" )     
                types["integer"] = t 
                self.schemaview.set_modified() 
                patch_integer_ranges(self.schemaview, self.schema_path) # Reassign datatypes to integer (were automatically assigned to string when loaded)
            except ValueError as e:
                logger.error(e)
                raise

    # Has been copied to processor
    def enrich_literal_datatypes(self, graph: Graph) -> Graph:
        """Enrich the Literals of a graph with datatypes collected from linkML SchemaView.
        
        - Cast value to correct format and log error when that is not possible.
        - Tag with the full URI of the primitive datatype.

        Parameters:
            graph (Graph): The graph to enrich.

        Returns:
            Graph: The enriched graph.
        """
        logger.info("Enriching literal datatypes")

        if not self.schemaview or not self.slot_index:
            logger.error("Missing schemaview or slot_index. Enriching not possible.")
            return graph

        unfound_predicates = set()
        casting_errors = []
        updated_count = 0

        for s, p, o in list(graph):
            if not isinstance(o, Literal) or o.datatype is not None or o.language is not None:
                continue

            slot = self.slot_index.get(str(p))
            if not slot:
                unfound_predicates.add(str(p))
                continue

            datatype_uri = resolve_datatype_from_slot(self.schemaview, slot)
            if not datatype_uri:
                logger.info(f"No datatype found for range: {slot.range}, for {slot.name}")
                continue

            try:
                new_literal = create_typed_literal(o.value, datatype_uri, self.schemaview)
            except LiteralCastingError as e:
                casting_errors.append(f"Error casting {o} for {s}, {p}: {e}")
                continue

            graph.remove((s, p, o))
            graph.add((s, p, new_literal))
            updated_count += 1

        if casting_errors:
            logger.error("\n".join(casting_errors))

        if unfound_predicates:
            logger.info(f"Did not find these predicates in model: {unfound_predicates}")
        logger.info(f"Enriching done. Added datatypes to {updated_count} triples.")

        return graph


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


def patch_integer_ranges(schemaview: SchemaView, schema_path: str) -> None:
    """Patch slots in schemaview which contain range: integer in raw yaml.
    
    Parameters:
        schemaview (SchemaView): The schemaview which is to be patched.
        schema_path (str): Path to the file with the raw yaml linkML data.

    Raises:
        ValueError: - If schemaview has no slots.
                    - If specific slot is not found in the schemaview.
    """
    integer_slots = find_slots_with_range(schema_path, "integer")

    if not integer_slots:
        logger.info("No attributes with range=integer found. No changes made.")
        return
    
    if schemaview.schema is None or not isinstance(schemaview.schema.slots, dict): 
        raise ValueError("SchemaView has no slots") 
    
    changed = False 
    for slot_name in integer_slots: 
        original_slot = schemaview.get_slot(slot_name) 
        if original_slot is None: 
            raise ValueError(f"{slot_name} not found in schemaview")
        
        if original_slot.range != "integer": 
            original_slot.range = "integer" 
            schemaview.add_slot(original_slot) 
            changed = True 
            
    if changed: 
        schemaview.set_modified()


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



# Moved to enriching.py and class_index removed
# def _build_slot_and_class_index(schemaview: SchemaView) -> tuple[dict, dict]:
#     """Build an index of classes and slots from a SchemaView.
    
#     Parameters:
#         schemaview (SchemaView): The schemaview to build the index from.

#     Returns:
#         tuple[dict, dict]: The slot index and the class index in that order.
#     """
#     slot_index = {}

#     for cls_name, cls in schemaview.all_classes().items():
#         if not isinstance(cls.attributes, dict):
#             continue

#         for slot_name, slot in cls.attributes.items():
#             slot = cast(SlotDefinition, slot)
#             if not slot.slot_uri:
#                 continue

#             expanded = schemaview.expand_curie(slot.slot_uri)

#             if expanded not in slot_index:
#                 slot_index[expanded] = slot
#                 continue

#             existing = slot_index[expanded]

#             if slots_equal(existing, slot):
#                 continue

#             logger.warning(f"Slot for URI '{expanded}' is overwritten by class slot '{slot_name}'.")
#             slot_index[expanded] = slot

#     class_index = {name: cls for name, cls in schemaview.all_classes().items()}
#     return slot_index, class_index


if __name__ == "__main__":
    print("cimxml plugin for rdflib")
