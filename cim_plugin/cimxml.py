from rdflib.parser import Parser, InputSource
from rdflib.plugins.parsers.rdfxml import RDFXMLParser
from rdflib import URIRef, Literal, RDF, Namespace, Graph
from rdflib.namespace import XSD
from linkml_runtime.utils.schemaview import SchemaView, SlotDefinition
import uuid
from linkml_runtime.linkml_model.meta import TypeDefinition 
import yaml
import logging
from typing import Optional, cast

logger = logging.getLogger('cimxml_logger')

# Namespaces
MD = Namespace("http://iec.ch/TC57/61970-552/ModelDescription/1#") 
DCAT = Namespace("http://www.w3.org/ns/dcat#")
CIM = Namespace("https://cim.ucaiug.io/ns#")
EU = Namespace("https://cim.ucaiug.io/ns/eu#")

class CIMXMLParser(Parser):
    name = "cimxml"
    format = "cimxml"

    def __init__(self, schema_path: str|None=None) -> None:
        super().__init__()
        self.schema_path: str|None = schema_path
        self.schemaview: SchemaView|None = None
        self.slot_index: dict|None = None
        self.class_index: dict|None = None
        self.model_uuid: uuid.UUID|None = None
        logger.info("CIMXMLParser loaded")

    def parse(self, source: InputSource, sink: Graph, **kwargs) -> None:
        logger.info("CIMXMLParser.parse called")
        rdfxml = RDFXMLParser()     # Parsing data as if it was RDF/XML format
        rdfxml.parse(source, sink, **kwargs)
        if "schema_path" in kwargs:
            self.schema_path = kwargs["schema_path"]
        if self.schema_path and self.schemaview is None:    # Load model from linkML file
            self.schemaview = SchemaView(self.schema_path)
            self.ensure_correct_namespace_model(prefix="cim", correct_namespace=CIM)  # Ensures that the linkML has correct namespace for the cim prefix
            self.ensure_correct_namespace_model(prefix="eu", correct_namespace=EU)  # Ensures that the linkML has correct namespace for the eu prefix
            # self.patch_missing_datatypes_in_model() # If linkML does not contain all necessary types, it is fixed here
            self.slot_index, self.class_index = _build_slot_index(self.schemaview)    # Build index for more effective retrieval of datatypes
            self.post_process(sink)
        else:
            logger.info("Cannot perform post processing without the model. Data parsed as RDF/XML.")
        
    def post_process(self, graph: Graph) -> None:
        logger.info("Running post-process")
        self.model_uuid = find_model_uuid(graph)    # Find uuid from md:FullModel or dcat:Dataset
        self.normalize_rdf_ids(graph)     # Fix rdf:ID errors created by the RDFXMLParser and remove _ and #_
        ensure_correct_namespace_graph(graph, prefix="cim", correct_namespace=CIM)  # Ensures that data has correct namespace for the cim prefix
        ensure_correct_namespace_graph(graph, prefix="eu", correct_namespace=EU)    # Ensures that data has correct namespace for the eu prefix
        # canonical_namespace = detect_cim_namespace(self.schemaview)
        # normalize_cim_uris(graph, canonical_ns=canonical_namespace)     # Fix when cim namespace in instance data differ from model     
        self.enrich_literal_datatypes(graph)    # Add datatypes from model


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

    # def fix_rdf_ids(self, graph: Graph, by: str = "urn:uuid") -> None:
    #     if by == "urn:uuid":
    #         self.normalize_rdf_ids(graph)   # Fix rdf:IDs by removing _ and adding urn:uuid
    #         logger.info("Filling in rdf:id with urn:uuid")
    #     elif by == "prefix":
    #         add_prefix_to_rdf_ids(graph)    # Fix rdf:IDs by adding prefix
    #         logger.info("Filling in rdf:id with prefix")
    #     else:
    #         raise ValueError(f"'{by}' is not an approved method.")

    def normalize_rdf_ids(self, graph: Graph) -> None: 
        """Remove _ and replace prefix set by RDFXMLparser with urn:uuid.
        
        Parameters:
            graph (Graph): The graph to be normalized.

        Raises:
            ValueError: If normalization makes different URIs identical. List of URIs affected is given.
        """
        id_set = set()

        for s in graph.subjects(): # Collect all relevant subjects
            s_str = str(s) 
            if "#" in s_str: 
                frag = s_str.split("#")[-1] 
                id_set.add(frag.lstrip("_"))

        try:
            detect_uri_collisions(graph, id_set)
        except ValueError as e:
            logger.error(e)
            raise

        uri_map = {} 
        for s, p, o in list(graph): 
            if isinstance(s, URIRef):   # Clean subjects 
                new_s = _clean_uri(s, uri_map, id_set) 
                if new_s != s: 
                    graph.remove((s, p, o)) 
                    graph.add((new_s, p, o)) 
                    s = new_s 
                                
            if isinstance(o, URIRef):   # Clean objects
                new_o = _clean_uri(o, uri_map, id_set) 
                if new_o != o: 
                    graph.remove((s, p, o)) 
                    graph.add((s, p, new_o))

    def enrich_literal_datatypes(self, graph: Graph) -> Graph:
        logger.info("Enriching literal datatypes")

        if self.schemaview is None or self.slot_index is None:
            logger.error("Missing schemaview or slot_index. Enriching not possible.")
            return graph

        triples_to_add = []
        triples_to_remove = []

        unfound_predicates = set()

        for s, p, o in graph:
            if isinstance(o, Literal) and o.datatype is None:
                slot = self.slot_index.get(str(p))

                if not slot: # Collecting predicates not found in the model
                    unfound_predicates.add(str(p))
                    continue

                datatype_uri = resolve_datatype_from_slot(self.schemaview, slot)

                if not datatype_uri:
                    logger.info(f"No datatype found for range: {slot.range}, for {slot.name}")
                    continue

                new_literal = create_typed_literal(o.value, datatype_uri, self.schemaview)

                triples_to_remove.append((s, p, o))
                triples_to_add.append((s, p, new_literal))

        for t in triples_to_remove:
            graph.remove(t)
        for t in triples_to_add:
            graph.add(t)
        logger.info(f"Did not find these predicates in model: {unfound_predicates}")
        logger.info(f"Enriching done. Added datatypes to {len(triples_to_add)} triples.")
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


def update_namespace_in_graph(graph: Graph, old_namespace: str, new_namespace: str) -> None:
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
        raise ValueError(f"No namespace is called by this prefix: '{prefix}'.")

    if current == stripped_namespace:
        logger.info(f"Graph has correct namespace for {prefix}.")
        return

    logger.info(f"Wrong namespace detected for {prefix} in graph. Correcting to {stripped_namespace}.")
    
    graph.bind(prefix, Namespace(stripped_namespace), replace=True)
    update_namespace_in_graph(graph, current, stripped_namespace)


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


def detect_uri_collisions(graph: Graph, id_set: set[str]) -> None:
    """Scan the graph for URI collisions that will happen if they are cleaned with _clean_uri.
   
    Parameters:
        graph (Graph): The graph to scan for collisions.
        id_set (set[str]): A set of uri that should be cleaned.
    
    Raises: 
        ValueError: with list of collisions if collisions are found.
    """
    uri_map = {}
    reverse_map = {}
    collisions = []

    for s, p, o in graph:
        # SUBJECT
        if isinstance(s, URIRef):
            new_s = _clean_uri(s, uri_map, id_set)
            if new_s != s:
                if new_s in reverse_map and reverse_map[new_s] != s:
                    collisions.append((s, reverse_map[new_s], new_s))
                else:
                    reverse_map[new_s] = s

        # OBJECT
        if isinstance(o, URIRef):
            new_o = _clean_uri(o, uri_map, id_set)
            if new_o != o:
                if new_o in reverse_map and reverse_map[new_o] != o:
                    collisions.append((o, reverse_map[new_o], new_o))
                else:
                    reverse_map[new_o] = o

    if collisions:
        msg_lines = ["IRI collisions detected:"]
        for old, existing, new in collisions:
            msg_lines.append(f"  {old} and {existing} both map to {new}")
        raise ValueError("\n".join(msg_lines))


def _clean_uri(uri: URIRef, uri_map: dict[str, URIRef], id_set: set[str]) -> URIRef:
    """Clean a uri for _ and # with everything before it, and add urn:uuid: as prefix.

    The uri is cleaned if:
        - It contains _ at the beginning of the fragment (after #)
        - It is in id_set
    
    Parameters:
        uri (URIRef): The uri to be cleaned.
        uri_map (dict[str, URIRef]): A map keeping track of cleaned uri.
        id_set (set[str]): A set of uri fragments that should be cleaned even if they don't contain _.

    Returns:
        URIRef: The cleaned uri.
    """
    uri_str = str(uri)

    if "#" not in uri_str:
        return uri

    if len(uri_str.split("#")) > 2:
        logger.warning(f"{uri_str} has more then one #")

    fragment = uri_str.split("#")[-1]
    if fragment not in id_set and not fragment.startswith("_"): 
        return uri

    clean = fragment.lstrip("_")
    if uri_str not in uri_map: 
        uri_map[uri_str] = URIRef(f"urn:uuid:{clean}") 
        
    return uri_map[uri_str]


def find_model_uuid(graph: Graph) -> uuid.UUID: 
    # Søk etter md:FullModel 
    for s in graph.subjects(RDF.type, MD.FullModel): 
        return _extract_uuid_from_urn(str(s)) 
    
    # Søk etter dcat:Dataset 
    for s in graph.subjects(RDF.type, DCAT.Dataset): 
        return _extract_uuid_from_urn(str(s)) 
    
    # Ingen global modell-ID funnet → raise error 
    raise ValueError( 
        "Fant verken md:FullModel eller dcat:Dataset i grafen. " 
        "Kan ikke bestemme global modell-UUID." 
    )


def _extract_uuid_from_urn(urn: str) -> uuid.UUID: 
    """ Tar en URI som 'urn:uuid:1234-...' og returnerer en uuid.UUID. """ 
    if not urn.startswith("urn:uuid:"): 
        raise ValueError(f"Ugyldig modell-URI: {urn}") 
    
    return uuid.UUID(urn[len("urn:uuid:"):])

# PRIMITIVE_MAP = {
#     "integer": "xsd:integer",
#     "int": "xsd:integer",
#     "float": "xsd:float",
#     "double": "xsd:double",
#     "string": "xsd:string",
#     "boolean": "xsd:boolean",
# }

PRIMITIVE_MAP = {
    "integer": "http://www.w3.org/2001/XMLSchema#integer",
    "int": "http://www.w3.org/2001/XMLSchema#integer",
    "float": "http://www.w3.org/2001/XMLSchema#float",
    "double": "http://www.w3.org/2001/XMLSchema#double",
    "string": "http://www.w3.org/2001/XMLSchema#string",
    "boolean": "http://www.w3.org/2001/XMLSchema#boolean",
}

# def resolve_datatype_from_slot(sv, slot):
#     rng = slot.range

#     if rng in PRIMITIVE_MAP:
#         return PRIMITIVE_MAP[rng]

#     if rng in sv.schema.types:
#         t = sv.get_type(rng)
#         return t.uri or PRIMITIVE_MAP.get(t.base) or t.base

#     if rng in sv.all_classes():
#         cls = sv.get_class(rng)
#         for s in cls.slots:
#             sub_slot = sv.get_slot(s)
#             if sub_slot.range in PRIMITIVE_MAP:
#                 return PRIMITIVE_MAP[sub_slot.range]

#     return None


# def _resolve_type(schemaview: SchemaView, type_name: str) -> str:
#         """Return the XSD URI or base type for any type, including primitives."""
#         t = schemaview.get_type(type_name)
#         print(t)
#         if t is None:
#             return type_name
#         print(t.uri)
#         if t.uri:
#             return t.uri 
#         print(t.base)
#         if t.base:
#             base = schemaview.get_type(t.base)
#             return base.uri or base.base or t.base 
        
#         return type_name


def _resolve_type(schemaview: SchemaView, type_name: str) -> str:
    """Resolve a LinkML type name to its canonical datatype.
    
    Works for both declared types and built-in primitives.
    
    Parameters:
        schemaview (SchemaView): The SchemaView to get the datatype from.
        type_name (str): Name of type.

    Returns:
        str: The datatype, either as a uri or base name. Last resort returns type_name.
    """
    t = schemaview.get_type(type_name)
    if t is None:
        return type_name

    if t.uri:
        return t.uri

    if t.base:
        return _resolve_type(schemaview, t.base)

    return type_name


def resolve_datatype_from_slot(sv, slot):
    
    rng = slot.range
    if not rng:
        return None

    # Case 1: range is a declared type or a primitive
    try:
        return _resolve_type(sv, rng)
    except KeyError:
        pass  # not a type, maybe a class

    # Case 2: range is a class → inspect its attributes
    if rng in sv.all_classes():
        cls = sv.get_class(rng)
        for attr_name, attr in (cls.attributes or {}).items():
            if attr.range:
                return _resolve_type(sv, attr.range)

    # Fallback: return the raw range
    return rng

# def resolve_range_datatype(schemaview, range_name):
#     """
#     Returnerer RDF-datatype for en LinkML-range.
#     Prioritet:
#       1. CIM-annotert datatype (annotations["uri"])
#       2. LinkML-type sin uri (xsd:float osv.)
#     """

#     t = schemaview.get_type(range_name)
#     if not t:
#         return None

#     # 1. CIM-type (fra annotations)
#     if t.annotations:
#         ann = t.annotations.get("uri")
#         if ann and ann.value:
#             return schemaview.expand_curie(ann.value)

#     # 2. XSD-type (fra type.uri)
#     if t.uri:
#         return schemaview.expand_curie(t.uri)

#     return None

def create_typed_literal(value, datatype_uri, schemaview):
    # 1. Expand CURIE if needed
    if ":" in datatype_uri and not datatype_uri.startswith("http"):
        datatype_uri = schemaview.expand_curie(datatype_uri)

    # 2. Cast lexical form based on datatype
    if datatype_uri == str(XSD.float):
        value = float(value)
    elif datatype_uri == str(XSD.integer):
        value = int(value)
    # legg til flere hvis du trenger

    return Literal(value, datatype=URIRef(datatype_uri))


def slots_equal(slot1: SlotDefinition, slot2: SlotDefinition) -> bool:
    """Show whether two SlotDefinition objects contain the same keys and values.

    Internal attributes are ignored (starts with _).

    Parameters:
        slot1 (SlotDefinition): First slot.
        slot2 (SlotDefinition): Second slot.
    
    Returns:
        bool: True if slot1 and slot2 are the identical.

    """
    d1 = {k: v for k, v in slot1.__dict__.items() if not k.startswith("_")}
    d2 = {k: v for k, v in slot2.__dict__.items() if not k.startswith("_")}

    return d1 == d2


def _build_slot_index(schemaview: SchemaView) -> tuple[dict, dict]:
    """Build an index of classes and slots from a SchemaView.
    
    Parameters:
        schemaview (SchemaView): The schemaview to build the index from.

    Returns:
        tuple[dict, dict]: The slot index and the class index in that order.
    """
    slot_index = {}

    for cls_name, cls in schemaview.all_classes().items():
        if not isinstance(cls.attributes, dict):
            continue

        for slot_name, slot in cls.attributes.items():
            slot = cast(SlotDefinition, slot)
            if not slot.slot_uri:
                continue

            expanded = schemaview.expand_curie(slot.slot_uri)

            if expanded not in slot_index:
                slot_index[expanded] = slot
                continue

            existing = slot_index[expanded]

            if slots_equal(existing, slot):
                continue

            logger.warning(f"Slot for URI '{expanded}' is overwritten by class slot '{slot_name}'.")
            slot_index[expanded] = slot

    class_index = {name: cls for name, cls in schemaview.all_classes().items()}
    return slot_index, class_index



# def get_cim_base(schemaview):
#     """
#     Returnerer CIM-base-URI fra en LinkML SchemaView.
#     Søker etter prefix_reference som inneholder 'CIM'.
#     Faller tilbake til første prefix som slutter med '#'.
#     """
#     prefixes = schemaview.schema.prefixes

#     # Førstevalg: prefix som inneholder 'CIM'
#     for pfx, prefix_obj in prefixes.items():
#         ref = prefix_obj.prefix_reference
#         if ref and "CIM" in ref:
#             return ref

#     # Andrevalg: første prefix som slutter med '#'
#     for pfx, prefix_obj in prefixes.items():
#         ref = prefix_obj.prefix_reference
#         if ref and ref.endswith("#"):
#             return ref

#     raise ValueError("Fant ingen CIM-base i schema prefixes")


# def get_cim_base_from_graph(graph: Graph) -> str:
#     """
#     Finn CIM-base-URI fra namespaces i grafen.

#     Strategi:
#     1. Hvis det finnes et prefix som heter 'cim' → bruk det.
#     2. Ellers: feile eksplisitt, så du ikke gjetter feil.
#     """
#     ns_map = dict(graph.namespaces())

#     # 1. Eksakt 'cim' er det vi stoler på
#     if "cim" in ns_map:
#         return str(ns_map["cim"])

#     # Hvis du vil være streng, stopper du her:
#     raise ValueError("Fant ikke 'cim' prefix i grafen. Kan ikke bestemme CIM-base.")


# def add_prefix_to_rdf_ids(graph):
#     cim_base = get_cim_base_from_graph(graph).rstrip("#") + "#"

#     candidates = []
#     for node in graph.all_nodes():
#         if isinstance(node, URIRef) and "#" in node:
#             base, frag = node.rsplit("#", 1)

#             # Hopp over hvis den allerede er i CIM-namespace
#             if base.startswith(cim_base.rstrip("#")):
#                 continue

#             candidates.append((node, frag))

    # for old_uri, frag in candidates:
    #     new_uri = URIRef(f"{cim_base}{frag}")

    #     # Flytt triples der old_uri er subjekt
    #     for p, o in list(graph.predicate_objects(old_uri)):
    #         graph.add((new_uri, p, o))
    #         graph.remove((old_uri, p, o))

    #     # Flytt triples der old_uri er objekt
    #     for s, p in list(graph.subject_predicates(old_uri)):
    #         graph.add((s, p, new_uri))
    #         graph.remove((s, p, old_uri))


# def graph_uses_canonical_namespace(graph, canonical_ns):
#     for _, uri in graph.namespaces():
#         if str(uri).rstrip("#/") == canonical_ns.rstrip("#/"):
#             return True
#     return False


# def detect_cim_namespace(schemaview):
#     prefixes = schemaview.schema.prefixes

#     if "cim" in prefixes:
#         pref = prefixes["cim"]

#         # LinkML Prefix object → extract actual URI string
#         if hasattr(pref, "prefix_reference"):
#             ns = pref.prefix_reference
#         elif hasattr(pref, "uri"):
#             ns = pref.uri
#         else:
#             raise ValueError("Prefix object has no usable URI field")

#         # Normalize
#         if not ns.endswith("#"):
#             ns = ns.rstrip("/") + "#"

#         return ns

#     raise ValueError("Model has no 'cim' prefix defined")

# def normalize_cim_uris(graph, canonical_ns):
#     if graph_uses_canonical_namespace(graph, canonical_ns):
#         print("CIM namespace matches model. Skip normalisation")
#         return
    
#     print("Normalising CIM namespaces...")
#     triples = list(graph)
#     for s, p, o in triples:
#         new_s = normalize_uri(s, canonical_ns)
#         new_p = normalize_uri(p, canonical_ns)
#         new_o = normalize_uri(o, canonical_ns)

#         if (s, p, o) != (new_s, new_p, new_o):
#             graph.remove((s, p, o))
#             graph.add((new_s, new_p, new_o))

# def normalize_uri(term, canonical_ns):
#     if not isinstance(term, URIRef):
#         return term
#     uri = str(term)
#     if looks_like_cim_uri(uri):
#         local = uri.split("#")[-1]
#         return URIRef(canonical_ns + local)
#     return term

# def looks_like_cim_uri(uri: str) -> bool:
#     u = uri.lower()
#     return (
#         "cim" in u or
#         "tc57" in u or
#         "ucaiug" in u or
#         "entsoe" in u
#     )

if __name__ == "__main__":
    print("cimxml plugin for rdflib")
    filepath = "../CoreEquipment.linkml.yaml"
    sv = SchemaView(filepath)
    slot = sv.get_slot("isInfiniteDuration")
    print(slot.range)
    rng = resolve_datatype_from_slot(sv, slot)
    print(rng)
    print(sv.get_type("integer").uri)
    # slots, classes = _build_slot_index(sv)
    # print(classes)
