from rdflib.plugin import register
from rdflib.parser import Parser, InputSource
from rdflib.plugins.parsers.rdfxml import RDFXMLParser
from rdflib import URIRef, Literal, RDF, Namespace, Graph
from rdflib.namespace import XSD
from linkml_runtime.utils.schemaview import SchemaView
import uuid
from linkml_runtime.linkml_model.meta import TypeDefinition 
import yaml


MD = Namespace("http://iec.ch/TC57/61970-552/ModelDescription/1#") 
DCAT = Namespace("http://www.w3.org/ns/dcat#")

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
        print("CIMXMLParser loaded")

    def parse(self, source: InputSource, sink: Graph, **kwargs) -> None:
        print("CIMXMLParser.parse called")
        rdfxml = RDFXMLParser()     # Parsing data as if it was RDF/XML format
        rdfxml.parse(source, sink, **kwargs)
        if "schema_path" in kwargs:
            self.schema_path = kwargs["schema_path"]
        if self.schema_path and self.schemaview is None:    # Load model from linkML file
            self.schemaview = SchemaView(self.schema_path)
            inject_integer_type(self.schemaview)    # Add integer to linkML model primitive types
            patch_integer_ranges(self.schemaview, self.schema_path) # Reassign datatypes to integer (were automatically assigned to string when loaded)
            self.slot_index, self.class_index = _build_slot_index(self.schemaview)    # Build index for more effective retrieval of datatypes
            self.post_process(sink)
        else:
            print("Cannot perform post processing without the model. Data parsed as RDF/XML.")
        
    def post_process(self, graph: Graph) -> None:
        print("Running post-process")
        self.model_uuid = find_model_uuid(graph)    # Find uuid from md:FullModel or dcat:Dataset
        self.fix_rdf_ids(graph)     # Fix rdf:ID errors created by the RDFXMLParser and remove _ and #_
        canonical_namespace = detect_cim_namespace(self.schemaview)
        normalize_cim_uris(graph, canonical_ns=canonical_namespace)     # Fix when cim namespace in instance data differ from model     
        self.enrich_literal_datatypes(graph)    # Add datatypes from model

    def fix_rdf_ids(self, graph: Graph, by: str = "urn:uuid") -> None:
        if by == "urn:uuid":
            self.normalize_rdf_ids(graph)   # Fix rdf:IDs by removing _ and adding urn:uuid
            print("Filling in rdf:id with urn:uuid")
        elif by == "prefix":
            add_prefix_to_rdf_ids(graph)    # Fix rdf:IDs by adding prefix
            print("Filling in rdf:id with prefix")
        else:
            raise ValueError(f"'{by}' is not an approved method.")

    def normalize_rdf_ids(self, graph: Graph) -> None: 
        """Remove _ and replace prefix set by RDFXMLparser with urn:uuid."""
        uri_map = {} 
        id_set = set()

        for s in graph.subjects(): # Collect all relevant subjects
            s_str = str(s) 
            if "#" in s_str: 
                frag = s_str.split("#")[-1] 
                id_set.add(frag.lstrip("_"))

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
        print("Enriching literal datatypes")

        if self.schemaview is None or self.slot_index is None:
            print("Missing schemaview or slot_index. Enriching not possible.")
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
                    print(f"  [DEBUG] Ingen datatype funnet for range: {slot.range}, for {slot.name}")
                    continue

                new_literal = create_typed_literal(o.value, datatype_uri, self.schemaview)

                triples_to_remove.append((s, p, o))
                triples_to_add.append((s, p, new_literal))

        for t in triples_to_remove:
            graph.remove(t)
        for t in triples_to_add:
            graph.add(t)
        print(f"Fant ikke: {unfound_predicates}")
        print("\nEnriching done. La til", len(triples_to_add), "tripler.")
        return graph

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
    if not isinstance(types, dict): 
        raise ValueError("Schema types is not a dictionary")

    if "integer" in types: 
        return 

    t = TypeDefinition( name="integer", base="int", uri="http://www.w3.org/2001/XMLSchema#integer" )     
    types["integer"] = t 

    schemaview.set_modified() # Oppdater interne indekser i SchemaView 


# def patch_integer_ranges(schemaview: SchemaView, schema_path: str) -> None:

#     with open(schema_path) as f:
#         raw = yaml.safe_load(f)

#     # Find all class attributes which had integer as range before import into schemaview by examining the raw file
#     # These were reassigned as string because integer was not a type in the linkML model
#     integer_attrs = []

#     for cls_name, cls in raw.get("classes", {}).items():
#         if cls is None:
#             continue

#         attrs = cls.get("attributes") or {}
#         for slot_name, slot_def in attrs.items():
#             if slot_def and slot_def.get("range") == "integer":
#                 integer_attrs.append((cls_name, slot_name))

#     # Reassign integer to the ranges of these class attributes
#     for cls_name, slot_name in integer_attrs:
#         cls = schemaview.get_class(cls_name)
#         if cls and cls.attributes and slot_name in cls.attributes:
#             cls.attributes[slot_name].range = "integer"

#     schemaview.set_modified()


def patch_integer_ranges(schemaview: SchemaView, schema_path: str) -> None:
    with open(schema_path) as f:
        raw = yaml.safe_load(f)

    integer_attrs = []

    # Find all class attribute slots which had integer as range before import into schemaview, by examining the raw file
    # These were reassigned as string because integer was not a type in the linkML model
    for cls_name, cls in raw.get("classes", {}).items():
        if not cls:
            continue

        attrs = cls.get("attributes") or {}
        for slot_name, slot_def in attrs.items():
            if slot_def and slot_def.get("range") == "integer":
                integer_attrs.append(slot_name)

    # Reassign integer to the ranges of these class attribute slots
    for slot_name in integer_attrs:
        slot = schemaview.get_slot(slot_name)
        if slot:
            slot.range = "integer"
            schemaview.add_slot(slot)

    schemaview.set_modified()

def _clean_uri(uri: URIRef, uri_map: dict[str, URIRef], id_set: set[str]) -> URIRef:
    uri_str = str(uri)

    # Bare URIer med fragment (#id) er aktuelle
    if "#" not in uri_str:
        return uri

    fragment = uri_str.split("#")[-1]

    # Normaliser KUN hvis fragmentet er en faktisk rdf:ID 
    # # dvs. det finnes i id_set, eller starter med "_" 
    if fragment not in id_set and not fragment.startswith("_"): 
        return uri

    # Fjern leading underscore
    clean = fragment.lstrip("_")

    if uri_str not in uri_map: 
        uri_map[uri_str] = URIRef(f"urn:uuid:{clean}") 
        
    return uri_map[uri_str]

    # def enrich_literal_datatypes(self, graph):
    #     print("Enriching")
    #     if self.schemaview is None or self.slot_index is None:
    #         print("No schemaview found.")
    #         return
        
    #     triples_to_add = []
    #     triples_to_remove = []

    #     for s, p, o in graph:
    #         if isinstance(o, Literal) and o.datatype is None:
    #             # print(p)
    #             slot = self.slot_index.get(str(p))
    #             # print(slot)
    #             if slot and slot.range:
    #                 t = self.schemaview.get_type(slot.range)
    #                 if t and t.uri:
    #                     datatype_uri = self.schemaview.expand_curie(t.uri)
    #                     # print(datatype_uri)
    #                     new_literal = Literal(o.value, datatype=URIRef(datatype_uri))

    #                     triples_to_remove.append((s, p, o))
    #                     triples_to_add.append((s, p, new_literal))

    #     for t in triples_to_remove:
    #         graph.remove(t)
    #     for t in triples_to_add:
    #         graph.add(t)

    #     return graph

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

def resolve_datatype_from_slot(sv, slot):
    rng = slot.range

    if rng in PRIMITIVE_MAP:
        return PRIMITIVE_MAP[rng]

    if rng in sv.schema.types:
        t = sv.get_type(rng)
        return t.uri or PRIMITIVE_MAP.get(t.base) or t.base

    if rng in sv.all_classes():
        cls = sv.get_class(rng)
        for s in cls.slots:
            sub_slot = sv.get_slot(s)
            if sub_slot.range in PRIMITIVE_MAP:
                return PRIMITIVE_MAP[sub_slot.range]

    return None

def resolve_range_datatype(schemaview, range_name):
    """
    Returnerer RDF-datatype for en LinkML-range.
    Prioritet:
      1. CIM-annotert datatype (annotations["uri"])
      2. LinkML-type sin uri (xsd:float osv.)
    """

    t = schemaview.get_type(range_name)
    if not t:
        return None

    # 1. CIM-type (fra annotations)
    if t.annotations:
        ann = t.annotations.get("uri")
        if ann and ann.value:
            return schemaview.expand_curie(ann.value)

    # 2. XSD-type (fra type.uri)
    if t.uri:
        return schemaview.expand_curie(t.uri)

    return None

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


def _build_slot_index(schemaview):
    slot_index = {}

    # 1. Globale slots (hvis noen finnes)
    for name, slot in schemaview.all_slots().items():
        if slot.slot_uri:
            expanded = schemaview.expand_curie(slot.slot_uri)
            slot_index[expanded] = slot

    # 2. Class-lokale attributes (CIM bruker nesten bare disse)
    for cls_name, cls in schemaview.all_classes().items():
        if not cls.attributes:
            continue

        for slot_name, slot in cls.attributes.items():
            if slot.slot_uri:
                expanded = schemaview.expand_curie(slot.slot_uri)
                slot_index[expanded] = slot

    # 3. Class index (som før)
    class_index = {name: cls for name, cls in schemaview.all_classes().items()}

    return slot_index, class_index

# def _build_slot_index(schemaview):
#     slot_index = {}

#     # Slots
#     for name, slot in schemaview.all_slots().items():
#         if slot.slot_uri:
#             expanded = schemaview.expand_curie(slot.slot_uri)
#             slot_index[expanded] = slot

#     # Classes (for å kunne slå opp range-klasser)
#     class_index = {}
#     for name, cls in schemaview.all_classes().items():
#         class_index[name] = cls

#     return slot_index, class_index


def get_cim_base(schemaview):
    """
    Returnerer CIM-base-URI fra en LinkML SchemaView.
    Søker etter prefix_reference som inneholder 'CIM'.
    Faller tilbake til første prefix som slutter med '#'.
    """
    prefixes = schemaview.schema.prefixes

    # Førstevalg: prefix som inneholder 'CIM'
    for pfx, prefix_obj in prefixes.items():
        ref = prefix_obj.prefix_reference
        if ref and "CIM" in ref:
            return ref

    # Andrevalg: første prefix som slutter med '#'
    for pfx, prefix_obj in prefixes.items():
        ref = prefix_obj.prefix_reference
        if ref and ref.endswith("#"):
            return ref

    raise ValueError("Fant ingen CIM-base i schema prefixes")


def get_cim_base_from_graph(graph: Graph) -> str:
    """
    Finn CIM-base-URI fra namespaces i grafen.

    Strategi:
    1. Hvis det finnes et prefix som heter 'cim' → bruk det.
    2. Ellers: feile eksplisitt, så du ikke gjetter feil.
    """
    ns_map = dict(graph.namespaces())

    # 1. Eksakt 'cim' er det vi stoler på
    if "cim" in ns_map:
        return str(ns_map["cim"])

    # Hvis du vil være streng, stopper du her:
    raise ValueError("Fant ikke 'cim' prefix i grafen. Kan ikke bestemme CIM-base.")


def add_prefix_to_rdf_ids(graph):
    cim_base = get_cim_base_from_graph(graph).rstrip("#") + "#"

    candidates = []
    for node in graph.all_nodes():
        if isinstance(node, URIRef) and "#" in node:
            base, frag = node.rsplit("#", 1)

            # Hopp over hvis den allerede er i CIM-namespace
            if base.startswith(cim_base.rstrip("#")):
                continue

            candidates.append((node, frag))

    for old_uri, frag in candidates:
        new_uri = URIRef(f"{cim_base}{frag}")

        # Flytt triples der old_uri er subjekt
        for p, o in list(graph.predicate_objects(old_uri)):
            graph.add((new_uri, p, o))
            graph.remove((old_uri, p, o))

        # Flytt triples der old_uri er objekt
        for s, p in list(graph.subject_predicates(old_uri)):
            graph.add((s, p, new_uri))
            graph.remove((s, p, old_uri))


def graph_uses_canonical_namespace(graph, canonical_ns):
    for _, uri in graph.namespaces():
        if str(uri).rstrip("#/") == canonical_ns.rstrip("#/"):
            return True
    return False


def detect_cim_namespace(schemaview):
    prefixes = schemaview.schema.prefixes

    if "cim" in prefixes:
        pref = prefixes["cim"]

        # LinkML Prefix object → extract actual URI string
        if hasattr(pref, "prefix_reference"):
            ns = pref.prefix_reference
        elif hasattr(pref, "uri"):
            ns = pref.uri
        else:
            raise ValueError("Prefix object has no usable URI field")

        # Normalize
        if not ns.endswith("#"):
            ns = ns.rstrip("/") + "#"

        return ns

    raise ValueError("Model has no 'cim' prefix defined")

def normalize_cim_uris(graph, canonical_ns):
    if graph_uses_canonical_namespace(graph, canonical_ns):
        print("CIM namespace matches model. Skip normalisation")
        return
    
    print("Normalising CIM namespaces...")
    triples = list(graph)
    for s, p, o in triples:
        new_s = normalize_uri(s, canonical_ns)
        new_p = normalize_uri(p, canonical_ns)
        new_o = normalize_uri(o, canonical_ns)

        if (s, p, o) != (new_s, new_p, new_o):
            graph.remove((s, p, o))
            graph.add((new_s, new_p, new_o))

def normalize_uri(term, canonical_ns):
    if not isinstance(term, URIRef):
        return term
    uri = str(term)
    if looks_like_cim_uri(uri):
        local = uri.split("#")[-1]
        return URIRef(canonical_ns + local)
    return term

def looks_like_cim_uri(uri: str) -> bool:
    u = uri.lower()
    return (
        "cim" in u or
        "tc57" in u or
        "ucaiug" in u or
        "entsoe" in u
    )

if __name__ == "__main__":
    print("cimxml plugin for rdflib")

