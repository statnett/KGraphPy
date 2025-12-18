from rdflib.plugin import register
from rdflib.parser import Parser
from rdflib.plugins.parsers.rdfxml import RDFXMLParser
from rdflib import URIRef, Literal
from rdflib.namespace import XSD
from rdflib import Graph
from linkml_runtime.utils.schemaview import SchemaView

class CIMXMLParser(Parser):
    name = "cimxml"
    format = "cimxml"

    def __init__(self, schema_path: str|None=None):
        super().__init__()
        self.schema_path: str|None = schema_path
        self.schemaview: SchemaView|None = None
        self.slot_index: dict|None = None
        print("CIMXMLParser loaded")

    def parse(self, source, sink, **kwargs):
        print("CIMXMLParser.parse called")
        rdfxml = RDFXMLParser()     # Parsing data as if it was RDF/XML format
        rdfxml.parse(source, sink, **kwargs)
        if "schema_path" in kwargs:
            self.schema_path = kwargs["schema_path"]
        if self.schema_path and self.schemaview is None:    # Load model from linkML file
            self.schemaview = SchemaView(self.schema_path)
            self.slot_index = _build_slot_index(self.schemaview)    # Build index for more effective retrieval of datatypes
            self.post_process(sink)
        else:
            print("Cannot perform post processing without the model. Data parsed as RDF/XML.")
        return sink


    def post_process(self, graph):
        print("Running post-process")
        fix_rdf_ids(graph, self.schemaview)     # Fix rdf:ID errors created by the RDFXMLParser
        canonical_namespace = detect_cim_namespace(self.schemaview)
        normalize_cim_uris(graph, canonical_ns=canonical_namespace)     # Fix when cim namespace in instance data differ from model     
        self.enrich_literal_datatypes(graph)    # Add datatypes from model


    def enrich_literal_datatypes(self, graph):
        print("Enriching")
        if self.schemaview is None or self.slot_index is None:
            print("No schemaview found.")
            return
        
        triples_to_add = []
        triples_to_remove = []

        for s, p, o in graph:
            if isinstance(o, Literal) and o.datatype is None:
                # print(p)
                slot = self.slot_index.get(str(p))
                # print(slot)
                if slot and slot.range:
                    t = self.schemaview.get_type(slot.range)
                    if t and t.uri:
                        datatype_uri = self.schemaview.expand_curie(t.uri)
                        # print(datatype_uri)
                        new_literal = Literal(o.value, datatype=URIRef(datatype_uri))

                        triples_to_remove.append((s, p, o))
                        triples_to_add.append((s, p, new_literal))

        for t in triples_to_remove:
            graph.remove(t)
        for t in triples_to_add:
            graph.add(t)

        return graph


def _build_slot_index(schemaview):
    index = {}

    for name, slot in schemaview.all_slots().items():

        # 1. Full URI (vanligst i CIM)
        if slot.slot_uri:
            try:
                expanded = schemaview.expand_curie(slot.slot_uri)
            except Exception as e:
                expanded = slot.slot_uri
                # print(e)
            index[expanded] = slot
    return index

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


def fix_rdf_ids(graph, schemaview):
    cim_base = get_cim_base_from_graph(schemaview).rstrip("#") + "#"

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

