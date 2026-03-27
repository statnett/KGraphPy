"""Validation of header according to format specification."""

# Trig:
#   - dcterms:PeriodOfTime is expressed through dcterms:temporal with 3 blank nodes (issue #14):
#    - rdf:type dcterms:PeriodOfTime
#    - dcat:startDate with xsd:dateTime literal in utc time zone (ending with "Z")
#    - dcat:endDate with xsd:dateTime literal in utc time zone (ending with "Z")
#   - There should not be a dcat:distribution or dcat:Distribution triple (issue #13)
#   - There should be a dcterms:issued triple which is expressed as a literal with xsd:dateTime datatype in utc time zone (ending with "Z"), and there should only be one such triple (issue #13).
#        (find out: if there are multiple with different values, which should I keep?)
#   - There should be a triple with rdf:type rdfg:Graph. This is in some trigs expressed like this: subject rdf:type rdfg:Graph, dcat:Dataset, 
#       but rdflib seems to ommit the rdf:type rdfg:Graph triple when parsing. It must be there when serializing. 
#       How to add it when the rdflib trig serializer uses a different style (subject a dcat:Dataset)? (issue #12)
#   - There should be no json-ld:base triple (issue #11)

# CIMXML:
#   - Cannot use rdf:type dcterms:PeriodOfTime because it only supports flat structures (issue #14). This triple must be ommitted. Two triples will be present instead:
#     - dcat:startDate literal in utc time zone (ending with "Z"), no datatype
#     - dcat:endDate literal in utc time zone (ending with "Z"), no datatype
#   - There should not be a dcat:distribution or dcat:Distribution triple (issue #13)
#   - There should be a dcterms:issued triple which is expressed as a literal without datatype in utc time zone (ending with "Z"), and there should only be one such triple (issue #13).
#        (find out: if there are multiple with different values, which should I keep?)
#   - Should not have rdf:type rdfg:Graph triple, but should have rdf:type dcat:Dataset or md:FullModel triple (issue #12)
#   - There should be no json-ld:base triple (issue #11)

from email import header
import re
from typing import Tuple, Optional

from rdflib import XSD, BNode, Literal, Node, Graph, URIRef
from rdflib.namespace import DCAT, DCTERMS, RDF

from cim_plugin.enriching import cast_datetime_utc
from cim_plugin.namespaces import RDFG, JSONLD, MD
from cim_plugin.header import CIMMetadataHeader

import logging

logger = logging.getLogger("cimxml_logger")

Triple = Tuple[Node, Node, Node]


# ── Common checks (both formats) ──────────────────────────────────────────────

def _remove_invalid_triples(graph: Graph, predicates: Optional[URIRef|list[URIRef]] = None, obj: Optional[URIRef|list[URIRef]] = None) -> None:
    """Remove any triples with the given predicate(s) and/or object(s).
    
    Parameters:
        graph (Graph): The graph to remove triples from.
        predicates (URIRef or list of URIRef, optional): The predicate(s) to remove. If None, no predicate-based filtering is applied.
        obj (URIRef or list of URIRef, optional): The object(s) to remove. If None, no object-based filtering is applied.
    """
    if isinstance(predicates, URIRef):
        predicates = [predicates]
    if isinstance(obj, URIRef):
        obj = [obj]

    if predicates is None and obj is None:
        return

    for s, p, o in graph:
        # If a predicate and/or an object is in one of the lists, remove the entire triple.
        if (predicates is not None and p in predicates) or (obj is not None and o in obj):
            logger.error(f"Invalid triple detected in header, removing: ({s}, {p}, {o})")
            graph.remove((s, p, o))


DATETIME_REGEX = re.compile(
    r"""(
        \d{2}:\d{2}                |  # HH:MM → definitely datetime
        ^\d{8}[Tt]\d{6}([Zz]|[+-]\d{2}:?\d{2})?$  # compact ISO datetime
    )""",
    re.VERBOSE
)

def _fix_datetime_format(obj: Node) -> Node:
    """Fix the format of a Literal object to datetime.
    
    An attempt to cast the literal to datetime is made if the value does not look like a datetime already.
    No correction is made if the value looks like a datetime.

    Parameters:
        obj (Node): The object to check and potentially fix.

    Returns:
        Node: The corrected object or the original object if no correction was needed or possible.
    """
    if isinstance(obj, Literal):
        if obj.value is None:
            return obj

        text = obj.value.strip() if isinstance(obj.value, str) else str(obj.value).strip()

        # If it matches datetime patterns → leave unchanged
        if DATETIME_REGEX.search(text):
            if obj.datatype is None or obj.datatype != XSD.dateTime:
                return Literal(obj.value, datatype=XSD.dateTime)

            return obj

        # Otherwise, try to cast as datetime
        try:
            return cast_datetime_utc(obj)
        except ValueError as e:
            logger.error(f"Failed to correct datetime format for literal {obj}: {e}")
            return obj

    logger.error(f"Expected a Literal for datetime correction, got: {obj!r}")
    return obj


def _fix_datetime_format_in_triples(graph: Graph) -> None:
    """Fix datetime format for triples with these predicates:
    
        - dcat:endDate
        - dcat:startDate
        - dcterms:issued

    Parameters:
        graph (Graph): The graph to fix.
    """
    predicates = {DCAT.endDate, DCAT.startDate, DCTERMS.issued}
    triples = {
    (s, p, o)
    for s, p, o in graph.triples((None, None, None))
    if p in predicates
    }

    for s, p, o in triples:
        new_obj = _fix_datetime_format(o)
        if new_obj is None:
            logger.error(f"Found None for {p}. Expected a datetime.")
            continue

        if new_obj != o:
            graph.remove((s, p, o))
            graph.add((s, p, new_obj))
            logger.error(f"Corrected date format for predicate {p}: from {o} to {new_obj}.")


def _check_dcterms_issued_count(graph: Graph, identifier: URIRef) -> None:
    """Check that there is exactly one dcterms:issued triple for any subject. 
    
    If there are none, add a dummy triple with "unknown" as object and an identifier as subject. 
    If there are multiple, log an error.
    
    Parameters:
        graph (Graph): The graph to check.
        identifier (URIRef): The identifier to use for the dummy triple.
    """
    issued_triples = list(graph.triples((None, DCTERMS.issued, None)))
    if not issued_triples:
        logger.error("Missing required dcterms:issued triple. Creating dummy triple without date.")
        graph.add((identifier, DCTERMS.issued, Literal("unknown")))
    if len(issued_triples) > 1:
        logger.error(f"Multiple dcterms:issued triples found ({len(issued_triples)}): {issued_triples}. All but one should be removed.")

def _correct_triple_representation_by_predicate(graph: Graph, predicate: URIRef, identifier: URIRef) -> None:
    """Check that there is at least one triple with the given predicate. 
    
    Subject and object is ignored in search.
    If not, add a dummy triple with "unknown" as object and an identifier as subject.
    Logs error if multiple triples with the given predicate are found, but does not remove them.
    
    Parameters:
        graph (Graph): The graph to check.
        predicate (URIRef): The predicate to check for.
        identifier (URIRef): The identifier to use for the dummy triple if missing.
    """
    triples = list(graph.triples((None, predicate, None)))
    if len(triples) > 1:
        logger.error(f"Multiple {predicate} triples found. All but one should be removed.")
    if not triples:
        logger.error(f"Missing required {predicate} triple. Creating dummy triple without date.")
        graph.add((identifier, predicate, Literal("unknown")))

# ── TriG-specific checks ──────────────────────────────────────────────────────
# Keeping this for now in case we want to use it for checking the presence of a complete temporal triple before fixing the format.
def has_complete_temporal(graph: Graph, identifier: URIRef) -> bool:
    # Find the blank node connected via dcterms:temporal
    for o in graph.objects(identifier, DCTERMS.temporal):
        if isinstance(o, BNode):
            # Check required triples
            type_ok = (o, RDF.type, DCTERMS.PeriodOfTime) in graph
            start_ok = any(graph.objects(o, DCAT.startDate))
            end_ok = any(graph.objects(o, DCAT.endDate))

            if type_ok and start_ok and end_ok:
                return True

    return False


def _fix_trig_period_of_time_format(graph: Graph, identifier: URIRef) -> None:
    """Fix the format of dcterms:PeriodOfTime representation in Trig header.
    
    Triples with dcat:startDate and dcat:endDate predicates are moved into a new blank node connected to the identifier via dcterms:temporal.

    Parameters:
        graph (Graph): The graph to fix.
        identifier (URIRef): The identifier for the node group.
    """
    for s, p, o in list(graph.triples((None, DCTERMS.temporal, None))):
        graph.remove((s, p, o))

    bnode = BNode()
    graph.add((identifier, DCTERMS.temporal, bnode))

    for s in graph.subjects(RDF.type, DCTERMS.PeriodOfTime):
        graph.remove((s, RDF.type, DCTERMS.PeriodOfTime))
    graph.add((bnode, RDF.type, DCTERMS.PeriodOfTime))

    _make_bnode_triple_for_given_predicate(graph, bnode, DCAT.startDate)
    _make_bnode_triple_for_given_predicate(graph, bnode, DCAT.endDate)


def _make_bnode_triple_for_given_predicate(graph: Graph, bnode: BNode, predicate: URIRef) -> None:
    """Remake triples into a blank node triple with the given predicate and blank node as subject.
    
    If there already are triples with the given predicate, they are removed and replaced with triples with the same objects but the blank node as subject.
    If there are no triples with the given predicate, a new triple with the blank node as subject and "unknown" as object is created.

    Parameters:
        graph (Graph): The graph to modify.
        bnode (BNode): The blank node to use as the subject.
        predicate (URIRef): The predicate to use for the triple.
    """
    triples = list(graph.triples((None, predicate, None)))

    if len(triples) > 1:
        logger.error(f"Multiple {predicate} triples. All but one should be removed.")

    if triples:
        for s, p, o in triples:
            graph.remove((s, p, o))
            graph.add((bnode, p, o))
    else:
        logger.error(f"Missing required {predicate} triple. Creating dummy triple with no date.")
        graph.add((bnode, predicate, Literal("unknown")))


def _check_trig_rdfg_graph(graph: Graph, identifier: URIRef) -> None:
    """Check that an rdf:type rdfg:Graph triple is present in trig header.
    
    Will add the triple if missing.

    Parameters:
        graph (Graph): The graph to check.
        identifier (URIRef): The identifier (subject) of the triple.
    """
    graphtype = next(graph.triples((identifier, RDF.type, RDFG.Graph)), None)
    if not graphtype:
        logger.error("Missing required rdf:type rdfg:Graph triple for Trig header. Adding it.")
        graph.bind("rdfg", RDFG, override=False) # Needs testing
        graph.add((identifier, RDF.type, RDFG.Graph))

# ── CIMXML-specific checks ────────────────────────────────────────────────────


def _fix_cimxml_period_of_time_format(graph: Graph, identifier: URIRef) -> None:
    """Fix the format of dcterms:PeriodOfTime representation in CIMXML header, including dcat:startDate and dcat:endDate triples.
    
    Parameters:
        graph (Graph): The graph to fix.
        identifier (URIRef): The identifier to use for the dummy triple of startDate and endDate if missing.
    """
    for s in graph.subjects(RDF.type, DCTERMS.PeriodOfTime):
        graph.remove((s, RDF.type, DCTERMS.PeriodOfTime))

    _correct_triple_representation_by_predicate(graph, DCAT.endDate, identifier)
    _correct_triple_representation_by_predicate(graph, DCAT.startDate, identifier)


# Cannot use _remove_invalid_triples here because it would remove all triples with rdf:type as predicate.
def _remove_cimxml_rdfg_graph(graph: Graph) -> None:
    """Remove rdf:type rdfg:Graph triple if present in CIMXML header.
    
    The triple is removed regardless of what the subject is.

    Parameters:
        graph (Graph): The graph to check and modify.
    """
    graphtype = list(graph.triples((None, RDF.type, RDFG.Graph)))
    if graphtype:
        logger.error("Invalid rdf:type rdfg:Graph triple detected in CIMXML header, removing it.")
        for triple in graphtype:
            graph.remove(triple)


# ── Public validators ─────────────────────────────────────────────────────────


def validate_header(header: CIMMetadataHeader, format: str="cimxml") -> None:
    """Validate a CIMMetadataHeader according to the format specification.
    
    The validation is only performed for dcat:Dataset headers. md:FullModel and custom headers are ignored.

    Parameters:
        header (CIMMetadataHeader): The header to validate.
        format (str): The format to validate against ("cimxml" or "trig"). Default is "cimxml".
    """
    if format is None:
        format = "cimxml"

    if header is None or len(header.graph) == 0:
        logger.error("Header graph is empty. No validation performed.")
        return

    if header.header_type not in CIMMetadataHeader.DEFAULT_METADATA_OBJECTS:
        logger.error(f"Unknown header type: {header.header_type}. No validation performed.")
        return

    if header.header_type == MD.FullModel:
        logger.error(f"Validation for MD.FullModel header is not implemented yet. No validation performed for this header type.")
        return
    

    format = format.lower().strip()
    identifier = header.subject
    
    # Common checks
    _remove_invalid_triples(header.graph, predicates=[DCAT.distribution, JSONLD.base], obj=DCAT.Distribution)
    _fix_datetime_format_in_triples(header.graph) # This should be done before checking DCTERMS.issued as correction may remove duplicates automatically.
    _correct_triple_representation_by_predicate(header.graph, DCTERMS.issued, identifier)

    if format == "cimxml":
        _fix_cimxml_period_of_time_format(header.graph, identifier)
        _remove_cimxml_rdfg_graph(header.graph)
        
    elif format == "trig":
        _fix_trig_period_of_time_format(header.graph, identifier)
        _check_trig_rdfg_graph(header.graph, identifier)

    elif format == "jsonld":
         logger.error("Header validation for JSON-LD format is not implemented yet.")
    
    else:
        logger.error(f"Unknown format specified for header validation: {format}. No validation performed.")


if __name__ == "__main__":
    print("Header validation.")