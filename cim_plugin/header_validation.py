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

import re
from typing import Tuple, Optional

from rdflib import XSD, BNode, Literal, Node, Graph, URIRef
from rdflib.namespace import DCAT, DCTERMS, RDF

from cim_plugin.enriching import cast_datetime_utc
from cim_plugin.namespaces import RDFG, JSONLD
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


def _fix_datetime_format_in_triples(graph: Graph) -> None:
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
            logger.error(f"Corrected date format for triple ({s}, {p}, {o}) to ({s}, {p}, {new_obj})")

DATETIME_REGEX = re.compile(
    r"""(
        \d{2}:\d{2}                |  # HH:MM → definitely datetime
        ^\d{8}[Tt]\d{6}([Zz]|[+-]\d{2}:?\d{2})?$  # compact ISO datetime
    )""",
    re.VERBOSE
)

def _fix_datetime_format(obj: Node) -> Node:
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


# def _fix_datetime_format(obj: Node) -> Node:
#     """Fix datetime of literal object if it does not have required format.

#     A casting to datetime is attempted. If it fails, the original object is returned and an error is logged.
    
#     Parameters:
#         obj (Node): The object node to check and correct if necessary.

#     Returns:
#         Node: The original node if no correction was needed, or a new Literal with corrected datetime format.
#     """
#     if isinstance(obj, Literal):
#         if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}+.*", str(obj)):
#             try:
#                 return cast_datetime_utc(obj)
#             except ValueError as e:
#                 logger.error(f"Failed to correct datetime format for literal {obj}: {e}")
#                 return obj
#         return obj
#     else:
#         logger.error(f"Expected a Literal for datetime correction, got: {obj!r}")
#         return obj

def _check_dcterms_issued_count(graph: Graph, identifier: URIRef) -> None:
    issued_triples = list(graph.triples((None, DCTERMS.issued, None)))
    if not issued_triples:
        logger.error("Missing required dcterms:issued triple. Creating dummy triple without date.")
        graph.add((identifier, DCTERMS.issued, Literal("unknown")))
    if len(issued_triples) > 1:
        logger.error(f"Multiple dcterms:issued triples found ({len(issued_triples)}): {issued_triples}. All but one should be removed.")


# ── TriG-specific checks ──────────────────────────────────────────────────────

def _fix_trig_period_of_time_format(graph: Graph, identifier: URIRef) -> None:
    bnode = BNode()

    graph.remove((None, RDF.type, DCTERMS.PeriodOfTime))
    graph.add((identifier, DCTERMS.temporal, bnode))
    graph.add((bnode, RDF.type, DCTERMS.PeriodOfTime))

    _make_bnode_date_triple_for_period_of_time(graph, bnode, DCAT.startDate)
    _make_bnode_date_triple_for_period_of_time(graph, bnode, DCAT.endDate)


def _make_bnode_date_triple_for_period_of_time(graph: Graph, bnode: BNode, predicate: Node) -> None:
    triples = list(graph.triples((None, predicate, None)))

    if len(triples) == 1:
        graph.remove(triples[0])
        new_obj = _fix_datetime_format(triples[0][2])
        if new_obj != triples[0][2]:
            logger.error(f"Corrected {predicate} format for {triples[0][2]} -> {new_obj})")
            graph.add((bnode, predicate, new_obj))
        else:
            graph.add((bnode, predicate, triples[0][2]))

    elif len(triples) > 1:
        logger.error(f"Multiple {predicate} triples found for PeriodOfTime. Keeping only the first one and removing the rest.")
        for triple in triples[1:]:
            graph.remove(triple)
        new_obj = _fix_datetime_format(triples[0][2])
        if new_obj != triples[0][2]:
            logger.error(f"Corrected {predicate} format for {triples[0][2]} -> {new_obj})")
            graph.add((bnode, predicate, new_obj))
        else:
            graph.add((bnode, predicate, triples[0][2]))

    else:
        logger.error(f"Missing required {predicate} triple for PeriodOfTime. Creating dummy triple with no date.")
        graph.add((bnode, predicate, Literal("unknown")))


def _check_trig_rdfg_graph(graph: Graph, identifier: URIRef) -> None:
    """Check that an rdf:type rdfg:Graph triple is present (issue #12, TriG)."""
    graphtype = next(graph.triples((identifier, RDF.type, RDFG.Graph)), None)
    if not graphtype:
        logger.error("Missing required rdf:type rdfg:Graph triple for TriG header (issue #12). Adding it.")
        graph.add((identifier, RDF.type, RDFG.Graph))

# ── CIMXML-specific checks ────────────────────────────────────────────────────


def _fix_cimxml_period_of_time_format(graph: Graph, identifier: URIRef) -> None:
    graph.remove((None, RDF.type, DCTERMS.PeriodOfTime))

    endate = list(graph.triples((None, DCAT.endDate, None)))
    startdate = list(graph.triples((None, DCAT.startDate, None)))
    
    if len(endate) > 1:
        logger.error(f"Multiple dcat:endDate triples found for PeriodOfTime. All but one should be removed.")
    elif not endate:
        logger.error(f"Missing required dcat:endDate triple for PeriodOfTime. Creating dummy triple with no date.")
        graph.add((identifier, DCAT.endDate, Literal("unknown")))

    
    if len(startdate) > 1:
        logger.error(f"Multiple dcat:startDate triples found for PeriodOfTime. All but one should be removed.")
    elif not startdate:
        logger.error(f"Missing required dcat:startDate triple for PeriodOfTime. Creating dummy triple with no date.")
        graph.add((identifier, DCAT.startDate, Literal("unknown")))


def _remove_cimxml_rdfg_graph(graph: Graph) -> None:
    graphtype = next(graph.triples((None, RDF.type, RDFG.Graph)), None)
    if graphtype:
        logger.error("Invalid rdf:type rdfg:Graph triple detected in CIMXML header, removing it.")
        graph.remove(graphtype)


# ── Public validators ─────────────────────────────────────────────────────────


def validate_header(header: CIMMetadataHeader, format: str="cimxml") -> None:
    """Validate a CIMMetadataHeader according to the format specification. Logs errors for any issues found, but does not raise exceptions or modify the header.
    
    Parameters:
        header (CIMMetadataHeader): The header to validate.
        format (str): The format to validate against ("cimxml" or "trig"). Default is "cimxml".
    """
    format = format.lower()
    identifier = header.subject
    
    # Common checks
    # _remove_distribution(header.graph)
    # _remove_jsonld_base(header.graph)
    _remove_invalid_triples(header.graph, predicates=[DCAT.distribution, JSONLD.base], obj=DCAT.Distribution)
    _fix_datetime_format_in_triples(header.graph) # This must be done before _check_dcterms_issued_count as correction may remove duplicates automatically.
    _check_dcterms_issued_count(header.graph, identifier)

    if format == "cimxml":
        _fix_cimxml_period_of_time_format(header.graph, identifier)
        _remove_cimxml_rdfg_graph(header.graph)
    elif format == "trig":
        # Not using _fix_trig_period_of_time_format because the trig serializer would have to be changed for it to give intended result.
        # Using the _fix_cimxml_period_of_time_format for now, which makes sure the format of the dates is correct and that they are present. 
        # _fix_trig_period_of_time_format(graph)
        _fix_cimxml_period_of_time_format(header.graph, identifier)
        _check_trig_rdfg_graph(header.graph, identifier)
    elif format == "jsonld":
         logger.error("Header validation for JSON-LD format is not implemented yet.")
    else:
        logger.error(f"Unknown format specified for header validation: {format}. No validation performed.")


if __name__ == "__main__":
    print("Header validation.")