from rdflib import BNode, Literal, Namespace, URIRef
from rdflib.namespace import DCAT, RDF

from cim_plugin.cimtrig_serializer import CIMTrigSerializer
from cim_plugin.graph import CIMGraph
from cim_plugin.header import CIMMetadataHeader


def test_orders_header_subject_first() -> None:
    g = CIMGraph()
    ex = Namespace("http://example.com/")
    g.bind("ex", ex)

    header_subject = URIRef("http://example.com/header")
    other_subject = URIRef("http://example.com/other")

    linked_bnode = BNode("linked")
    unlinked_bnode = BNode("unlinked")

    header = CIMMetadataHeader.empty(header_subject)
    header.graph.bind("ex", ex)
    header.add_triple(RDF.type, DCAT.Dataset)
    g.metadata_header = header

    # Header triples are in the graph that gets serialized.
    g.add((header_subject, RDF.type, DCAT.Dataset))
    g.add((header_subject, ex.p, linked_bnode))

    g.add((other_subject, ex.p, linked_bnode))
    g.add((linked_bnode, ex.p, Literal("x")))
    g.add((unlinked_bnode, ex.p, Literal("y")))

    ser = CIMTrigSerializer(g)
    ser.reset()
    ser.preprocess()

    ordered = ser.orderSubjects()

    assert ordered[0] == header_subject
    assert ordered.index(linked_bnode) > ordered.index(unlinked_bnode)


def test_uses_default_order_for_plain_graph() -> None:
    from rdflib import Graph

    g = Graph()
    ex = Namespace("http://example.com/")
    g.bind("ex", ex)

    s1 = URIRef("http://example.com/s1")
    s2 = URIRef("http://example.com/s2")
    g.add((s1, ex.p, s2))

    ser = CIMTrigSerializer(g)
    ser.reset()
    ser.preprocess()

    assert ser.orderSubjects() == super(CIMTrigSerializer, ser).orderSubjects()


def test_preprocess_merges_header_namespaces() -> None:
    g = CIMGraph()
    ex = Namespace("http://example.com/")
    foo = Namespace("http://foo.org/ns#")

    g.bind("ex", ex)

    header_subject = URIRef("http://example.com/header")
    header = CIMMetadataHeader.empty(header_subject)
    header.graph.bind("foo", foo)
    g.metadata_header = header

    # The graph uses a URI from header namespace, but graph does not bind foo.
    g.add((URIRef("http://foo.org/ns#s"), ex.p, Literal("v")))

    ser = CIMTrigSerializer(g)
    ser.reset()
    ser.preprocess()

    assert "foo" in ser.namespaces
