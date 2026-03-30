from unittest.mock import MagicMock, patch, call
import pytest
from rdflib import BNode, Literal, Namespace, URIRef, Graph
from rdflib.namespace import DCAT, RDF
from rdflib.plugins.serializers.turtle import OBJECT, SUBJECT

from cim_plugin.cimtrig_serializer import CIMTrigSerializer
from cim_plugin.graph import CIMGraph
from cim_plugin.header import CIMMetadataHeader


# Unit tests .reset
def test_reset() -> None:
    g = CIMGraph()
    g.add((URIRef("http://example.com/s"), DCAT.keyword, BNode("o")))
    ser = CIMTrigSerializer(g)
    ser.reset()

    assert hasattr(ser, "_object_refs")
    assert ser._object_refs == {}

# Unit tests .preprocess
def test_preprocess_mergesnamespaces() -> None:
    graph = CIMGraph()
    header = CIMMetadataHeader.empty(URIRef("http://example.com/header"))
    header.graph.bind("ex", URIRef("http://example.com/"))
    graph.metadata_header = header
    
    ser = CIMTrigSerializer(graph)
    ser.preprocess()

    assert graph.namespace_manager.store.namespace("ex") == URIRef("http://example.com/")


def test_preprocess_mergestriples() -> None:
    graph = CIMGraph()
    header = CIMMetadataHeader.empty(URIRef("http://example.com/header"))
    header.graph.bind("ex", URIRef("http://example.com/"))
    header.add_triple(DCAT.keyword, Literal("test"))
    graph.metadata_header = header
    
    ser = CIMTrigSerializer(graph)
    ser.preprocess()

    assert (URIRef("http://example.com/header"), DCAT.keyword, Literal("test")) in graph


def test_preprocess_noncimgraph() -> None:
    graph = Graph()
    header = CIMMetadataHeader.empty(URIRef("http://example.com/header"))
    header.graph.bind("ex", URIRef("http://example.com/"))
    header.add_triple(DCAT.keyword, Literal("test"))
    # Pylance silenced to test graph type handling
    graph.metadata_header = header  # type: ignore
    
    ser = CIMTrigSerializer(graph)
    ser.preprocess()

    assert graph.namespace_manager.store.namespace("ex") == None
    assert (URIRef("http://example.com/header"), DCAT.keyword, Literal("test")) not in graph


def test_preprocess_headerisnone() -> None:
    graph = CIMGraph()
    graph.metadata_header = None
    
    ser = CIMTrigSerializer(graph)
    ser.preprocess()

    assert len(graph) == 0


def test_preprocess_namespaceconflict() -> None:
    graph = CIMGraph()
    graph.bind("ex", Namespace("http://different.com/"))
    header = CIMMetadataHeader.empty(URIRef("http://example.com/header"))
    header.graph.bind("ex", URIRef("http://example.com/"))
    header.add_triple(DCAT.keyword, Literal("test"))
    graph.metadata_header = header
    
    ser = CIMTrigSerializer(graph)
    ser.preprocess()

    # Graph namespace is kept, but header namespace is in the triples.
    assert graph.namespace_manager.store.namespace("ex") == URIRef("http://different.com/")
    assert (URIRef("http://example.com/header"), DCAT.keyword, Literal("test")) in graph


# Unit tests .preprocessTriple
def test_preprocessTriple_nobnode() -> None:
    graph = CIMGraph()
    triple = (URIRef("http://example.com/s"), DCAT.keyword, Literal("o"))
    
    ser = CIMTrigSerializer(graph)
    ser.reset()
    ser.preprocessTriple(triple)

    assert ser._object_refs == {}


def test_preprocessTriple_onebnode() -> None:
    graph = CIMGraph()
    bnode = BNode("o")
    triple = (URIRef("http://example.com/s"), DCAT.keyword, bnode)
    
    ser = CIMTrigSerializer(graph)
    ser.reset()
    ser.preprocessTriple(triple)

    assert ser._object_refs == {bnode: 1}


def test_preprocessTriple_severalbnodes() -> None:
    graph = CIMGraph()
    bnode = BNode("o")
    old_bnode = BNode("old")
    triple = (URIRef("http://example.com/s"), DCAT.keyword, bnode)
    
    ser = CIMTrigSerializer(graph)
    ser.reset()
    ser._object_refs[old_bnode] = 2
    ser.preprocessTriple(triple)

    assert ser._object_refs == {old_bnode: 2, bnode: 1}


# Unit tests .orderSubjects
def test_orders_header_subject_first() -> None:
    g = CIMGraph()
    ex = Namespace("http://example.com/")
    g.bind("ex", ex)

    header_subject = URIRef("http://example.com/header")
    other_subject = URIRef("http://example.com/other")
    another_subject = URIRef("http://example.com/another")

    linked_bnode = BNode("linked")
    unlinked_bnode = BNode("unlinked")
    object_bnode = BNode("object")

    header = CIMMetadataHeader.empty(header_subject)
    header.graph.bind("ex", ex)
    header.add_triple(RDF.type, DCAT.Dataset)
    header.add_triple(ex.p, linked_bnode)
    header.graph.add((linked_bnode, ex.p, Literal("x")))
    g.metadata_header = header


    g.add((other_subject, ex.p, Literal("v")))
    g.add((unlinked_bnode, ex.p, Literal("y")))
    g.add((another_subject, ex.p, object_bnode))  # Object node does not affect ordering unless linked

    ser = CIMTrigSerializer(g)
    ser.reset()
    ser.preprocess()

    ordered = ser.orderSubjects()

    assert ordered[0] == header_subject
    assert ordered.index(linked_bnode) < ordered.index(unlinked_bnode)
    assert ordered.index(unlinked_bnode) < ordered.index(another_subject)
    assert ordered.index(another_subject) < ordered.index(other_subject)

@pytest.mark.parametrize(
        "graphtype, header", 
        [
            pytest.param("cimgraph", None, id="CIMGraph without header"),
            pytest.param("cimgraph", CIMMetadataHeader.empty(URIRef("h1")), id="CIMGraph with empty header"),
            pytest.param("rdflib graph", None, id="rdflib Graph")
        ])
def test_orderSubjects_defaultorder(graphtype: str, header: CIMMetadataHeader | None) -> None:
    if graphtype == "cimgraph":
        g = CIMGraph()
        g.metadata_header = header
    else:
        g = Graph()
    
    ex = Namespace("http://example.com/")
    g.bind("ex", ex)

    s1 = URIRef("http://example.com/s1")
    s2 = URIRef("http://example.com/s2")
    g.add((s2, ex.p, Literal("v2")))
    g.add((s1, ex.p, Literal("v1")))

    ser = CIMTrigSerializer(g)
    ser.reset()
    ser.preprocess()

    assert ser.orderSubjects() == super(CIMTrigSerializer, ser).orderSubjects()
    assert ser.orderSubjects() == [s1, s2]

# Unit tests .p_squared
bnode = BNode("test")

@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param((URIRef("http://example.com/node"), {}, 1, OBJECT), id="Non-BNode node should not be p-squared"),
        pytest.param((bnode, {bnode: True}, 1, OBJECT), id="Already serialized node should not be p-squared"),
        pytest.param((bnode, {}, 2, OBJECT), id="Object referenced multiple times should not be p-squared"),
        pytest.param((bnode, {}, 1, SUBJECT), id="Wrong position"),
    ]
)
@patch.object(CIMTrigSerializer, "write")
@patch.object(CIMTrigSerializer, "doList")
@patch.object(CIMTrigSerializer, "predicateList")
@patch.object(CIMTrigSerializer, "subjectDone")
@patch.object(CIMTrigSerializer, "isValidList")
def test_p_squared_falsereturns(mock_is_list: MagicMock, mock_subject_done: MagicMock, mock_pred_list: MagicMock, mock_do_list: MagicMock, mock_write: MagicMock, parameters: tuple) -> None:
    n, serialized, object_refs, position = parameters
    ser = CIMTrigSerializer(Graph())
    ser._serialized = serialized
    ser._object_refs = {}

    node = n
    ser._object_refs[node] = object_refs

    result = ser.p_squared(node, position=position, newline=False)

    assert result is False
    mock_is_list.assert_not_called()
    mock_subject_done.assert_not_called()
    mock_pred_list.assert_not_called()
    mock_write.assert_not_called()
    mock_do_list.assert_not_called()


@patch.object(CIMTrigSerializer, "write")
@patch.object(CIMTrigSerializer, "subjectDone")
@patch.object(CIMTrigSerializer, "isValidList", return_value=True)
@patch.object(CIMTrigSerializer, "doList")
def test_p_squared_listbranch(mock_do_list: MagicMock, mock_is_list: MagicMock, mock_subject_done: MagicMock, mock_write: MagicMock) -> None:
    ser = CIMTrigSerializer(Graph())
    ser._serialized = {}
    ser._object_refs = {}

    node = BNode()
    ser._object_refs[node] = 1
    
    result = ser.p_squared(node, position=OBJECT, newline=False)

    assert result is True
    mock_is_list.assert_called_once_with(node)
    mock_write.assert_any_call(" ")
    mock_write.assert_any_call("(")
    mock_do_list.assert_called_once_with(node)
    mock_write.assert_any_call(")")

    mock_subject_done.assert_not_called()


@patch.object(CIMTrigSerializer, "write")
@patch.object(CIMTrigSerializer, "doList")
@patch.object(CIMTrigSerializer, "predicateList")
@patch.object(CIMTrigSerializer, "subjectDone")
@patch.object(CIMTrigSerializer, "isValidList", return_value=False)
def test_p_squared_nonlistbranch(mock_is_list: MagicMock, mock_subject_done: MagicMock, mock_pred_list: MagicMock, mock_do_list: MagicMock, mock_write: MagicMock) -> None:
    ser = CIMTrigSerializer(Graph())
    ser._serialized = {}
    ser._object_refs = {}

    node = BNode()
    ser._object_refs[node] = 1

    result = ser.p_squared(node, position=OBJECT, newline=False)

    assert result is True
    mock_is_list.assert_called_once_with(node)
    mock_subject_done.assert_called_once_with(node)
    mock_pred_list.assert_called_once_with(node, newline=False)
    mock_write.assert_any_call("[")
    mock_write.assert_any_call("]")

    mock_do_list.assert_not_called()


@patch.object(CIMTrigSerializer, "write")
@patch.object(CIMTrigSerializer, "subjectDone")
@patch.object(CIMTrigSerializer, "predicateList")
@patch.object(CIMTrigSerializer, "isValidList", return_value=False)
def test_p_squared_newlinetrue(mock_is_list: MagicMock, mock_pred_list: MagicMock, mock_subject_done: MagicMock, mock_write: MagicMock) -> None:
    ser = CIMTrigSerializer(Graph())
    ser._serialized = {}
    ser._object_refs = {}

    node = BNode()
    ser._object_refs[node] = 1

    result = ser.p_squared(node, position=OBJECT, newline=True)

    assert result is True
    mock_write.assert_has_calls([call("["), call("]")], any_order=True)

if __name__ == "__main__":
    pytest.main()