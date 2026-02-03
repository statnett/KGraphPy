from pathlib import Path
import pytest
from unittest.mock import MagicMock, Mock, patch
import uuid
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, DCAT
from rdflib.exceptions import ParserError
from typing import Callable
from cim_plugin.exceptions import CIMXMLParseError
from cim_plugin.namespaces import MD
from tests.fixtures import cimxml_plugin, mock_extract_uuid
from cim_plugin.utilities import (
    _extract_uuid_from_urn, 
    get_graph_uuid, 
    load_cimxml_graph, 
    collect_cimxml_to_dataset, 
    extract_subjects_by_object_type,
)

import logging

logger = logging.getLogger('cimxml_logger')


# Unit tests get_graph_uuid

@pytest.mark.parametrize(
    "rdf_type",
    [
        pytest.param(lambda: MD.FullModel, id="md:FullModel"),
        pytest.param(lambda: DCAT.Dataset, id="dcat:Dataset"),
    ]
)
def test_get_graph_uuid_correctsubject(rdf_type: Callable, mock_extract_uuid: Mock) -> None:
    graph = Graph()
    subject = URIRef("urn:uuid:12345678-1234-5678-1234-567812345678")

    graph.add((subject, RDF.type, rdf_type()))

    result = get_graph_uuid(graph)

    assert result == uuid.UUID("12345678-1234-5678-1234-567812345678")
    mock_extract_uuid.assert_called_once_with(str(subject))

def test_get_graph_uuid_emptygraph(mock_extract_uuid: Mock) -> None:
    graph = Graph()

    with pytest.raises(ValueError):
        get_graph_uuid(graph)

    mock_extract_uuid.assert_not_called()


def test_get_graph_uuid_bothtypesandnoise() -> None:
    graph = Graph()

    fullmodel_uuid = uuid.uuid4()
    dataset_uuid = uuid.uuid4()
    fullmodel_subject = URIRef(f"urn:uuid:{fullmodel_uuid}")
    dataset_subject = URIRef(f"urn:uuid:{dataset_uuid}")
    graph.add((fullmodel_subject, RDF.type, MD.FullModel))
    graph.add((dataset_subject, RDF.type, DCAT.Dataset))
    graph.add((URIRef("http://example.org/noise"), RDFS.label, Literal("irrelevant")))

    result = get_graph_uuid(graph)

    assert result == fullmodel_uuid


def test_get_graph_uuid_nofullmodel() -> None:
    graph = Graph()

    dataset_uuid = uuid.uuid4()
    dataset_subject = URIRef(f"urn:uuid:{dataset_uuid}")
    graph.add((dataset_subject, RDF.type, DCAT.Dataset))

    result = get_graph_uuid(graph)

    assert result == dataset_uuid


def test_get_graph_uuid_multiplefullmodels() -> None:
    graph = Graph()

    uuid1 = uuid.uuid4()
    uuid2 = uuid.uuid4()
    subject1 = URIRef(f"urn:uuid:{uuid1}")
    subject2 = URIRef(f"urn:uuid:{uuid2}")
    graph.add((subject1, RDF.type, MD.FullModel))
    graph.add((subject2, RDF.type, MD.FullModel))

    result = get_graph_uuid(graph)

    assert result == uuid1 # First encountered is returned


def test_get_graph_uuid_malformedurn() -> None:
    graph = Graph()

    bad_subject = URIRef("urn:uuid:not-a-valid-uuid")
    graph.add((bad_subject, RDF.type, MD.FullModel))

    with pytest.raises(ValueError):
        get_graph_uuid(graph)


def test_get_graph_uuid_ignoresblanknodes() -> None:
    graph = Graph()

    uid = uuid.uuid4()
    subject = URIRef(f"urn:uuid:{uid}")

    graph.add((subject, RDF.type, MD.FullModel))
    graph.add((BNode(), RDF.type, BNode()))

    result = get_graph_uuid(graph)

    assert result == uid
    assert len(graph) == 2


def test_get_graph_uuid_uppercaseurn() -> None:
    graph = Graph()

    uid = uuid.uuid4()
    subject = URIRef(f"urn:uuid:{str(uid).upper()}")    # Uppercase uuid, but not prefix

    graph.add((subject, RDF.type, MD.FullModel))

    result = get_graph_uuid(graph)

    assert result == uid


def test_get_graph_uuid_ignoresotherrdftypes() -> None:
    graph = Graph()

    uid = uuid.uuid4()
    subject = URIRef(f"urn:uuid:{uid}")

    graph.add((subject, RDF.type, RDFS.Class))

    with pytest.raises(ValueError):
        get_graph_uuid(graph)


# Unit tests _extract_uuid_from_urn
@pytest.mark.parametrize(
    "input, expected, error_match", [
        pytest.param("urn:uuid:6ba7b810-9dad-11d1-80b4-00c04fd430c8", 
                     "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
                     None, 
                     id="Version 1 (time-based)"),
        pytest.param("urn:uuid:f81d4fae-7dec-3f00-a7a5-21f12f3f3e21",
                     "f81d4fae-7dec-3f00-a7a5-21f12f3f3e21",
                     None,
                     id="Version 3 (namespace + MD5)"),
        pytest.param("urn:uuid:550e8400-e29b-41d4-a716-446655440000",
                     "550e8400-e29b-41d4-a716-446655440000",
                     None,
                     id="Version 4 (random)"),
        pytest.param("urn:uuid:6ba7b811-9dad-11d1-80b4-00c04fd430c8",
                     "6ba7b811-9dad-11d1-80b4-00c04fd430c8",
                     None,
                     id="Version 5 (namespace + SHA-1)"),
        pytest.param("urn:uuid:550E8400-E29B-41D4-A716-446655440000", "550e8400-e29b-41d4-a716-446655440000", None, id="Uppercase UUID" ),
        pytest.param("urn:uuid:550e8400-e29b-41d4-a716-446655440000 ", "550e8400-e29b-41d4-a716-446655440000", "badly formed hexadecimal UUID string", id="Trailing whitespace" ),
        pytest.param(" urn:uuid:550e8400-e29b-41d4-a716-446655440000", None, "Invalid model URI:", id="Leading whitespace breaks prefix" ),
        pytest.param("urn1234", None, "Invalid model URI:", id="Invalid prefix"),
        pytest.param("urn:UUID:550e8400-e29b-41d4-a716-446655440000", "550e8400-e29b-41d4-a716-446655440000", None, id="Wrong case in prefix" ),
        pytest.param("urn:uuid:", None, "badly formed hexadecimal UUID string", id="Missing UUID after prefix" ),
        pytest.param("urn:uuid:not-a-uuid", None, "badly formed hexadecimal UUID string", id="Malformed UUID - non-hex" ),
        pytest.param("urn:uuid:1234", None, "badly formed hexadecimal UUID string", id="Malformed UUID - too short" ),
        pytest.param("urn:uuid:550e8400-e29b-41d4-a716-446655440000-extra", None, "badly formed hexadecimal UUID string", id="Extra characters after UUID" ),
    ]
)
def test_extract_uuid_from_urn(input: str, expected: str|None, error_match: str|None) -> None:
    if error_match:
        with pytest.raises(ValueError, match=error_match):
            _extract_uuid_from_urn(input)
    else:
        result = _extract_uuid_from_urn(input)
        assert result == uuid.UUID(expected)


# Unit tests load_cimxml_graph

@patch("cim_plugin.utilities.get_graph_uuid")
@patch("cim_plugin.utilities.Graph")
def test_load_cimxml_graph_success(mock_graph_cls: MagicMock, mock_get_uuid: MagicMock) -> None:
    mock_graph = MagicMock(spec=Graph)
    mock_graph_cls.return_value = mock_graph
    mock_get_uuid.return_value = "12345678-1234-5678-1234-567812345678"

    uuid, graph = load_cimxml_graph("dummy.xml")

    mock_graph.parse.assert_called_once_with("dummy.xml", format="cimxml", schema_path=None)
    mock_get_uuid.assert_called_once_with(mock_graph)
    assert uuid == mock_get_uuid.return_value
    assert graph is mock_graph


@patch("cim_plugin.utilities.get_graph_uuid", return_value="uuid")
@patch("cim_plugin.utilities.Graph")
def test_load_cimxml_graph_schema_path(mock_graph_cls: MagicMock, mock_get_uuid: MagicMock) -> None:
    mock_graph = mock_graph_cls.return_value
    
    uuid, graph = load_cimxml_graph("file.xml", schema_path="schema.xsd")

    mock_graph.parse.assert_called_once_with("file.xml", format="cimxml", schema_path="schema.xsd")
    mock_get_uuid.assert_called_once_with(mock_graph)
    assert uuid == "uuid"
    assert graph is mock_graph



@pytest.mark.parametrize(
        "exception", [
            pytest.param(FileNotFoundError("missing"), id="FileNotFound"),
            pytest.param(ParserError("bad rdf"), id="ParserError"),
            pytest.param(ValueError("invalid"), id="ValueError"),
            # Note: SAXParseException is not tested because constructing a valid Locator
            # object requires implementing the full SAX interface. The other parsing errors
            # already verify the exception-wrapping behavior.
        ]
)
@patch("cim_plugin.utilities.get_graph_uuid")
@patch("cim_plugin.utilities.Graph")
def test_load_cimxml_graph_exceptions(mock_graph_cls: MagicMock, mock_get_uuid: MagicMock, exception: Exception) -> None:
    mock_graph = mock_graph_cls.return_value
    mock_graph.parse.side_effect = exception

    with pytest.raises(CIMXMLParseError) as exc:
        load_cimxml_graph("bad.xml")

    assert "bad.xml" in str(exc.value)
    mock_get_uuid.assert_not_called()


# Unit tests collect_cimxml_to_dataset
def test_collect_cimxml_to_dataset_emptylist() -> None:
    ds = collect_cimxml_to_dataset([])

    contexts = list(ds.graphs())
    assert len(contexts) == 1

    default_graph = ds.default_graph
    assert contexts[0].identifier == default_graph.identifier
    assert len(default_graph) == 0


@patch("cim_plugin.utilities.load_cimxml_graph")
def test_collect_cimxml_to_dataset_singlefile(mock_loader: MagicMock) -> None:
    g = Graph()
    g.add((URIRef("s"), URIRef("p"), URIRef("o")))
    g.namespace_manager.bind("ex", "http://example.com/")

    mock_loader.return_value = ("uuid1", g)

    ds = collect_cimxml_to_dataset(["file1.xml"])

    named = ds.graph(URIRef("urn:uuid:uuid1"))

    assert len(named) == 1
    assert (URIRef("s"), URIRef("p"), URIRef("o")) in named

    assert ds.namespace_manager.store.namespace("ex") == URIRef("http://example.com/")
    assert named.namespace_manager.store.namespace("ex") == URIRef("http://example.com/")
    assert mock_loader.call_count == 1


@patch("cim_plugin.utilities.load_cimxml_graph")
def test_collect_cimxml_to_dataset_multiplefiles(mock_loader: MagicMock) -> None:
    g1 = Graph()
    g1.add((URIRef("s1"), URIRef("p1"), URIRef("o1")))

    g2 = Graph()
    g2.add((URIRef("s2"), URIRef("p2"), URIRef("o2")))

    mock_loader.side_effect = [("uuid1", g1), ("uuid2", g2),]

    ds = collect_cimxml_to_dataset(["a.xml", "b.xml"])

    g1_named = ds.graph(URIRef("urn:uuid:uuid1"))
    g2_named = ds.graph(URIRef("urn:uuid:uuid2"))

    assert (URIRef("s1"), URIRef("p1"), URIRef("o1")) in g1_named
    assert (URIRef("s2"), URIRef("p2"), URIRef("o2")) in g2_named
    assert mock_loader.call_count == 2

@patch("cim_plugin.utilities.load_cimxml_graph")
def test_collect_cimxml_to_dataset_multiplenssameprefix(mock_loader: MagicMock) -> None:
    # Two graphs with different namespaces to same prefix. First added wins in dataset. 
    # Namespace changed for second graph.
    g1 = Graph()
    g1.bind("foo", "bar.com")
    g1.add((URIRef("foo:s1"), URIRef("foo:p1"), URIRef("foo:o1")))

    g2 = Graph()
    g2.bind("foo", "foo.com")
    g2.add((URIRef("foo:s2"), URIRef("foo:p2"), URIRef("foo:o2")))

    mock_loader.side_effect = [("uuid1", g1), ("uuid2", g2),]

    ds = collect_cimxml_to_dataset(["a.xml", "b.xml"])

    g1_named = ds.graph(URIRef("urn:uuid:uuid1"))
    g2_named = ds.graph(URIRef("urn:uuid:uuid2"))

    assert (URIRef("foo:s1"), URIRef("foo:p1"), URIRef("foo:o1")) in g1_named
    assert (URIRef("foo:s2"), URIRef("foo:p2"), URIRef("foo:o2")) in g2_named
    assert mock_loader.call_count == 2
    assert ds.namespace_manager.store.namespace("foo") == URIRef("bar.com")
    assert g1_named.namespace_manager.store.namespace("foo") == URIRef("bar.com")
    assert g2_named.namespace_manager.store.namespace("foo") == URIRef("bar.com")


@patch("cim_plugin.utilities.load_cimxml_graph")
def test_collect_cimxml_to_dataset_multipleprefixsamens(mock_loader: MagicMock) -> None:
    # Two graphs with same namespaces bound to different prefixes. Only one is kept; the last. 
    # Namespace changed to None for first graph.
    g1 = Graph()
    g1.bind("foo", "bar.com")
    g1.add((URIRef("foo:s1"), URIRef("foo:p1"), URIRef("foo:o1")))

    g2 = Graph()
    g2.bind("bar", "bar.com")
    g2.add((URIRef("bar:s2"), URIRef("bar:p2"), URIRef("bar:o2")))

    mock_loader.side_effect = [("uuid1", g1), ("uuid2", g2),]

    ds = collect_cimxml_to_dataset(["a.xml", "b.xml"])

    g1_named = ds.graph(URIRef("urn:uuid:uuid1"))
    g2_named = ds.graph(URIRef("urn:uuid:uuid2"))

    assert (URIRef("foo:s1"), URIRef("foo:p1"), URIRef("foo:o1")) in g1_named
    assert (URIRef("bar:s2"), URIRef("bar:p2"), URIRef("bar:o2")) in g2_named
    assert mock_loader.call_count == 2
    assert ds.namespace_manager.store.namespace("bar") == URIRef("bar.com")
    assert ds.namespace_manager.store.namespace("foo") == None
    assert g1_named.namespace_manager.store.namespace("bar") == URIRef("bar.com")
    assert g1_named.namespace_manager.store.namespace("foo") == None
    assert g2_named.namespace_manager.store.namespace("bar") == URIRef("bar.com")


@patch("cim_plugin.utilities.load_cimxml_graph")
def test_collect_cimxml_to_dataset_samefileinputtwice(mock_loader: MagicMock) -> None:
    g1 = Graph()
    g1.add((URIRef("s1"), URIRef("p1"), URIRef("o1")))

    mock_loader.side_effect = [("uuid1", g1), ("uuid1", g1),]

    ds = collect_cimxml_to_dataset(["a.xml", "a.xml"])

    g1_named = ds.graph(URIRef("urn:uuid:uuid1"))

    assert len(list(ds.graphs())) == 2  # 1 graph + default graph
    assert len(g1_named) == 1   # No duplicate triples
    assert (URIRef("s1"), URIRef("p1"), URIRef("o1")) in g1_named
    assert mock_loader.call_count == 2

@patch("cim_plugin.utilities.load_cimxml_graph")
def test_collect_cimxml_to_dataset_nonamespaces(mock_loader: MagicMock) -> None:
    g1 = Graph()
    g1.add((URIRef("s1"), URIRef("p1"), URIRef("o1")))

    blank = Graph()

    mock_loader.return_value = ("uuid1", g1)

    ds = collect_cimxml_to_dataset(["a.xml"])

    g1_named = ds.graph(URIRef("urn:uuid:uuid1"))

    assert (URIRef("s1"), URIRef("p1"), URIRef("o1")) in g1_named
    assert mock_loader.call_count == 1
    # No user defined namespaces added
    blank_ns = set(blank.namespaces())
    ds_ns = set(ds.namespaces())
    g1_ns = set(g1_named.namespaces())
    assert ds_ns - blank_ns == set()
    assert g1_ns - blank_ns == set()


@patch("cim_plugin.utilities.load_cimxml_graph")
def test_collect_cimxml_to_dataset_nondata(mock_loader: MagicMock) -> None:
    g1 = Graph()
    g1.bind("ex", "http://example.com/")

    mock_loader.return_value = ("uuid1", g1)

    ds = collect_cimxml_to_dataset(["a.xml"])

    g1_named = ds.graph(URIRef("urn:uuid:uuid1"))
    assert len(g1_named) == 0
    assert mock_loader.call_count == 1
    assert ds.namespace_manager.store.namespace("ex") == URIRef("http://example.com/")
    assert g1_named.namespace_manager.store.namespace("ex") == URIRef("http://example.com/")


@patch("cim_plugin.utilities.load_cimxml_graph")
def test_collect_cimxml_to_dataset_failedfile(mock_loader: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    good_graph = Graph()
    good_graph.add((URIRef("s"), URIRef("p"), URIRef("o")))

    mock_loader.side_effect = [
        CIMXMLParseError("bad.xml", Exception("fail")),
        ("uuid2", good_graph),
    ]

    with caplog.at_level("ERROR"):
        ds = collect_cimxml_to_dataset(["bad.xml", "good.xml"])

    named = ds.graph(URIRef("urn:uuid:uuid2"))
    assert len(named) == 1
    assert any("bad.xml" in msg for msg in caplog.messages)
    assert len(list(ds.graphs())) == 2  # 1 named graph in addition to default graph
    assert mock_loader.call_count == 2
     

@pytest.mark.parametrize(
        "schema", [
            pytest.param(None, id="No schema"), 
            pytest.param("schema.yaml", id="Schema present")
        ])
@patch("cim_plugin.utilities.load_cimxml_graph")
def test_collect_cimxml_to_dataset_passesschema(mock_loader: MagicMock, schema: str|None) -> None:
    g = Graph()
    mock_loader.return_value = ("uuid1", g)

    ds = collect_cimxml_to_dataset(["file.xml"], schema_path=schema)

    mock_loader.assert_called_once_with("file.xml", schema)
    named = ds.graph(URIRef("urn:uuid:uuid1"))
    assert len(named) == 0


def test_collect_cimxml_to_dataset_integrationrealparse(tmp_path: Path, caplog: pytest.LogCaptureFixture, cimxml_plugin: None) -> None:
    uuid = "12345678-1234-5678-1234-567812345678"
    subject = f"urn:uuid:{uuid}"

    xml = f"""<?xml version="1.0"?>
        <rdf:RDF
            xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
            xmlns:md="{MD}">
            <md:FullModel rdf:about="{subject}" />
        </rdf:RDF>
    """

    file_path = tmp_path / "model.xml"
    file_path.write_text(xml, encoding="utf-8")

    with caplog.at_level("INFO"):
        ds = collect_cimxml_to_dataset([str(file_path)], schema_path=None)

    named = ds.graph(URIRef(f"urn:uuid:{uuid}"))
    assert len(named) == 1
    assert (URIRef(subject), RDF.type, MD.FullModel) in named
    assert len(ds.default_graph) == 0
    assert any(
        "Cannot perform post processing without the model" in msg
        for msg in caplog.messages
    )


@pytest.mark.parametrize(
        "object_type, expected",
        [
            pytest.param([MD.FullModel], {URIRef("s2"), URIRef("s4")}, id="Multiple catches"),
            pytest.param([DCAT.Dataset], {URIRef("s3")}, id="Single catch"),
            pytest.param([URIRef("NotFound")], set(), id="Nothing found"),
            pytest.param([MD.FullModel, DCAT.Dataset], {URIRef("s2"), URIRef("s3"), URIRef("s4")}, id="Catching more than one type"),
            pytest.param([URIRef("o1")], set(), id="Not rdf:type"),
            pytest.param([123, None, DCAT.Dataset], {URIRef("s3")}, id="Invalid input format"),
            pytest.param([], set(), id="Empty input"),
        ]
)
def test_extract_subjects_by_type_various(object_type: list[URIRef], expected: set[URIRef]) -> None:
    g = Graph()
    g.add((URIRef("s1"), URIRef("noise"), URIRef("o1")))
    g.add((URIRef("s2"), RDF.type, MD.FullModel))
    g.add((URIRef("s3"), RDF.type, DCAT.Dataset))
    g.add((URIRef("s4"), RDF.type, MD.FullModel))

    result = extract_subjects_by_object_type(g, object_type=object_type)
    assert set(result) == expected


@pytest.mark.parametrize(
        "triples",
        [
            pytest.param(None, id="Empty graph"),
            pytest.param((URIRef("s1"), URIRef("p"), URIRef("o")), id="No rdf:type in graph")
        ]
)
def test_extract_subjects_by_object_type_emptyinput(triples: tuple[URIRef, URIRef, URIRef]) -> None:
    g = Graph()
    if triples:
        g.add(triples)
    result = extract_subjects_by_object_type(g, [MD.FullModel])
    assert result == []


@pytest.mark.parametrize(
        "input, expected",
        [
            pytest.param([DCAT.Dataset], [URIRef("s1")], id="Catching one"),
            pytest.param([DCAT.Dataset, MD.FullModel], [URIRef("s1"), URIRef("s1")], id="Catching both"),
        ]
)
def test_extract_subjects_by_object_type_multipletypespersubject(input: list[URIRef], expected: list[URIRef]) -> None:
    g = Graph()
    g.add((URIRef("s1"), RDF.type, MD.FullModel))
    g.add((URIRef("s1"), RDF.type, DCAT.Dataset))
    result = extract_subjects_by_object_type(g, input)
    assert result == expected


def test_extract_subjects_by_object_type_performance():
    g = Graph()
    g.add((URIRef("s1"), RDF.type, MD.FullModel))
    many_types = [URIRef(f"t{i}") for i in range(1000)] + [MD.FullModel]
    result = extract_subjects_by_object_type(g, many_types)
    assert result == [URIRef("s1")]


def test_extract_subjects_by_object_type_blanksubjects():
    b = BNode()
    g = Graph()
    g.add((b, RDF.type, MD.FullModel))
    result = extract_subjects_by_object_type(g, [MD.FullModel])
    assert result == [b]

def test_extract_subjects_by_object_type_literalsubject():
    g = Graph()
    g.add((Literal("s1"), RDF.type, MD.FullModel))
    result = extract_subjects_by_object_type(g, [MD.FullModel])
    assert result == [Literal("s1")]


if __name__ == "__main__":
    pytest.main()