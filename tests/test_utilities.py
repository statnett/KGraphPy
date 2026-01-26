import pytest
from unittest.mock import MagicMock, Mock
import uuid
from cim_plugin.utilities import _extract_uuid_from_urn, get_graph_uuid, MD, DCAT
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS
from typing import Callable
import logging

logger = logging.getLogger('cimxml_logger')


# Unit tests get_graph_uuid

@pytest.fixture 
def mock_extract_uuid(monkeypatch: pytest.MonkeyPatch) -> Mock: 
    mock = Mock() 
    mock.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678") 
    monkeypatch.setattr("cim_plugin.utilities._extract_uuid_from_urn", mock) 
    return mock


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

if __name__ == "__main__":
    pytest.main()