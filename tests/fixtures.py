import pytest
from typing import Callable, Generator
from linkml_runtime import SchemaView
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model.meta import SchemaDefinition
from rdflib import Graph, URIRef, Namespace, BNode, Literal
from rdflib.namespace import RDF, DCAT
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock
from cim_plugin.cimxml_parser import CIMXMLParser
from cim_plugin.header import CIMMetadataHeader
from cim_plugin.cimxml_serializer import CIMXMLSerializer
from cim_plugin.graph import CIMGraph
import uuid
import textwrap
from rdflib.plugin import register
from rdflib.parser import Parser


# Schemaview fixtures
@pytest.fixture
def make_schemaview() -> Callable[..., SchemaView]:
    """Factory for creating SchemaView objects that mimic real LinkML schemas."""

    def _factory(*, prefixes=None, types=None, slots=None, classes=None, enums=None) -> SchemaView:
        # Build SchemaDefinition in the same structure as real LinkML YAML
        schema = SchemaDefinition(
            id="test",
            name="test",
            imports=["linkml:types"],
            prefixes=prefixes,
            types=types,
            slots=slots,
            classes=classes,
            enums=enums
        )

        return SchemaView(schema=schema) # type: ignore

    return _factory


# Graph making fixtures
@pytest.fixture
def make_graph_with_prefixes() -> Graph:
    g = Graph()
    g.bind("ex", Namespace("www.example.com/"))
    g.bind("same", Namespace("www.same.com/"))
    g.bind("ws", Namespace(" www.whitespace.com/ "))
    g.add((URIRef("www.example.com/a"), URIRef("www.same.com/b"), URIRef(" www.whitespace.com/ c")))

    return g


@pytest.fixture
def make_graph() -> Callable[..., Graph]: 
    """Helper to build a graph from a list of (s, p, o) triples."""
    def _make_graph(triples: list[tuple]) -> Graph:
        g = Graph()
        g.bind("ex", Namespace("http://example.org/"))
        for s, p, o in triples: 
            g.add((s, p, o)) 
        return g
    return _make_graph


@pytest.fixture
def make_cimgraph():
    """Create a CIMGraph with a metadata header and some predefined namespaces."""
    g = CIMGraph()

    # Register namespaces
    g.bind("ex", Namespace("http://example.com/"))
    g.bind("foo", Namespace("http://foo.org/ns#"))
    g.bind("bar", Namespace("http://bar.org/"))

    # Create metadata header
    header = CIMMetadataHeader.empty(URIRef("http://example.com/header"))
    header.add_triple(RDF.type, DCAT.Dataset)
    g.metadata_header = header

    return g

@pytest.fixture
def build_graph_with_blank_header() -> tuple[Graph, BNode, set[BNode]]: 
    g = Graph() 
    header = BNode() 
    b1 = BNode() 
    b2 = BNode() 
    # Header type triple 
    g.add((header, RDF.type, URIRef("urn:meta:Header"))) 
    # Reachable chain 
    g.add((header, URIRef("urn:p:1"), b1)) 
    g.add((b1, URIRef("urn:p:2"), b2)) 
    g.add((b2, URIRef("urn:p:3"), Literal("value"))) 
    
    return g, header, {header, b1, b2}

# CIMXMLParser
@pytest.fixture
def cimxmlinstance_w_prefixes(make_schemaview):
    """Create an instance with a real SchemaView."""
    obj = CIMXMLParser()
    obj.schemaview = make_schemaview(prefixes={"ex": {"prefix_prefix": "ex", "prefix_reference": "www.example.org"}})
    return obj


@pytest.fixture
def make_cimxmlparser() -> Callable[..., CIMXMLParser]:
    def _factory(schemaview: SchemaView) -> CIMXMLParser:
        obj = CIMXMLParser()
        obj.schema_path = "schema.yaml"
        obj.schemaview = schemaview
        return obj
    return _factory


# CIMXMLSerializer
@pytest.fixture
def capture_writer() -> tuple[list, Callable]:
    output = []

    def writer(text: str) -> int:
        output.append(text)
        return 0  # mimic stream.write return type

    return output, writer


@pytest.fixture
def serializer(capture_writer: tuple[list, Callable]) -> tuple[CIMXMLSerializer, list]:
    output, writer = capture_writer
    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty()
    ser = CIMXMLSerializer(g)
    ser.write = writer
    ser.qualifier_resolver = Mock()
    ser.qualifier_resolver.convert_to_special_qualifier.side_effect = lambda x: str(x)
    ser.qualifier_resolver.convert_to_default_qualifier.side_effect = lambda x: str(x)
    return ser, output


# Function specific fixtures
@dataclass
class PatchMocks:
    """Collecting mocks for all functions used by patch_integer_ranges."""
    find_slots: MagicMock
    add_slot: MagicMock
    set_modified: MagicMock
    calls: list


@pytest.fixture
def mock_patch_integer_ranges(monkeypatch: pytest.MonkeyPatch) -> PatchMocks:
    """Patching all functions used by patch_integer_ranges."""

    mocks = PatchMocks(find_slots=MagicMock(), add_slot=MagicMock(), set_modified=MagicMock(), calls=[])
    monkeypatch.setattr("cim_plugin.cimxml_parser.find_slots_with_range", mocks.find_slots)
    monkeypatch.setattr(SchemaView, "add_slot", mocks.add_slot)
    monkeypatch.setattr(SchemaView, "set_modified", mocks.set_modified)
    
    mocks.calls = []
    mocks.find_slots.side_effect = lambda *a, **kw: (mocks.calls.append("find_slots_with_range"), mocks.find_slots.return_value)[1]
    mocks.add_slot.side_effect = lambda *a, **kw: mocks.calls.append("add_slot")
    mocks.set_modified.side_effect = lambda *a, **kw: mocks.calls.append("set_modified")

    return mocks


# Various
@pytest.fixture
def set_prefixes() -> dict:
    return {"ex": {"prefix_prefix": "ex", "prefix_reference": "http://example.org/"}}


@pytest.fixture
def sample_yaml() -> str: 
    return textwrap.dedent("""
    classes:
        Season:
            attributes:
                endDate:
                    range: integer
                    description: 'Something here'                    
                startDate: 
                    range: Date
                    description: 'Something there'
        Foo:
            attributes:
                bar:
                    range: string
                tend:
                    range: integer
        Activity:
            attributes:
                Updates:
                    range: Count
        Software:
            attributes:
                Updates:
                    range: Count
    """)


@pytest.fixture 
def cimxml_plugin() -> Generator: 
    register( "cimxml", Parser, "cim_plugin.cimxml_parser", "CIMXMLParser" ) 
    # yield so the test can run after registration 
    yield


@pytest.fixture 
def mock_extract_uuid(monkeypatch: pytest.MonkeyPatch) -> Mock: 
    mock = Mock() 
    mock.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678") 
    monkeypatch.setattr("cim_plugin.utilities._extract_uuid_from_urn", mock) 
    return mock


if __name__ == "__main__":
    print("Fixtures for tests")
