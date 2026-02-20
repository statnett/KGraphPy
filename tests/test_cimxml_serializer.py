from typing import Callable, Type, cast, IO
import pytest
from unittest.mock import MagicMock, call, patch, Mock
import uuid
import io
import logging

from rdflib import Namespace, URIRef, Graph, Literal, Node, BNode
from rdflib.plugins.serializers.xmlwriter import ESCAPE_ENTITIES
from rdflib.namespace import XSD, RDF, DCAT
from xml.sax.saxutils import escape
from cim_plugin.cimxml_serializer import _subject_sort_key, CIMXMLSerializer
from cim_plugin.qualifiers import CIMQualifierStrategy, UnderscoreQualifier, URNQualifier, NamespaceQualifier, CIMQualifierResolver, uuid_namespace
from cim_plugin.header import CIMMetadataHeader
from cim_plugin.graph import CIMGraph
from cim_plugin.namespaces import MD
from tests.fixtures import capture_writer, serializer, make_cimgraph


logger = logging.getLogger("cimxml_logger")

# Unit tests ._init_qualifier_resolver
@pytest.mark.parametrize(
        "input, resolver",
        [
            pytest.param("urn", URNQualifier, id="Urn qualifier"),
            pytest.param("underscore", UnderscoreQualifier, id="Underscore qualifier"),
            pytest.param("namespace", NamespaceQualifier, id="Namespace qualifier"),
            pytest.param("UNderSCore", UnderscoreQualifier, id="Mixed letters"),
            pytest.param(None, UnderscoreQualifier, id="None input"),
            pytest.param("", UnderscoreQualifier, id="Empty input")
        ]
)
def test_init_qualifier_resolver_basic(input: str, resolver: Type[CIMQualifierStrategy]) -> None:
    g = Graph()
    ser = CIMXMLSerializer(g)
    ser._init_qualifier_resolver(input)
    assert type(ser.qualifier_resolver.output) == resolver
    assert isinstance(ser.qualifier_resolver, CIMQualifierResolver)
    assert isinstance(ser.qualifier_resolver.output, resolver)

def test_init_qualifier_resolver_wronginput() -> None:
    g = Graph()
    ser = CIMXMLSerializer(g)
    with pytest.raises(ValueError) as exc:
        ser._init_qualifier_resolver("wrong")

    assert str(exc.value) == "Unknown qualifier: wrong"


# Unit tests ._ensure_header
def test_ensure_header_headerexists() -> None:
    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty(URIRef("s1"))
    ser = CIMXMLSerializer(g)

    header = ser._ensure_header()
    assert header.subject == g.metadata_header.subject
    store_header = cast(CIMGraph, ser.store).metadata_header
    assert store_header is not None
    assert store_header.subject == URIRef("s1")
    assert header is g.metadata_header


def test_ensure_header_noheader(caplog: pytest.LogCaptureFixture) -> None:
    g = CIMGraph()
    ser = CIMXMLSerializer(g)
    assert getattr(ser.store, "metadata_header", None) == None
    
    header = ser._ensure_header()
    store_header = cast(CIMGraph, ser.store).metadata_header
    assert store_header is not None
    assert store_header is header
    assert store_header.subject == header.subject
    assert "Random id generated for graph" in caplog.text

@patch("cim_plugin.cimxml_serializer.create_header_attribute")
def test_ensure_header_createcalled(mock_create: MagicMock) -> None:
    g = CIMGraph()
    ser = CIMXMLSerializer(g)
    mock_create.return_value = CIMMetadataHeader.empty(URIRef("s1"))
    header = ser._ensure_header()

    mock_create.assert_called_once()
    store_header = cast(CIMGraph, ser.store).metadata_header
    assert store_header is header


@patch("cim_plugin.cimxml_serializer.create_header_attribute")
def test_ensure_header_createnotcalled(mock_create: MagicMock) -> None:
    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty(URIRef("s1"))
    ser = CIMXMLSerializer(g)
    header = ser._ensure_header()

    mock_create.assert_not_called()
    store_header = cast(CIMGraph, ser.store).metadata_header
    assert store_header is header



# Unit tests ._collect_used_namespaces
@pytest.mark.parametrize(
    "uri,expected_prefix",
    [
        ("http://example.com/Thing", "ex"),
        ("http://foo.org/ns#Item", "foo"),
        ("http://bar.org/Value", "bar"),
    ],
)
def test__collect_used_namespaces_onlyregisterednamespaces(make_cimgraph: CIMGraph, uri: str, expected_prefix: str) -> None:
    # Collecting namespace if it exist in the namespace_manager
    g = make_cimgraph
    g.add((URIRef(uri), URIRef("http://example.com/p"), URIRef("http://example.com/o")))

    ser = CIMXMLSerializer(g)
    ns_list = dict(ser._collect_used_namespaces())

    assert expected_prefix in ns_list
    # Check that the namespace collected matches the namespace by the same prefix in the namespace_manager
    assert str(ns_list[expected_prefix]).startswith(
        str(dict(g.namespace_manager.namespaces())[expected_prefix])
    )

@pytest.mark.parametrize(
    "uri",
    [
        "http://not-registered.com/A",
        "http://another.org/ns#B",
        "http://www.fake.org/",
    ],
)
def test_collect_used_namespaces_unregisterednamespaces(make_cimgraph: CIMGraph, uri: str) -> None:
    # Namespace not collected if it is not in namespace_manager
    g = make_cimgraph
    g.add((URIRef(uri), URIRef("http://example.com/p"), URIRef("http://example.com/o")))

    ser = CIMXMLSerializer(g)
    ns_list = dict(ser._collect_used_namespaces())

    assert ns_list.keys() == {'dcat', 'ex', 'rdf'}
    # None of these should appear
    assert all(not str(ns).startswith(uri.rsplit("/", 1)[0]) for ns in ns_list.values())
    

def test_urns_are_ignored(make_cimgraph: CIMGraph) -> None:
    g = make_cimgraph
    g.add((URIRef("urn:uuid:1234"), URIRef("http://example.com/p"), URIRef("http://example.com/o")))

    ser = CIMXMLSerializer(g)
    ns_list = dict(ser._collect_used_namespaces())

    assert len(ns_list) == 3
    assert ns_list.keys() == {'dcat', 'ex', 'rdf'}  # No urn in the the prefix list
    

def test_collect_used_namespaces_headertriples(make_cimgraph: CIMGraph) -> None:
    g = make_cimgraph

    # Add header triples
    assert g.metadata_header    # Without this pylance reacts to add_triple
    g.metadata_header.add_triple(
        URIRef("http://foo.org/ns#headerPredicate"),
        URIRef("http://foo.org/ns#headerObject"),
    )

    ser = CIMXMLSerializer(g)
    ns_list = dict(ser._collect_used_namespaces())

    assert "foo" in ns_list
    assert "ex" in ns_list  # header subject is in example.com namespace


def test_collect_used_namespaces_sortedandunique(make_cimgraph: CIMGraph) -> None:
    g = make_cimgraph

    # Add multiple triples using same namespace
    g.add((URIRef("http://example.com/A"), URIRef("http://example.com/p"), URIRef("http://example.com/o")))
    g.add((URIRef("http://example.com/B"), URIRef("http://example.com/p2"), URIRef("http://example.com/o2")))

    ser = CIMXMLSerializer(g)
    ns_list = ser._collect_used_namespaces()

    # Should contain only one entry for "ex"
    prefixes = [p for p, _ in ns_list]
    assert prefixes.count("ex") == 1

    # Should be sorted by prefix
    assert prefixes == sorted(prefixes)

def test_collect_used_namespaces_blanknodes(make_cimgraph: CIMGraph) -> None:
    g = make_cimgraph

    b = BNode("http://bar.org/bnode")
    g.add((b, URIRef("http://example.com/p"), URIRef("http://example.com/o")))

    ser = CIMXMLSerializer(g)
    ns_list = dict(ser._collect_used_namespaces())

    assert ns_list.keys() == {'dcat', 'ex', 'rdf'}  # No blank node in the the prefix list

def test_collect_used_namespaces_literals(make_cimgraph: CIMGraph) -> None:
    g = make_cimgraph

    g.add((
        URIRef("http://example.com/s"),
        URIRef("http://example.com/p"),
        Literal("http://bar.org/bnode")
    ))

    ser = CIMXMLSerializer(g)
    ns_list = dict(ser._collect_used_namespaces())
    assert ns_list.keys() == {'dcat', 'ex', 'rdf'}  # No literal in the the prefix list
    assert "bar" not in ns_list


def test_collect_used_namespaces_nousednamespaces():
    g = CIMGraph()

    # Add triples with URIs that look like namespaces
    g.add((
        URIRef("http://example.com/A"),
        URIRef("http://example.com/p"),
        URIRef("http://example.com/o")
    ))
    g.metadata_header = CIMMetadataHeader.empty(URIRef("http://example.com/header"))

    ser = CIMXMLSerializer(g)
    ns_list = dict(ser._collect_used_namespaces())

    assert ns_list == {}


def test_collect_used_namespaces_overlappingnamespaces() -> None:
    g = CIMGraph()
    g.bind("ex", Namespace("http://example.com/"))
    g.bind("exns", Namespace("http://example.com/ns/"))
    g.metadata_header = CIMMetadataHeader.empty(URIRef("http://example.com/header"))
    g.add((
        URIRef("http://example.com/ns/Thing"),
        URIRef("http://example.com/p"),
        URIRef("http://example.com/ns/Object")
    ))

    ser = CIMXMLSerializer(g)
    ns_list = dict(ser._collect_used_namespaces())
    assert "ex" in ns_list
    assert "exns" in ns_list
    assert str(ns_list["ex"]) == "http://example.com/"
    assert str(ns_list["exns"]) == "http://example.com/ns/"


def test_collect_used_namespaces_collisions() -> None:
    g = CIMGraph()

    # Two prefixes bound to the same namespace URI
    g.bind("ex", Namespace("http://example.com/"))
    g.bind("alt", Namespace("http://example.com/"))

    g.metadata_header = CIMMetadataHeader.empty(URIRef("http://example.com/header"))

    g.add((
        URIRef("http://example.com/Thing"),
        URIRef("http://example.com/p"),
        URIRef("http://example.com/o")
    ))

    ser = CIMXMLSerializer(g)
    ns_list = dict(ser._collect_used_namespaces())
    print(ns_list)

    assert len(ns_list) == 1    # Only one prefix should survive
    assert list(ns_list.values())[0] == URIRef("http://example.com/")   # It must map to the correct namespace URI
    assert list(ns_list.keys())[0] in {"ex", "alt"} # And the prefix must be one of the registered ones


def test_collect_used_namespaces_rebindingnamespace() -> None:
    g = CIMGraph()

    g.bind("ex", Namespace("http://old.example.com/"))
    g.bind("ex", Namespace("http://new.example.com/"), replace=True)    # Rebinding to new namespace

    g.metadata_header = CIMMetadataHeader.empty(URIRef("http://new.example.com/header"))

    g.add((
        URIRef("http://new.example.com/Thing"),
        URIRef("http://new.example.com/p"),
        URIRef("http://new.example.com/o")
    ))

    ser = CIMXMLSerializer(g)
    ns_list = dict(ser._collect_used_namespaces())

    assert "ex" in ns_list
    assert ns_list["ex"] == URIRef("http://new.example.com/")
    assert URIRef("http://old.example.com/") not in ns_list.values()

# Unit tests .serialize
@patch("cim_plugin.cimxml_serializer._subject_sort_key")
@patch("cim_plugin.cimxml_serializer.group_subjects_by_type")
def test_serialize_allcalls(mock_group: MagicMock, mock_sort: MagicMock) -> None:
    buf = io.BytesIO()
    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty(URIRef("h1"))
    ser = CIMXMLSerializer(g)

    ser._ensure_header = Mock(return_value=g.metadata_header)
    ser._init_qualifier_resolver = Mock()
    ser._collect_used_namespaces = Mock(return_value=[("ex", "example.com/")])
    ser.write_header = Mock()
    mock_group.return_value = {"ex:o": [URIRef("s1"), URIRef("s2")]}
    mock_sort.side_effect = [(1, "s1"), (0, "s2")]
    ser.subject = Mock()

    ser.serialize(buf, qualifier="foo")
    result = buf.getvalue().decode()

    ser._ensure_header.assert_called_once()
    ser._init_qualifier_resolver.assert_called_once_with("foo")
    ser._collect_used_namespaces.assert_called_once()
    ser.write_header.assert_called_once()
    mock_group.assert_called_once()
    assert mock_sort.call_count == 2
    assert ser.subject.call_count == 2
    assert result == '<?xml version="1.0" encoding="utf-8"?>\n<rdf:RDF\n    xmlns:ex="example.com/"\n    >\n\n</rdf:RDF>\n'
    
def test_serialize_namespaces() -> None:
    buf = io.BytesIO()
    g = CIMGraph()
    g.bind("ex", "http://example.com/")
    g.bind("foo", "http://bar.com/")
    g.add((URIRef("s1"), URIRef("http://example.com/p"), Literal("o")))
    g.metadata_header = CIMMetadataHeader.empty(URIRef("h1"))
    g.metadata_header.add_triple(RDF.type, DCAT.Dataset)

    ser = CIMXMLSerializer(g)

    ser.serialize(buf)
    out = buf.getvalue().decode()
    assert 'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"' in out
    assert 'xmlns:ex="http://example.com/"' in out
    assert 'xmlns:foo="http://bar.com/' not in out


def test_serialize_multipleserializations() -> None:
    buf1 = io.BytesIO()
    buf2 = io.BytesIO()
    g1 = CIMGraph()
    g1.bind("ex", "http://example.com/")
    g2 = CIMGraph()
    g2.bind("foo", "http://bar.com/")
    g1.add((URIRef("s1"), URIRef("http://example.com/p"), Literal("o")))
    g2.add((URIRef("s2"), URIRef("http://bar.com/p"), Literal("o")))
    g1.metadata_header = CIMMetadataHeader.empty(URIRef("h1"))
    g1.metadata_header.add_triple(RDF.type, DCAT.Dataset)
    g2.metadata_header = CIMMetadataHeader.empty(URIRef("h2"))
    g2.metadata_header.add_triple(RDF.type, DCAT.Dataset)

    ser1 = CIMXMLSerializer(g1)
    ser2 = CIMXMLSerializer(g2)
    ser1.serialize(buf1)
    ser2.serialize(buf2)
    out1 = buf1.getvalue().decode()
    out2 = buf2.getvalue().decode()
    assert 'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"' in out1
    assert 'xmlns:ex="http://example.com/"' in out1
    assert 'xmlns:foo="http://bar.com/' not in out1
    assert 'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"' in out2
    assert 'xmlns:ex="http://example.com/"' not in out2
    assert 'xmlns:foo="http://bar.com/' in out2


@pytest.mark.parametrize(
    "subjects, expected",
    [
        pytest.param([URIRef("beta"), URIRef("alfa")], ["alfa", "beta"], id="Two subjects"),
        pytest.param([URIRef("ex:xeta"), URIRef("zeta"), URIRef("mu")], ["ex:xeta", "mu", "zeta"], id="Three subjects, one with namespace"),
    ],
)
@patch("cim_plugin.cimxml_serializer.group_subjects_by_type")
def test_serialize_subjectsorting(mock_group: MagicMock, subjects: list[URIRef], expected: list[str]) -> None:
    buf = io.BytesIO()
    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty(URIRef("h1"))

    ser = CIMXMLSerializer(g)

    # Fake grouping
    def fake_group(*args, **kwargs):
        return {"ex:o": subjects}

    mock_group.side_effect = fake_group
    ser.serialize(buf)

    out = buf.getvalue().decode()
    lines = [line.strip() for line in out.splitlines() if "<" in line]

    # Extract subject IDs
    found = [s for s in expected if any(s in line for line in lines)]
    assert found == expected

@pytest.mark.parametrize("enc", ["utf-8", "latin-1"])
def test_serialize_encoding(enc: str) -> None:
    buf = io.BytesIO()
    g = CIMGraph()
    g.bind("ex", "http://example.com/")
    g.metadata_header = CIMMetadataHeader.empty(URIRef("h1"))
    g.metadata_header.add_triple(RDF.type, DCAT.dataset)
    g.metadata_header.add_triple(URIRef("http://example.com/p"), Literal("æøå"))

    ser = CIMXMLSerializer(g)
    ser.encoding = enc

    ser.serialize(buf)
    out = buf.getvalue().decode(enc)

    assert f'encoding="{enc}"' in out
    assert '<ex:p>æøå</ex:p>' in out


def test_serialize_header() -> None:
    buf = io.BytesIO()
    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty(URIRef("h1"))
    g.metadata_header.add_triple(RDF.type, DCAT.Dataset)

    ser = CIMXMLSerializer(g)

    ser.serialize(buf)
    out = buf.getvalue().decode()
    assert '<dcat:Dataset rdf:about="urn:uuid:h1"/>\n' in out


def test_serialize_nonamespaces() -> None:
    buf = io.BytesIO()
    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty(URIRef("h1"))
    
    ser = CIMXMLSerializer(g)

    ser.serialize(buf)
    out = buf.getvalue().decode()
    # Header is malformed because it lacks the rdf:type, which is excluded to keep it from being collected in the namespaces
    assert out == '<?xml version="1.0" encoding="utf-8"?>\n<rdf:RDF\n    >\n  <<MALFORMED> rdf:about="urn:uuid:h1"/>\n\n</rdf:RDF>\n'


def test_serialize_namespacewithnoprefix() -> None:
    buf = io.BytesIO()
    g = CIMGraph()
    g.bind("ex", "http://example.com/")
    g.bind("", "http://noprefix.com/")
    g.add((URIRef("http://example.com/s1"), URIRef("http://noprefix.com/p"), Literal("o")))
    g.metadata_header = CIMMetadataHeader.empty(URIRef("h1"))
    
    ser = CIMXMLSerializer(g)

    ser.serialize(buf)
    out = buf.getvalue().decode()
    assert 'xmlns:ex="http://example.com/"' in out
    assert 'xmlns="http://noprefix.com/"' in out


def test_serialize_nosubjects() -> None:
    buf = io.BytesIO()
    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty(URIRef("h1"))
    g.metadata_header.add_triple(RDF.type, DCAT.Dataset)
    
    ser = CIMXMLSerializer(g)

    ser.serialize(buf)
    out = buf.getvalue().decode()
    assert out == '<?xml version="1.0" encoding="utf-8"?>\n<rdf:RDF\n    xmlns:dcat="http://www.w3.org/ns/dcat#"\n    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"\n    >\n  <dcat:Dataset rdf:about="urn:uuid:h1"/>\n\n</rdf:RDF>\n'


def test_serialize_multiplegroups() -> None:
    buf = io.BytesIO()
    g = CIMGraph()
    g.bind("ex", "http://example.com/")
    s3 = "123e4567-e89b-12d3-a456-426614174000"
    s2 = "123e4567-e89b-12d3-a456-426614174001"
    s1 = "123e4567-e89b-12d3-a456-426614174002"
    s4 = "123e4567-e89b-12d3-a456-426614174003"
    s5 = "http://example.com/a5"
    g.add((URIRef(f"urn:uuid:{s1}"), RDF.type, URIRef("http://example.com/TypeA")))
    g.add((URIRef(f"urn:uuid:{s2}"), RDF.type, URIRef("http://example.com/TypeA")))
    g.add((URIRef(f"urn:uuid:{s3}"), RDF.type, URIRef("http://example.com/TypeA")))
    g.add((URIRef(f"urn:uuid:{s4}"), RDF.type, URIRef("http://example.com/TypeB")))
    g.add((URIRef(s5), RDF.type, URIRef("http://example.com/TypeA")))

    g.metadata_header = CIMMetadataHeader.empty(URIRef("h1"))
    
    ser = CIMXMLSerializer(g)

    ser.serialize(buf)
    out = buf.getvalue().decode()
    # Extract rdf:about values in order 
    abouts = [ 
        line.split('"')[1] 
        for line in out.splitlines() 
        if 'rdf:about="#_' in line 
    ] 
    assert abouts == [ 
        f"#_{s3}", 
        f"#_{s2}", 
        f"#_{s1}", 
        f"#_{s5}", # invalid UUID → sorted last among TypeA 
        f"#_{s4}", # TypeB comes after TypeA 
    ]
    
def test_serialize_streamwritefailure() -> None:
    class BadStream():
        def write(self, data: bytes) -> int:
            raise IOError("boom")

    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty(URIRef("h1"))
    ser = CIMXMLSerializer(g)

    bad_stream = BadStream()

    with pytest.raises(IOError) as excinfo:
        # Pylance silenced to test bad input
        ser.serialize(bad_stream)   # type: ignore

    assert "boom" in str(excinfo.value)


def test_serialize_streamwritefailurepartial() -> None:
    class BadStream:
        def __init__(self):
            self.calls = 0

        def write(self, data: bytes) -> int:
            self.calls += 1
            raise IOError("boom")

    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty(URIRef("h1"))
    ser = CIMXMLSerializer(g)

    bad_stream = BadStream()

    with pytest.raises(IOError):
        # Pylance silenced to test bad input
        ser.serialize(bad_stream)   # type: ignore

    # Should have attempted exactly one write (XML header)
    assert bad_stream.calls == 1


# Unit tests .write_header
def test_write_header_basic(capture_writer: tuple[list, Callable]) -> None:
    output, writer = capture_writer
    g = CIMGraph()
    g.bind("ex", "http://example.com/")
    header = CIMMetadataHeader.empty(subject=URIRef("s1"))
    header.add_triple(RDF.type, DCAT.Dataset)
    header.add_triple(URIRef("http://example.com/p"), Literal("o"))
    g.metadata_header = header
    ser = CIMXMLSerializer(g)
    ser.write = writer
    ser.qualifier_resolver = CIMQualifierResolver(UnderscoreQualifier())

    ser.write_header(header)

    result = "".join(output)

    assert result == '  <dcat:Dataset rdf:about="urn:uuid:s1">\n    <ex:p>o</ex:p>\n  </dcat:Dataset>\n'
    assert type(ser.qualifier_resolver.output) == UnderscoreQualifier


def test_write_header_emptybody(capture_writer: tuple[list, Callable]) -> None:
    output, writer = capture_writer
    g = CIMGraph()
    g.bind("ex", "http://example.com/")
    header = CIMMetadataHeader.empty(subject=URIRef("s1"))
    header.add_triple(RDF.type, DCAT.Dataset)
    g.metadata_header = header
    ser = CIMXMLSerializer(g)
    ser.write = writer
    ser.qualifier_resolver = CIMQualifierResolver(UnderscoreQualifier())
    ser.predicate = Mock()

    ser.write_header(header)

    result = "".join(output)

    assert result == '  <dcat:Dataset rdf:about="urn:uuid:s1"/>\n'
    assert type(ser.qualifier_resolver.output) == UnderscoreQualifier
    ser.predicate.assert_not_called()


def test_write_header_predicatesorting(capture_writer: tuple[list, Callable]) -> None:
    output, writer = capture_writer
    g = CIMGraph()
    g.bind("ex", "http://example.com/")
    header = CIMMetadataHeader.empty(subject=URIRef("s1"))
    header.add_triple(RDF.type, DCAT.Dataset)
    header.add_triple(URIRef("http://example.com/p"), Literal("o"))
    header.add_triple(URIRef("http://example.com/a"), URIRef("o"))
    g.metadata_header = header
    ser = CIMXMLSerializer(g)
    ser.write = writer
    ser.qualifier_resolver = CIMQualifierResolver(UnderscoreQualifier())

    ser.write_header(header)

    result = "".join(output)
    print(result)
    assert result == '  <dcat:Dataset rdf:about="urn:uuid:s1">\n    <ex:a rdf:resource="o"/>\n    <ex:p>o</ex:p>\n  </dcat:Dataset>\n'
    assert type(ser.qualifier_resolver.output) == UnderscoreQualifier


def test_write_header_rdftypehandling(capture_writer: tuple[list, Callable]) -> None:
    output, writer = capture_writer
    g = CIMGraph()
    g.bind("ex", "http://example.com/")
    header = CIMMetadataHeader.empty(subject=URIRef("s1"))
    header.add_triple(RDF.type, DCAT.Dataset)   # The first rdf.type in metadata_object is set as main_type
    header.add_triple(RDF.type, MD.FullModel)   # The next rdf.type is treated as any other triple
    header.add_triple(RDF.type, URIRef("o"))
    g.metadata_header = header
    ser = CIMXMLSerializer(g)
    ser.write = writer
    ser.qualifier_resolver = CIMQualifierResolver(UnderscoreQualifier())

    ser.write_header(header)

    result = "".join(output)
    print(result)
    assert result == '  <dcat:Dataset rdf:about="urn:uuid:s1">\n    <rdf:type rdf:resource="http://iec.ch/TC57/61970-552/ModelDescription/1#FullModel"/>\n    <rdf:type rdf:resource="o"/>\n  </dcat:Dataset>\n'
    assert type(ser.qualifier_resolver.output) == UnderscoreQualifier


def test_write_header_predicatecalls(capture_writer: tuple[list, Callable]) -> None:
    output, writer = capture_writer
    g = CIMGraph()
    g.bind("ex", "http://example.com/")
    g.bind("MODEL", "https://model4powersystem.no/")
    header = CIMMetadataHeader.empty(subject=URIRef("s1")) 
    header.add_triple(RDF.type, DCAT.Dataset)
    header.add_triple(URIRef("http://example.com/p"), Literal("o"))
    header.add_triple(URIRef("http://example.com/a"), URIRef("o"))
    header.add_triple(URIRef("http://example.com/f"), URIRef("urn:uuid:o"))
    header.add_triple(URIRef("http://example.com/s"), URIRef("https://model4powersystem.no/:o"))
    header.add_triple(URIRef("http://example.com/b"), URIRef("#_o"))

    g.metadata_header = header
    ser = CIMXMLSerializer(g)
    ser.write = writer
    ser.qualifier_resolver = CIMQualifierResolver(NamespaceQualifier())
    ser.predicate = Mock()

    ser.write_header(header)
    calls = [
        call(URIRef("http://example.com/a"), URIRef("o"), 2, use_qualifier=False), 
        call(URIRef("http://example.com/b"), URIRef("#_o"), 2, use_qualifier=True),
        call(URIRef("http://example.com/f"), URIRef("urn:uuid:o"), 2, use_qualifier=True),
        call(URIRef("http://example.com/p"), Literal("o"), 2, use_qualifier=False),
        call(URIRef("http://example.com/s"), URIRef("https://model4powersystem.no/:o"), 2, use_qualifier=True),
    ]
    assert ser.predicate.call_count == 5
    ser.predicate.assert_has_calls(calls)
    assert type(ser.qualifier_resolver.output) == NamespaceQualifier


def test_write_header_resolverrestored(capture_writer: tuple[list, Callable]) -> None:
    output, writer = capture_writer
    g = CIMGraph()
    g.bind("ex", "http://example.com/")
    header = CIMMetadataHeader.empty(subject=URIRef("s1"))
    header.add_triple(RDF.type, DCAT.Dataset)
    header.add_triple(URIRef("http://example.com/p"), Literal("o"))
    g.metadata_header = header
    ser = CIMXMLSerializer(g)
    ser.write = writer
    ser.qualifier_resolver = CIMQualifierResolver(NamespaceQualifier())
    ser.predicate = Mock(side_effect=ValueError)

    with pytest.raises(ValueError):
        ser.write_header(header)

    assert type(ser.qualifier_resolver.output) == NamespaceQualifier

def test_write_header_nomaintype(capture_writer: tuple[list, Callable], caplog: pytest.LogCaptureFixture) -> None:
    output, writer = capture_writer
    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty(subject=URIRef("s1"))
    ser = CIMXMLSerializer(g)
    ser.qualifier_resolver = CIMQualifierResolver(NamespaceQualifier())
    ser.write = writer
    
    ser.write_header(g.metadata_header)
    result = "".join(output)
    assert "MALFORMED" in result
    assert "Header type missing:" in caplog.text


def test_write_header_noqualifierresolver(capture_writer: tuple[list, Callable]) -> None:
    # Documents what happends if the qualifier_resolver has not been set
    output, writer = capture_writer
    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty(subject=URIRef("s1"))
    ser = CIMXMLSerializer(g)
    ser.write = writer
    
    with pytest.raises(AttributeError):
        ser.write_header(g.metadata_header)

# Unit tests .subject
def test_subject_nonuriref(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    subject = Literal("not-a-uri")

    ser.subject(subject)

    result = "".join(output)

    assert "<MALFORMED" in result
    assert "Subject is not a URIRef" in result
    assert "</MALFORMED>" in result


def test_subject_missingtype(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    g = ser.store

    s = URIRef("http://example.com/s")
    g.add((s, URIRef("http://example.com/p"), Literal("x")))

    ser.subject(s)

    result = "".join(output)

    assert "<MALFORMED" in result
    assert "No rdf:type found" in result

@patch("cim_plugin.cimxml_serializer.is_uuid_qualified")
def test_subject_valid(mock_qualified: MagicMock, serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    mock_qualified.return_value = False
    g = ser.store

    g.bind("ex", "http://example.com/")

    s = URIRef("s123")
    t = URIRef("http://example.com/Class")
    p = URIRef("http://example.com/p")

    g.add((s, RDF.type, t))
    g.add((s, p, Literal("value")))

    ser.subject(s)

    result = "".join(output)

    assert '<ex:Class rdf:about="s123"' in result
    assert "<ex:p>value</ex:p>" in result
    assert "</ex:Class>" in result


@pytest.mark.parametrize(
        "qualifier_return", [True, False]
)
@patch("cim_plugin.cimxml_serializer.is_uuid_qualified")
def test_subject_objectuuid(mock_qualified: MagicMock, serializer: tuple[CIMXMLSerializer, list], qualifier_return: bool) -> None:
    ser, output = serializer
    mock_qualified.return_value = qualifier_return
    g = ser.store

    g.bind("ex", "http://example.com/")

    s = URIRef("s123")
    t = URIRef("http://example.com/Class")
    p = URIRef("http://example.com/p")

    g.add((s, RDF.type, t))
    g.add((s, p, Literal("value")))
    ser.predicate = Mock()

    ser.subject(s)

    mock_qualified.assert_called_once_with(ser.qualifier_resolver, Literal("value"))
    ser.predicate.assert_called_once_with(p, Literal("value"), 2, use_qualifier=qualifier_return)


@patch("cim_plugin.cimxml_serializer.is_uuid_qualified")
def test_subject_rdfid(mock_qualified: MagicMock, serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    mock_qualified.return_value = False
    g = ser.store
    g.metadata_header.profile = "http://cim-profile.ucaiug.io/grid/Dynamics/2.0"    # pyright: ignore[reportAttributeAccessIssue]

    g.bind("ex", "http://example.com/")

    s = URIRef("s123")
    t = URIRef("http://example.com/Class")
    p = URIRef("http://example.com/p")

    g.add((s, RDF.type, t))
    g.add((s, p, Literal("value")))

    ser.subject(s)

    result = "".join(output)

    assert '<ex:Class rdf:ID="s123"' in result
    assert "<ex:p>value</ex:p>" in result
    assert "</ex:Class>" in result


def test_subject_onlyrdftype(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    g = ser.store

    g.bind("ex", "http://example.com/")

    s = URIRef("s123")
    t = URIRef("http://example.com/Class")
    p = URIRef("http://example.com/p")

    g.add((s, RDF.type, t))

    ser.subject(s)

    result = "".join(output)
    assert '  <ex:Class rdf:about="s123">\n  </ex:Class>\n' in result
    

def test_subject_alreadyserialized(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    g = ser.store

    s = URIRef("http://example.com/s")
    g.add((s, RDF.type, URIRef("http://example.com/Class")))

    ser.subject(s)
    output.clear()  # Remove output so the method can be run again, but now the subject is registered

    ser.subject(s)

    assert output == []

@patch("cim_plugin.cimxml_serializer.is_uuid_qualified")
def test_subject_malformedpredicate(mock_qualified: MagicMock, serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    mock_qualified.return_value = False
    g = ser.store

    g.bind("ex", "http://example.com/")

    s = URIRef("http://example.com/s")
    t = URIRef("http://example.com/Class")

    g.add((s, RDF.type, t))
    g.add((s, Literal("not-a-uri"), Literal("x")))  # malformed predicate

    ser.subject(s)

    result = "".join(output)

    assert "MALFORMED_" in result  # from predicate()

@patch("cim_plugin.cimxml_serializer.is_uuid_qualified")
def test_subject_malformedobject(mock_qualified, serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    mock_qualified.return_value = False
    g = ser.store

    g.bind("ex", "http://example.com/")

    s = URIRef("s123")
    t = URIRef("http://example.com/Class")
    p = URIRef("http://example.com/p")

    g.add((s, RDF.type, t))
    g.add((s, p, BNode("value")))

    ser.subject(s)

    result = "".join(output)
    assert '<ex:Class rdf:about="s123"' in result
    assert "<ex:p>MALFORMED_value</ex:p>" in result
    assert "</ex:Class>" in result

@patch("cim_plugin.cimxml_serializer.is_uuid_qualified")
def test_subject_predicatesorting(mock_qualified: MagicMock, serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    mock_qualified.return_value = False
    g = ser.store

    g.bind("ex", "http://example.com/")
    g.bind("foo", "http://bar.com/")

    s = URIRef("s123")
    t = URIRef("http://example.com/Class")
    p = URIRef("http://example.com/p")

    g.add((s, RDF.type, t))
    g.add((s, p, Literal("value")))
    g.add((s, URIRef("http://example.com/ad"), Literal("o")))
    g.add((s, URIRef("http://bar.com/ad"), Literal("o2")))
    g.add((s, URIRef("http://example.com/a_d"), Literal("o3")))

    ser.subject(s)

    result = "".join(output)

    assert result == '  <ex:Class rdf:about="s123">\n    <ex:a_d>o3</ex:a_d>\n    <ex:ad>o</ex:ad>\n    <ex:p>value</ex:p>\n    <foo:ad>o2</foo:ad>\n  </ex:Class>\n'


def test_subject_multipletypes(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    g = ser.store
    g.bind("ex", "http://example.com/")

    s = URIRef("http://example.com/s")
    t1 = URIRef("http://example.com/ClassA")
    t2 = URIRef("http://example.com/ClassB")

    g.add((s, RDF.type, t1))
    g.add((s, RDF.type, t2))
    g.add((s, URIRef("http://example.com/p"), Literal("x")))

    ser.subject(s)

    result = "".join(output)

    assert "<MALFORMED" in result
    assert "Multiple rdf:type values" in result
    assert "ClassA" in result
    assert "ClassB" in result
    assert "<ex:p>x</ex:p>" in result

@patch("cim_plugin.cimxml_serializer.is_uuid_qualified")
def test_subject_rdftypewithoutprefix(mock_qualified: MagicMock, serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    mock_qualified.return_value = False
    g = ser.store

    g.bind("ex", "http://example.com/")

    s = URIRef("s123")
    t = URIRef("http://unknown.com/Class")   # No prefix registered
    p = URIRef("http://example.com/p")

    g.add((s, RDF.type, t))
    g.add((s, p, Literal("value")))

    ser.subject(s)

    result = "".join(output)
    # .qname_strict adds <> around the type.
    assert result == '  <<http://unknown.com/Class> rdf:about="s123">\n    <ex:p>value</ex:p>\n  </<http://unknown.com/Class>>\n'


def test_subject_rdftypenoturi(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    g = ser.store

    g.bind("ex", "http://example.com/")

    s = URIRef("s123")
    t = Literal("Not-a-uri")   # Not a uri
    p = URIRef("http://example.com/p")

    g.add((s, RDF.type, t))
    g.add((s, p, Literal("value")))

    ser.subject(s)

    result = "".join(output)
    print(result)
    assert result == '  <MALFORMED rdf:about="s123">\n    <message>The rdf:type object is not a uri: Not-a-uri</message>\n    <rdf:type>Not-a-uri</rdf:type>\n    <ex:p>value</ex:p>\n  </MALFORMED>\n'


@patch("cim_plugin.cimxml_serializer.is_uuid_qualified")
def test_subject_rdftypemalformed(mock_qualified: MagicMock, serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    mock_qualified.return_value = False
    g = ser.store

    g.bind("ex", "http://example.com/")

    s = URIRef("s123")
    t = URIRef("http://example.com/1Class")   # Name starts with number
    p = URIRef("http://example.com/p")

    g.add((s, RDF.type, t))
    g.add((s, p, Literal("value")))

    ser.subject(s)

    result = "".join(output)
    
    assert result == '  <ex:1Class rdf:about="s123">\n    <ex:p>value</ex:p>\n  </ex:1Class>\n'


@patch("cim_plugin.cimxml_serializer.is_uuid_qualified")
def test_subject_circulartriples(mock_qualifier: MagicMock, serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    mock_qualifier.return_value = False
    g = ser.store

    g.bind("ex", "http://example.com/")

    s = URIRef("s123")
    t = URIRef("http://example.com/Class")   # Name starts with number
    p = URIRef("http://example.com/p")

    g.add((s, RDF.type, t))
    g.add((s, p, s))

    ser.subject(s)

    result = "".join(output)
    # .normalizeURI does not write out the whole name.
    assert result == '  <ex:Class rdf:about="s123">\n    <ex:p rdf:resource="s123"/>\n  </ex:Class>\n'
    

@patch("cim_plugin.cimxml_serializer.is_uuid_qualified")
def test_subject_predicatecalls(mock_qualified: MagicMock, serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    mock_qualified.side_effect = [False, False]
    g = ser.store

    s = URIRef("http://example.com/s")
    t = URIRef("http://example.com/Class")
    p1 = URIRef("http://example.com/p1")
    p2 = URIRef("http://example.com/p2")

    g.add((s, RDF.type, t))
    g.add((s, p1, Literal("x")))
    g.add((s, p2, Literal("y")))

    ser.predicate = Mock()

    ser.subject(s)

    qual_calls = [call(ser.qualifier_resolver, Literal("x")), call(ser.qualifier_resolver, Literal("y"))]
    pred_calls = [call(p1, Literal("x"), 2, use_qualifier=False), call(p2, Literal("y"), 2, use_qualifier=False)]
    mock_qualified.assert_has_calls(qual_calls)
    ser.predicate.assert_has_calls(pred_calls)


b = BNode("bad")    # Creating a shared bnode for test below

@pytest.mark.parametrize(
        "input, reachable",
        [
            pytest.param(Literal("bad"), set(), id="Literal subject"),
            pytest.param(BNode("bad"), set(), id="BNode not reachable"),
            pytest.param(b, {b}, id="BNode reachable"),
        ]
)
def test_subject_malformedsubjectcalls(input: Literal|BNode, reachable: set, serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    ser._write_malformed_subject = Mock()
    header_mock = Mock()
    header_mock.reachable_nodes = reachable
    ser.store.metadata_header = header_mock # pyright: ignore[reportAttributeAccessIssue]
    
    subject = input

    ser.subject(subject)

    if len(reachable) == 0:
        ser._write_malformed_subject.assert_called_once()
    else:
        ser._write_malformed_subject.assert_not_called()


def test_subject_multipletypesmalformedcalls(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    g = ser.store

    s = URIRef("http://example.com/s")
    g.add((s, RDF.type, URIRef("http://example.com/A")))
    g.add((s, RDF.type, URIRef("http://example.com/B")))

    ser._write_malformed_subject = Mock()

    ser.subject(s)

    ser._write_malformed_subject.assert_called_once()


# Unit tests .predicate
@pytest.mark.parametrize("literal", [
    pytest.param(Literal("simple"), id="Simple text"),
    pytest.param(Literal(42), id="Integer"),
    pytest.param(Literal(4.2), id="Float"),
    pytest.param(Literal("with <xml> chars"), id="Contains <>"),
    pytest.param(Literal("ampersand & test"), id="Contains ampersand"),
    pytest.param(Literal('with "quotes" inside'), id="Contains double quotes"),
    pytest.param(Literal("with 'single quotes'"), id="Contains single quotes"),
    pytest.param(Literal("with ]]>"), id="CDATA edge case"),
    pytest.param(Literal("with \n"), id="Contains new line"),
    pytest.param(Literal("with \t"), id="Contains tab"),
    pytest.param(Literal(""), id="Empty string"),
    pytest.param(Literal("  "), id="Whitespaces"),
    pytest.param(Literal("With datatype", datatype=XSD.string), id="Has a datatype"),
    pytest.param(Literal("Has language", lang="en"), id="Has language specification"),
])
def test_predicate_literal(literal: Literal, capture_writer: tuple[list, Callable]) -> None:
    output, writer = capture_writer
    g = Graph()
    g.bind("ex", "http://example.com/")
    pred = URIRef("http://example.com/p")

    ser = CIMXMLSerializer(g)
    ser.qualifier_resolver = Mock()
    ser.write = writer

    ser.predicate(pred, literal, depth=2)

    result = "".join(output)

    litval = escape(str(literal.value), ESCAPE_ENTITIES)
    assert f"    <ex:p>{litval}</ex:p>" in result
    assert "xml:lang" not in result
    assert "rdf:datatype" not in result


def test_predicate_booleanliteral(capture_writer: tuple[list, Callable]) -> None:
    output, writer = capture_writer
    g = Graph()
    g.bind("ex", "http://example.com/")
    pred = URIRef("http://example.com/p")
    obj = Literal(True)

    ser = CIMXMLSerializer(g)
    ser.qualifier_resolver = Mock()
    ser.write = writer

    ser.predicate(pred, obj, depth=2)

    result = "".join(output)

    assert f"    <ex:p>true</ex:p>" in result   # The boolean value True is made lowercase


@pytest.mark.parametrize(
        "object_value, return_value",[
            pytest.param("http://example.com/o", "#_o", id="Regular URIRef"),
            pytest.param("http://example.com/o s", "urn:uuid:o s", id="Contains spaces"),
            pytest.param("http://example.com/o&b", "ex:#_o&b", id="Contains ampersand"),
            pytest.param("http://example.com/o<b>", "#_o<b>", id="Contains <>"),
            pytest.param("http://example.com/o", "", id="Qualifier returns empty string"),
            pytest.param("foo/bar", "#_foo/bar", id="Relative uri")
        ]
)
def test_predicate_uriref(object_value: str, return_value: str, capture_writer: tuple[list, Callable]) -> None:
    output, writer = capture_writer
    g = Graph()
    g.bind("ex", "http://example.com/")
    pred = URIRef("http://example.com/p")
    obj = URIRef(object_value)

    ser = CIMXMLSerializer(g)
    ser.qualifier_resolver = Mock()
    ser.qualifier_resolver.convert_to_default_qualifier.return_value = return_value

    ser.write = writer

    ser.predicate(pred, obj, depth=1)
    result = "".join(output)

    esc_return = escape(return_value, ESCAPE_ENTITIES)
    assert f'<ex:p rdf:resource="{esc_return}"/>' in result
    ser.qualifier_resolver.convert_to_default_qualifier.assert_called_once_with(obj)


def test_predicate_noqualifier(capture_writer: tuple[list, Callable]) -> None:
    output, writer = capture_writer
    g = Graph()
    g.bind("ex", "http://example.com/")
    pred = URIRef("http://example.com/p")
    obj = URIRef("http://example.com/o")

    ser = CIMXMLSerializer(g)
    ser.qualifier_resolver = Mock()

    ser.write = writer

    ser.predicate(pred, obj, depth=1, use_qualifier=False)
    result = "".join(output)

    assert f'<ex:p rdf:resource="http://example.com/o"/>' in result
    ser.qualifier_resolver.convert_to_default_qualifier.assert_not_called()


def test_predicate_qnameerror(capture_writer: tuple[list, Callable]) -> None:
    output, writer = capture_writer
    g = Graph()
    pred = URIRef("http://unknown/p")
    obj = Literal("x")

    ser = CIMXMLSerializer(g)
    ser.qualifier_resolver = Mock()
    ser.write = writer

    ser.predicate(pred, obj)
    result = "".join(output)
    assert '  <ns1:p>x</ns1:p>\n' in result # If .qname_strict cannot find the namespace, it creates one


@pytest.mark.parametrize("depth,spaces", [
    pytest.param(0, "", id="Depth zero, no indentation"), 
    pytest.param(1, "  ", id="Depth 1, 2 indentation"), 
    pytest.param(3, "      ", id="Depth 3, 6 indentations"),
    pytest.param(-1, "", id="Negative depth, no indentation"),
    pytest.param(20, "                                        ", id="Very large indentation") # 40 spaces
])
def test_predicate_indentation(depth: int, spaces: str, capture_writer: tuple[list, Callable]) -> None:
    output, writer = capture_writer
    g = Graph()
    g.bind("ex", "http://example.com/")
    pred = URIRef("http://example.com/p")
    obj = Literal("x")

    ser = CIMXMLSerializer(g)
    ser.qualifier_resolver = Mock()
    ser.write = writer

    ser.predicate(pred, obj, depth=depth)
    result = "".join(output)

    assert result.startswith(spaces)


@pytest.mark.parametrize("predicate,expected,log_error", [
    pytest.param(URIRef("http://example.com/p"), "ex:p", False, id="URIRef"),
    pytest.param(URIRef("http://noprefix.com/p"), ":p", False, id="Namespace with no prefix"),
    pytest.param(URIRef("http://noneprefix.com/p"), ":p", False, id="Namespace with prefix None"),
    pytest.param(URIRef("p"), "MALFORMED_p", False, id="Prefix with no namespace"),
    pytest.param(URIRef("http://example.com/pære"), "ex:pære", False, id="Unicode letters"),
    pytest.param(URIRef("http://example.com/per%cent%"), "ex:per%cent%", False, id="Percent encoded"),
    pytest.param(URIRef("http://example.com/#foo"), "ns1:foo", False, id="Fragment identifier"),
    pytest.param(URIRef("http://example.com/?x=1"), "MALFORMED_http://example.com/?x=1", False, id="Query parameters"),
    pytest.param(Literal("p"), "MALFORMED_p", True, id="Literal"), 
    pytest.param(BNode("p"), "MALFORMED_p", True, id="BNode")
])
def test_predicate_predicatetypes(predicate: Node, expected: str, log_error: bool, capture_writer: tuple[list, Callable], caplog: pytest.LogCaptureFixture) -> None:
    output, writer = capture_writer
    g = Graph()
    g.bind("ex", "http://example.com/")
    g.bind("", "http://noprefix.com/p")
    g.bind(None, "http://noneprefix.com")
    g.bind("ex2", "")
    pred = predicate
    obj = Literal("x")

    ser = CIMXMLSerializer(g)
    ser.qualifier_resolver = Mock()
    ser.write = writer

    ser.predicate(pred, obj)
    result = "".join(output)
    assert expected in result
    if log_error:
        assert "Predicate p not a valid predicate." in caplog.text


def test_predicate_noobject(capture_writer: tuple[list, Callable], caplog: pytest.LogCaptureFixture) -> None:
    # This test documents what happends if object does not exist or is not a URIRef or Literal.
    output, writer = capture_writer
    g = Graph()
    pred = URIRef("http://unknown/p")
    obj = None

    ser = CIMXMLSerializer(g)
    ser.qualifier_resolver = Mock()
    ser.write = writer

    # Pylance silenced to test invalid input
    ser.predicate(pred, obj)    # type: ignore
    result = "".join(output)
    assert "MALFORMED_None" in result
    assert "Invalid object detected." in caplog.text

# Integration tests .predicate and .subject
@pytest.mark.parametrize(
    "input_uri,output_strategy,expected_resource",
    [
        pytest.param("_1234", UnderscoreQualifier(), "#_1234", id="Underscore input, underscore output"),
        pytest.param("urn:uuid:abcd", UnderscoreQualifier(), "#_abcd", id="Urn input, underscore output"),
        pytest.param(f"{uuid_namespace}:xyz", UnderscoreQualifier(), "#_xyz", id="Namespace input, underscore output"),
        pytest.param("weird", UnderscoreQualifier(), "#_weird", id="Fallback, underscore output"),
        pytest.param("_1234", URNQualifier(), "urn:uuid:1234", id="Underscore input, urn output"),
        pytest.param("urn:uuid:abcd", URNQualifier(), "urn:uuid:abcd", id="Urn input, urn output"),
        pytest.param(f"{uuid_namespace}:xyz", URNQualifier(), "urn:uuid:xyz", id="Namespace input, urn output"),
        pytest.param("weird", URNQualifier(), "urn:uuid:weird", id="Fallback, urn output"),
        pytest.param("_1234", NamespaceQualifier(), f"{uuid_namespace}:1234", id="Underscore input, namespace output"),
        pytest.param("urn:uuid:abcd", NamespaceQualifier(), f"{uuid_namespace}:abcd", id="Urn input, namespace output"),
        pytest.param(f"{uuid_namespace}:xyz", NamespaceQualifier(), f"{uuid_namespace}:xyz", id="Namespace input, namespace output"),
        pytest.param("weird", NamespaceQualifier(), f"{uuid_namespace}:weird", id="Fallback, namespace output"),
    ]
)
def test_predicate_resolver_integration(capture_writer: tuple[list, Callable], input_uri: str, output_strategy: CIMQualifierStrategy, expected_resource: str) -> None:
    output, writer = capture_writer

    g = Graph()
    g.bind("ex", "http://example.com/")

    s = URIRef("http://example.com/s")
    p = URIRef("http://example.com/p")
    o = URIRef(input_uri)

    g.add((s, p, o))

    ser = CIMXMLSerializer(g)
    ser.write = writer
    ser.qualifier_resolver = CIMQualifierResolver(output_strategy)

    ser.predicate(p, o)

    result = "".join(output)

    assert f'rdf:resource="{expected_resource}"' in result


@pytest.mark.parametrize(
    "input_uri,output_strategy,expected_about",
    [
        pytest.param("_1234", UnderscoreQualifier(), "#_1234", id="Underscore input, underscore output"),
        pytest.param("urn:uuid:abcd", UnderscoreQualifier(), "#_abcd", id="Urn input, underscore output"),
        pytest.param(f"{uuid_namespace}:xyz", UnderscoreQualifier(), "#_xyz", id="Namespace input, underscore output"),
        pytest.param("weird", UnderscoreQualifier(), "#_weird", id="Fallback, underscore output"),
        pytest.param("_1234", URNQualifier(), "urn:uuid:1234", id="Underscore input, urn output"),
        pytest.param("urn:uuid:abcd", URNQualifier(), "urn:uuid:abcd", id="Urn input, urn output"),
        pytest.param(f"{uuid_namespace}:xyz", URNQualifier(), "urn:uuid:xyz", id="Namespace input, urn output"),
        pytest.param("weird", URNQualifier(), "urn:uuid:weird", id="Fallback, urn output"),
        pytest.param("_1234", NamespaceQualifier(), f"{uuid_namespace}:1234", id="Underscore input, namespace output"),
        pytest.param("urn:uuid:abcd", NamespaceQualifier(), f"{uuid_namespace}:abcd", id="Urn input, namespace output"),
        pytest.param(f"{uuid_namespace}:xyz", NamespaceQualifier(), f"{uuid_namespace}:xyz", id="Namespace input, namespace output"),
        pytest.param("weird", NamespaceQualifier(), f"{uuid_namespace}:weird", id="Fallback, namespace output"),
    ]
)
def test_subject_resolver_integration(capture_writer: tuple[list, Callable], input_uri: str, output_strategy: CIMQualifierStrategy, expected_about: str) -> None:
    output, writer = capture_writer

    g = Graph()
    g.metadata_header = CIMMetadataHeader.empty()   # pyright: ignore[reportAttributeAccessIssue]
    g.bind("ex", "http://example.com/")

    s = URIRef(input_uri)
    t = URIRef("http://example.com/Class")

    g.add((s, RDF.type, t))

    ser = CIMXMLSerializer(g)
    ser.write = writer
    ser.qualifier_resolver = CIMQualifierResolver(output_strategy)

    ser.subject(s)

    result = "".join(output)

    assert f'rdf:about="{expected_about}"' in result


@pytest.mark.parametrize(
    "subject_uri,object_uri,output_strategy,expected_about,expected_resource",
    [
        pytest.param("_s", "_o", UnderscoreQualifier(), "#_s", "#_o", id="Underscore input, underscore output"),
        pytest.param("urn:uuid:abcd", "urn:uuid:efgh", UnderscoreQualifier(), "#_abcd", "#_efgh", id="Urn input, underscore output"),
        pytest.param(f"{uuid_namespace}:x", f"{uuid_namespace}:y", UnderscoreQualifier(), "#_x", "#_y", id="Namespace input, underscore output"),
        pytest.param("weird", "strange", UnderscoreQualifier(), "#_weird", "strange", id="Fallback, underscore output"),
        pytest.param("_s", "_o", URNQualifier(), "urn:uuid:s", "urn:uuid:o", id="Underscore input, urn output"),
        pytest.param("urn:uuid:abcd", "urn:uuid:efgh", URNQualifier(), "urn:uuid:abcd", "urn:uuid:efgh", id="Urn input, urn output"),
        pytest.param(f"{uuid_namespace}:x", f"{uuid_namespace}:y", URNQualifier(), "urn:uuid:x", "urn:uuid:y", id="Namespace input, urn output"),
        pytest.param("weird", "strange", URNQualifier(), "urn:uuid:weird", "strange", id="Fallback, urn output"),
        pytest.param("_s", "_o", NamespaceQualifier(), f"{uuid_namespace}:s", f"{uuid_namespace}:o", id="Underscore input, namespace output"),
        pytest.param("urn:uuid:abcd", "urn:uuid:efgh", NamespaceQualifier(), f"{uuid_namespace}:abcd", f"{uuid_namespace}:efgh", id="Urn input, namespace output"),
        pytest.param(f"{uuid_namespace}:x", f"{uuid_namespace}:y", NamespaceQualifier(), f"{uuid_namespace}:x", f"{uuid_namespace}:y", id="Namespace input, namespace output"),
        pytest.param("weird", "strange", NamespaceQualifier(), f"{uuid_namespace}:weird", "strange", id="Fallback, namespace output"),
    ]
)
def test_subject_and_predicate_resolver_integration_with_default_qualifier(
    capture_writer: tuple[list, Callable],
    subject_uri: str,
    object_uri: str,
    output_strategy: CIMQualifierStrategy,
    expected_about: str,
    expected_resource: str,
):
    output, writer = capture_writer

    g = Graph()
    g.metadata_header = CIMMetadataHeader.empty()   # pyright: ignore[reportAttributeAccessIssue]
    g.bind("ex", "http://example.com/")

    s = URIRef(subject_uri)
    t = URIRef("http://example.com/Class")
    p = URIRef("http://example.com/p")
    o = URIRef(object_uri)

    g.add((s, RDF.type, t))
    g.add((s, p, o))

    ser = CIMXMLSerializer(g)
    ser.write = writer
    ser.qualifier_resolver = CIMQualifierResolver(output_strategy)

    ser.subject(s)

    result = "".join(output)

    assert f'rdf:about="{expected_about}"' in result
    assert f'rdf:resource="{expected_resource}"' in result


@pytest.mark.parametrize(
    "subject_uri,object_uri,output_strategy,expected_about,expected_resource",
    [
        pytest.param("_s", "_o", UnderscoreQualifier(), "_s", "#_o", id="Underscore input, underscore output"),
        pytest.param("urn:uuid:abcd", "urn:uuid:efgh", UnderscoreQualifier(), "_abcd", "#_efgh", id="Urn input, underscore output"),
        pytest.param(f"{uuid_namespace}:x", f"{uuid_namespace}:y", UnderscoreQualifier(), "_x", "#_y", id="Namespace input, underscore output"),
        pytest.param("weird", "strange", UnderscoreQualifier(), "_weird", "strange", id="Fallback, underscore output"),
        pytest.param("_s", "_o", URNQualifier(), "urn:uuid:s", "urn:uuid:o", id="Underscore input, urn output"),
        pytest.param("urn:uuid:abcd", "urn:uuid:efgh", URNQualifier(), "urn:uuid:abcd", "urn:uuid:efgh", id="Urn input, urn output"),
        pytest.param(f"{uuid_namespace}:x", f"{uuid_namespace}:y", URNQualifier(), "urn:uuid:x", "urn:uuid:y", id="Namespace input, urn output"),
        pytest.param("weird", "strange", URNQualifier(), "urn:uuid:weird", "strange", id="Fallback, urn output"),
        pytest.param("_s", "_o", NamespaceQualifier(), f"{uuid_namespace}:s", f"{uuid_namespace}:o", id="Underscore input, namespace output"),
        pytest.param("urn:uuid:abcd", "urn:uuid:efgh", NamespaceQualifier(), f"{uuid_namespace}:abcd", f"{uuid_namespace}:efgh", id="Urn input, namespace output"),
        pytest.param(f"{uuid_namespace}:x", f"{uuid_namespace}:y", NamespaceQualifier(), f"{uuid_namespace}:x", f"{uuid_namespace}:y", id="Namespace input, namespace output"),
        pytest.param("weird", "strange", NamespaceQualifier(), f"{uuid_namespace}:weird", "strange", id="Fallback, namespace output"),
    ]
)
def test_subject_and_predicate_resolver_integration_with_special_qualifier(
    capture_writer: tuple[list, Callable],
    subject_uri: str,
    object_uri: str,
    output_strategy: CIMQualifierStrategy,
    expected_about: str,
    expected_resource: str,
):
    output, writer = capture_writer

    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty()
    g.bind("ex", "http://example.com/")
    g.metadata_header.profile = "http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0"

    s = URIRef(subject_uri)
    t = URIRef("http://example.com/Class")
    p = URIRef("http://example.com/p")
    o = URIRef(object_uri)

    g.add((s, RDF.type, t))
    g.add((s, p, o))

    ser = CIMXMLSerializer(g)
    ser.write = writer
    ser.qualifier_resolver = CIMQualifierResolver(output_strategy)

    ser.subject(s)

    result = "".join(output)

    assert f'rdf:ID="{expected_about}"' in result
    assert f'rdf:resource="{expected_resource}"' in result

# Unit tests ._write_malformed_subject
def test_write_malformed_subject_success(capture_writer: tuple[list, Callable], caplog: pytest.LogCaptureFixture) -> None:
    output, writer = capture_writer
    g = Graph()
    g.bind("ex", "http://example.com/")
    sub = URIRef("s1")
    pred = URIRef("http://example.com/p")
    obj = Literal(True)
    g.add((sub, pred, obj))
    ser = CIMXMLSerializer(g)
    ser.write = writer

    ser._write_malformed_subject(sub, "error", depth=1)

    result = "".join(output)
    print(result)
    assert result == '  <MALFORMED rdf:about="s1">\n    <message>error</message>\n    <ex:p>true</ex:p>\n  </MALFORMED>\n'
    assert "error" in caplog.text


def test_write_malformed_subject_predicatecalls(capture_writer: tuple[list, Callable], caplog: pytest.LogCaptureFixture) -> None:
    output, writer = capture_writer
    g = Graph()
    g.bind("ex", "http://example.com/")
    sub = URIRef("s1")
    pred = URIRef("http://example.com/p")
    obj = Literal(True)
    g.add((sub, pred, obj))
    g.add((sub, URIRef("http://example.com/p2"), Literal("1")))
    ser = CIMXMLSerializer(g)
    ser.predicate = Mock()
    ser.write = writer

    ser._write_malformed_subject(sub, "oops", depth=1)

    assert ser.predicate.call_count == 2
    error_logs = [rec for rec in caplog.records if rec.levelname == "ERROR"] 
    assert len(error_logs) == 1
    assert error_logs[0].message == "oops"


def test_write_malformed_subject_nopredicates(capture_writer: tuple[list, Callable], caplog: pytest.LogCaptureFixture) -> None:
    output, writer = capture_writer
    g = Graph()
    g.bind("ex", "http://example.com/")
    sub = URIRef("s1")
    ser = CIMXMLSerializer(g)
    ser.predicate = Mock()
    ser.write = writer

    ser._write_malformed_subject(sub, "error", depth=1)
    result = "".join(output)

    assert ser.predicate.call_count == 0
    assert result == '  <MALFORMED rdf:about="s1">\n    <message>error</message>\n  </MALFORMED>\n'
    assert "error" in caplog.text

@pytest.mark.parametrize(
        "depth, expected",
        [
            pytest.param(1, '  <MALFORMED rdf:about="s1">\n    <message>error</message>\n  </MALFORMED>\n', id="depth 1"),
            pytest.param(0, '<MALFORMED rdf:about="s1">\n  <message>error</message>\n</MALFORMED>\n', id="depth 0"),
            pytest.param(3, '      <MALFORMED rdf:about="s1">\n        <message>error</message>\n      </MALFORMED>\n', id="depth 3")
        ]
)
def test_write_malformed_subject_indents(depth: int, expected: str, capture_writer: tuple[list, Callable], caplog: pytest.LogCaptureFixture) -> None:
    output, writer = capture_writer
    g = Graph()
    g.bind("ex", "http://example.com/")
    sub = URIRef("s1")
    ser = CIMXMLSerializer(g)
    ser.predicate = Mock()
    ser.write = writer

    ser._write_malformed_subject(sub, "error", depth=depth)
    result = "".join(output)

    assert ser.predicate.call_count == 0
    assert result == expected
    assert "error" in caplog.text


# Unit tests _subject_sort_key

@patch("cim_plugin.cimxml_serializer._extract_uuid_from_urn")
def test_subject_sort_key_uuidfound(mock_extract: MagicMock) -> None:
    mock_extract.return_value = uuid.UUID('12345678123456781234567812345678')
    subject = URIRef('urn:uuid:12345678123456781234567812345678')
    result = _subject_sort_key(subject)
    assert result[0] == 0
    assert result[1] == '12345678-1234-5678-1234-567812345678'
    mock_extract.assert_called_once_with('urn:uuid:12345678123456781234567812345678')


@patch("cim_plugin.cimxml_serializer._extract_uuid_from_urn")
def test_subject_sort_key_notfound(mock_extract: MagicMock) -> None:
    mock_extract.side_effect = ValueError("Invalid model URI: notuuid")
    subject = URIRef('notuuid')
    result = _subject_sort_key(subject)
    assert result[0] == 1
    assert result[1] == 'notuuid'
    mock_extract.assert_called_once_with('notuuid')


def test_subject_sort_key_sortingbehavior() -> None:
    items = [
        ("notuuid", (1, "notuuid")),
        ("urn:uuid:12345678123456781234567812345678", (0, "12345678-1234-5678-1234-567812345678")),
    ]

    uris = [URIRef(i[0]) for i in items]
    sorted_uris = sorted(uris, key=_subject_sort_key)

    assert str(sorted_uris[0]) == "urn:uuid:12345678123456781234567812345678"
    assert str(sorted_uris[1]) == "notuuid"


@patch("cim_plugin.cimxml_serializer._extract_uuid_from_urn")
def test_subject_sort_key_unexpectedexception(mock_extract: MagicMock) -> None:
    mock_extract.side_effect = TypeError("boom")
    subject = URIRef("whatever")

    with pytest.raises(TypeError):
        _subject_sort_key(subject)

@patch("cim_plugin.cimxml_serializer._extract_uuid_from_urn")
def test_subject_sort_key_nonstringuri(mock_extract: MagicMock) -> None:
    class Weird:
        def __str__(self):
            return "weird"

    mock_extract.side_effect = ValueError("Invalid model URI: weird")
    # Pylance silenced to test invalid input
    result = _subject_sort_key(Weird()) # type: ignore

    assert result == (1, "weird")


if __name__ == "__main__":
    pytest.main()