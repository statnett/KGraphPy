from typing import Callable, Type, cast, Any
import pytest
from unittest.mock import MagicMock, call, patch, Mock
import uuid
import io
import logging

from rdflib import Namespace, URIRef, Graph, Literal, Node, BNode
from rdflib.plugins.serializers.xmlwriter import ESCAPE_ENTITIES
from rdflib.namespace import XSD, RDF, DCAT, DCTERMS
from xml.sax.saxutils import escape
from cim_plugin.cimxml_serializer import _subject_sort_key, CIMXMLSerializer
from cim_plugin.qualifiers import CIMQualifierStrategy, UnderscoreQualifier, URNQualifier, NamespaceQualifier, CIMQualifierResolver, uuid_namespace
from cim_plugin.header import CIMMetadataHeader
from cim_plugin.graph import CIMGraph
from cim_plugin.namespaces import MD, DCAT_EXT
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
    assert ser.qualifier_resolver
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
def test_collect_used_namespaces_onlyregisterednamespaces(make_cimgraph: CIMGraph, uri: str, expected_prefix: str) -> None:
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
    

def test_collect_used_namespaces_urnsignored(make_cimgraph: CIMGraph) -> None:
    g = make_cimgraph
    g.add((URIRef("urn:uuid:1234"), URIRef("http://example.com/p"), URIRef("http://example.com/o")))

    ser = CIMXMLSerializer(g)
    ns_list = dict(ser._collect_used_namespaces())

    assert len(ns_list) == 3
    assert ns_list.keys() == {'dcat', 'ex', 'rdf'}  # No urn in the the prefix list
    

def test_collect_used_namespaces_headertriples(make_cimgraph: CIMGraph) -> None:
    g = make_cimgraph

    assert g.metadata_header
    g.metadata_header.graph.bind("foo", "http://foo.org/ns#")
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

    g.add((
        URIRef("http://example.com/A"),
        URIRef("http://example.com/p"),
        URIRef("http://example.com/o")
    ))
    g.metadata_header = CIMMetadataHeader.empty(URIRef("http://example.com/header"))

    ser = CIMXMLSerializer(g)
    ns_list = dict(ser._collect_used_namespaces())

    assert ns_list == {}    # The namespace was not bound to the namespace manager


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


def test_collect_used_namespaces_overlappingnamespacesshortnotused() -> None:
    g = CIMGraph()
    g.bind("ex", Namespace("http://example.com/"))
    g.bind("exns", Namespace("http://example.com/ns/"))
    g.metadata_header = CIMMetadataHeader.empty(URIRef("http://example.com/ns/header"))
    g.add((
        URIRef("http://example.com/ns/Thing"),
        URIRef("http://example.com/ns/p"),
        URIRef("http://example.com/ns/Object")
    ))

    ser = CIMXMLSerializer(g)
    ns_list = dict(ser._collect_used_namespaces())
    assert "ex" not in ns_list
    assert "exns" in ns_list
    assert str(ns_list["exns"]) == "http://example.com/ns/"


def test_collect_used_namespaces_collisions() -> None:
    # This test documents that g.bind decides which prefix is kept with namespace collisions.
    
    g = CIMGraph()

    # Three prefixes bound to the same namespace URI in different ways
    g.bind("ex0", Namespace("http://example.com/"))
    g.bind("ex", Namespace("http://example.com/"), override=True) # New prefix replaces old
    g.bind("alt", Namespace("http://example.com/"), override=False) # Old prefix kept, new not added

    g.metadata_header = CIMMetadataHeader.empty(URIRef("http://example.com/header"))

    g.add((
        URIRef("http://example.com/Thing"),
        URIRef("http://example.com/p"),
        URIRef("http://example.com/o")
    ))

    ser = CIMXMLSerializer(g)
    ns_list = dict(ser._collect_used_namespaces())

    assert len(ns_list) == 1    # Only one prefix should survive
    assert "ex" in ns_list
    assert ns_list["ex"] == URIRef("http://example.com/")   # It must map to the correct namespace URI


def test_collect_used_namespaces_collisionsfromdifferentsources() -> None:
    # Documents namespace collisions where one namespace comes from the graph and another from the metadata header.
    g = CIMGraph()
    g.bind("ex", Namespace("http://example.com/"))

    g.metadata_header = CIMMetadataHeader.empty(URIRef("http://example.com/header"))
    g.metadata_header.graph.bind("alt", Namespace("http://example.com/"))
    g.metadata_header.add_triple(URIRef("http://example.com/p"), Literal("o"))

    g.add((
        URIRef("http://example.com/Thing"),
        URIRef("http://example.com/p"),
        URIRef("http://example.com/o")
    ))

    ser = CIMXMLSerializer(g)
    ns_list = dict(ser._collect_used_namespaces())

    assert len(ns_list) == 2    # Both namespaces are collected
    assert "alt" in ns_list
    assert "ex" in ns_list
    assert ns_list["ex"] == ns_list["alt"]  # Both prefixes points to the same namespace


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


# Unit tests _build_subject_index
def test_build_subject_index_emptygraph() -> None:
    g = Graph()
    ser = CIMXMLSerializer(g)
    index = ser._build_subject_index(skip_subjects=set())
    assert index == {}

def test_build_subject_index_basic() -> None:
    g = Graph()
    g.bind("ex", Namespace("http://example.com/"))
    
    g.add((URIRef("s1"), RDF.type, URIRef("http://example.com/TypeA")))
    g.add((URIRef("s2"), RDF.type, URIRef("http://example.com/TypeB")))
    g.add((URIRef("s3"), RDF.type, URIRef("http://example.com/TypeC"))) # Exluded by skip_subjects
    g.add((URIRef("s4"), URIRef("not_rdftype"), URIRef("http://example.com/TypeA")))    # Excluded because not RDF.type
    g.add((URIRef("s1"), URIRef("http://example.com/p"), URIRef("o"))) # Excluded because not RDF.type, is indexed by RDF.type triples
    g.add((URIRef("s1"), RDF.type, URIRef("http://example.com/TypeB"))) # s1 has two types, is indexed under both.
    
    ser = CIMXMLSerializer(g)
    index = ser._build_subject_index(skip_subjects={URIRef("s3")})
    
    assert index == {
        "ex:TypeA": {URIRef("s1")},
        "ex:TypeB": {URIRef("s2"), URIRef("s1")},
        "ErrorMissingType": {URIRef("s4")},
    }


bn = BNode()    # Creating a blank node for test below

@pytest.mark.parametrize(
    "triples, expected_result",
    [
        pytest.param([(bn, RDF.type, URIRef("http://example.com/TypeA"))], {"ex:TypeA": {bn}}, id="Blank node as subject and RDF.type triple"),
        pytest.param([(Literal("NotURI"), RDF.type, URIRef("http://example.com/TypeA"))], {"ex:TypeA": {Literal("NotURI")}}, id="Literal as subject and RDF.type triple"),
        pytest.param([(URIRef("s1"), RDF.type, Literal("NotAURI"))], {"NotAURI": {URIRef("s1")}}, id="URI subject with literal object in RDF.type triple"),
        pytest.param([(URIRef("s1"), RDF.type, URIRef("NotAURI"))], {"<NotAURI>": {URIRef("s1")}}, id="URI subject with URI object that cannot be qname in RDF.type triple"),
        pytest.param([(URIRef("s1"), RDF.type, URIRef("http://example.com/TypeA")), 
                      (URIRef("s1"), RDF.type, URIRef("http://example.com/TypeA"))], {"ex:TypeA": {URIRef("s1")}}, id="Duplicate RDF.type triples"), 
    ]
)
def test_build_subject_index_edgecases(triples: list[tuple[Node, Node, Node]], expected_result: dict[str, set[Node]]) -> None:
    g = Graph()
    g.bind("ex", Namespace("http://example.com/"))
    
    for s, p, o in triples:
        g.add((s, p, o))

    ser = CIMXMLSerializer(g)
    index = ser._build_subject_index(skip_subjects=set())
    
    assert index == expected_result

# Unit tests .serialize
@patch("cim_plugin.cimxml_serializer._subject_sort_key")
def test_serialize_allcalls(mock_sort: MagicMock) -> None:
    buf = io.BytesIO()
    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty(URIRef("h1"))
    ser = CIMXMLSerializer(g)

    ser._ensure_header = Mock(return_value=g.metadata_header)
    ser._init_qualifier_resolver = Mock()
    ser._collect_used_namespaces = Mock(return_value=[("ex", "example.com/")])
    ser.write_header = Mock()
    ser._build_subject_index = Mock(return_value={"ex:o": [URIRef("s1"), URIRef("s2")]})
    mock_sort.side_effect = [(1, "s1"), (0, "s2")]
    ser.subject = Mock()

    ser.serialize(buf, qualifier="foo")
    result = buf.getvalue().decode()

    ser._ensure_header.assert_called_once()
    ser._init_qualifier_resolver.assert_called_once_with("foo")
    ser._collect_used_namespaces.assert_called_once()
    ser.write_header.assert_called_once()
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
def test_serialize_subjectsorting(subjects: list[URIRef], expected: list[str]) -> None:
    buf = io.BytesIO()
    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty(URIRef("h1"))

    ser = CIMXMLSerializer(g)

    # Fake grouping
    def fake_group(*args, **kwargs):
        return {"ex:o": subjects}

    ser._build_subject_index = Mock(side_effect=fake_group)
    ser.subject = Mock()
    ser.serialize(buf)
    
    ser.subject.assert_has_calls([call(URIRef(s), depth=1) for s in expected])

@pytest.mark.parametrize("enc", ["utf-8", "latin-1"])
def test_serialize_encoding(enc: str) -> None:
    buf = io.BytesIO()
    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty(URIRef("h1"))
    g.metadata_header.graph.bind("ex", "http://example.com/")
    g.metadata_header.add_triple(RDF.type, DCAT_EXT.Dataset)
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
    assert out == '<?xml version="1.0" encoding="utf-8"?>\n<rdf:RDF\n    >\n  <dcat:Dataset rdf:about="urn:uuid:h1"/>\n\n</rdf:RDF>\n'


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
    ser.subject = Mock()  
    ser.serialize(buf)

    ser.subject.assert_has_calls([
        call(URIRef(f"urn:uuid:{s3}"), depth=1),
        call(URIRef(f"urn:uuid:{s2}"), depth=1),
        call(URIRef(f"urn:uuid:{s1}"), depth=1),
        call(URIRef(s5), depth=1),  # invalid UUID → sorted last among TypeA 
        call(URIRef(f"urn:uuid:{s4}"), depth=1) # TypeB comes after TypeA 
    ])

    
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
    header = CIMMetadataHeader.empty(subject=URIRef("s1"))
    header.graph.bind("ex", "http://example.com/")
    header.add_triple(RDF.type, DCAT_EXT.Dataset)
    header.add_triple(URIRef("http://example.com/p"), Literal("o"))
    g.metadata_header = header
    ser = CIMXMLSerializer(g)
    ser._namespace_lookup = [("http://example.com/", "ex")]
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
    ser._namespace_lookup = [("http://example.com/", "ex")]
    ser.write = writer
    ser.qualifier_resolver = CIMQualifierResolver(UnderscoreQualifier())

    ser.write_header(header)

    result = "".join(output)
    
    assert result == '  <dcat:Dataset rdf:about="urn:uuid:s1">\n    <ex:a rdf:resource="o"/>\n    <ex:p>o</ex:p>\n  </dcat:Dataset>\n'
    assert type(ser.qualifier_resolver.output) == UnderscoreQualifier


def test_write_header_multiplerdftypes(capture_writer: tuple[list, Callable]) -> None:
    output, writer = capture_writer
    g = CIMGraph()
    g.bind("ex", "http://example.com/")
    header = CIMMetadataHeader.empty(subject=URIRef("s1"))
    header.add_triple(RDF.type, DCAT_EXT.Dataset)   # dcat:Dataset is prioritized as header type.
    header.add_triple(RDF.type, MD.FullModel)   # Other rdf.type triples are treated as any other triple
    header.add_triple(RDF.type, URIRef("o"))
    g.metadata_header = header
    ser = CIMXMLSerializer(g)
    ser._namespace_lookup = [("http://example.com/", "ex"), ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf")]
    ser.write = writer
    ser.qualifier_resolver = CIMQualifierResolver(UnderscoreQualifier())

    ser.write_header(header)

    result = "".join(output)
    
    assert '<dcat:Dataset rdf:about="urn:uuid:s1">' in result
    assert '<rdf:type rdf:resource="http://iec.ch/TC57/61970-552/ModelDescription/1#FullModel"/>' in result
    assert type(ser.qualifier_resolver.output) == UnderscoreQualifier


def test_write_header_multiplerdftypesnodcatdataset(capture_writer: tuple[list, Callable]) -> None:
    # If there are multiple rdf:type triples in the header, but none of them is dcat:Dataset, the first is used as header type
    output, writer = capture_writer
    g = CIMGraph()
    g.bind("ex", "http://example.com/")
    g.bind("md", MD)
    header = CIMMetadataHeader.empty(subject=URIRef("s1"))
    header.graph.bind("md", MD)
    header.add_triple(RDF.type, MD.FullModel)
    header.add_triple(RDF.type, URIRef("o"))
    g.metadata_header = header
    ser = CIMXMLSerializer(g)
    ser._namespace_lookup = [("http://example.com/", "ex"), ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf")]
    ser.write = writer
    ser.qualifier_resolver = CIMQualifierResolver(UnderscoreQualifier())

    ser.write_header(header)

    assert output[0] == '  <md:FullModel rdf:about="urn:uuid:s1"'
    assert output[2] == '    <rdf:type rdf:resource="o"/>\n'


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
    g.metadata_header.add_triple(URIRef("http://example.com/p"), Literal("o"))
    ser = CIMXMLSerializer(g)
    ser.qualifier_resolver = CIMQualifierResolver(NamespaceQualifier())
    ser.write = writer
    
    ser.write_header(g.metadata_header)

    assert output[0] == '  <dcat:Dataset rdf:about="urn:uuid:s1"'
    assert output[2] == '    <http://example.com/p>o</http://example.com/p>\n'
    assert "Header type missing. dcat:Dataset used as default." in caplog.text


def test_write_header_noqualifierresolver(capture_writer: tuple[list, Callable]) -> None:
    # Documents what happens if the qualifier_resolver has not been set
    output, writer = capture_writer
    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty(subject=URIRef("s1"))
    ser = CIMXMLSerializer(g)
    ser.write = writer
    
    with pytest.raises(AssertionError):
        ser.write_header(g.metadata_header)

# Unit tests .subject
def test_subject_nonuriref(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    ser.store.bind("ex", "http://example.com/")
    subject = Literal("not-a-uri")
    ser.store.add((subject, RDF.type, URIRef("http://example.com/TypeA")))

    ser.subject(subject)
    print(output)
    assert output[0] == '  <ex:TypeA rdf:about="not-a-uri">\n'  # Non-uriref subjects gets written out as string
    # assert "Subject is not a URIRef: not-a-uri" in caplog.text # If logging is uncommented

@patch("cim_plugin.cimxml_serializer.find_rdf_id_or_about")
def test_subject_missingtype(mock_find: MagicMock, serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    g = ser.store
    s = URIRef("s1")
    g.add((s, URIRef("http://example.com/p"), Literal("x")))
    ser._write_untyped_subject = Mock()

    ser.subject(s)

    ser._write_untyped_subject.assert_called_once_with(s, 1)
    mock_find.assert_not_called()
    # assert "No rdf:type triple detected for s1." in caplog.text # If logging is uncommented

@patch("cim_plugin.cimxml_serializer.is_uuid_qualified", return_value=False)
@patch("cim_plugin.cimxml_serializer.find_rdf_id_or_about", return_value="about")
def test_subject_multipletypes(mock_find: MagicMock, mock_uuid: MagicMock, serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    ser._write_untyped_subject = Mock()
    ser.predicate = Mock()
    g = ser.store
    g.bind("ex", "http://example.com/")
    s = URIRef("http://example.com/s")
    t1 = URIRef("http://example.com/ClassA")
    t2 = URIRef("http://example.com/ClassB")

    g.add((s, RDF.type, t1))
    g.add((s, RDF.type, t2))
    g.add((s, URIRef("http://example.com/p"), Literal("x")))

    ser.subject(s)

    ser._write_untyped_subject.assert_not_called()
    mock_find.assert_called_once_with(None, "http://example.com/ClassA")  # First rdf:type triple is picked as subject type
    predicate_calls = [
        call(URIRef("http://example.com/p"), Literal("x"), 2, use_qualifier=False),
        call(RDF.type, URIRef("http://example.com/ClassB"), 2, use_qualifier=False) # Other rdf:type triples are treated as normal triples
    ]
    ser.predicate.assert_has_calls(predicate_calls, any_order=True)
    # assert "Multiple rdf:type triples detected for http://example.com/s" in caplog.text # If logging is uncommented

@patch("cim_plugin.cimxml_serializer.is_uuid_qualified", return_value=False)
@patch("cim_plugin.cimxml_serializer.find_rdf_id_or_about", return_value="about")
def test_subject_valid(mock_find: MagicMock, mock_uuid: MagicMock, serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    ser._write_untyped_subject = Mock()
    ser.predicate = Mock()
    g = ser.store

    g.bind("ex", "http://example.com/")

    s = URIRef("s123")
    t = URIRef("http://example.com/Class")
    p = URIRef("http://example.com/p")

    g.add((s, RDF.type, t))
    g.add((s, p, Literal("value")))

    ser.subject(s)

    ser._write_untyped_subject.assert_not_called()
    mock_find.assert_called_once_with(None, "http://example.com/Class")
    ser.predicate.assert_called_once_with(p, Literal("value"), 2, use_qualifier=False)
    
    assert output[0] == '  <ex:Class rdf:about="s123">\n' # The rdf:type triple is written first


@pytest.mark.parametrize("qualifier_return", [True, False])
@patch("cim_plugin.cimxml_serializer.is_uuid_qualified")
def test_subject_objectuuid(mock_uuid: MagicMock, serializer: tuple[CIMXMLSerializer, list], qualifier_return: bool) -> None:
    # If the object is a uuid it needs to be written with the correct qualifier. This test checks that the predicate is called correctly.
    ser, output = serializer
    mock_uuid.return_value = qualifier_return
    g = ser.store

    g.bind("ex", "http://example.com/")

    s = URIRef("s123")
    t = URIRef("http://example.com/Class")
    p = URIRef("http://example.com/p")

    g.add((s, RDF.type, t))
    g.add((s, p, Literal("value")))
    ser.predicate = Mock()

    ser.subject(s)

    mock_uuid.assert_called_once_with(ser.qualifier_resolver, Literal("value"))
    ser.predicate.assert_called_once_with(p, Literal("value"), 2, use_qualifier=qualifier_return)


@pytest.mark.parametrize("find_return", ["ID", "about", None])
@patch("cim_plugin.cimxml_serializer.is_uuid_qualified", return_value=False)
@patch("cim_plugin.cimxml_serializer.find_rdf_id_or_about")
def test_subject_rdfid(mock_find: MagicMock, mock_uuid: MagicMock, find_return: str, serializer: tuple[CIMXMLSerializer, list]) -> None:
    # The return of find_rdf_id_or_about desides how the rdf:type triple is written
    ser, output = serializer
    ser.predicate = Mock()
    mock_find.return_value = find_return
    g = ser.store
    g.bind("ex", "http://example.com/")
    ser._namespace_lookup = [("http://example.com/", "ex")]

    s = URIRef("s123")
    t = URIRef("http://example.com/Class")
    p = URIRef("http://example.com/p")

    g.add((s, RDF.type, t))
    g.add((s, p, Literal("value")))

    ser.subject(s)

    mock_find.assert_called_once_with(None, "http://example.com/Class")
    mock_uuid.assert_called_once_with(ser.qualifier_resolver, Literal("value"))
    ser.predicate.assert_called_once_with(p, Literal("value"), 2, use_qualifier=False)
    
    assert output[0] == f'  <ex:Class rdf:{find_return}="s123">\n'
    assert ser.qualifier_resolver
    assert isinstance(ser.qualifier_resolver, Mock) # To silence pylance

    if find_return == "ID":
        ser.qualifier_resolver.convert_to_special_qualifier.assert_called_once_with(URIRef("s123"))
        ser.qualifier_resolver.convert_to_default_qualifier.assert_not_called()
    if find_return == "about" or find_return is None:
        ser.qualifier_resolver.convert_to_default_qualifier.assert_called_once_with(URIRef("s123"))
        ser.qualifier_resolver.convert_to_special_qualifier.assert_not_called()
    
def test_subject_onlyrdftype(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    g = ser.store

    g.bind("ex", "http://example.com/")

    s = URIRef("s123")
    t = URIRef("http://example.com/Class")

    g.add((s, RDF.type, t))

    ser.subject(s)

    result = "".join(output)
    assert '  <ex:Class rdf:about="s123">\n  </ex:Class>\n' in result
    

def test_subject_alreadyserialized(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    g = ser.store
    g.bind("ex", "http://example.com/")
    s = URIRef("http://example.com/s")
    g.add((s, RDF.type, URIRef("http://example.com/Class")))

    ser.subject(s)
    
    result = "".join(output)
    assert str.count(result, "<ex:Class ") == 1  # Subject should be serialized only once
    

@patch("cim_plugin.cimxml_serializer.is_uuid_qualified", return_value=False)
def test_subject_malformedpredicateandobject(mock_uuid: MagicMock, serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    ser.predicate = Mock()
    g = ser.store

    g.bind("ex", "http://example.com/")

    s = URIRef("http://example.com/s")
    t = URIRef("http://example.com/Class")
    bn = BNode("b")

    g.add((s, RDF.type, t))
    g.add((s, Literal("not-a-uri"), bn))

    ser.subject(s)
    # Predicates and objects are passed on to .predicate whether they are valid or not
    ser.predicate.assert_called_once_with(Literal("not-a-uri"), bn, 2, use_qualifier=False)


@patch("cim_plugin.cimxml_serializer.is_uuid_qualified", return_value=False)
def test_subject_predicatesorting(mock_uuid: MagicMock, serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    ser.predicate = Mock()
    g = ser.store

    g.bind("ex", "http://example.com/")
    g.bind("foo", "http://bar.com/")
    ser._namespace_lookup = [("http://example.com/", "ex"), ("http://bar.com/", "foo")]

    s = URIRef("s123")
    t = URIRef("http://example.com/Class")
    p = URIRef("http://example.com/p")

    g.add((s, RDF.type, t))
    g.add((s, p, Literal("value")))
    g.add((s, URIRef("http://example.com/ad"), Literal("o")))
    g.add((s, URIRef("http://bar.com/ad"), Literal("o2")))
    g.add((s, URIRef("http://example.com/a_d"), Literal("o3")))

    ser.subject(s)

    predicate_calls = [
        call(URIRef("http://example.com/a_d"), Literal("o3"), 2, use_qualifier=False),
        call(URIRef("http://example.com/ad"), Literal("o"), 2, use_qualifier=False),
        call(URIRef("http://example.com/p"), Literal("value"), 2, use_qualifier=False),
        call(URIRef("http://bar.com/ad"), Literal("o2"), 2, use_qualifier=False)
    ]
    ser.predicate.assert_has_calls(predicate_calls, any_order=False) # Triples sent to .predicate in correct order
    

bn = BNode("b")    # Creating a shared bnode for test below
@pytest.mark.parametrize(
    "pred, pred_output",
    [
        pytest.param(URIRef("http://unknown.com/Class"), "<http://unknown.com/Class>", id="Unknown namespace"),
        pytest.param(Literal("Not-a-uri"), "<Not-a-uri>", id="Literal type rdf:type"),
        pytest.param(URIRef("http://example.com/1Class"), "ex:1Class", id="Name starting with number"),
        pytest.param(bn, "<b>", id="BNode type rdf:type")
    ]
)
@patch("cim_plugin.cimxml_serializer.is_uuid_qualified", return_value=False)
def test_subject_predicateedgecases(mock_uuid: MagicMock, pred: Any, pred_output: str, serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    ser.predicate = Mock()
    g = ser.store

    g.bind("ex", "http://example.com/")
    ser._namespace_lookup = [("http://example.com/", "ex")]

    s = URIRef("s123")
    t = pred   # No prefix registered
    p = URIRef("http://example.com/p")

    g.add((s, RDF.type, t))
    g.add((s, p, Literal("value")))

    ser.subject(s)

    # .normalizeUri adds <> around the type.
    assert output[0] == f'  <{pred_output} rdf:about="s123">\n'
    ser.predicate.assert_called_once_with(p, Literal("value"), 2, use_qualifier=False)
    

@patch("cim_plugin.cimxml_serializer.is_uuid_qualified", return_value=False)
def test_subject_circulartriples(mock_uuid: MagicMock, serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    g = ser.store

    g.bind("ex", "http://example.com/")
    ser._namespace_lookup = [("http://example.com/", "ex")]

    s = URIRef("s123")
    t = URIRef("http://example.com/Class")
    p = URIRef("http://example.com/p")

    g.add((s, RDF.type, t))
    g.add((s, p, s))

    ser.subject(s)

    result = "".join(output)
    assert result == '  <ex:Class rdf:about="s123">\n    <ex:p rdf:resource="s123"/>\n  </ex:Class>\n'
    

b = BNode("bad")    # Creating a shared bnode for test below
@pytest.mark.parametrize(
        "subject, reachable",
        [
            pytest.param(b, set(), id="BNode not reachable"),
            pytest.param(b, {b}, id="BNode reachable"),
        ]
)
@patch("cim_plugin.cimxml_serializer.is_uuid_qualified", return_value=False)
@patch("cim_plugin.cimxml_serializer.find_rdf_id_or_about", return_value="about")
def test_subject_bnodesubjects(mock_find: MagicMock, mock_uuid: MagicMock, subject: Literal|BNode, reachable: set, serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    ser.predicate = Mock()
    header_mock = Mock()
    header_mock.reachable_nodes = reachable
    g = cast(CIMGraph, ser.store)
    g.metadata_header = header_mock
    g.bind("ex", "http://example.com/")
    g.add((subject, RDF.type, URIRef("http://example.com/Class")))
    g.add((subject, URIRef("http://example.com/p"), Literal("value")))

    ser.subject(subject)

    if len(reachable) > 0:   # If the bnode is in the header reachable nodes it is handled by the header writer
        mock_find.assert_not_called()
        ser.predicate.assert_not_called()
    else: # If the bnode is not in the header reachable nodes it is treated as a regular subject
        mock_find.assert_called_once()
        assert output[0] == '  <ex:Class rdf:about="bad">\n'
        ser.predicate.assert_called_once_with(URIRef("http://example.com/p"), Literal("value"), 2, use_qualifier=False)


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
    ser._namespace_lookup = [("http://example.com/", "ex")]
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
    ser._namespace_lookup = [("http://example.com/", "ex")]
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
    ser._namespace_lookup = [("http://example.com/", "ex")]
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
    ser._namespace_lookup = [("http://example.com/", "ex")]
    ser.qualifier_resolver = Mock()

    ser.write = writer

    ser.predicate(pred, obj, depth=1, use_qualifier=False)
    result = "".join(output)

    assert f'<ex:p rdf:resource="http://example.com/o"/>' in result
    ser.qualifier_resolver.convert_to_default_qualifier.assert_not_called()


def test_predicate_prefixnotregistered(capture_writer: tuple[list, Callable]) -> None:
    output, writer = capture_writer
    g = Graph()
    pred = URIRef("http://unknown/p")
    obj = Literal("x")

    ser = CIMXMLSerializer(g)
    ser.qualifier_resolver = Mock()
    ser.write = writer

    ser.predicate(pred, obj)
    result = "".join(output)

    assert '  <http://unknown/p>x</http://unknown/p>\n' in result
    # Earlier versions used .qname_strict, which would generate a new prefix
    # assert '  <ns1:p>x</ns1:p>\n' in result


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
    ser._namespace_lookup = [("http://example.com/", "ex")]
    ser.qualifier_resolver = Mock()
    ser.write = writer

    ser.predicate(pred, obj, depth=depth)
    result = "".join(output)

    assert result.startswith(spaces)

# This entire tests was changed drastically when .predicate not longer relied on .qname_strict.
# The New changes marker show which would behave differently with .qname_strict.
@pytest.mark.parametrize("predicate,expected", [
    pytest.param(URIRef("http://example.com/p"), "ex:p", id="URIRef"),
    pytest.param(URIRef("http://noprefix.com/p"), "<http://noprefix.com/p", id="Namespace with no prefix"), # New changes
    pytest.param(URIRef("http://noneprefix.com/p"), "default1:/p", id="Namespace with prefix None"), # New changes
    pytest.param(URIRef("p"), "ex2:p", id="Prefix with no namespace"),   # New changes
    pytest.param(URIRef("http://example.com/pære"), "ex:pære", id="Unicode letters"),
    pytest.param(URIRef("http://example.com/per%cent%"), "ex:per%cent%", id="Percent encoded"),
    pytest.param(URIRef("http://example.com/#foo"), "ex:#foo", id="Fragment identifier"), # New changes
    pytest.param(URIRef("http://example.com/?x=1"), "ex:?x=1", id="Query parameters"), # New changes
    pytest.param(Literal("p"), "p", id="Literal"), # New changes
    pytest.param(BNode("p"), "p", id="BNode") # New changes
])
def test_predicate_predicatetypes(predicate: Node, expected: str, capture_writer: tuple[list, Callable]) -> None:
    output, writer = capture_writer
    g = Graph()
    g.bind("ex", "http://example.com/")
    g.bind("", "http://noprefix.com/p")
    g.bind(None, "http://noneprefix.com")
    g.bind("ex2", "")
    pred = predicate
    obj = Literal("x")
    g.add((URIRef("http://example.com/s"), pred, obj))

    ser = CIMXMLSerializer(g)
    ser.qualifier_resolver = Mock()
    used = ser._collect_used_namespaces()
    ser._namespace_lookup = sorted([(str(ns), prefix) for prefix, ns in used], key=lambda item: len(item[0]), reverse=True) #[("http://example.com/", "ex"), ("http://noprefix.com/p", "")] # New changes
    ser.write = writer

    ser.predicate(pred, obj)
    result = "".join(output)
    assert expected in result


def test_predicate_noobject(capture_writer: tuple[list, Callable]) -> None:
    # Documents what happens if object does not exist or is not a URIRef or Literal.
    output, writer = capture_writer
    g = Graph()
    pred = URIRef("http://unknown/p")
    obj = None

    ser = CIMXMLSerializer(g)
    ser.qualifier_resolver = Mock()
    ser._namespace_lookup = []
    ser.write = writer

    # Pylance silenced to test invalid input
    ser.predicate(pred, obj)    # type: ignore
    result = "".join(output)
    assert output[0] == '  <http://unknown/p>None</http://unknown/p>\n'
    # assert "Invalid object detected: None.\n" in caplog.text # If logging is uncommented

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

    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty()
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

    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty()
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
    g.metadata_header.add_triple(DCTERMS.conformsTo, URIRef("http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0"))
    
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


# Unit tests ._write_untyped_subject
def test_write_untyped_subject_basic(capture_writer: tuple[list, Callable]) -> None:
    output, writer = capture_writer
    g = Graph()
    g.bind("ex", "http://example.com/")
    sub = URIRef("s1")
    pred = URIRef("http://example.com/p")
    obj1 = Literal(True)
    obj2 = URIRef("http://example.com/o")
    obj3 = URIRef("urn:uuid:abcd")
    g.add((sub, pred, obj1))
    g.add((sub, pred, obj2))
    g.add((sub, pred, obj3))
    ser = CIMXMLSerializer(g)
    ser._namespace_lookup = [("http://example.com/", "ex"), ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf")]
    ser.write = writer
    ser.qualifier_resolver = CIMQualifierResolver(UnderscoreQualifier())

    ser._write_untyped_subject(sub, depth=1)

    assert output[0] == '  <rdf:Description rdf:about="s1">\n'
    assert output[1] == '    <ex:p>true</ex:p>\n'
    assert output[2] == '    <ex:p rdf:resource="http://example.com/o"/>\n'
    assert output[3] == '    <ex:p rdf:resource="#_abcd"/>\n'
    assert output[4] == '  </rdf:Description>\n'
    
    # Indentations:
    assert len(output[0]) - len(output[0].lstrip()) == 2
    assert len(output[4]) - len(output[4].lstrip()) == 2
    for line in output[1:3]:
        assert len(line) - len(line.lstrip()) == 4


def test_write_untyped_subject_rdftypepresent(capture_writer: tuple[list, Callable]) -> None:
    output, writer = capture_writer
    g = Graph()
    g.bind("ex", "http://example.com/")
    sub = URIRef("s1")
    pred = URIRef("http://example.com/p")
    obj1 = Literal(True)
    obj2 = URIRef("http://example.com/o")
    g.add((sub, pred, obj1))
    g.add((sub, RDF.type, obj2))
    ser = CIMXMLSerializer(g)
    ser._namespace_lookup = [("http://example.com/", "ex"), ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf")]
    ser.write = writer
    ser.qualifier_resolver = CIMQualifierResolver(UnderscoreQualifier())

    ser._write_untyped_subject(sub, depth=1)

    assert output == [] # Nothing is written when one of the subjects carries an rdf:type

def test_write_untyped_subject_noqualifier(capture_writer: tuple[list, Callable]) -> None:
    output, writer = capture_writer
    g = Graph()
    sub = URIRef("s1")
    ser = CIMXMLSerializer(g)
    ser._namespace_lookup = [("http://example.com/", "ex")]
    ser.write = writer

    with pytest.raises(AssertionError):
        ser._write_untyped_subject(sub, depth=1)

    assert ser.qualifier_resolver is None
    assert output == [] 


# Unit tests ._resolve_qname
@pytest.mark.parametrize("lookup", [[], None])
def test_resolve_qname_nonamespaces(lookup: list|None) -> None:
    ser = CIMXMLSerializer(Graph())
    ser._namespace_lookup = lookup
    ser.store.bind("ex", "http://example.com/") 

    result = ser._resolve_qname("http://example.com/p")
    # The namespace manager is ignored, so the result is not "ex:p" but the full uri.
    assert result == "http://example.com/p"


@pytest.mark.parametrize(
    "uri, lookup, expected",
    [
        pytest.param("http://example.com/p", [("http://example.com/", "ex")], "ex:p", id="Simple qname"),
        pytest.param("http://example.com/p", [("http://example.com/", "ex"), ("http://other.com/p", "ot")], "ex:p", id="Multiple namespaces in lookup"),
        pytest.param("http://example.com/extended/p", [("http://example.com/", "ex")], "ex:extended/p", id="Partial namespace match"),
        pytest.param("http://example.com/p", [("http://other.com/", "ot")], "http://example.com/p", id="No matching namespace"),
        pytest.param(URIRef("http://example.com/p"), [("http://example.com/", "ex")], "ex:p", id="URIRef input with match"),
        pytest.param("http://example.com/p", [("http://example.com/", "")], "http://example.com/p", id="Empty prefix"),
        pytest.param("http://example.com/p", [("http://example.com/p", "ex")], "ex:", id="Namespace matches uri entirely"),
        
        # ._namespace_lookup is ordered in reverse length order when made by the .serialize method. This ensures the longest match. 
        # The below tests document that this behaviour is not enforced by ._resolve_qname.
        pytest.param("http://example.com/p", [("http://example.com/longer/", "exl"), ("http://example.com/", "ex")], "ex:p", id="Overlapping namespaces, shortest match"),
        pytest.param("http://example.com/longer/p", [("http://example.com/longer/", "exl"), ("http://example.com/", "ex")], "exl:p", id="Overlapping namespaces, longest match"),
        pytest.param("http://example.com/longer/p", [("http://example.com/", "ex"), ("http://example.com/longer/", "exl")], "ex:longer/p", id="Overlapping namespaces, the first match wins"),
        
        # The method is at the moment only used for predicates, which should never be BNodes or Literals. The below tests document the behavior with such inputs.
        pytest.param(Literal("http://example.com/p"), [("http://example.com/", "ex")], "ex:p", id="Literal input with match"),
        pytest.param(BNode("http://example.com/bnode"), [("http://example.com/", "ex")], "ex:bnode", id="BNode input with match")
    ]
)
def test_resolve_qname_various(uri: Any, lookup: list[tuple[str, str]], expected: str) -> None:
    g = Graph()
    ser = CIMXMLSerializer(g)
    ser._namespace_lookup = lookup  
    result = ser._resolve_qname(uri)
    assert result == expected


def test_resolve_qname_cache() -> None:
    ser = CIMXMLSerializer(Graph())
    ser._namespace_lookup = [("http://example.com/", "ex")]

    first = ser._resolve_qname("http://example.com/p")
    assert first == "ex:p"

    # Change lookup after caching
    ser._namespace_lookup = [("http://example.com/", "changed")]

    second = ser._resolve_qname("http://example.com/p")
    assert second == "ex:p"  # still cached

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