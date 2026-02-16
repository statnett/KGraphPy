from typing import Callable
import pytest
from unittest.mock import MagicMock, call, patch, Mock
import uuid
import logging

from rdflib import URIRef, Graph, Literal, Node, BNode
from rdflib.plugins.serializers.xmlwriter import ESCAPE_ENTITIES
from rdflib.namespace import XSD, RDF, DCAT
from xml.sax.saxutils import escape
from cim_plugin.cimxml_serializer import _subject_sort_key, CIMXMLSerializer
from cim_plugin.qualifiers import CIMQualifierStrategy, UnderscoreQualifier, URNQualifier, NamespaceQualifier, CIMQualifierResolver, uuid_namespace
from cim_plugin.header import CIMMetadataHeader
from cim_plugin.graph import CIMGraph
from cim_plugin.namespaces import MD
from tests.fixtures import capture_writer, serializer


logger = logging.getLogger("cimxml_logger")

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


def test_subject_valid(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
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


def test_subject_rdfid(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
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


def test_subject_malformedpredicate(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    g = ser.store

    g.bind("ex", "http://example.com/")

    s = URIRef("http://example.com/s")
    t = URIRef("http://example.com/Class")

    g.add((s, RDF.type, t))
    g.add((s, Literal("not-a-uri"), Literal("x")))  # malformed predicate

    ser.subject(s)

    result = "".join(output)

    assert "MALFORMED_" in result  # from predicate()

def test_subject_malformedobject(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    g = ser.store

    g.bind("ex", "http://example.com/")

    s = URIRef("s123")
    t = URIRef("http://example.com/Class")
    p = URIRef("http://example.com/p")

    g.add((s, RDF.type, t))
    g.add((s, p, BNode("value")))

    ser.subject(s)

    result = "".join(output)
    print(result)
    assert '<ex:Class rdf:about="s123"' in result
    assert "<ex:p>MALFORMED_value</ex:p>" in result
    assert "</ex:Class>" in result

def test_subject_predicatesorting(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
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


def test_subject_rdftypewithoutprefix(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
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


def test_subject_rdftypemalformed(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
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


def test_subject_circulartriples(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
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
    

def test_subject_predicatecalls(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
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

    assert ser.predicate.call_count == 2


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
        pytest.param("weird", "strange", UnderscoreQualifier(), "#_weird", "#_strange", id="Fallback, underscore output"),
        pytest.param("_s", "_o", URNQualifier(), "urn:uuid:s", "urn:uuid:o", id="Underscore input, urn output"),
        pytest.param("urn:uuid:abcd", "urn:uuid:efgh", URNQualifier(), "urn:uuid:abcd", "urn:uuid:efgh", id="Urn input, urn output"),
        pytest.param(f"{uuid_namespace}:x", f"{uuid_namespace}:y", URNQualifier(), "urn:uuid:x", "urn:uuid:y", id="Namespace input, urn output"),
        pytest.param("weird", "strange", URNQualifier(), "urn:uuid:weird", "urn:uuid:strange", id="Fallback, urn output"),
        pytest.param("_s", "_o", NamespaceQualifier(), f"{uuid_namespace}:s", f"{uuid_namespace}:o", id="Underscore input, namespace output"),
        pytest.param("urn:uuid:abcd", "urn:uuid:efgh", NamespaceQualifier(), f"{uuid_namespace}:abcd", f"{uuid_namespace}:efgh", id="Urn input, namespace output"),
        pytest.param(f"{uuid_namespace}:x", f"{uuid_namespace}:y", NamespaceQualifier(), f"{uuid_namespace}:x", f"{uuid_namespace}:y", id="Namespace input, namespace output"),
        pytest.param("weird", "strange", NamespaceQualifier(), f"{uuid_namespace}:weird", f"{uuid_namespace}:strange", id="Fallback, namespace output"),
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
        pytest.param("weird", "strange", UnderscoreQualifier(), "_weird", "#_strange", id="Fallback, underscore output"),
        pytest.param("_s", "_o", URNQualifier(), "urn:uuid:s", "urn:uuid:o", id="Underscore input, urn output"),
        pytest.param("urn:uuid:abcd", "urn:uuid:efgh", URNQualifier(), "urn:uuid:abcd", "urn:uuid:efgh", id="Urn input, urn output"),
        pytest.param(f"{uuid_namespace}:x", f"{uuid_namespace}:y", URNQualifier(), "urn:uuid:x", "urn:uuid:y", id="Namespace input, urn output"),
        pytest.param("weird", "strange", URNQualifier(), "urn:uuid:weird", "urn:uuid:strange", id="Fallback, urn output"),
        pytest.param("_s", "_o", NamespaceQualifier(), f"{uuid_namespace}:s", f"{uuid_namespace}:o", id="Underscore input, namespace output"),
        pytest.param("urn:uuid:abcd", "urn:uuid:efgh", NamespaceQualifier(), f"{uuid_namespace}:abcd", f"{uuid_namespace}:efgh", id="Urn input, namespace output"),
        pytest.param(f"{uuid_namespace}:x", f"{uuid_namespace}:y", NamespaceQualifier(), f"{uuid_namespace}:x", f"{uuid_namespace}:y", id="Namespace input, namespace output"),
        pytest.param("weird", "strange", NamespaceQualifier(), f"{uuid_namespace}:weird", f"{uuid_namespace}:strange", id="Fallback, namespace output"),
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