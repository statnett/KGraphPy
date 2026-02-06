from typing import Callable
from unittest import result
import pytest
from unittest.mock import MagicMock, patch, Mock
import uuid
import logging

from rdflib import URIRef, Graph, Literal, Node, BNode
from rdflib.plugins.serializers.xmlwriter import ESCAPE_ENTITIES
from rdflib.namespace import XSD, RDF
from xml.sax.saxutils import escape
from cim_plugin.cimxml_serializer import _subject_sort_key, CIMXMLSerializer
from tests.fixtures import capture_writer, serializer


logger = logging.getLogger("cimxml_logger")


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


def test_subject_with_malformedpredicate(serializer: tuple[CIMXMLSerializer, list]) -> None:
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

    assert '<ex:Class rdf:about="s123"' in result
    assert "<ex:p>INVALID OBJECT</ex:p>" in result
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


def test_subject_rdftypewoprefix(serializer: tuple[CIMXMLSerializer, list]) -> None:
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


def test_subject_rdftypnoturi(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    g = ser.store

    g.bind("ex", "http://example.com/")

    s = URIRef("s123")
    t = Literal("Class")   # Not a uri
    p = URIRef("http://example.com/p")

    g.add((s, RDF.type, t))
    g.add((s, p, Literal("value")))

    ser.subject(s)

    result = "".join(output)
    # .qname_strict adds <> around the type.
    assert result == '  <<Class> rdf:about="s123">\n    <ex:p>value</ex:p>\n  </<Class>>\n'


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

def test_subject_malformedsubjectcalls(serializer: tuple[CIMXMLSerializer, list]) -> None:
    ser, output = serializer
    ser._write_malformed_subject = Mock()

    subject = Literal("bad")

    ser.subject(subject)

    ser._write_malformed_subject.assert_called_once()

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
    ser.qualifier_resolver.convert_resource.return_value = return_value

    ser.write = writer

    ser.predicate(pred, obj, depth=1)
    result = "".join(output)

    esc_return = escape(return_value, ESCAPE_ENTITIES)
    assert f'<ex:p rdf:resource="{esc_return}"/>' in result
    ser.qualifier_resolver.convert_resource.assert_called_once_with(obj)


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

    assert "INVALID OBJECT" in result
    assert "Invalid object detected." in caplog.text

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