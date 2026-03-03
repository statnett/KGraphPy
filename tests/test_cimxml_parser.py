from cim_plugin.cimxml_parser import (
    _get_current_namespace_from_graph,
    update_namespace_in_graph,
    ensure_correct_namespace_graph,
    find_slots_with_range,
    _clean_uri,
    fix_qualifier_for_all_uuids,
    cast_float,
    cast_bool
)
import pytest
from unittest.mock import patch, MagicMock, mock_open, call
from rdflib import URIRef, Graph, Namespace, Literal, BNode
import logging
from pytest import LogCaptureFixture
from tests.fixtures import make_graph_with_prefixes, sample_yaml
import textwrap
from typing import Any


logger = logging.getLogger("cimxml_logger")


# Unit tests _get_current_namespace_from_graph

@pytest.mark.parametrize(
    "prefix, expected",
    [
        pytest.param("ex", "http://example.org/", id="Simple test 1"),
        pytest.param("foo", "http://foo.bar/", id="Simple test 2"),
        pytest.param("missing", None, id="Prefix not found"),
        pytest.param("", "http://empty_string.prefix", id="Prefix empty string"),
        pytest.param("EX", None, id="Wrong case"),
        pytest.param("ex ", None, id="Extra whitespace"),
        pytest.param(None, None, id="Prefix None")
    ],
)
def test_get_current_namespace_from_graph_basic(prefix: str, expected: str|None) -> None:
    g = Graph()
    g.namespace_manager.bind("ex", Namespace("http://example.org/"))
    g.namespace_manager.bind("foo", Namespace("http://foo.bar/"))
    g.namespace_manager.bind("", Namespace("http://empty_string.prefix"))
    result = _get_current_namespace_from_graph(g, prefix)
    assert result == expected


def test_get_current_namespace_from_graph_empty_graph() -> None:
    g = Graph()
    assert _get_current_namespace_from_graph(g, "ex") is None


def test_get_current_namespace_from_graph_prefixessameuri() -> None:
    # This test shows that rdflib does not allow the same namespace with multiple prefixes.
    # Technically not a test of this function, but included for documentation.
    g = Graph()
    g.namespace_manager.bind("ex", Namespace("http://example.org/"))

    assert _get_current_namespace_from_graph(g, "ex") == "http://example.org/"
    
    # If override is False ex is kept and ex2 is ignored 
    # If override is True ex2 is kept and ex is removed
    g.namespace_manager.bind("ex2", Namespace("http://example.org/"), override=True)

    assert _get_current_namespace_from_graph(g, "ex") is None # Because ex has been replaced by ex2
    assert _get_current_namespace_from_graph(g, "ex2") == "http://example.org/"

    prefixes = {pfx for pfx, _ in g.namespace_manager.namespaces()} 
    assert "ex" not in prefixes 
    assert "ex2" in prefixes


def test_get_current_namespace_from_graph_namespaceuriisuriref() -> None:
    g = Graph()
    g.namespace_manager.bind("ex", URIRef("http://example.org/uri"))

    result = _get_current_namespace_from_graph(g, "ex")
    assert result == "http://example.org/uri"


def test_get_current_namespace_from_graph_rebindingprefixoverwritesprevious() -> None:
    g = Graph()
    g.namespace_manager.bind("ex", Namespace("http://old.example.org/"))
    g.namespace_manager.bind("ex", Namespace("http://new.example.org/"), replace=True)

    result = _get_current_namespace_from_graph(g, "ex")
    assert result == "http://new.example.org/"


def test_get_current_namespace_from_graph_namespaceuriinvalidstring() -> None:
    g = Graph()
    g.namespace_manager.bind("weird", Namespace("not a uri at all"))

    result = _get_current_namespace_from_graph(g, "weird")
    assert result == "not a uri at all"


@pytest.mark.parametrize(
    "prefix, uri",
    [
        pytest.param("ø", "http://example.org/unicode1", id="ø in prefix"),
        pytest.param("π", "http://example.org/unicode2", id="pi in prefix"),
        pytest.param("префикс", "http://example.org/unicode3", id="Greek prefix"),
    ],
)
def test_get_current_namespace_from_graph_unicodeprefix(prefix: str, uri: str) -> None:
    g = Graph()
    g.namespace_manager.bind(prefix, Namespace(uri))

    result = _get_current_namespace_from_graph(g, prefix)
    assert result == uri


@pytest.mark.parametrize(
    "triple, old_ns, new_ns, expected",
    [
        pytest.param(
            (URIRef("http://old.com/a"), URIRef("x"), URIRef("y")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("http://new.com/a"), URIRef("x"), URIRef("y")),
            id="Subject update"
        ),
        pytest.param(
            (URIRef("s"), URIRef("http://old.com/p"), URIRef("o")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("s"), URIRef("http://new.com/p"), URIRef("o")),
            id="Predicate update"
        ),
        pytest.param(
            (URIRef("s"), URIRef("p"), URIRef("http://old.com/o")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("s"), URIRef("p"), URIRef("http://new.com/o")),
            id="Object update"
        ),
        pytest.param(
            (URIRef("s"), URIRef("p"), URIRef("o")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("s"), URIRef("p"), URIRef("o")),
            id="Simple no update"
        ),
        pytest.param(
            (URIRef("http://unrecognized.com/s"), URIRef("http://unrecognized.com/p"), URIRef("http://unrecognized.com/o")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("http://unrecognized.com/s"), URIRef("http://unrecognized.com/p"), URIRef("http://unrecognized.com/o")),
            id="No update, all uris."
        ),
        pytest.param(
            (BNode("s"), URIRef("http://old.com/p"), URIRef("o")),
            "http://old.com/",
            "http://new.com/",
            (BNode("s"), URIRef("http://new.com/p"), URIRef("o")),
            id="Subject is BNode"
        ),
        pytest.param(
            (URIRef("s"), URIRef("http://old.com/p"), BNode("o")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("s"), URIRef("http://new.com/p"), BNode("o")),
            id="Object is BNode"
        ),
        pytest.param(
            (URIRef("s"), URIRef("http://old.com/p"), Literal("o")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("s"), URIRef("http://new.com/p"), Literal("o")),
            id="Object is Literal"
        ),
        pytest.param(
            (URIRef("http://old.comX/a"), URIRef("x"), URIRef("y")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("http://old.comX/a"), URIRef("x"), URIRef("y")),
            id="Partial match -> No update"
        ),
        # This may cause problems!!
        pytest.param(
            (URIRef("http://old.com/a/http://old.com/b"), URIRef("x"), URIRef("y")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("http://new.com/a/http://new.com/b"), URIRef("x"), URIRef("y")),
            id="Multiple occurence of same uri"
        ),
        pytest.param(
            (URIRef("http://old.complex/foo/a"), URIRef("x"), URIRef("y")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("http://old.complex/foo/a"), URIRef("x"), URIRef("y")),
            id="Uri is part of another uri -> No update"
        ),
        pytest.param(
            (URIRef("http://old.com/a"), URIRef("x"), URIRef("y")),
            "http://old.com/",
            "",
            (URIRef("a"), URIRef("x"), URIRef("y")),
            id="New namespace empty"
        ),
    ]
)
def test_update_namespace_in_graph_singletriple(triple: tuple, old_ns: str, new_ns: str, expected: tuple) -> None:
    g = Graph()
    g.add(triple)

    update_namespace_in_graph(g, old_ns, new_ns)
    for s, p, o in g:
        print(s, p, o)
    assert len(g) == 1
    assert expected in g


def test_update_namespace_in_graph_mixedtriples() -> None:
    old_ns = "http://old.com/"
    new_ns = "http://new.com/"

    g = Graph()
    t1 = (URIRef("http://old.com/a"), URIRef("p"), URIRef("o"))
    t2 = (URIRef("s"), URIRef("http://old.com/p"), URIRef("o"))
    t3 = (URIRef("s"), URIRef("p"), URIRef("http://old.com/o"))
    t4 = (URIRef("s"), URIRef("p"), URIRef("o"))  # unchanged

    for t in (t1, t2, t3, t4):
        g.add(t)

    update_namespace_in_graph(g, old_ns, new_ns)

    expected = {
        (URIRef("http://new.com/a"), URIRef("p"), URIRef("o")),
        (URIRef("s"), URIRef("http://new.com/p"), URIRef("o")),
        (URIRef("s"), URIRef("p"), URIRef("http://new.com/o")),
        t4,
    }

    assert len(g) == 4
    for triple in expected:
        assert triple in g


def test_update_namespace_in_graph_multiplereplacements() -> None:
    g = Graph()
    old_ns = "http://old.com/"
    new_ns = "http://new.com/"

    triple = (URIRef("http://old.com/s"), URIRef("http://old.com/p"), URIRef("http://old.com/o"),)
    g.add(triple)

    update_namespace_in_graph(g, old_ns, new_ns)

    expected = (URIRef("http://new.com/s"), URIRef("http://new.com/p"), URIRef("http://new.com/o"),)

    assert len(g) == 1
    assert expected in g
    assert triple not in g


def test_update_namespace_in_graph_duplicatesafterreplacement() -> None:
    # This test shows what happends if replacements leads to two uris being made identical
    g = Graph()
    old_ns = "http://old.com/"
    new_ns = "http://new.com/"

    t1 = (URIRef("http://old.com/a"), URIRef("p"), URIRef("o"))
    t2 = (URIRef("http://new.com/a"), URIRef("p"), URIRef("o"))
    g.add(t1)
    g.add(t2)

    assert len(g) == 2  # Number of triples before replacement

    update_namespace_in_graph(g, old_ns, new_ns)

    expected = (URIRef("http://new.com/a"), URIRef("p"), URIRef("o"))

    assert len(g) == 1  # Number of triples after replacement
    assert expected in g


def test_update_namespace_in_graph_emptygraph() -> None:
    g = Graph()

    update_namespace_in_graph(g, "http://old.com/", "http://new.com/")

    assert len(g) == 0


def test_update_namespace_in_graph() -> None:
    g = Graph()
    g.add((URIRef("a"), URIRef("x"), URIRef("y")))
    assert len(g) == 1

    with pytest.raises(ValueError, match="old_namespace cannot be an empty string"):
        update_namespace_in_graph(g, "", "http://new.com/")


# Unit tests ensure_correct_namespace_graph
@pytest.mark.parametrize(
    "prefix, current, new_ns, update",
    [
        pytest.param("ex", "www.example.com/", "www.newexample.com/", True, id="Current namespace not correct -> update"),
        pytest.param("same", "www.same.com/", "www.same.com/", False, id="Current namespace correct -> no update"),
        pytest.param("ws", " www.whitespace.com/ ", "www.whitespace.com/", True, id="Current namespace has whitespace -> update"),
        pytest.param("ex", "www.example.com/", " www.newexample.com/ ", True, id="New namespace has whitespace -> update"),
    ]
)
@patch("cim_plugin.cimxml_parser.update_namespace_in_graph")
@patch("cim_plugin.cimxml_parser._get_current_namespace_from_graph")
def test_ensure_correct_namespace_graph_namespacehandling(mock_get: MagicMock, mock_update: MagicMock, make_graph_with_prefixes: Graph, prefix: str, current: str, new_ns: str, update: bool, caplog: LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    mock_get.return_value = current
    g = make_graph_with_prefixes

    ensure_correct_namespace_graph(g, prefix, new_ns)

    mock_get.assert_called_once_with(g, prefix)
    bound_ns = g.namespace_manager.store.namespace(prefix)
    if update:
        mock_update.assert_called_once_with(g, current, new_ns.strip())
        assert bound_ns == URIRef(new_ns.strip())
        assert f"Wrong namespace detected for {prefix} in graph. Correcting to {new_ns.strip()}." in caplog.text
    else:
        mock_update.assert_not_called()
        assert bound_ns == URIRef(current)
        assert f"Graph has correct namespace for {prefix}." in caplog.text

@patch("cim_plugin.cimxml_parser.update_namespace_in_graph")
@patch("cim_plugin.cimxml_parser._get_current_namespace_from_graph")
def test_ensure_correct_namespace_graph_currentisnone(mock_get: MagicMock, mock_update: MagicMock) -> None:
    mock_get.return_value = None
    g = Graph()

    with patch.object(g, "bind") as mock_bind:
    # with pytest.raises(ValueError, match="No namespace is called by this prefix: 'ex'."):
        ensure_correct_namespace_graph(g, "ex", "www.example.com")
        mock_bind.assert_not_called()

    mock_get.assert_called_once()
    mock_update.assert_not_called()


@patch("cim_plugin.cimxml_parser.update_namespace_in_graph")
@patch("cim_plugin.cimxml_parser._get_current_namespace_from_graph")
def test_ensure_correct_namespace_graph_newisonlywhitespace(mock_get: MagicMock, mock_update: MagicMock, make_graph_with_prefixes: Graph) -> None:
    mock_get.return_value = "www.example.com/"
    g = make_graph_with_prefixes

    with pytest.raises(ValueError, match="Namespace cannot be an empty string."):
        ensure_correct_namespace_graph(g, "ex", " ")

    mock_get.assert_not_called()
    mock_update.assert_not_called()
    assert g.namespace_manager.store.namespace("ex") == URIRef("www.example.com/")


@patch("cim_plugin.cimxml_parser.update_namespace_in_graph")
@patch("cim_plugin.cimxml_parser._get_current_namespace_from_graph")
def test_ensure_correct_namespace_graph_nocorruptionofnewns(mock_get: MagicMock, mock_update: MagicMock, make_graph_with_prefixes: Graph) -> None:
    mock_get.return_value = "www.example.com/"
    g = make_graph_with_prefixes

    ensure_correct_namespace_graph(g, "ex", " www.new.org/ ")

    mock_update.assert_called_once_with(g, "www.example.com/", "www.new.org/")
    assert g.namespace_manager.store.namespace("ex") == URIRef("www.new.org/")


@patch("cim_plugin.cimxml_parser.update_namespace_in_graph")
@patch("cim_plugin.cimxml_parser._get_current_namespace_from_graph")
def test_ensure_correct_namespace_graph_bindcalledcorrectly(mock_get: MagicMock, mock_update: MagicMock, make_graph_with_prefixes: Graph) -> None:
    mock_get.return_value = "www.example.com/"
    g = make_graph_with_prefixes
    new_ns = " www.new.org/ "

    with patch.object(g, "bind") as mock_bind:
        ensure_correct_namespace_graph(g, "ex", new_ns)
        mock_bind.assert_called_once_with("ex", Namespace(new_ns.strip()), replace=True)

    mock_update.assert_called_once_with(g, "www.example.com/", "www.new.org/")


@patch("cim_plugin.cimxml_parser.update_namespace_in_graph")
@patch("cim_plugin.cimxml_parser._get_current_namespace_from_graph")
def test_ensure_correct_namespace_graph_nswrongtype(mock_get: MagicMock, mock_update: MagicMock, make_graph_with_prefixes: Graph) -> None:
    # This test documents what happends if _get_current_namespace_from_graph brings back a namespace with wrong datatype.
    # This should never happen, though, as rdflib does not allow int as namespace.
    mock_get.return_value = 123
    g = make_graph_with_prefixes
    # Pylance ignored to check wrong datatypes
    g.bind("wrong", Namespace(123)) # type: ignore
    new_ns = "www.new.org/"
    assert g.namespace_manager.store.namespace("wrong") == URIRef("123")

    ensure_correct_namespace_graph(g, "wrong", new_ns)
    mock_update.assert_called_once_with(g, 123, "www.new.org/")
    assert g.namespace_manager.store.namespace("wrong") == URIRef("www.new.org/")


@patch("cim_plugin.cimxml_parser.update_namespace_in_graph")
@patch("cim_plugin.cimxml_parser._get_current_namespace_from_graph")
def test_ensure_correct_namespace_graph_idempotence(mock_get: MagicMock, mock_update: MagicMock, make_graph_with_prefixes: Graph) -> None:
    mock_get.side_effect = ["www.example.com/", "www.new.org/"]
    g = make_graph_with_prefixes

    ensure_correct_namespace_graph(g, "ex", "www.new.org/")
    assert mock_get.call_count == 1
    mock_update.assert_called_once_with(g, "www.example.com/", "www.new.org/")
    assert g.namespace_manager.store.namespace("ex") == URIRef("www.new.org/")

    # Second call
    ensure_correct_namespace_graph(g, "ex", "www.new.org/")
    assert mock_get.call_count == 2 # This one will now have been called twice
    assert mock_update.call_count == 1 # Should not be called the second time as the namespace has already been corrected
    assert g.namespace_manager.store.namespace("ex") == URIRef("www.new.org/")


# Unit tests find_slots_with_range

@pytest.mark.parametrize(
        "datatype, exp_result",
        [
            pytest.param("string", ["bar"], id="One slot"),
            pytest.param("integer", ["endDate", "tend"], id="Multiple slots"),
            pytest.param("Date", ["startDate"], id="Class datatype"),
            pytest.param("float", [], id="No matching slot"),
            pytest.param("Count", ["Updates"], id="Same slot in pultiple classes")
        ]
)
def test_find_slots_with_range_basic(datatype: str, exp_result: list, sample_yaml: str) -> None:
    m = mock_open(read_data=sample_yaml)
    with patch("builtins.open", m):
        result = find_slots_with_range("schema.yaml", datatype=datatype)
    assert result == set(exp_result)


@pytest.mark.parametrize(
        "yaml, raises",
        [
            pytest.param("classes: {}", False, id="No classes"),
            pytest.param(
                """classes: 
                    EmptyClass: {}
                """, 
                False, id="No attributes"),
            pytest.param(
                """classes: 
                    BadClass: 
                        attributes: 
                            weird: 123
                """, 
                True, id="Attributes not a dict"),
            pytest.param(
                """classes:
                    C:
                        attributes:
                            a: {}
                """,
                False, id="No range in attributes"
            )
        ]
)
def test_find_slots_with_range_attributeerrors(yaml: str, raises: bool) -> None:
    m = mock_open(read_data=textwrap.dedent(yaml))
    with patch("builtins.open", m):
        if raises:
            with pytest.raises(ValueError) as exc_info:
                find_slots_with_range("schema.yaml", "integer")
            assert "Attributes must be dictionaries." in str(exc_info.value)
        else:
            result = find_slots_with_range("schema.yaml", "integer")
            assert result == set()


# Unit tests _clean_uri
@pytest.mark.parametrize(
    "uri,expected",
    [
        pytest.param(URIRef("http://example.com/nohash"), URIRef("http://example.com/nohash"),
                     id="Not uuid → returned unchanged"),
        pytest.param(URIRef("http://example.com#_00000000-0000-4000-8000-000000000001"), URIRef("urn:uuid:00000000-0000-4000-8000-000000000001"),
                     id="Fragment with #_ → cleaned"),
        pytest.param(URIRef("http://example.com#00000000-0000-4000-8000-000000000001"), URIRef("urn:uuid:00000000-0000-4000-8000-000000000001"),
                     id="Fragment with # → cleaned"),
        pytest.param(URIRef("http://example.com_00000000-0000-4000-8000-000000000001"), URIRef("urn:uuid:00000000-0000-4000-8000-000000000001"),
                     id="Fragment with _ → cleaned"),
        pytest.param(URIRef("http://example.com#"), URIRef("http://example.com#"),
                     id="No fragment → unchanged"),
        pytest.param(URIRef("http://example.com#_"), URIRef("http://example.com#_"),
                     id="Just _ fragment → unchanged"),
    ]
)
def test_clean_uri_basic(uri: URIRef, expected: URIRef) -> None:
    uri_map = {}
    result = _clean_uri(uri, uri_map)
    assert result == expected

def test_clean_uri_uses_cache() -> None:
    uri = URIRef("http://example.com#_00000000-0000-4000-8000-000000000001")
    uri_map = {}

    first = _clean_uri(uri, uri_map)
    second = _clean_uri(uri, uri_map)

    assert first is second  # Same object → same uri_map
    assert list(uri_map.values()) == [first]    # uri_map should only have one entry

def test_clean_uri_cacheonlyforcleaned() -> None:
    uri = URIRef("http://example.com#not_a_uuid")
    uri_map = {}

    result = _clean_uri(uri, uri_map)

    assert uri_map == {} # No caching when nothing cleaned
    assert result == uri


def test_clean_uri_multipleurissamefragments() -> None:
    # This test shows that two identical fragments with different URIs will end up the same object in the graph 
    uri1 = URIRef("http://example.com#_00000000-0000-4000-8000-000000000001")
    uri2 = URIRef("http://other.com#_00000000-0000-4000-8000-000000000001")

    uri_map = {}

    cleaned1 = _clean_uri(uri1, uri_map)
    cleaned2 = _clean_uri(uri2, uri_map)

    assert cleaned1 is not cleaned2 # They are not the same object
    assert cleaned1 == cleaned2 # They look identical
    assert cleaned1 == URIRef("urn:uuid:00000000-0000-4000-8000-000000000001")
    assert len(uri_map) == 2
    assert set(uri_map.values()) == {cleaned1}


@patch("cim_plugin.cimxml_parser.extract_uuid")
def test_clean_uri_extractcalls(mock_extract: MagicMock) -> None:
    mock_extract.return_value = "00000000-0000-4000-8000-000000000001"
    uri1 = URIRef("http://example.com#_00000000-0000-4000-8000-000000000001")

    uri_map = {}

    cleaned1 = _clean_uri(uri1, uri_map)

    assert cleaned1 == URIRef("urn:uuid:00000000-0000-4000-8000-000000000001")
    assert set(uri_map.values()) == {cleaned1}
    mock_extract.assert_called_once_with("http://example.com#_00000000-0000-4000-8000-000000000001")


# Unit tests normalize_rdf_ids

@pytest.mark.parametrize(
        "s, o, calls",
        [
            pytest.param(URIRef("a"), Literal("b"), [call(URIRef("a"), {})], id="Only first call"),
            pytest.param(URIRef("a"), URIRef("b"), [call(URIRef('a'), {}), call(URIRef('b'), {})], id="Both called"),
            pytest.param(BNode("x"), Literal("b"),[], id="Subject is BNode → no calls"),
            pytest.param(BNode("x"), URIRef("b"),[call(URIRef("b"), {})], id="Subject is BNode → object cleaned"),
            pytest.param(URIRef("a"), BNode("x"), [call(URIRef("a"), {})], id="Object is BNode → only subject cleaned"),
        ]
)
@patch("cim_plugin.cimxml_parser._clean_uri")
def test_normalize_rdf_ids_callscleanuri(mock_clean: MagicMock, s: URIRef|BNode, o: Literal|URIRef|BNode, calls: list) -> None:
    mock_clean.side_effect = lambda uri, *_: URIRef("urn:uuid:test")

    g = Graph()
    g.add((s, URIRef("p"), o))

    fix_qualifier_for_all_uuids(g)

    assert mock_clean.mock_calls == calls


@patch("cim_plugin.cimxml_parser._clean_uri")
def test_normalize_rdf_ids_mutatesgraph(mock_clean: MagicMock) -> None:
    mock_clean.side_effect = [
        URIRef("urn:uuid:a"),  # new_s
        URIRef("urn:uuid:b"),  # new_o
    ]

    g = Graph()
    s = URIRef("http://ex.com#_a")
    p = URIRef("p")
    o = URIRef("http://ex.com#_b")
    g.add((s, p, o))

    fix_qualifier_for_all_uuids(g)

    triples = list(g)
    assert len(triples) == 1
    new_s, new_p, new_o = triples[0]
    assert new_s == URIRef("urn:uuid:a")
    assert new_o == URIRef("urn:uuid:b")
    assert new_p == p


def test_normalize_rdf_ids_withoutmocking() -> None:
    g = Graph()

    # Two triples sharing the same subject
    s = URIRef("http://example.com#_00000000-0000-4000-8000-000000000001")
    p = URIRef("p")
    o1 = URIRef("http://example.com#_00000000-0000-4000-8000-000000000002")
    o2 = URIRef("http://example.com#_00000000-0000-4000-8000-000000000003")

    g.add((s, p, o1))
    g.add((s, p, o2))

    fix_qualifier_for_all_uuids(g)

    assert all("urn:uuid:" in str(tr[0]) for tr in g)
    assert all("http://example.com" not in str(tr[0]) for tr in g)
    assert len(list(g)) == 2


@patch("cim_plugin.cimxml_parser._clean_uri")
def test_normalize_rdf_ids_reusesurimap(mock_clean: MagicMock) -> None:
    mock_clean.side_effect = lambda uri, *_: URIRef("urn:uuid:test")

    g = Graph()
    s = URIRef("http://ex.com#_a")
    o = URIRef("http://ex.com#_a")
    p = URIRef("p")
    g.add((s, p, o))
    g.add((s, p, o))  # Same triple again

    fix_qualifier_for_all_uuids(g)
    assert mock_clean.call_count == 2 # _clean_uri should only be called once each for s and o


@patch("cim_plugin.cimxml_parser._clean_uri")
def test_normalize_rdf_ids_emptygraph(mock_clean: MagicMock) -> None:
    g = Graph()

    fix_qualifier_for_all_uuids(g)

    mock_clean.assert_not_called()
    

# Unit tests cast_bool
@pytest.mark.parametrize(
    "input, output",
    [
        pytest.param("true", True, id="Input 'true'"),
        pytest.param("1", True, id="Input '1'"),
        pytest.param("false", False, id="Input 'false'"),
        pytest.param("0", False, id="Input '0'"),
        pytest.param("gibberish", None, id="Nonsense"),
        pytest.param("123", None, id="Numeric nonsense"),
        pytest.param(True, True, id="Input boolean True"),
        pytest.param(False, False, id="Input boolean False"),
        pytest.param("True", True, id="Upper case True"),
        pytest.param("FALSE", False, id="Upper case False"),
        pytest.param(1, True, id="Integer"),
        pytest.param(123, None, id="Wrong integer")
    ]
)
def test_cast_bool_various(input: str|bool, output: bool) -> None:
    if output is None:
        with pytest.raises(ValueError, match="Invalid boolean lexical form"):
            cast_bool(input)
    else:
        # Pylance silenced to test incorrect input type
        assert cast_bool(input) == output   # type: ignore



# Unit tests cast_float
@pytest.mark.parametrize(
    "input, output",
    [
        pytest.param("1", 1.0, id="Input '1'"),
        pytest.param("0.5", 0.5, id="Input '0.5'"),
        pytest.param("0,5", 0.5, id="Comma error"),
        pytest.param("1,567.89", None, id="Comma as thousand mark"),
        pytest.param("123", 123.0, id="Hundreds"),
        pytest.param(True, None, id="Input boolean True"),
        pytest.param("Hey", None, id="Invalid float string"),
        pytest.param(123, 123.0, id="Integer input")
    ]
)
def test_cast_float_various(input: Any, output: float|None) -> None:
    if output is None:
        with pytest.raises(ValueError, match="Invalid float"):
            cast_float(input)
    else:
        # Pylance silenced to test incorrect input type
        assert cast_float(input) == output   # type: ignore


if __name__ == "__main__":
    pytest.main()