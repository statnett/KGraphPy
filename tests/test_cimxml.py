from cim_plugin.cimxml import (
    CIMXMLParser,
    # _get_current_namespace_from_model,
    _get_current_namespace_from_graph,
    # update_namespace_in_model,
    # update_namespace_in_graph,
    # inject_integer_type, 
    find_slots_with_range,
    patch_integer_ranges,
    detect_uri_collisions, 
    _clean_uri,
    # looks_like_cim_uri, 
)
import pytest
from unittest.mock import patch, MagicMock, call, mock_open
# from linkml_runtime.linkml_model import TypeDefinition
# import yaml
from pathlib import Path
from rdflib import URIRef, Graph, Literal, BNode, Namespace
import logging
from pytest import LogCaptureFixture
import textwrap
# from types import SimpleNamespace

logger = logging.getLogger("cimxml_logger")


# @pytest.fixture
# def mock_schemaview():
#     def _factory(types=None, slots=None):
#         mock_schema = MagicMock()
#         mock_schema.types = types
#         mock_schema.slots = slots or {}

#         mock_schemaview = MagicMock()
#         mock_schemaview.schema = mock_schema
#         mock_schemaview.set_modified = MagicMock()

#         mock_schemaview.get_slot.side_effect = lambda name: mock_schema.slots.get(name)

#         def add_slot(slot):
#             mock_schema.slots[slot.name] = slot

#         mock_schemaview.add_slot.side_effect = add_slot

#         return mock_schemaview

#     return _factory

# Unit tests CIMXMLParser.normalize_rdf_ids

@patch("cim_plugin.cimxml._clean_uri")
@patch("cim_plugin.cimxml.detect_uri_collisions")
def test_normalize_rdf_ids_stops_on_collision(mock_detect: MagicMock, mock_clean: MagicMock) -> None:
    mock_detect.side_effect = ValueError("collision!")

    parser = CIMXMLParser()
    g = Graph()
    g.add((URIRef("s"), URIRef("p"), URIRef("o")))
    g.add((URIRef('www.something.com#_ab'), URIRef("p"), URIRef("o")))

    with pytest.raises(ValueError):
        parser.normalize_rdf_ids(g)

    mock_detect.assert_called_once_with(g, {"ab"})
    mock_clean.assert_not_called()


@pytest.mark.parametrize(
        "s, o, calls",
        [
            pytest.param(URIRef("a"), Literal("b"), [call(URIRef('a'), {}, set())], id="Only first call"),
            pytest.param(URIRef("s#_a"), Literal("b"), [call(URIRef('s#_a'), {}, set("a"))], id="One call with id_set"),
            pytest.param(URIRef("s#a"), URIRef("b"), [call(URIRef('s#a'), {}, set("a")), call(URIRef('b'), {}, set("a"))], id="Subject no _"),
            pytest.param(URIRef("a"), URIRef("b"), [call(URIRef('a'), {}, set()), call(URIRef('b'), {}, set())], id="Both called"),
            pytest.param(URIRef("s#_a"), URIRef("b"), [call(URIRef('s#_a'), {}, set("a")), call(URIRef('b'), {}, set("a"))], id="Both called, with id_set"),
            pytest.param(BNode("x"), Literal("b"),[], id="Subject is BNode → no calls"),
            pytest.param(BNode("x"), URIRef("b"),[call(URIRef("b"), {}, set())], id="Subject is BNode → object cleaned"),
            pytest.param(URIRef("a"), BNode("x"), [call(URIRef("a"), {}, set())], id="Object is BNode → only subject cleaned"),
            pytest.param(URIRef("http://ex.com/foo"), Literal("b"), [call(URIRef("http://ex.com/foo"), {}, set())], id="Subject without fragment still cleaned"),
            pytest.param(URIRef("s#_a"), URIRef("http://ex.com#not_id"), [call(URIRef("s#_a"), {}, {"a"}), call(URIRef("http://ex.com#not_id"), {}, {"a"})], id="Object fragment not in id_set → not cleaned, but _clean_uri is called.")
        ]
)
@patch("cim_plugin.cimxml._clean_uri")
@patch("cim_plugin.cimxml.detect_uri_collisions")
def test_normalize_rdf_ids_callscleanuri(mock_detect: MagicMock, mock_clean: MagicMock, s: URIRef|BNode, o: Literal|URIRef|BNode, calls: list) -> None:
    mock_detect.return_value = None
    mock_clean.side_effect = lambda uri, *_: URIRef("urn:uuid:test")

    parser = CIMXMLParser()
    g = Graph()
    g.add((s, URIRef("p"), o))

    parser.normalize_rdf_ids(g)

    mock_detect.assert_called_once()
    assert mock_clean.mock_calls == calls


@patch("cim_plugin.cimxml._clean_uri")
@patch("cim_plugin.cimxml.detect_uri_collisions")
def test_normalize_rdf_ids_mutatesgraph(mock_detect: MagicMock, mock_clean: MagicMock) -> None:
    mock_detect.return_value = None

    mock_clean.side_effect = [
        URIRef("urn:uuid:a"),  # new_s
        URIRef("urn:uuid:b"),  # new_o
    ]

    parser = CIMXMLParser()
    g = Graph()
    s = URIRef("http://ex.com#_a")
    p = URIRef("p")
    o = URIRef("http://ex.com#_b")
    g.add((s, p, o))

    parser.normalize_rdf_ids(g)

    triples = list(g)
    assert len(triples) == 1
    new_s, new_p, new_o = triples[0]
    assert new_s == URIRef("urn:uuid:a")
    assert new_o == URIRef("urn:uuid:b")
    assert new_p == p


@patch("cim_plugin.cimxml._clean_uri")
@patch("cim_plugin.cimxml.detect_uri_collisions")
def test_normalize_rdf_ids_reusesurimap(mock_detect: MagicMock, mock_clean: MagicMock) -> None:
    mock_detect.return_value = None

    mock_clean.side_effect = lambda uri, *_: URIRef("urn:uuid:test")

    parser = CIMXMLParser()
    g = Graph()
    s = URIRef("http://ex.com#_a")
    o = URIRef("http://ex.com#_a")
    p = URIRef("p")
    g.add((s, p, o))
    g.add((s, p, o))  # Same triple again

    parser.normalize_rdf_ids(g)
    assert mock_clean.call_count == 2 # _clean_uri should only be called once each for s and o


@patch("cim_plugin.cimxml._clean_uri")
@patch("cim_plugin.cimxml.detect_uri_collisions")
def test_normalize_rdf_ids_emptygraph(mock_detect: MagicMock, mock_clean: MagicMock) -> None:
    mock_detect.return_value = None

    parser = CIMXMLParser()
    g = Graph()

    parser.normalize_rdf_ids(g)

    mock_clean.assert_not_called()
    mock_detect.assert_called_once()


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

# Unit tests find_slots_with_range
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


# Unit tests detect_uri_collisions
@pytest.mark.parametrize(
    "triples",
    [
        pytest.param([(URIRef("http://ex.com#a"), URIRef("p"), URIRef("http://ex.com#b"))],
                     id="Completely different uris"),
        pytest.param([(URIRef("http://ex.com#_a"), URIRef("p"), URIRef("http://ex.com#_b"))],
                     id="Completely different, but require cleaning"),
        pytest.param([
            (URIRef("http://ex.com#_a"), URIRef("p"), URIRef("http://ex.com#_b")),
            (URIRef("http://ex.com#_c"), URIRef("p"), URIRef("http://ex.com#_d")),
        ], id="Several triples, no collision")
    ]
)
def test_detect_uri_collisions_nocollision(triples: list) -> None:
    g = Graph()
    for s, p, o in triples:
        g.add((s, p, o))

    id_set = {"a", "b", "c", "d"}

    detect_uri_collisions(g, id_set)


def test_detect_uri_collisions_multiple() -> None:
    g = Graph()

    g.add((URIRef("http://ex.com#_a"), URIRef("p"), URIRef("o")))
    g.add((URIRef("http://other.com#a"), URIRef("p"), URIRef("o")))
    g.add((URIRef("http://ex.com#_b"), URIRef("p"), URIRef("o")))
    g.add((URIRef("http://other.com#b"), URIRef("p"), URIRef("o")))

    id_set = {"a", "b"}

    with pytest.raises(ValueError) as exc:
        detect_uri_collisions(g, id_set)

    msg = str(exc.value)

    assert "http://ex.com#_a" in msg
    assert "http://other.com#a" in msg
    assert "urn:uuid:a" in msg

    assert "http://ex.com#_b" in msg
    assert "http://other.com#b" in msg
    assert "urn:uuid:b" in msg


def test_detect_uri_collisionsinobject() -> None:
    g = Graph()
    g.add((URIRef("s1"), URIRef("p"), URIRef("http://ex.com#_x")))
    g.add((URIRef("s2"), URIRef("p"), URIRef("http://other.com#x")))

    id_set = {"x"}

    with pytest.raises(ValueError) as exc:
        detect_uri_collisions(g, id_set)

    msg = str(exc.value)
    assert "http://ex.com#_x" in msg
    assert "http://other.com#x" in msg
    assert "urn:uuid:x" in msg


def test_detect_uri_collisionssubjectvsobject() -> None:
    g = Graph()
    g.add((URIRef("http://ex.com#_x"), URIRef("p"), URIRef("o1")))
    g.add((URIRef("s2"), URIRef("p"), URIRef("http://other.com#x")))

    id_set = {"x"}

    with pytest.raises(ValueError) as exc:
        detect_uri_collisions(g, id_set)

    msg = str(exc.value)
    assert "http://ex.com#_x" in msg
    assert "http://other.com#x" in msg
    assert "urn:uuid:x" in msg


def test_detect_uri_collisionemptyfragments() -> None:
    g = Graph()
    g.add((URIRef("http://ex.com#_"), URIRef("p"), URIRef("o1")))
    g.add((URIRef("s2"), URIRef("p"), URIRef("http://other.com#__")))

    id_set = {"x"}

    with pytest.raises(ValueError) as exc:
        detect_uri_collisions(g, id_set)

    msg = str(exc.value)
    assert "http://ex.com#_" in msg
    assert "http://other.com#__" in msg
    assert "urn:uuid:" in msg


# Unit tests _clean_uri
@pytest.mark.parametrize(
    "uri,id_set,expected",
    [
        pytest.param(URIRef("http://example.com/nohash"), {"id1"}, URIRef("http://example.com/nohash"),
                     id="URI wo # → returned unchanged"),
        pytest.param(URIRef("http://example.com#abc"), {"id1"}, URIRef("http://example.com#abc"),
                     id="Fragment not in id_set and wo _ → returned unchanged"),
        pytest.param(URIRef("http://example.com#_abc"), {"id1"}, URIRef("urn:uuid:abc"),
                     id="Fragment with _ → cleaned"),
        pytest.param(URIRef("http://example.com#id1"), {"id1"}, URIRef("urn:uuid:id1"),
                     id="Fragment in id_set wo _ → cleaned"),
        pytest.param(URIRef("http://example.com#_id1"), {"id1"}, URIRef("urn:uuid:id1"),
                     id="Fragment in id_set and has _ → cleaned"),
        pytest.param(URIRef("http://example.com#"), {"id1"}, URIRef("http://example.com#"),
                     id="No fragment → unchanged"),
        pytest.param(URIRef("http://example.com#_"), {"id1"}, URIRef("urn:uuid:"),
                     id="Just _ fragment → cleaned"),
        pytest.param(URIRef("http://example.com#__id1"), {"id1"}, URIRef("urn:uuid:id1"),
                     id="Fragment with 2+ _ → cleaned"),

    ]
)
def test_clean_uri_basic(uri: URIRef, id_set: set[str], expected: URIRef) -> None:
    uri_map = {}
    result = _clean_uri(uri, uri_map, id_set)
    assert result == expected

def test_clean_uri_uses_cache() -> None:
    uri = URIRef("http://example.com#_abc")
    id_set = {"abc"}
    uri_map = {}

    first = _clean_uri(uri, uri_map, id_set)
    second = _clean_uri(uri, uri_map, id_set)

    assert first is second  # Same object → same uri_map
    assert list(uri_map.values()) == [first]    # uri_map should only have one entry

def test_clean_uri_cacheonlyforcleaned() -> None:
    uri = URIRef("http://example.com#not_in_set")
    id_set = {"something_else"}
    uri_map = {}

    result = _clean_uri(uri, uri_map, id_set)

    assert uri_map == {} # No caching when nothing cleaned
    assert result == uri

@pytest.mark.parametrize(
        "input, ids, output",
        [
            pytest.param("http://example.com#a#b#_c", {"id"}, "urn:uuid:c", id="Cleaned because of _ in last fragment"),
            pytest.param("http://example.com#_a#b#c", {"id"}, "http://example.com#_a#b#c", id="Not cleaned because _ in first fragment"),
            pytest.param("http://example.com#a#_b#c", {"id"}, "http://example.com#a#_b#c", id="Not cleaned because _ in second fragment"),
            pytest.param("http://example.com#a#b#c", {"id"}, "http://example.com#a#b#c", id="Not cleaned because there are no _"),
            pytest.param("http://example.com#a#b#c", {"c"}, "urn:uuid:c", id="Cleaned because fragment in id_set"),
            pytest.param("http://example.com#a#b#c", {"b"}, "http://example.com#a#b#c", id="Not cleaned because wrong fragment in id_set"),
        ]
)
def test_clean_uri_multiplehashes(input: str, ids: set[str], output: str, caplog: LogCaptureFixture) -> None:
    # These tests show what happends if there are multiple # in the URIRef.
    # It relies on a warning being logged and manually checking that the result is correct.
    # Consider if the function should fail instead.
    uri = URIRef(input)
    id_set = ids
    uri_map = {}

    result = _clean_uri(uri, uri_map, id_set)

    assert result == URIRef(output)
    assert f"{input} has more then one #" in caplog.text


def test_clean_uri_multipleurissamefragments() -> None:
    # This test shows that two identical fragments with different URIs will end up the same object in the graph 
    uri1 = URIRef("http://example.com#_abc")
    uri2 = URIRef("http://other.com#_abc")

    id_set = {"abc"}
    uri_map = {}

    cleaned1 = _clean_uri(uri1, uri_map, id_set)
    cleaned2 = _clean_uri(uri2, uri_map, id_set)

    assert cleaned1 is not cleaned2 # They are not the same object
    assert cleaned1 == cleaned2 # They look identical
    assert cleaned1 == URIRef("urn:uuid:abc")
    assert len(uri_map) == 2
    assert set(uri_map.values()) == {cleaned1}

# Unit tests looks_like_cim_uri

# @pytest.mark.parametrize(
#     "input, output",
#     [
#         pytest.param("https://cim.ucaiug.io/ns", True, id="Namespace match: cim"),
#         pytest.param("http://iec.ch/TC57", True, id="Namespace match: tc57"),
#         pytest.param("UCaiug", True, id="Namespace match upper letters: ucaiug"),
#         pytest.param("tac57", False, id="Not matched"),
#         pytest.param("57", False, id="Numerical input as string")
#     ]
# )
# def test_looks_like_cim_uri_various(input, output):
#     assert looks_like_cim_uri(input) == output

# def test_looks_like_cim_uri_numericinput():
#     with pytest.raises(AttributeError):
#         looks_like_cim_uri(57) # type: ignore

if __name__ == "__main__":
    pytest.main()