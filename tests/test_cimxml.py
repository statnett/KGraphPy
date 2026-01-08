from cim_plugin.cimxml import (
    looks_like_cim_uri, 
    inject_integer_type, 
    patch_integer_ranges,
    detect_uri_collisions, 
    _clean_uri
)
import pytest
from unittest.mock import MagicMock
from linkml_runtime.linkml_model import TypeDefinition
from collections import defaultdict
import yaml
from pathlib import Path
from rdflib import URIRef, Graph
import logging
from pytest import LogCaptureFixture

logger = logging.getLogger("cimxml_logger")

# @pytest.fixture
# def mock_schemaview():
#     """
#     Returnerer en funksjon som lager en mock SchemaView med valgfritt types-dict.
#     Brukes slik:

#         schemaview = mock_schemaview(types={"integer": ...})
#     """
#     def _factory(types=None, slots=None):
#         mock_schema = MagicMock()
#         mock_schema.types = types
#         mock_schema.slots = slots

#         mock_schemaview = MagicMock()
#         mock_schemaview.schema = mock_schema
#         mock_schemaview.set_modified = MagicMock()

#         return mock_schemaview

#     return _factory
@pytest.fixture
def mock_schemaview():
    def _factory(types=None, slots=None):
        mock_schema = MagicMock()
        mock_schema.types = types
        mock_schema.slots = slots or {}

        mock_schemaview = MagicMock()
        mock_schemaview.schema = mock_schema
        mock_schemaview.set_modified = MagicMock()

        # get_slot henter fra schema.slots
        mock_schemaview.get_slot.side_effect = lambda name: mock_schema.slots.get(name)

        # add_slot oppdaterer schema.slots
        def add_slot(slot):
            mock_schema.slots[slot.name] = slot

        mock_schemaview.add_slot.side_effect = add_slot

        return mock_schemaview

    return _factory

# Unit tests inject_integer_type

@pytest.mark.parametrize(
        "typeset, set_modified_called",
        [
            pytest.param({}, True, id="Integer added to schemaview"),
            pytest.param({"integer": TypeDefinition(name="integer", base="int")}, False, id="Integer already in schemaview"),
            pytest.param({"wierd": "Not a TypeDefinition"}, True, id="Integer added when other non-conventional types are present"),
            pytest.param({123: TypeDefinition(name="num")}, True, id="Integer added when other non-string keys are present"),
            pytest.param({"integer": "Not a TypeDefinition"}, False, id="Integer already in schemaview, but with invalid type"),
        ]
)
def test_inject_integer_type_various(typeset: dict, set_modified_called: bool, mock_schemaview: MagicMock) -> None:
    schemaview = mock_schemaview(types=typeset)
    inject_integer_type(schemaview)

    assert "integer" in schemaview.schema.types
    if set_modified_called:
        schemaview.set_modified.assert_called_once()
    else:
        schemaview.set_modified.assert_not_called()

def test_inject_integer_type_schemamissing(mock_schemaview: MagicMock) -> None:
    schemaview = mock_schemaview(types={})
    schemaview.schema = None  # overskriv schema

    with pytest.raises(ValueError):
        inject_integer_type(schemaview)


def test_inject_integer_type_raises_if_types_not_dict(mock_schemaview: MagicMock) -> None:
    schemaview = mock_schemaview(types=None)

    with pytest.raises(ValueError):
        inject_integer_type(schemaview)


def test_inject_integer_type_set_modifiederror(mock_schemaview: MagicMock) -> None:
    schemaview = mock_schemaview(types={})
    schemaview.set_modified.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError):
        inject_integer_type(schemaview)

def test_inject_integer_type_supportsdictsubclasses(mock_schemaview: MagicMock) -> None:
    schemaview = mock_schemaview(types=defaultdict(dict))
    inject_integer_type(schemaview)
    assert "integer" in schemaview.schema.types

# Unit tests patch_integer_ranges
def test_patch_integer_ranges(tmp_path: Path, mock_schemaview: MagicMock) -> None:
    yaml_content = """
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
    """
    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text(yaml_content)

    endDate_slot = MagicMock()
    endDate_slot.name = "endDate"
    endDate_slot.range = "string"

    sv = mock_schemaview(slots={"endDate": endDate_slot})

    patch_integer_ranges(sv, str(schema_file))

    assert sv.schema.slots["endDate"].range == "integer"
    sv.set_modified.assert_called_once()

def test_patch_integer_ranges_noninteger(tmp_path: Path, mock_schemaview: MagicMock) -> None:
    yaml_content = """
        classes:
            Season:
                attributes:
                    endDate:
                        range: float
    """
    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text(yaml_content)

    endDate_slot = MagicMock()
    endDate_slot.name = "endDate"
    endDate_slot.range = "string"

    sv = mock_schemaview(slots={"endDate": endDate_slot})

    patch_integer_ranges(sv, str(schema_file))

    assert sv.schema.slots["endDate"].range == "string"
    sv.add_slot.assert_not_called()
    sv.set_modified.assert_not_called()


def test_patch_integer_ranges_missingattributes(tmp_path: Path, mock_schemaview: MagicMock) -> None:
    yaml_content = """
        classes:
            EmptyClass: {}
    """
    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text(yaml_content)

    sv = mock_schemaview(slots={})

    patch_integer_ranges(sv, str(schema_file))

    sv.add_slot.assert_not_called() # No attributes to patch, no patching
    sv.set_modified.assert_not_called()


def test_patch_integer_ranges_multipleintegerslots(tmp_path: Path, mock_schemaview: MagicMock) -> None:
    yaml_content = """
        classes:
            A:
                attributes:
                    x:
                        range: integer
                    y:
                        range: integer
    """
    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text(yaml_content)

    x_slot = MagicMock()
    x_slot.name = "x"
    x_slot.range = "string"

    y_slot = MagicMock()
    y_slot.name = "y"
    y_slot.range = "string"

    sv = mock_schemaview(slots={"x": x_slot, "y": y_slot})

    patch_integer_ranges(sv, str(schema_file))

    assert sv.schema.slots["x"].range == "integer"
    assert sv.schema.slots["y"].range == "integer"
    assert sv.add_slot.call_count == 2
    sv.set_modified.assert_called_once()


def test_patch_integer_ranges_stringattributeentries(tmp_path: Path, mock_schemaview: MagicMock) -> None:
    yaml_content = """
        classes:
            Season:
                attributes:
                    endDate: endDate
    """
    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text(yaml_content)

    sv = mock_schemaview(slots={})

    with pytest.raises(ValueError) as exc_info:
        patch_integer_ranges(sv, str(schema_file))

    assert "unexpected structure" in str(exc_info.value)
    sv.add_slot.assert_not_called()
    sv.set_modified.assert_not_called()


def test_patch_integer_ranges_attributesnull(tmp_path: Path, mock_schemaview: MagicMock) -> None:
    # This test is included to show that the function will work even if a class attributes is null.
    # Consider if an error should be raised here, as it is not a valid linkML format.

    yaml_content = """
        classes:
            Season:
                attributes:
                    endDate:
                        range: integer
            Foo:
                attributes: null
    """
    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text(yaml_content)
    endDate_slot = MagicMock()
    endDate_slot.name = "endDate"
    endDate_slot.range = "string"

    sv = mock_schemaview(slots={"endDate": endDate_slot})

    patch_integer_ranges(sv, str(schema_file))

    sv.add_slot.assert_called_once()    # Would have been called twice if Foo contained a range: integer
    sv.set_modified.assert_called_once()

def test_patch_integer_ranges_attributesempty(tmp_path: Path, mock_schemaview: MagicMock) -> None:
    # This test is included to show that the function will work even if a class attributes is an empty dict.
    # This is a valid linkML.

    yaml_content = """
        classes:
            Season:
                attributes:
                    endDate:
                        range: integer
            Foo:
                attributes: {}
    """
    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text(yaml_content)
    endDate_slot = MagicMock()
    endDate_slot.name = "endDate"
    endDate_slot.range = "string"

    sv = mock_schemaview(slots={"endDate": endDate_slot})

    patch_integer_ranges(sv, str(schema_file))

    sv.add_slot.assert_called_once()    # Would have been called twice if Foo contained a range: integer
    sv.set_modified.assert_called_once()


def test_patch_integer_ranges_rangenull(tmp_path: Path, mock_schemaview: MagicMock) -> None:
    # This test is included to show that the function will work even if a range is null.
    # Consider if an error should be raised here, as it is not a valid linkML format.

    yaml_content = """
        classes:
            Season:
                attributes:
                    endDate:
                        range: null
    """
    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text(yaml_content)
    endDate_slot = MagicMock()
    endDate_slot.name = "endDate"
    endDate_slot.range = "string"

    sv = mock_schemaview(slots={"endDate": endDate_slot})

    patch_integer_ranges(sv, str(schema_file))

    assert sv.schema.slots["endDate"].range == "string"
    sv.add_slot.assert_not_called() # range is not an integer, and therefore no changes are made
    sv.set_modified.assert_not_called()


def test_patch_integer_ranges_rangelist(tmp_path: Path, mock_schemaview: MagicMock) -> None:
    # This test is included to show that the function will work even if a range is a list.
    # Consider if an error should be raised here, as it is not a valid linkML format.

    yaml_content = """
        classes:
            Season:
                attributes:
                    endDate:
                        range:
                            - integer
                            - string
    """
    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text(yaml_content)
    endDate_slot = MagicMock()
    endDate_slot.name = "endDate"
    endDate_slot.range = "string"

    sv = mock_schemaview(slots={"endDate": endDate_slot})

    patch_integer_ranges(sv, str(schema_file))

    assert sv.schema.slots["endDate"].range == "string"
    sv.add_slot.assert_not_called() # Even though one of the items is an integer, no changes are made
    sv.set_modified.assert_not_called()

def test_patch_integer_ranges_mismatchedschemaviews(tmp_path: Path, mock_schemaview: MagicMock) -> None:
    # Mismatch between the schemaview and the linkML yaml, 
    # which should never happen because both are loaded from same file location

    yaml_content = """
        classes:
            Season:
                attributes:
                    endDate:
                        range: integer
    """
    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text(yaml_content)
    endDate_slot = MagicMock()
    endDate_slot.name = "startDate"
    endDate_slot.range = "string"

    sv = mock_schemaview(slots={"startDate": endDate_slot})

    with pytest.raises(ValueError) as exc_info:
        patch_integer_ranges(sv, str(schema_file))

    assert "endDate not found in schemaview" in str(exc_info.value)
    assert sv.schema.slots["startDate"].range == "string"
    sv.add_slot.assert_not_called() 
    sv.set_modified.assert_not_called()



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

@pytest.mark.parametrize(
    "input, output",
    [
        pytest.param("https://cim.ucaiug.io/ns", True, id="Namespace match: cim"),
        pytest.param("http://iec.ch/TC57", True, id="Namespace match: tc57"),
        pytest.param("UCaiug", True, id="Namespace match upper letters: ucaiug"),
        pytest.param("tac57", False, id="Not matched"),
        pytest.param("57", False, id="Numerical input as string")
    ]
)
def test_looks_like_cim_uri_various(input, output):
    assert looks_like_cim_uri(input) == output

def test_looks_like_cim_uri_numericinput():
    with pytest.raises(AttributeError):
        looks_like_cim_uri(57) # type: ignore

if __name__ == "__main__":
    pytest.main()