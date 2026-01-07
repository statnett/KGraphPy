from cim_plugin.cimxml import looks_like_cim_uri, inject_integer_type, patch_integer_ranges
import pytest
from unittest.mock import MagicMock
from linkml_runtime.linkml_model import TypeDefinition
from collections import defaultdict
import yaml
from pathlib import Path

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