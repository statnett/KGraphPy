from cim_plugin.cimxml import looks_like_cim_uri, inject_integer_type
import pytest
from unittest.mock import MagicMock
from linkml_runtime.linkml_model import TypeDefinition
from collections import defaultdict


@pytest.fixture
def mock_schemaview():
    """
    Returnerer en funksjon som lager en mock SchemaView med valgfritt types-dict.
    Brukes slik:

        schemaview = mock_schemaview(types={"integer": ...})
    """
    def _factory(types=None, slots=None):
        mock_schema = MagicMock()
        mock_schema.types = types
        mock_schema.slots = slots

        mock_schemaview = MagicMock()
        mock_schemaview.schema = mock_schema
        mock_schemaview.set_modified = MagicMock()

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