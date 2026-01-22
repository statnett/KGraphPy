from unittest.mock import MagicMock, patch, call
import pytest
from pytest import LogCaptureFixture
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model.meta import (
    SchemaDefinition, 
    Prefix, 
    TypeDefinition, 
    SlotDefinition, 
    ClassDefinition,
    EnumDefinition,
    PermissibleValue,
)
from typing import Callable, Any    #, Optional, Dict
from rdflib import Literal, URIRef
from rdflib.namespace import XSD
from datetime import date, datetime
import copy
from collections import defaultdict
from dataclasses import dataclass
import logging

from cim_plugin.cimxml import (
    _get_current_namespace_from_model, 
    update_namespace_in_model,
    inject_integer_type,
    patch_integer_ranges,
    slots_equal,
    _build_slot_index,
    _resolve_type,
    resolve_datatype_from_slot,
    create_typed_literal,
    LiteralCastingError
)

logger = logging.getLogger("cimxml_logger")

# There are numerous type: ignore in this file. Where not otherwise stated, this is to silence pylance 
# about schemaview.schema. Pylance complains because linkML are using too wide type hints.


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

        return SchemaView(schema=schema)

    return _factory

# Unit tests for _get_current_namespace_from_model
@pytest.mark.parametrize(
    "prefixes,prefix,expected",
    [
        pytest.param(
            [{"ex": Prefix(prefix_prefix="ex", prefix_reference="http://example.org/")}],
            "ex",
            "http://example.org/",
            id="Prefix found"
        ),
        pytest.param(
            [
                {"foo": Prefix(prefix_prefix="foo", prefix_reference="http://foo.org/")},
                {"bar": Prefix(prefix_prefix="bar", prefix_reference="http://bar.org/")},
            ],
            "bar",
            "http://bar.org/",
            id="Multiple prefixes registered"
        ),
        pytest.param(
            [
                {"foo": Prefix(prefix_prefix="foo", prefix_reference="http://foo.org/")},
                {"bar": Prefix(prefix_prefix="bar", prefix_reference="http://bar.org/")},
            ],
            "missing",
            None,
            id="No prefix found"
        ),
    ]
)
def test_get_current_namespace_from_model_basic(make_schemaview: Callable[..., SchemaView], prefixes: list[dict], prefix: str, expected: str|None) -> None:
    sv = make_schemaview(prefixes=prefixes)
    result = _get_current_namespace_from_model(sv, prefix)
    assert result == expected


def test_get_current_namespace_from_model_missingschema(make_schemaview: Callable[..., SchemaView]) -> None:
    sv = make_schemaview()
    sv.schema = None

    with pytest.raises(ValueError) as exc_info:
        _get_current_namespace_from_model(sv, "ex")

    assert "Schemaview not found or schemaview is missing schema." in str(exc_info.value)


def test_get_current_namespace_from_model_schemahasnamespaces(make_schemaview: Callable[..., SchemaView]) -> None:
    # The field "namespaces" is deprecated from linkML. 
    # This test shows that a schemaview that contain namespaces is outdated and cannot be handled.
    sv = make_schemaview(prefixes=[{"foo": Prefix(prefix_prefix="foo", prefix_reference="http://foo.org/")}])
    # Ignoring pylance for testing wrong input
    sv.schema.namespaces = {}   # type: ignore

    with pytest.raises(ValueError) as exc_info:
        _get_current_namespace_from_model(sv, "ex")

    assert "The attribute 'namespaces' found in schema. This schemaview is outdated." in str(exc_info.value)


@pytest.mark.parametrize(
    "prefixes,prefix,expected",
    [
        pytest.param({}, "ex", None, id="Empty prefixes"),
        pytest.param(None, "ex", None, id="None prefixes"),
        pytest.param(None, None, None, id="Prefix is None"),
        pytest.param({}, 1, None, id="Prefix is int"),
    ]
)
def test__get_current_namespace_from_model_edgecases(make_schemaview: Callable[..., SchemaView], prefixes: dict|None|int, prefix: str|None, expected: str|None) -> None:
    sv = make_schemaview(prefixes=prefixes)

    # Ignoring pylance for testing wrong input
    result = _get_current_namespace_from_model(sv, prefix)   # type: ignore
    assert result == expected


def test_get_current_namespace_from_model_nomutationofschemaview(make_schemaview: Callable[..., SchemaView]) -> None:
    prefixes = [{"foo": Prefix(prefix_prefix="foo", prefix_reference="http://foo.org/")},
                {"bar": Prefix(prefix_prefix="bar", prefix_reference="http://bar.org/")}]
    sv = make_schemaview(prefixes=prefixes)
    schema_before_id = id(sv.schema)
    prefixes_before = copy.deepcopy(sv.schema.prefixes) # type: ignore
    name_before = copy.deepcopy(getattr(sv.schema, "name", None))

    result = _get_current_namespace_from_model(sv, "foo")

    assert result == "http://foo.org/"

    # Checking that nothing has been changed in the schemaview
    assert id(sv.schema) == schema_before_id
    assert sv.schema.prefixes == prefixes_before    # type: ignore
    assert getattr(sv.schema, "name", None) == name_before


# Unit tests update_namespace_in_model

def test_update_namespace_in_model_noschema(make_schemaview: Callable[..., SchemaView]) -> None:
    sv = make_schemaview()
    sv.schema = None

    with pytest.raises(ValueError):
        update_namespace_in_model(sv, "ex", "http://new/")


@pytest.mark.parametrize(
    "prefixes,expected",
    [
        pytest.param(
            {"ex": Prefix(prefix_prefix="ex", prefix_reference="http://old/")},
            "http://updated/",
            id="Prefix exists and is updated"
        ),
        pytest.param(
            {"other": Prefix(prefix_prefix="other", prefix_reference="http://old/")},
            "http://old/",
            id="Prefix not in schema, no update"
        ),
        pytest.param(
            {"ex": Prefix(prefix_prefix="ex", prefix_reference="")},
            "http://updated/",
            id="Prefix exist, but prefix_reference is empty -> Updated"
        ),
        pytest.param(
            {"ex": {"prefix_reference": "http://old/"}},
            "http://updated/",
            id="Prefix is not a prefix object -> Updated"
        ),
        pytest.param(
            {"ex": "http://old/"},
            "http://updated/",
            id="Prefix is a simple dict -> Updated"
        ),
        pytest.param(
            {},
            "",
            id="Prefix is empty -> Not updated"
        ),
    ],
)
def test_update_namespace_in_model_prefixes(make_schemaview: Callable[..., SchemaView], prefixes: dict[str, Any], expected: str) -> None:
    sv = make_schemaview(prefixes=prefixes)

    update_namespace_in_model(sv, "ex", "http://updated/")

    # Using type: ignore to cancel pylance issues with linkML prefixes
    if "ex" in prefixes:
        assert sv.schema.prefixes["ex"].prefix_reference == expected    # type: ignore
    elif "other" in prefixes:
        assert sv.schema.prefixes["other"].prefix_reference == expected # type: ignore
    else:
        assert len(prefixes) == 0


def test_update_namespace_in_model_reinitializing_check(make_schemaview: Callable[..., SchemaView], monkeypatch):
    sv = make_schemaview()

    called = False

    def fake_init(self, schema):
        nonlocal called
        called = True

    monkeypatch.setattr(SchemaView, "__init__", fake_init)

    update_namespace_in_model(sv, "ex", "http://updated/")

    assert called is True   # Checking that schemaview.__init__(schema) is called


def test_update_namespace_in_model_prefixesisnone(make_schemaview: Callable[..., SchemaView]) -> None: 
    sv = make_schemaview() 
    update_namespace_in_model(sv, "ex", "http://updated/") 
    assert sv.schema.prefixes == {} # type: ignore


def test_update_namespace_in_model_missingprefixesattribute(make_schemaview: Callable[..., SchemaView]) -> None: 
    sv = make_schemaview(prefixes=[{"ex": Prefix(prefix_prefix="ex", prefix_reference="http://old/")}]) 
    del sv.schema.prefixes  # type: ignore
    update_namespace_in_model(sv, "ex", "http://updated/") 
    assert not hasattr(sv.schema, "prefixes")

def test_update_namespace_in_model_reinitraisesexception(monkeypatch: pytest.MonkeyPatch, make_schemaview: Callable[..., SchemaView]) -> None: 
    sv = make_schemaview(prefixes=[{"ex": Prefix(prefix_prefix="ex", prefix_reference="http://old/")}]) 
    
    def fake_init(self, schema): 
        raise RuntimeError("boom") 
    
    monkeypatch.setattr(SchemaView, "__init__", fake_init) 
    with pytest.raises(RuntimeError): 
        update_namespace_in_model(sv, "ex", "http://updated/")

def test_update_namespace_in_model_nonstringinput(make_schemaview: Callable[..., SchemaView]) -> None: 
    prefixes = {"ex": Prefix(prefix_prefix="ex", prefix_reference="http://old/")} 
    sv = make_schemaview(prefixes=prefixes) 
    # Using type: ignore to test wrong datatype input without pylance complaining.
    update_namespace_in_model(sv, "ex", 12345)  # type: ignore
    sv.schema.prefixes["ex"].prefix_reference == 12345  # type: ignore


def test_update_namespace_in_model_prefixesislist(make_schemaview: Callable[..., SchemaView]) -> None: 
    prefixes = [ {"ex": {"prefix_reference": "http://old/"}} ] 
    sv = make_schemaview(prefixes=prefixes) 
    update_namespace_in_model(sv, "ex", "http://updated/") 
    assert sv.schema.prefixes == {"ex": Prefix(prefix_prefix="ex", prefix_reference="http://updated/")} # type: ignore

def test_update_namespace_in_model_schemawrongtype(make_schemaview: Callable[..., SchemaView]) -> None: 
    sv = make_schemaview(prefixes=[{"ex": Prefix(prefix_prefix="ex", prefix_reference="http://old/")}]) 
    # Using type: ignore to test wrong datatype input without pylance complaining.
    sv.schema = "not-a-schema-object"   # type: ignore
    with pytest.raises(Exception): 
        update_namespace_in_model(sv, "ex", "http://updated/")


# Unit tests inject_integer_type

@pytest.mark.parametrize(
        "typeset, set_modified_called",
        [
            pytest.param({}, True, id="Integer added to schemaview"),
            pytest.param({"integer": TypeDefinition(name="integer", base="int")}, False, id="Integer already in schemaview"),
            pytest.param({123: TypeDefinition(name="123")}, True, id="Integer added when other non-string keys are present"),
        ]
)
def test_inject_integer_type_various(typeset: dict, set_modified_called: bool, make_schemaview: Callable[..., SchemaView], monkeypatch: pytest.MonkeyPatch) -> None:
    called = False
    def fake_set_modified(self):
        nonlocal called
        called = True

    monkeypatch.setattr(SchemaView, "set_modified", fake_set_modified)

    schemaview = make_schemaview(types=typeset)
    inject_integer_type(schemaview)

    assert "integer" in schemaview.schema.types # type: ignore
    if set_modified_called:
        assert called is True
    else:
        assert called is False
        

def test_inject_integer_type_schemamissing(make_schemaview: Callable[..., SchemaView]) -> None:
    schemaview = make_schemaview()
    schemaview.schema = None  # overskriv schema

    with pytest.raises(ValueError):
        inject_integer_type(schemaview)


def test_inject_integer_type_set_modifiederror(make_schemaview: Callable[..., SchemaView], monkeypatch: pytest.MonkeyPatch) -> None:
    schemaview = make_schemaview()

    def fake_set_modified(self): 
        raise RuntimeError("boom") 
    
    monkeypatch.setattr(SchemaView, "set_modified", fake_set_modified)

    with pytest.raises(RuntimeError):
        inject_integer_type(schemaview)

def test_inject_integer_type_supportsdictsubclasses(make_schemaview: Callable[..., SchemaView]) -> None:
    schemaview = make_schemaview(types=defaultdict(dict))
    inject_integer_type(schemaview)
    assert "integer" in schemaview.schema.types # type: ignore


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
    monkeypatch.setattr("cim_plugin.cimxml.find_slots_with_range", mocks.find_slots)
    monkeypatch.setattr(SchemaView, "add_slot", mocks.add_slot)
    monkeypatch.setattr(SchemaView, "set_modified", mocks.set_modified)
    
    mocks.calls = []
    mocks.find_slots.side_effect = lambda *a, **kw: (mocks.calls.append("find_slots_with_range"), mocks.find_slots.return_value)[1]
    mocks.add_slot.side_effect = lambda *a, **kw: mocks.calls.append("add_slot")
    mocks.set_modified.side_effect = lambda *a, **kw: mocks.calls.append("set_modified")

    return mocks

# Unit tests patch_integer_ranges
def test_patch_integer_ranges_basic(make_schemaview: Callable[..., SchemaView], mock_patch_integer_ranges: PatchMocks) -> None:
    mock_patch_integer_ranges.find_slots.return_value = {"endDate"}
    
    slots = [{"endDate": SlotDefinition(name="endDate", range="string")},
             {"startDate": SlotDefinition(name="startDate", range="Date")}]
    sv = make_schemaview(slots=slots)
    
    patch_integer_ranges(sv, "schema.yaml")

    assert sv.schema.slots["endDate"].range == "integer" # type: ignore
    assert sv.schema.slots["startDate"].range == "Date" # type: ignore
    mock_patch_integer_ranges.find_slots.assert_called_once_with("schema.yaml", "integer")
    mock_patch_integer_ranges.add_slot.assert_called_once_with(sv.schema.slots["endDate"])  # type: ignore
    mock_patch_integer_ranges.set_modified.assert_called_once()

def test_patch_integer_ranges_multiple(make_schemaview: Callable[..., SchemaView], mock_patch_integer_ranges: PatchMocks) -> None:
    mock_patch_integer_ranges.find_slots.return_value = {"endDate", "startDate"}

    slots = [{"endDate": SlotDefinition(name="endDate", range="string")},
             {"startDate": SlotDefinition(name="startDate", range="string")},]
    sv = make_schemaview(slots=slots)

    patch_integer_ranges(sv, "schema.yaml")

    assert sv.schema.slots["endDate"].range == "integer"    # type: ignore
    assert sv.schema.slots["startDate"].range == "integer"  # type: ignore
    assert mock_patch_integer_ranges.add_slot.call_count == 2
    mock_patch_integer_ranges.set_modified.assert_called_once()


def test_patch_integer_ranges_noslotstochange(make_schemaview: Callable[..., SchemaView], mock_patch_integer_ranges: PatchMocks, caplog: LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    mock_patch_integer_ranges.find_slots.return_value = {}
    slots = [{"endDate": SlotDefinition(name="endDate", range="string")},
             {"startDate": SlotDefinition(name="startDate", range="Date")}]
    sv = make_schemaview(slots=slots)
    patch_integer_ranges(sv, "schema.yaml")

    assert sv.schema.slots["endDate"].range == "string" # type: ignore
    assert sv.schema.slots["startDate"].range == "Date" # type: ignore
    mock_patch_integer_ranges.add_slot.assert_not_called()
    mock_patch_integer_ranges.set_modified.assert_not_called()
    assert "No attributes with range=integer found. No changes made." in caplog.text 


def test_patch_integer_ranges_slotnotinschema(make_schemaview: Callable[..., SchemaView], mock_patch_integer_ranges: PatchMocks) -> None:
    mock_patch_integer_ranges.find_slots.return_value = {"middleDate"}
    slots = [{"endDate": SlotDefinition(name="endDate", range="string")},
             {"startDate": SlotDefinition(name="startDate", range="Date")}]
    sv = make_schemaview(slots=slots)
    with pytest.raises(ValueError) as exc_info:
        patch_integer_ranges(sv, "schema.yaml")

    assert sv.schema.slots["endDate"].range == "string" # type: ignore
    assert sv.schema.slots["startDate"].range == "Date" # type: ignore
    mock_patch_integer_ranges.add_slot.assert_not_called()
    mock_patch_integer_ranges.set_modified.assert_not_called()
    assert "middleDate not found in schemaview" in str(exc_info.value) 


def test_patch_integer_ranges_missingattributes(make_schemaview: Callable[..., SchemaView], mock_patch_integer_ranges: PatchMocks) -> None:
    mock_patch_integer_ranges.find_slots.return_value = {"middleDate"}
    sv = make_schemaview(slots={})
    with pytest.raises(ValueError) as exc_info:
        patch_integer_ranges(sv, "schema.yaml")

    mock_patch_integer_ranges.add_slot.assert_not_called()
    mock_patch_integer_ranges.set_modified.assert_not_called()
    assert "middleDate not found in schemaview" in str(exc_info.value) 

def test_patch_integer_ranges_find_slots_input_not_mutated(make_schemaview: Callable[..., SchemaView], mock_patch_integer_ranges: PatchMocks) -> None:
    slots_set = {"endDate"}
    mock_patch_integer_ranges.find_slots.return_value = slots_set

    sv = make_schemaview(slots=[{"endDate": SlotDefinition(name="endDate", range="string")}])
    patch_integer_ranges(sv, "schema.yaml")

    assert slots_set == {"endDate"}  # Not modified


def test_patch_integer_ranges_rangealreadyinteger(make_schemaview: Callable[..., SchemaView], mock_patch_integer_ranges: PatchMocks) -> None:
    mock_patch_integer_ranges.find_slots.return_value = {"endDate"}

    slots = [{"endDate": SlotDefinition(name="endDate", range="integer")}]
    sv = make_schemaview(slots=slots)

    patch_integer_ranges(sv, "schema.yaml")

    mock_patch_integer_ranges.add_slot.assert_not_called()
    mock_patch_integer_ranges.set_modified.assert_not_called()


def test_patch_integer_ranges_call_order(make_schemaview, mock_patch_integer_ranges):
    mock_patch_integer_ranges.find_slots.return_value = {"endDate", "startDate"}

    slots = [{"endDate": SlotDefinition(name="endDate", range="string")},
             {"startDate": SlotDefinition(name="startDate", range="string")}]
    sv = make_schemaview(slots=slots)

    patch_integer_ranges(sv, "schema.yaml")

    # Checking that calls are in correct order
    assert mock_patch_integer_ranges.calls == [
        "find_slots_with_range", 
        "add_slot",
        "add_slot", 
        "set_modified"
    ]


def test_patch_integer_ranges_noslotsadded(make_schemaview: Callable[..., SchemaView], mock_patch_integer_ranges: PatchMocks) -> None:
    mock_patch_integer_ranges.find_slots.return_value = {"endDate"}
    
    slots = [{"endDate": SlotDefinition(name="endDate", range="string")}]
    sv = make_schemaview(slots=slots)
    
    assert sv.schema is not None
    assert isinstance(sv.schema.slots, dict)
    
    slot_before = sv.schema.slots["endDate"] 
    before_count = len(sv.schema.slots)
    patch_integer_ranges(sv, "schema.yaml")
    slot_after = sv.schema.slots["endDate"] 
    after_count = len(sv.schema.slots)
    
    # Showing that the slot is changed, not added
    assert before_count == after_count
    assert slot_before is slot_after 
    assert sv.schema.slots["endDate"].range == "integer" # type: ignore

def test_patch_integer_ranges_class_attributes(make_schemaview: Callable[..., SchemaView], mock_patch_integer_ranges: PatchMocks) -> None:
    # This test ensures the function works with the exact structure used by cim linkML
    mock_patch_integer_ranges.find_slots.return_value = {"endDate"}
    schema_dict = {
            "MyClass": {
                "attributes": {
                    "endDate": {"range": "string"},
                    "startDate": {"range": "Date"},
                }
            }
        }

    sv = make_schemaview(classes=schema_dict)

    patch_integer_ranges(sv, "schema.yaml")

    slot = sv.get_slot("endDate")
    assert slot.range == "integer"
    mock_patch_integer_ranges.add_slot.assert_called_once()
    mock_patch_integer_ranges.set_modified.assert_called_once()

# Unit tests slots_equal

@pytest.mark.parametrize(
    "attrs1, attrs2, expected",
    [
        pytest.param({"name": "age", "range": "integer"}, {"name": "age", "range": "integer"}, True, id="Identical simple attributes"),
        pytest.param({"name": "age", "range": "integer"}, {"name": "age", "range": "string"}, False, id="Different values"),
        pytest.param({"name": "age"}, {"name": "age", "description": "Age of person"}, False, id="One has extra attribute"),
        pytest.param({"name": "age", "range": "123"}, {"name": "age", "range": 123}, True, id="Different datatypes"),
        pytest.param({"name": "age", "multivalued": True}, {"name": "age", "multivalued": "True"}, True, id="Different datatypes, boolean"),
        pytest.param({"name": "age", "range": "integer"}, {"range": "integer", "name": "age"}, True, id="Same keys but different order"),
        pytest.param({"name": "age"}, {"name": "age", "description": None}, True, id="Attribute None vs. missing attribute"),
        pytest.param({"name": "age"}, {"name": "age", "description": ""}, False, id="Attribute empty vs. missing attribute"),
        pytest.param({"name": "age", "description": ["B", "A"]}, {"name": "age", "description": ["A", "B"]}, False, id="Attributes lists with different orders"),
        pytest.param({"name": "age", "description": {"Name": "B", "Type": "A"}}, {"name": "age", "description": {"Type": "A", "Name": "B"}}, False, id="Attributes dicts with different orders"),
        pytest.param({"name": "age", "description": ["A", "B"]}, {"name": "age", "description": ("A", "B")}, False, id="Attributes lists vs. tuples"),
    ],
)
def test_slots_equal_basic(attrs1: dict, attrs2: dict, expected: bool) -> None:
    slot1 = SlotDefinition(**attrs1)
    slot2 = SlotDefinition(**attrs2)
    assert slots_equal(slot1, slot2) is expected


def test_slots_equal_oneprivateattributes() -> None:
    slot1 = SlotDefinition(name="height")
    slot2 = SlotDefinition(name="height")
    slot1._internal = "secret"

    assert slots_equal(slot1, slot2) is True


def test_slots_equal_privateattributes() -> None:
    slot1 = SlotDefinition(name="height")
    slot2 = SlotDefinition(name="height")
    slot1._internal = "secret"
    slot2._internal = "different_secret"

    assert slots_equal(slot1, slot2) is True


def test_slots_equal_emptyslots() -> None:
    slot1 = SlotDefinition("field")
    slot2 = SlotDefinition("field")
    # To force the SlotDefinitions to be empty.
    del slot1.name
    del slot2.name
    assert slots_equal(slot1, slot2) is True


def test_slots_equal_explicitvsimplicitdefaults() -> None:
    s1 = SlotDefinition(name="age") # multivalued default is False
    s2 = SlotDefinition(name="age", multivalued=False)

    assert slots_equal(s1, s2) is (s1.__dict__ == s2.__dict__)


def test_slots_equal_mutationchangesresult() -> None:
    s1 = SlotDefinition(name="x")
    s2 = SlotDefinition(name="x")

    assert slots_equal(s1, s2)

    s1.description = "A slot"

    assert not slots_equal(s1, s2)

def test_slots_equal_dynamicattributes() -> None: 
    s1 = SlotDefinition(name="x") 
    s2 = SlotDefinition(name="x") 
    s1.new_field = "value" 
    
    assert not slots_equal(s1, s2)

    s2.new_field = "value" 
    
    assert slots_equal(s1, s2)

@pytest.fixture
def set_prefixes() -> dict:
    return {"ex": {"prefix_prefix": "ex", "prefix_reference": "http://example.org/"}}

# Unit tests _build_slot_index
def test_build_slot_index_emptyschema(make_schemaview: Callable[..., SchemaView]) -> None:
    sv = make_schemaview()
    slot_index, class_index = _build_slot_index(sv)

    assert slot_index == {}
    assert class_index == {}


def test_build_slot_index_globalslots(make_schemaview: Callable[..., SchemaView], set_prefixes: dict) -> None:
    # Documents what happends when global slots are not used in a class
    slots = {"s1": SlotDefinition(name="s1", slot_uri="ex:s1"),}
    classes = {"C": ClassDefinition(name="C", attributes=None)}

    sv = make_schemaview(slots=slots, classes=classes, prefixes=set_prefixes)
    slot_index, class_index = _build_slot_index(sv)

    assert slot_index == {} # Global slots are not collected when not used by a class
    assert "C" in class_index

def test_build_slot_index_noclasses(make_schemaview: Callable[..., SchemaView], set_prefixes: dict) -> None:
    # Documents what happends when global slots are not used in a class
    slots = {"s1": SlotDefinition(name="s1", slot_uri="ex:s1"),}

    sv = make_schemaview(slots=slots, prefixes=set_prefixes)
    slot_index, class_index = _build_slot_index(sv)

    assert slot_index == {}
    assert class_index == {}

def test_build_slot_index_classattributeslots(make_schemaview: Callable[..., SchemaView], set_prefixes: dict) -> None:
    classes = {
        "C": ClassDefinition(
            name="C",
            attributes={
                "a1": SlotDefinition(name="a1", slot_uri="ex:a1"),
                "a2": SlotDefinition(name="a2", slot_uri="ex:a2"),
            },
        )
    }

    sv = make_schemaview(classes=classes, prefixes=set_prefixes)
    slot_index, class_index = _build_slot_index(sv)

    assert len(slot_index) == 2
    assert "http://example.org/a1" in slot_index
    assert "http://example.org/a2" in slot_index
    assert "C" in class_index
    assert class_index["C"].name == "C"


def test_build_slot_index_overwriting(make_schemaview: Callable[..., SchemaView], set_prefixes: dict) -> None:
    # This test documents that global slot is normalized with class slot when the uri is the same
    # Metadata in the global slot is overwritten by the class attribute slot
    # This is done by linkML, not by _build_slot_index.
    slots = {
        "s1": SlotDefinition(
            name="s1",
            slot_uri="ex:prop",
            range="string",
            required=True,
            multivalued=False,
        )
    }

    classes = {
        "C": ClassDefinition(
            name="C",
            attributes={
                "a1": SlotDefinition(
                    name="a1",
                    slot_uri="ex:prop",
                    range="integer",
                    required=False,
                    multivalued=True,
                )
            }
        )
    }

    sv = make_schemaview(prefixes=set_prefixes, slots=slots, classes=classes)
    slot_index, class_index = _build_slot_index(sv)

    expanded = "http://example.org/prop"

    assert expanded in slot_index
    slot = slot_index[expanded]

    assert slot.range == "integer"
    assert slot.required is False
    assert slot.multivalued is True
    assert "C" in class_index


def test_build_slot_index_classwithoutattributes(make_schemaview: Callable[..., SchemaView]) -> None:
    classes = {
        "C1": ClassDefinition(name="C1", attributes=None),
        "C2": ClassDefinition(name="C2", attributes={}),  # empty dict
    }

    sv = make_schemaview(classes=classes)
    slot_index, class_index = _build_slot_index(sv)

    assert slot_index == {}
    assert set(class_index.keys()) == {"C1", "C2"}
    # The classes are stored with standards
    assert class_index["C1"] == ClassDefinition(name='C1', from_schema='test')
    assert class_index["C2"] == ClassDefinition(name='C2', from_schema='test')

def test_build_slot_index_duplicatedclassslots(make_schemaview: Callable[..., SchemaView], set_prefixes: dict, caplog: LogCaptureFixture) -> None:
    # Documents what happends if two slots have the same slot_uri with different metadata.
    classes = {
        "C": ClassDefinition(name="C", attributes={"a1": SlotDefinition(name="a1", slot_uri="ex:a1", range="string")}),
        "B": ClassDefinition(name="B", attributes={"b1": SlotDefinition(name="b1", slot_uri="ex:a1", range="integer")})
    }

    sv = make_schemaview(classes=classes, prefixes=set_prefixes)
    slot_index, class_index = _build_slot_index(sv)

    slot = slot_index["http://example.org/a1"]
    assert len(slot_index) == 1
    assert "B" in class_index
    assert "C" in class_index
    assert slot.name == "b1"
    assert slot.range == "integer"
    assert caplog.records[0].message == "Slot for URI 'http://example.org/a1' is overwritten by class slot 'b1'."


def test_build_slot_index_unexpandablecuries(make_schemaview: Callable[..., SchemaView], set_prefixes: dict, caplog: LogCaptureFixture) -> None:
    # Documents what happends to curies that cannot be expanded
    classes = {
        "C": ClassDefinition(name="C", attributes={"a1": SlotDefinition(name="a1", slot_uri="ex:a1")}),
        "B": ClassDefinition(name="B", attributes={"b1": SlotDefinition(name="b1", slot_uri="foo:bar")})
    }

    sv = make_schemaview(classes=classes, prefixes=set_prefixes)
    slot_index, class_index = _build_slot_index(sv)

    assert len(slot_index) == 2
    assert "http://example.org/a1" in slot_index
    assert "foo:bar" in slot_index
    assert slot_index["foo:bar"].name == "b1"
    
@pytest.mark.parametrize(
        "prefix, map, expecteduri",
        [
            pytest.param("ex", {}, "ex:a1", id="No prefix map"),
            pytest.param("schema.org", {"schema.org": {"prefix_prefix": "schema.org", "prefix_reference": "http://fullschema.org/"}}, "http://fullschema.org/a1", id="Prefix with ."),
        ]
)
def test_build_slot_index_unusualprefixes(prefix: str, map: dict, expecteduri:str, make_schemaview: Callable[..., SchemaView]) -> None:
    classes = {
        "C": ClassDefinition(name="C", attributes={"a1": SlotDefinition(name="a1", slot_uri=f"{prefix}:a1")}),
    }
    sv = make_schemaview(classes=classes, prefixes=map)
    slot_index, class_index = _build_slot_index(sv)

    assert "C" in class_index
    assert slot_index[expecteduri].name == "a1"


def test_build_slot_index_identicalslots(make_schemaview: Callable[..., SchemaView], set_prefixes: dict, caplog: LogCaptureFixture) -> None:
    # Documents that when there are identical slots in different classes, only the first is kept
    classes = {
        "C": ClassDefinition(
            name="C",
            attributes={"a1": SlotDefinition(name="a1", slot_uri="ex:a1", range="string")}
        ),
        "D": ClassDefinition(
            name="D",
            attributes={"a1": SlotDefinition(name="a1", slot_uri="ex:a1", range="string")}
        ),
    }

    sv = make_schemaview(classes=classes, prefixes=set_prefixes)

    with caplog.at_level(logging.WARNING):
        slot_index, class_index = _build_slot_index(sv)

    assert len(slot_index) == 1
    assert "http://example.org/a1" in slot_index
    assert caplog.text == ""  # No warnings because slots are identical


def test_build_slot_index_noslot_uri(make_schemaview: Callable[..., SchemaView]) -> None:
    classes = {
        "C": ClassDefinition(name="C", attributes={
            "a1": SlotDefinition(name="a1", slot_uri=None),
            "a2": SlotDefinition(name="a2"),  # slot_uri missing
            },
        )
    }

    sv = make_schemaview(classes=classes)
    slot_index, class_index = _build_slot_index(sv)

    assert slot_index == {}
    assert "C" in class_index


def test_build_slot_index_multipleoverwrites(make_schemaview: Callable[..., SchemaView], set_prefixes: dict, caplog: LogCaptureFixture) -> None:
    classes = {
        "C": ClassDefinition(name="C", attributes={"a1": SlotDefinition(name="a1", slot_uri="ex:a1", range="string")}),
        "D": ClassDefinition(name="D", attributes={"d1": SlotDefinition(name="d1", slot_uri="ex:a1", range="integer")}),
        "E": ClassDefinition(name="E", attributes={"e1": SlotDefinition(name="e1", slot_uri="ex:a1", range="float")}),
    }

    sv = make_schemaview(classes=classes, prefixes=set_prefixes)

    slot_index, class_index = _build_slot_index(sv)

    assert slot_index["http://example.org/a1"].range == "float"
    assert len([r for r in caplog.records if "overwritten" in r.message]) == 2


@patch("cim_plugin.cimxml.slots_equal")
def test_build_slot_index_callinghelperfunction(mock_patch: MagicMock, make_schemaview: Callable[..., SchemaView], set_prefixes: dict, caplog: LogCaptureFixture) -> None:
    classes = {
        "C": ClassDefinition(name="C", attributes={"a1": SlotDefinition(name="a1", slot_uri="ex:a1", range="string")}),
        "D": ClassDefinition(name="D", attributes={"d1": SlotDefinition(name="d1", slot_uri="ex:a1", range="integer")}),
        "E": ClassDefinition(name="E", attributes={"d1": SlotDefinition(name="d1", slot_uri="ex:a1", range="integer")}),
    }
    mock_patch.side_effect = [False, True]

    sv = make_schemaview(classes=classes, prefixes=set_prefixes)    
    slot_index, class_index = _build_slot_index(sv)

    calls = [
        call(SlotDefinition(name='a1', from_schema='test', slot_uri='ex:a1', range='string'), 
             SlotDefinition(name='d1', from_schema='test', slot_uri='ex:a1', range='integer')),
        call(SlotDefinition(name='d1', from_schema='test', slot_uri='ex:a1', range='integer'), 
             SlotDefinition(name='d1', from_schema='test', slot_uri='ex:a1', range='integer'))
            ]
    mock_patch.assert_has_calls(calls)
    assert slot_index["http://example.org/a1"].range == "integer"
    assert len([r for r in caplog.records if "overwritten" in r.message]) == 1
    class_list = ["C", "D", "E"]
    assert all(c in class_index for c in class_list)
    

# Unit tests _resolve_type

@pytest.mark.parametrize(
    "type_name, base, expected",
    [
        pytest.param("string", None, "xsd:string", id="Primitive type: string"),
        pytest.param("boolean", None, "xsd:boolean", id="Primitive type: boolean"),
        pytest.param("integer", None, "xsd:integer", id="Primitive type: integer"),
        pytest.param("MyString", "string", "xsd:string", id="Custom type: string"),
        pytest.param("FlagType", "boolean", "xsd:boolean", id="Custom type: boolean"),
        pytest.param("NonExistent", None, None, id="Type not found, returns None"),
        pytest.param("String", None, None, id="Case sensitivity, returns None")
    ]
)
def test_resolve_type_basic(make_schemaview: Callable[..., SchemaView], type_name: str, base: str|None, expected: str) -> None:
    if not base:
        sv = make_schemaview()
    else:
        sv = make_schemaview(types={type_name: TypeDefinition(name=type_name, base=base)})
    
    assert _resolve_type(sv, type_name) == expected

@pytest.mark.parametrize(
        "type_name, types, expected",
        [
            pytest.param(
                "MyType", 
                {"MyType": TypeDefinition(name="MyType", uri="http://example.org/MyType", base="string")}, 
                "http://example.org/MyType", 
                id="Uri preferred over base"),
            pytest.param(
                "MyType", 
                {"MyType": TypeDefinition(name="MyType", uri="http://example.org/MyType")}, 
                "http://example.org/MyType", 
                id="Uri, but no base"),
            pytest.param(
                "A",
                {"A": TypeDefinition(name="A", base="B"), "B": TypeDefinition(name="B", base="C"), "C": TypeDefinition(name="C", base="string")},
                "xsd:string",
                id="Recursive chain"
            ),
            pytest.param("Foo", {"Foo": TypeDefinition(name="Foo")}, "Foo", id="No base, no uri"),
            pytest.param(
                "Child", 
                {"Base": TypeDefinition(name="Base", uri="http://example.org/Base"), "Child": TypeDefinition(name="Child", uri="http://example.org/Child", base="Base")},
                "http://example.org/Child",
                id="Uri overrides base uri"
            ),
            pytest.param("string", {"string": TypeDefinition(name="string", uri="http://example.org/CustomString")}, "http://example.org/CustomString", id="Override of primitive")
        ]
)
def test_resolve_type_various(make_schemaview: Callable[..., SchemaView], type_name: str, types: dict, expected: str) -> None:
    sv = make_schemaview(types=types)
    assert _resolve_type(sv, type_name=type_name) == expected


def test_resolve_type_circularinheritance(make_schemaview: Callable[..., SchemaView]) -> None:
    sv = make_schemaview(
        types={
            "A": TypeDefinition(name="A", base="B"),
            "B": TypeDefinition(name="B", base="A"),
        }
    )
    with pytest.raises(RecursionError):
        _resolve_type(sv, "A")


# Unit tests resolve_datatype_from_slot

@pytest.mark.parametrize(
    "slot_def, expected",
    [
        pytest.param(SlotDefinition(name="s1", range="string"), "xsd:string", id="Primitive datatype"),
        pytest.param(SlotDefinition(name="s2", range="MyInt"), "xsd:integer", id="Custom datatype"),
    ]
)
def test_resolve_datatype_from_slot_direct(make_schemaview: Callable[..., SchemaView], slot_def: SlotDefinition, expected: str) -> None:
    sv = make_schemaview(
        types={
            "MyInt": TypeDefinition(name="MyInt", base="integer")
        },
        slots={slot_def.name: slot_def}
    )
    slot = sv.get_slot(slot_def.name)
    assert resolve_datatype_from_slot(sv, slot) == expected


def test_resolve_datatype_from_slot_enumwithmeaning(make_schemaview: Callable[..., SchemaView]) -> None:
    enums = {"FuelType": EnumDefinition(
        name="FuelType", 
        enum_uri="http://iec.ch/TC57/CIM100#FuelType", 
        permissible_values={ "coal": PermissibleValue(text="coal", meaning="cim:FuelType.coal") } 
    )}
    slots = {"fuel": SlotDefinition(name="fuel", range="FuelType")}
    sv = make_schemaview(slots=slots, enums=enums)
    slot = sv.get_slot("fuel")
    assert resolve_datatype_from_slot(sv, slot) == "xsd:string"


def test_resolve_datatype_from_slot_enumnomeaning(make_schemaview: Callable[..., SchemaView]) -> None:
    enums = {"ColorEnum": EnumDefinition(
        name="ColorEnum", 
        permissible_values={ "red": PermissibleValue(text="red"), "blue": PermissibleValue(text="blue")}
    )}
    slots = {"color": SlotDefinition(name="color", range="ColorEnum")}
    sv = make_schemaview(slots=slots, enums=enums)
    slot = sv.get_slot("color")
    assert resolve_datatype_from_slot(sv, slot) == "xsd:string"


def test_resolve_datatype_from_slot_mixedpermissibles(make_schemaview: Callable[..., SchemaView], caplog: LogCaptureFixture) -> None:
    enums = {"MixedEnum": EnumDefinition(
        name="MixedEnum", 
        permissible_values={"a": PermissibleValue(text="a", meaning="ex:A"), "b": PermissibleValue(text="b")}
    )}
    slots = {"s1": SlotDefinition(name="s1", range="MixedEnum")}
    sv = make_schemaview(slots=slots, enums=enums)
    slot = sv.get_slot("s1")
    result = resolve_datatype_from_slot(sv, slot) 
    assert result == "xsd:string"
    assert "Literal encountered for enum MixedEnum with meaning." in caplog.text

def test_resolve_datatype_from_slot_enumnopermissibles(make_schemaview: Callable[..., SchemaView]) -> None:
    enums = {"ColorEnum": EnumDefinition(name="ColorEnum")}
    slots = {"color": SlotDefinition(name="color", range="ColorEnum")}
    sv = make_schemaview(slots=slots, enums=enums)
    slot = sv.get_slot("color")
    assert resolve_datatype_from_slot(sv, slot) == "xsd:string"


def test_resolve_datatype_from_slot_unknowntype(make_schemaview: Callable[..., SchemaView]) -> None:
    sv = make_schemaview(
        slots={
            "mystery": SlotDefinition(name="mystery", range="UnknownType")
        }
    )
    slot = sv.get_slot("mystery")
    assert resolve_datatype_from_slot(sv, slot) == "UnknownType"


def test_resolve_datatype_from_slot_norange(make_schemaview: Callable[..., SchemaView]) -> None:
    sv = make_schemaview(slots={"mystery": SlotDefinition(name="mystery")})
    slot = sv.get_slot("mystery")
    assert resolve_datatype_from_slot(sv, slot) == None


@patch("cim_plugin.cimxml._resolve_type")
def test_resolve_datatype_from_slot_funccalledonce(mock_resolve: MagicMock, make_schemaview: Callable[..., SchemaView]) -> None:
    mock_resolve.return_value = "xsd:integer"
    sv = make_schemaview(
        slots={"s1": SlotDefinition(name="s1", range="integer")}
    )
    slot = sv.get_slot("s1")
    assert resolve_datatype_from_slot(sv, slot) == "xsd:integer"
    mock_resolve.assert_called_once_with(sv, "integer")
    assert mock_resolve.call_count == 1


@patch("cim_plugin.cimxml._resolve_type")
def test_resolve_datatype_from_slot_funcerror(mock_resolve: MagicMock, make_schemaview: Callable[..., SchemaView]) -> None:
    mock_resolve.side_effect = RecursionError
    sv = make_schemaview(
        slots={"s1": SlotDefinition(name="s1", range="integer")}
    )
    slot = sv.get_slot("s1")

    with pytest.raises(RecursionError):
        resolve_datatype_from_slot(sv, slot)


def test_resolve_datatype_from_slot_classrange(make_schemaview: Callable[..., SchemaView], caplog: LogCaptureFixture) -> None:
    classes = {"Person": ClassDefinition(name="Person")}
    slots = {"friend": SlotDefinition(name="friend", range="Person")}
    sv = make_schemaview(slots=slots, classes=classes)

    slot = sv.get_slot("friend")

    result = resolve_datatype_from_slot(sv, slot)
    assert result is None
    assert "slot.range 'Person' is a class" in caplog.text


# Unit tests create_typed_literal  # adjust import

@pytest.mark.parametrize(
    "value, datatype, exp_val, exp_type",
    [
        pytest.param("42", "xsd:integer", 42, str(XSD.integer), id="Integer"),
        pytest.param("3.14", "xsd:float", 3.14, str(XSD.float), id="Float"),
        pytest.param("true", "xsd:boolean", True, str(XSD.boolean), id="Boolean True"),
        pytest.param("1", "xsd:boolean", True, str(XSD.boolean), id="Numeric boolean True"),
        pytest.param("false", "xsd:boolean", False, str(XSD.boolean), id="Boolean False"),
        pytest.param("2024-01-01", "xsd:date", date(2024, 1, 1), str(XSD.date), id="Date as string"),
        pytest.param(date(2024, 1, 1), "xsd:date", date(2024, 1, 1), str(XSD.date), id="Date as datetime.date"),
        pytest.param("2024-01-01T12:30:00", "xsd:dateTime", datetime(2024, 1, 1, 12, 30), str(XSD.dateTime), id="Datetime as string"),
        pytest.param(datetime(2024, 1, 1, 12, 30), "xsd:dateTime", datetime(2024, 1, 1, 12, 30), str(XSD.dateTime), id="Datetime as datetime.datetime"),
        pytest.param("42", str(XSD.integer), 42, str(XSD.integer), id="Already expanded uri"),
        pytest.param("hello", "", "hello", 'None', id="Datatype empty string"),
        pytest.param("hello", None, "hello", "None", id="Datatype None"),
        pytest.param(Literal("42", datatype=XSD.integer), "xsd:integer", 42, str(XSD.integer), id="Value already Literal"),
        pytest.param("42", URIRef(str(XSD.integer)), 42, str(XSD.integer), id="Datatype as URIRef"),
        # For all below: Value set to None by rdflib because datatype is unavailable
        pytest.param("42", "XSD:INTEGER", None, "http://www.w3.org/2001/XMLSchema#INTEGER", id="Uppercase CURIE"),  # Passes because linkML normalizes prefixes during lookup in internal maps
        pytest.param("hello", "xsd:unknown", None, "http://www.w3.org/2001/XMLSchema#unknown", id="Unknown datatype"),
        pytest.param("hello", "http://example.org/CustomType", None, "http://example.org/CustomType", id="Custom URI datatype"),
        pytest.param("hello", "foo:bar", None, "foo:bar", id="Unknown prefix CURIE"),
        pytest.param("42", " xsd:integer ", None, " xsd:integer ", id="Datatype with whitespace"),  
    ]
)
def test_create_typed_literal_basic(make_schemaview: Callable[..., SchemaView], value: Any, datatype: str|None, exp_val: Any, exp_type: str|None) -> None:
    sv = make_schemaview(prefixes={"xsd": "http://www.w3.org/2001/XMLSchema#"})
    
    # Pylance silenced to test incorrect datatypes
    lit = create_typed_literal(value, datatype, sv) # type: ignore
    
    assert isinstance(lit, Literal)
    assert lit.value == exp_val
    assert str(lit.datatype) == exp_type

@pytest.mark.parametrize(
    "value, datatype, exp_val",
    [
        pytest.param("abc", "xsd:integer", "abc", id="Invalid integer lexical form"),
        pytest.param("not-a-float", "xsd:float", "not-a-float", id="Invalid float lexical form"),
        pytest.param("yes", "xsd:boolean", "yes", id="Invalid boolean lexical form"),
    ]
)
def test_create_typed_literal_errors(make_schemaview: Callable[..., SchemaView], value: Any, datatype: str|None, exp_val: str, caplog: LogCaptureFixture) -> None:
    sv = make_schemaview(prefixes={"xsd": "http://www.w3.org/2001/XMLSchema#"})

    with pytest.raises(LiteralCastingError):
        # Pylance silenced to test incorrect datatypes
        create_typed_literal(value, datatype, sv)   # type: ignore

    # lit = create_typed_literal(value, datatype, sv)
    # assert isinstance(lit, Literal)
    # assert lit.value == exp_val
    # assert str(lit.datatype) == 'None'
    # assert "Failed to cast" in caplog.text


if __name__ == "__main__":
    pytest.main()