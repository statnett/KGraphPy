import pytest
from unittest.mock import patch, MagicMock, call
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model.meta import SlotDefinition, ClassDefinition
from typing import Callable
from tests.fixtures import make_schemaview, set_prefixes
from cim_plugin.enriching import slots_equal, _build_slot_index

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


# Unit tests _build_slot_index
def test_build_slot_index_emptyschema(make_schemaview: Callable[..., SchemaView]) -> None:
    sv = make_schemaview()
    slot_index = _build_slot_index(sv)

    assert slot_index == {}


def test_build_slot_index_globalslots(make_schemaview: Callable[..., SchemaView], set_prefixes: dict) -> None:
    # Documents what happends when global slots are not used in a class
    slots = {"s1": SlotDefinition(name="s1", slot_uri="ex:s1"),}
    classes = {"C": ClassDefinition(name="C", attributes=None)}

    sv = make_schemaview(slots=slots, classes=classes, prefixes=set_prefixes)
    slot_index = _build_slot_index(sv)

    assert slot_index == {} # Global slots are not collected when not used by a class


def test_build_slot_index_noclasses(make_schemaview: Callable[..., SchemaView], set_prefixes: dict) -> None:
    # Documents what happends when global slots are not used in a class
    slots = {"s1": SlotDefinition(name="s1", slot_uri="ex:s1"),}

    sv = make_schemaview(slots=slots, prefixes=set_prefixes)
    slot_index = _build_slot_index(sv)

    assert slot_index == {}

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
    slot_index = _build_slot_index(sv)

    assert len(slot_index) == 2
    assert "http://example.org/a1" in slot_index
    assert "http://example.org/a2" in slot_index
    

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
    slot_index = _build_slot_index(sv)

    expanded = "http://example.org/prop"

    assert expanded in slot_index
    slot = slot_index[expanded]

    assert slot.range == "integer"
    assert slot.required is False
    assert slot.multivalued is True
    

def test_build_slot_index_classwithoutattributes(make_schemaview: Callable[..., SchemaView]) -> None:
    classes = {
        "C1": ClassDefinition(name="C1", attributes=None),
        "C2": ClassDefinition(name="C2", attributes={}),  # empty dict
    }

    sv = make_schemaview(classes=classes)
    slot_index = _build_slot_index(sv)

    assert slot_index == {}

def test_build_slot_index_duplicatedclassslots(make_schemaview: Callable[..., SchemaView], set_prefixes: dict, caplog: pytest.LogCaptureFixture) -> None:
    # Documents what happends if two slots have the same slot_uri with different metadata.
    classes = {
        "C": ClassDefinition(name="C", attributes={"a1": SlotDefinition(name="a1", slot_uri="ex:a1", range="string")}),
        "B": ClassDefinition(name="B", attributes={"b1": SlotDefinition(name="b1", slot_uri="ex:a1", range="integer")})
    }

    sv = make_schemaview(classes=classes, prefixes=set_prefixes)
    slot_index = _build_slot_index(sv)

    slot = slot_index["http://example.org/a1"]
    assert len(slot_index) == 1
    assert slot.name == "b1"
    assert slot.range == "integer"
    assert caplog.records[0].message == "Slot for URI 'http://example.org/a1' is overwritten by class slot 'b1'."


def test_build_slot_index_unexpandablecuries(make_schemaview: Callable[..., SchemaView], set_prefixes: dict, caplog: pytest.LogCaptureFixture) -> None:
    # Documents what happends to curies that cannot be expanded
    classes = {
        "C": ClassDefinition(name="C", attributes={"a1": SlotDefinition(name="a1", slot_uri="ex:a1")}),
        "B": ClassDefinition(name="B", attributes={"b1": SlotDefinition(name="b1", slot_uri="foo:bar")})
    }

    sv = make_schemaview(classes=classes, prefixes=set_prefixes)
    slot_index = _build_slot_index(sv)

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
    slot_index = _build_slot_index(sv)

    assert slot_index[expecteduri].name == "a1"


def test_build_slot_index_identicalslots(make_schemaview: Callable[..., SchemaView], set_prefixes: dict, caplog: pytest.LogCaptureFixture) -> None:
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

    slot_index = _build_slot_index(sv)

    assert len(slot_index) == 1
    assert "http://example.org/a1" in slot_index
    assert caplog.text == ""  # No warnings because slots are identical


def test_build_slot_index_nosloturi(make_schemaview: Callable[..., SchemaView]) -> None:
    classes = {
        "C": ClassDefinition(name="C", attributes={
            "a1": SlotDefinition(name="a1", slot_uri=None),
            "a2": SlotDefinition(name="a2"),  # slot_uri missing
            },
        )
    }

    sv = make_schemaview(classes=classes)
    slot_index = _build_slot_index(sv)

    assert slot_index == {}


def test_build_slot_index_multipleoverwrites(make_schemaview: Callable[..., SchemaView], set_prefixes: dict, caplog: pytest.LogCaptureFixture) -> None:
    classes = {
        "C": ClassDefinition(name="C", attributes={"a1": SlotDefinition(name="a1", slot_uri="ex:a1", range="string")}),
        "D": ClassDefinition(name="D", attributes={"d1": SlotDefinition(name="d1", slot_uri="ex:a1", range="integer")}),
        "E": ClassDefinition(name="E", attributes={"e1": SlotDefinition(name="e1", slot_uri="ex:a1", range="float")}),
    }

    sv = make_schemaview(classes=classes, prefixes=set_prefixes)

    slot_index = _build_slot_index(sv)

    assert slot_index["http://example.org/a1"].range == "float"
    assert len([r for r in caplog.records if "overwritten" in r.message]) == 2



@patch("cim_plugin.enriching.slots_equal")
def test_build_slot_index_callinghelperfunction(mock_patch: MagicMock, make_schemaview: Callable[..., SchemaView], set_prefixes: dict, caplog: pytest.LogCaptureFixture) -> None:
    classes = {
        "C": ClassDefinition(name="C", attributes={"a1": SlotDefinition(name="a1", slot_uri="ex:a1", range="string")}),
        "D": ClassDefinition(name="D", attributes={"d1": SlotDefinition(name="d1", slot_uri="ex:a1", range="integer")}),
        "E": ClassDefinition(name="E", attributes={"d1": SlotDefinition(name="d1", slot_uri="ex:a1", range="integer")}),
    }
    mock_patch.side_effect = [False, True]

    sv = make_schemaview(classes=classes, prefixes=set_prefixes)    
    slot_index = _build_slot_index(sv)

    calls = [
        call(SlotDefinition(name='a1', from_schema='test', slot_uri='ex:a1', range='string'), 
             SlotDefinition(name='d1', from_schema='test', slot_uri='ex:a1', range='integer')),
        call(SlotDefinition(name='d1', from_schema='test', slot_uri='ex:a1', range='integer'), 
             SlotDefinition(name='d1', from_schema='test', slot_uri='ex:a1', range='integer'))
            ]
    
    assert mock_patch.call_count == 2
    mock_patch.assert_has_calls(calls)
    assert slot_index["http://example.org/a1"].range == "integer"
    assert len([r for r in caplog.records if "overwritten" in r.message]) == 1
    

if __name__ == "__main__":
    pytest.main()