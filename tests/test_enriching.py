import pytest
from unittest.mock import patch, MagicMock, call
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model.meta import SlotDefinition, ClassDefinition, TypeDefinition, EnumDefinition, PermissibleValue
from rdflib import Literal, URIRef
from rdflib.namespace import XSD
from datetime import date, datetime
from typing import Callable, Any
from tests.fixtures import make_schemaview, set_prefixes
from cim_plugin.exceptions import LiteralCastingError
from cim_plugin.enriching import (
    slots_equal, 
    _build_slot_index, 
    _resolve_type, 
    resolve_datatype_from_slot, 
    create_typed_literal, 
    cast_bool, 
    cast_float, 
    _parse_slash_date, 
    cast_datetime_utc
)

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


def test_resolve_datatype_from_slot_mixedpermissibles(make_schemaview: Callable[..., SchemaView], caplog: pytest.LogCaptureFixture) -> None:
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


@patch("cim_plugin.enriching._resolve_type")
def test_resolve_datatype_from_slot_funccalledonce(mock_resolve: MagicMock, make_schemaview: Callable[..., SchemaView]) -> None:
    mock_resolve.return_value = "xsd:integer"
    sv = make_schemaview(
        slots={"s1": SlotDefinition(name="s1", range="integer")}
    )
    slot = sv.get_slot("s1")
    assert resolve_datatype_from_slot(sv, slot) == "xsd:integer"
    mock_resolve.assert_called_once_with(sv, "integer")
    assert mock_resolve.call_count == 1


@patch("cim_plugin.enriching._resolve_type")
def test_resolve_datatype_from_slot_funcerror(mock_resolve: MagicMock, make_schemaview: Callable[..., SchemaView]) -> None:
    mock_resolve.side_effect = RecursionError
    sv = make_schemaview(
        slots={"s1": SlotDefinition(name="s1", range="integer")}
    )
    slot = sv.get_slot("s1")

    with pytest.raises(RecursionError):
        resolve_datatype_from_slot(sv, slot)


def test_resolve_datatype_from_slot_classrange(make_schemaview: Callable[..., SchemaView], caplog: pytest.LogCaptureFixture) -> None:
    classes = {"Person": ClassDefinition(name="Person")}
    slots = {"friend": SlotDefinition(name="friend", range="Person")}
    sv = make_schemaview(slots=slots, classes=classes)

    slot = sv.get_slot("friend")

    result = resolve_datatype_from_slot(sv, slot)
    assert result is None
    assert "slot.range 'Person' is a class" in caplog.text

# Unit tests parse_slash_date
@pytest.mark.parametrize(
    "input, output, raises",
    [
        pytest.param("01/13/2024", datetime(2024, 1, 13), "", id="American valid date"),
        pytest.param("13/01/2024", datetime(2024, 1, 13), "", id="European valid date"),
        pytest.param("2024/01/13", None, "Date format not supported for date parsing or invalid.", id="Not a slash-separated date"),
        pytest.param("01/01/2024", None, "Ambiguous date literal: 01/01/2024", id="Ambiguous date raises error"),
        pytest.param("what/ever/here", None, "Date format not supported for date parsing or invalid.", id="Random text"),
        pytest.param("02/30/2024", None, "day is out of range for month", id="Invalid date - american"),
        pytest.param("30/02/2024", None, "day is out of range for month", id="Invalid date - european"),
        pytest.param("29/02/2024", datetime(2024, 2, 29), "", id="Leap year - european"),
        pytest.param("02/29/2024", datetime(2024, 2, 29), "", id="Leap year - american"),
        pytest.param("29/02/2023", None, "day is out of range for month", id="Leap year invalid"),
        pytest.param("1/13/2024", datetime(2024, 1, 13), "", id="American valid date - no zeros"),
        pytest.param("13/1/2024", datetime(2024, 1, 13), "", id="European valid date - no zeros"),
        pytest.param("1/1/2024", None, "Ambiguous date literal: 1/1/2024", id="Ambiguous date raises error - no zeros"),
        pytest.param(" 13/01/2024 ", None, "Date format not supported for date parsing or invalid.", id="Whitespaces"),
    ]
)
def test_parse_slash_date_various(input: str, output: datetime, raises: str) -> None:
    print(output)
    if raises:
        with pytest.raises(ValueError, match=raises):
            _parse_slash_date(input)
    else:
        assert _parse_slash_date(input) == output


# Unit tests cast_datetime_utc
@pytest.mark.parametrize(
    "input, output",
    [
        pytest.param(Literal("2024-01-01", datatype=XSD.dateTime), Literal("2024-01-01T00:00:00+00:00", datatype=XSD.dateTime), id="Date with datetime format"),
        pytest.param(Literal("2024-01-01T12:33:44+01:00", datatype=XSD.dateTime), Literal("2024-01-01T00:00:00+00:00", datatype=XSD.dateTime), id="Datetime with datetime forma"), # Content gets deleted except for date
        pytest.param(Literal("2024-01-01", datatype=XSD.date), Literal("2024-01-01T00:00:00+00:00", datatype=XSD.dateTime), id="Date with date format"),
        pytest.param(Literal("2024-01-01", datatype=XSD.string), Literal("2024-01-01T00:00:00+00:00", datatype=XSD.dateTime), id="Date with explicit string datatype"),
        pytest.param(Literal("2024-01-01"), Literal("2024-01-01T00:00:00+00:00", datatype=XSD.dateTime), id="%Y-%m-%d format"),
        pytest.param(Literal("20240101"), Literal("2024-01-01T00:00:00+00:00", datatype=XSD.dateTime), id="%Y%m%d format"),
        pytest.param(Literal("01.01.2024"), Literal("2024-01-01T00:00:00+00:00", datatype=XSD.dateTime), id="%d.%m.%Y format"),
        pytest.param(Literal("01-01-2024"), Literal("2024-01-01T00:00:00+00:00", datatype=XSD.dateTime), id="%d-%m-%Y format"),
        pytest.param(Literal("2024/01/01"), Literal("2024-01-01T00:00:00+00:00", datatype=XSD.dateTime), id="%Y/%m/%d format"),
        pytest.param(Literal("01/13/2024"), Literal("2024-01-13T00:00:00+00:00", datatype=XSD.dateTime), id="American slash format"),
        pytest.param(Literal("13/01/2024"), Literal("2024-01-13T00:00:00+00:00", datatype=XSD.dateTime), id="European slash format"),
        pytest.param(Literal(" 2024-01-01 "), Literal("2024-01-01T00:00:00+00:00", datatype=XSD.dateTime), id="Whitespace"),
        pytest.param(Literal("2024-02-29"), Literal("2024-02-29T00:00:00+00:00", datatype=XSD.dateTime), id="Leap year"),
    ]
)
def test_cast_datetime_utc_various(input: Literal, output: Literal) -> None:
    assert cast_datetime_utc(input) == output


@pytest.mark.parametrize(
    "input, raises",
    [
        pytest.param(Literal(20240101, datatype=XSD.integer), "Datatype cannot be cast to datetime: 20240101", id="Integer literal"),
        pytest.param(Literal("01/01/2024", datatype=XSD.string), "Ambiguous date literal: 01/01/2024", id="Ambiguous slash date"),
        # All objects not parsed by datetime.strptime is passed to _parse_slash_date, which raises "Date format not supported for date parsing or invalid." error
        pytest.param(Literal("not a date", datatype=XSD.string), "Date format not supported for date parsing or invalid.", id="Invalid date string"),
        pytest.param(Literal("2024-01-41"), "Date format not supported for date parsing or invalid.", id="Invalid date - day out of range"),
        pytest.param(Literal("2023-02-29"), "Date format not supported for date parsing or invalid.", id="Invalid date - leap year"),
    ]
)
def test_cast_datetime_utc_errors(input: Literal, raises: str) -> None:
    with pytest.raises(ValueError, match=raises):
        cast_datetime_utc(input)


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
        pytest.param("3,14", "xsd:float", 3.14, str(XSD.float), id="Float with comma as separator"),
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
        pytest.param("3,14.15", "xsd:float", "3,14.15", id="Invalid float with comma as separator"),
    ]
)
def test_create_typed_literal_errors(make_schemaview: Callable[..., SchemaView], value: Any, datatype: str|None, exp_val: str, caplog: pytest.LogCaptureFixture) -> None:
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