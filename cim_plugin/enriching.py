from linkml_runtime.utils.schemaview import SchemaView, SlotDefinition
from rdflib import Literal, URIRef
from rdflib.namespace import XSD
from cim_plugin.exceptions import LiteralCastingError
from typing import cast
from datetime import datetime, timezone, date
import logging

logger = logging.getLogger('cimxml_logger')

def slots_equal(slot1: SlotDefinition, slot2: SlotDefinition) -> bool:
    """Show whether two SlotDefinition objects contain the same keys and values.

    Internal attributes are ignored (starts with _).

    Parameters:
        slot1 (SlotDefinition): First slot.
        slot2 (SlotDefinition): Second slot.
    
    Returns:
        bool: True if slot1 and slot2 are the identical.

    """
    d1 = {k: v for k, v in slot1.__dict__.items() if not k.startswith("_")}
    d2 = {k: v for k, v in slot2.__dict__.items() if not k.startswith("_")}

    return d1 == d2


def _build_slot_index(schemaview: SchemaView) -> dict:
    """Build an index of slots from a SchemaView.
    
    Parameters:
        schemaview (SchemaView): The schemaview to build the index from.

    Returns:
        dict: The slot index.
    """
    slot_index: dict = {}

    for cls_name, cls in schemaview.all_classes().items():
        if not isinstance(cls.attributes, dict):
            continue

        for slot_name, slot in cls.attributes.items():
            slot = cast(SlotDefinition, slot)
            if not slot.slot_uri:
                continue

            expanded = schemaview.expand_curie(slot.slot_uri)

            if expanded not in slot_index:
                slot_index[expanded] = slot
                continue

            existing = slot_index[expanded]

            if slots_equal(existing, slot):
                continue

            logger.warning(f"Slot for URI '{expanded}' is overwritten by class slot '{slot_name}'.")
            slot_index[expanded] = slot
    
    return slot_index


def _resolve_type(schemaview: SchemaView, type_name: str) -> str|None:
    """Resolve a LinkML type name to its canonical datatype.
    
    Works for both declared types and built-in primitives.
    
    Parameters:
        schemaview (SchemaView): The SchemaView to get the datatype from.
        type_name (str): Name of type.

    Returns:
        str: The datatype, either as a uri or base name. Last resort returns type_name.
    """
    t = schemaview.get_type(type_name)
    if t is None:
        return None

    if t.uri:
        return t.uri

    if t.base:
        return _resolve_type(schemaview, t.base)

    return type_name


def resolve_datatype_from_slot(schemaview: SchemaView, slot: SlotDefinition) -> str|None:
    """Resolve primitive datatype by collecting range from linkML slots.
    
    If range is a custom type, the primitive type of the custom type will be returned.

    This function should only be used on predicates whos object are literals.
    If the object is a URI, the return could be erronous.

    Parameters:
        schemaview (SchemaView): The linkML SchemaView to collect the datatype from.
        slot (SlotDefinition): The slot which contains the datatype range.

    Returns:
        str: The datatype if found.
        None: If range is nonexistent or a class.
    """
    rng = slot.range
    if not rng:
        return None

    # Case 1: range is a declared type or primitive
    resolved = _resolve_type(schemaview, rng)
    if resolved:
        return resolved

    # Case 2: range is an enum
    if rng in schemaview.all_enums():
        enum = schemaview.get_enum(rng)

        # If any permissible value has a meaning (URI), then enum values are URIs
        has_meaning = any(
            pv.meaning for pv in (enum.permissible_values or {}).values()   # type: ignore
        )

        if has_meaning:
            # Enum values will appear as URIs, not literals
            logger.warning(f"Literal encountered for enum {rng} with meaning. Literal enums should not have meaning. Is object a URI?")

        return "xsd:string"
    
    # Case 3: range is a class → literals should never appear 
    if rng in schemaview.all_classes(): 
        logger.warning( f"slot.range '{rng}' is a class. Is object a URI?" ) 
        return None

    return rng


def cast_float(value: str) -> float:
    """Cast string value to float.

    Corrects the common error of using , as decimal point (3,14 -> 3.14).
    
    Parameters:
        value (str): The value to be cast.
    
    Raises:
        ValueError: If input is not possible to cast.
    """
    s = str(value).strip()

    if "," in s and "." not in s:
        s = s.replace(",", ".")

    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Invalid float: {value}")


def cast_bool(value: str) -> bool:
    """Cast string value to boolean.
    
    Parameters:
        value (str): The value to be cast.
    
    Raises:
        ValueError: If input does not match true or false.
    """
    s = str(value).lower()
    if s in ("true", "1"):
        return True
    if s in ("false", "0"):
        return False
    raise ValueError(f"Invalid boolean lexical form: {value}")


# Needs testing
def cast_datetime_utc(lit: Literal) -> Literal:
    value = lit.toPython()

    if isinstance(value, date):
        dt = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
        return Literal(dt, datatype=XSD.dateTime)

    if isinstance(value, str):
        parsed = datetime.strptime(value.strip(), "%Y-%m-%d").date()
        dt = datetime(parsed.year, parsed.month, parsed.day, tzinfo=timezone.utc)
        return Literal(dt, datatype=XSD.dateTime)
        # Will raise ValueError if not in correct format, which should be sent forward.
        
    return lit


CASTERS = {
    str(XSD.integer): int,
    str(XSD.float): cast_float,
    str(XSD.boolean): cast_bool,
    str(XSD.date): lambda v: v.isoformat() if hasattr(v, "isoformat") else str(v),
    str(XSD.dateTime): lambda v: v.isoformat() if hasattr(v, "isoformat") else str(v),  # Consider using the caster above here too.
}


def create_typed_literal(value: str, datatype_uri: str, schemaview: SchemaView) -> Literal:
    """Cast Literal to correct format based on datatype uri.
    
    Parameters:
        value (str): The value to be cast.
        datatype_uri (str): The datatype in format "prefix:datatype".
    
    Raises:
        LiteralCastingError: If the casting fails.

    Returns:
        Literal with the new value format and datatype set.
    """
    if datatype_uri and ":" in datatype_uri and not datatype_uri.startswith("http"):
        datatype_uri = schemaview.expand_curie(datatype_uri)

    caster = CASTERS.get(datatype_uri)
    if caster:
        try:
            value = caster(value)
        except (ValueError, TypeError):
            raise LiteralCastingError(f"{value}, {datatype_uri}")

    # RDFLib will set .value = None for unknown datatypes
    return Literal(value, datatype=URIRef(datatype_uri) if datatype_uri else None)


if __name__ == "__main__":
    print("Functions for enriching cim graphs.")