from linkml_runtime.utils.schemaview import SchemaView, SlotDefinition
from typing import cast
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

if __name__ == "__main__":
    print("Functions for enriching cim graphs.")