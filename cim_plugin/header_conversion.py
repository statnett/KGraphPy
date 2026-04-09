"""Conversion between DCAT Dataset header and 552 MD FullModel header."""

from rdflib import Node, Literal, URIRef
from rdflib.namespace import DCTERMS, XSD
from cim_plugin.namespaces import MD, DCAT_EXT
from typing import Any

# Conversion mapping
TO_FULLMODEL: dict[URIRef, dict[str, Any]] = {
    DCTERMS.issued: {"pred": MD.Model.created, "object_type": "literal"},
    DCAT_EXT.startDate: {"pred": MD.Model.scenarioTime, "object_type": "literal"},
    DCTERMS.description: {"pred": MD.Model.description, "object_type": "literal"},
    DCAT_EXT.isVersionOf: {"pred": MD.Model.modelingAuthoritySet, "object_type": "uri"},
    DCTERMS.conformsTo: {"pred": MD.Model.profile, "object_type": "uri"},
    DCAT_EXT.version: {"pred": MD.Model.version, "object_type": "literal"},
    DCTERMS.references: {"pred": MD.Model.DependentOn, "object_type": "uri"},
    DCTERMS.requires: {"pred": MD.Model.DependentOn, "object_type": "uri"},
    DCTERMS.replaces: {"pred": MD.Model.Supersedes, "object_type": "uri"},
}
TO_DCAT: dict[URIRef, dict[str, Any]] = {
    MD.Model.created: {"pred": DCTERMS.issued, "object_type": "literal", "datatype": XSD.dateTime},
    MD.Model.scenarioTime: {"pred": DCAT_EXT.startDate, "object_type": "literal", "datatype": XSD.dateTime},
    MD.Model.description: {"pred": DCTERMS.description, "object_type": "literal", "datatype": XSD.string},
    MD.Model.modelingAuthoritySet: {"pred": DCAT_EXT.isVersionOf, "object_type": "uri"},
    MD.Model.profile: {"pred": DCTERMS.conformsTo, "object_type": "uri"},
    MD.Model.version: {"pred": DCAT_EXT.version, "object_type": "literal", "datatype": XSD.string},
    MD.Model.DependentOn: {"pred": DCTERMS.requires, "object_type": "uri"}, # Could also be DCTERMS.references. requires is preferred.
    MD.Model.Supersedes: {"pred": DCTERMS.replaces, "object_type": "uri"},
}


def convert_triple(triple: tuple[Node, Node, Node], target_format: str="md_fullmodel") -> tuple[Node, Node, Node]|None:
    """Convert a triple from one header format to another.

    Bidirectional conversion between DCAT Dataset header and 552 MD FullModel header. 
    If no conversion is possible for the given triple, None is returned.
    
    Parameters:
        triple (tuple[Node, Node, Node]): The triple to be converted.
        target_format (str): The target format to convert to. Supported values are "md_fullmodel" and "dcat_dataset". Default is "md_fullmodel".
    
    Raises:
        AssertionError: If the predicate of the triple is not a URIRef (should not happen since predicates should always be URIRefs).
        ValueError: If the target_format is not supported.

    Returns:
        tuple[Node, Node, Node]|None: The converted triple. If the predicate is not in the mapping for the target format, None is returned.
    """
    s, p, o = triple
    
    assert isinstance(p, URIRef)    # p should always be a URIRef. This is for the type checker, not an actual check we need to do at runtime.

    if target_format == "md_fullmodel":
        mapping = TO_FULLMODEL
    elif target_format == "dcat_dataset":
        mapping = TO_DCAT
    else:
        raise ValueError(f"Unknown target format: {target_format}")
    
    if p in mapping:
        new_pred = mapping[p]["pred"]
        object_type = mapping[p]["object_type"]
        o = convert_object(o, object_type, mapping[p].get("datatype"))
        return (s, new_pred, o)

    return None


def convert_object(o: Node, object_type: str, datatype: URIRef | None = None) -> Node:
    """Convert an object to given object_type with given datatype.
    
    Parameters:
        o (Node): The object to be converted.
        object_type (str): The object_type to convert the object to. Supported values are "literal" and "uri".
        datatype (URIRef | None): The datatype to be used if the object is a literal. Default is None.

    Returns:
        Node: The converted object. The original object is returned unchanged if it cannot be converted to the target type.
    """
    if object_type == "literal":
        if isinstance(o, URIRef):
            return Literal(o, datatype=datatype)
        elif isinstance(o, Literal) and datatype:
            return Literal(o.value, datatype=datatype)
        
    if object_type == "uri":
        if isinstance(o, Literal):
            return URIRef(o.value)
        return o
    
    return o


if __name__ == "__main__":
    print("Functions for converting between DCAT Dataset and MD FullModel headers.")