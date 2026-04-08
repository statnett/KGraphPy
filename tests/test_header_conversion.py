import pytest
from unittest.mock import patch, MagicMock
from rdflib import BNode, URIRef, Literal
from rdflib.namespace import XSD, DCTERMS
from cim_plugin.namespaces import MD, DCAT_EXT
from typing import Any
import copy

from cim_plugin.header_conversion import convert_triple, convert_object, TO_DCAT, TO_FULLMODEL

# Unit tests convert_triple
@patch("cim_plugin.header_conversion.convert_object")
def test_convert_triple_unknownformat(mock_convert: MagicMock) -> None:
    triple = (URIRef("http://example.com/subject"), URIRef("http://example.com/predicate"), Literal("object"))
    with pytest.raises(ValueError) as excinfo:
        convert_triple(triple, target_format="unknown")

    assert "Unknown target format: unknown" in str(excinfo.value)
    mock_convert.assert_not_called()


@patch("cim_plugin.header_conversion.convert_object")
def test_convert_triple_targetformatmismatch(mock_convert: MagicMock) -> None:
    triple = (URIRef("http://example.com/subject"), URIRef(MD.FullModel), Literal("object"))
    result = convert_triple(triple, target_format="md_fullmodel")

    mock_convert.assert_not_called()  # convert_object should not be called since the predicate is not in the mapping
    assert result == None  # The original triple should be returned unchanged


@pytest.mark.parametrize(
    "predicate_in, predicate_out, object_type",
    [
        pytest.param(DCTERMS.issued, MD.Model.created, "literal", id="DCTERMS.issued to MD.Model.created"),
        pytest.param(DCAT_EXT.startDate, MD.Model.scenarioTime, "literal", id="DCAT_CIM.startDate to MD.Model.scenarioTime"),
        pytest.param(DCTERMS.description, MD.Model.description, "literal", id="DCTERMS.description to MD.Model.description"),
        pytest.param(DCAT_EXT.isVersionOf, MD.Model.modelingAuthoritySet, "uri", id="DCAT_CIM.isVersionOf to MD.Model.modelingAuthoritySet"),
        pytest.param(DCTERMS.conformsTo, MD.Model.profile, "uri", id="DCTERMS.conformsTo to MD.Model.profile"),
        pytest.param(DCAT_EXT.version, MD.Model.version, "literal", id="DCAT_CIM.version to MD.Model.version"),
        pytest.param(DCTERMS.references, MD.Model.DependentOn, "uri", id="DCTERMS.references to MD.Model.DependentOn"),
        pytest.param(DCTERMS.requires, MD.Model.DependentOn, "uri", id="DCTERMS.requires to MD.Model.DependentOn"),
        pytest.param(DCTERMS.replaces, MD.Model.Supersedes, "uri", id="DCTERMS.replaces to MD.Model.Supersedes"),
    ]
)
@patch("cim_plugin.header_conversion.convert_object", return_value=Literal("converted_object"))
def test_convert_triple_mdfullmodel(mock_convert: MagicMock, predicate_in: URIRef, predicate_out: URIRef, object_type: str) -> None:
    triple = (URIRef("http://example.com/subject"), predicate_in, Literal("unconverted_object"))
    expected = (URIRef("http://example.com/subject"), predicate_out, Literal("converted_object"))

    result = convert_triple(triple, target_format="md_fullmodel")
    
    assert result == expected
    mock_convert.assert_called_once_with(Literal("unconverted_object"), object_type, None)


@pytest.mark.parametrize(
    "predicate_in, predicate_out, object_type, datatype",
    [
        pytest.param(MD.Model.created, DCTERMS.issued, "literal", XSD.dateTime, id="MD.Model.created to DCTERMS.issued with datetime datatype"),
        pytest.param(MD.Model.scenarioTime, DCAT_EXT.startDate, "literal", XSD.dateTime, id="MD.Model.scenarioTime to DCAT_CIM.startDate with datetime datatype"),
        pytest.param(MD.Model.description, DCTERMS.description, "literal", XSD.string, id="MD.Model.description to DCTERMS.description with string datatype"),
        pytest.param(MD.Model.modelingAuthoritySet, DCAT_EXT.isVersionOf, "uri", None, id="MD.Model.modelingAuthoritySet to DCAT_CIM.isVersionOf"),
        pytest.param(MD.Model.profile, DCTERMS.conformsTo, "uri", None, id="MD.Model.profile to DCTERMS.conformsTo"),
        pytest.param(MD.Model.version, DCAT_EXT.version, "literal", XSD.string, id="MD.Model.version to DCAT_CIM.version with string datatype"),
        pytest.param(MD.Model.DependentOn, DCTERMS.requires, "uri", None, id="MD.Model.DependentOn to DCTERMS.requires"),
        pytest.param(MD.Model.Supersedes, DCTERMS.replaces, "uri", None, id="MD.Model.Supersedes to DCTERMS.replaces"),
    ]
)
@patch("cim_plugin.header_conversion.convert_object", return_value=Literal("converted_object"))
def test_convert_triple_dcatdataset(mock_convert: MagicMock, predicate_in: URIRef, predicate_out: URIRef, object_type: str, datatype: URIRef | None) -> None:
    triple = (URIRef("http://example.com/subject"), predicate_in, Literal("unconverted_object"))
    expected = (URIRef("http://example.com/subject"), predicate_out, Literal("converted_object"))

    result = convert_triple(triple, target_format="dcat_dataset")
    
    assert result == expected
    mock_convert.assert_called_once_with(Literal("unconverted_object"), object_type, datatype)


@patch("cim_plugin.header_conversion.convert_object", return_value=URIRef("converted_object"))
def test_convert_triple_sanitycheck(mock_convert: MagicMock) -> None:
    # Checking that the function works as well with URIRef objects (the two parametrized tests only use Literals).
    # Bonus check that the original triple is not modified.
    s, p, o = (URIRef("http://example.com/subject"), MD.Model.profile, URIRef("unconverted_object"))
    triple = (s, p, o)
    expected = (URIRef("http://example.com/subject"), DCTERMS.conformsTo, URIRef("converted_object"))

    result = convert_triple(triple, target_format="dcat_dataset")
    
    assert result == expected
    mock_convert.assert_called_once_with(URIRef("unconverted_object"), "uri", None)
    assert triple == (s, p, o)  # The original triple should not be modified


@patch("cim_plugin.header_conversion.convert_object")
def test_convert_triple_nonuriinput(mock_convert: MagicMock) -> None:
    triple = (URIRef("http://example.com/subject"), Literal(MD.FullModel), Literal("object"))
    with pytest.raises(AssertionError):
        convert_triple(triple, target_format="dcat_dataset")

    mock_convert.assert_not_called()


def test_convert_triple_mappingunchanged() -> None:
    triple1 = (URIRef("http://example.com/subject"), DCTERMS.issued, Literal("test_object"))
    triple2 = (URIRef("http://example.com/subject"), MD.Model.profile, URIRef("test_object"))
    before_fullmodel = copy.deepcopy(TO_FULLMODEL)
    before_dcat = copy.deepcopy(TO_DCAT)

    result1 = convert_triple(triple1, target_format="md_fullmodel")
    result2 = convert_triple(triple2, target_format="dcat_dataset")
    
    assert before_fullmodel == TO_FULLMODEL  # The mapping should not be modified
    assert before_dcat == TO_DCAT  # The mapping should not be modified
    assert before_fullmodel is not TO_FULLMODEL
    assert before_dcat is not TO_DCAT

    
# Unit tests convert_object
bnode = BNode("test")

@pytest.mark.parametrize(
    "input, object_type, datatype, expected",
    [
        pytest.param(URIRef("http://example.com/resource"), "literal", None, Literal("http://example.com/resource"), id="Uri in, literal out, no datatype"),
        pytest.param(URIRef("http://example.com/resource"), "literal", XSD.string, Literal("http://example.com/resource", datatype=XSD.string), id="Uri in, literal out, with datatype"),
        pytest.param(URIRef("http://example.com/resource"), "uri", None, URIRef("http://example.com/resource"), id="Uri in, uri out"),
        pytest.param(URIRef("http://example.com/resource"), "uri", XSD.string, URIRef("http://example.com/resource"), id="Uri in, uri out, datatype ignored"),
        pytest.param(Literal("http://example.com/resource"), "literal", None, Literal("http://example.com/resource"), id="Literal in, literal out, no datatype"),
        pytest.param(Literal("http://example.com/resource"), "literal", XSD.string, Literal("http://example.com/resource", datatype=XSD.string), id="Literal in, literal out, with datatype"),
        pytest.param(Literal("http://example.com/resource"), "uri", None, URIRef("http://example.com/resource"), id="Literal in, uri out"),
        pytest.param(bnode, "literal", None, bnode, id="BNode in, literal type, no changes"),
        pytest.param(bnode, "uri", None, bnode, id="BNode in, uri type, no changes"),
        pytest.param(Literal("http://example.com/resource"), "unknown", None, Literal("http://example.com/resource"), id="Unknown object type, no changes"),
        pytest.param(Literal(42), "literal", XSD.string, Literal(42, datatype=XSD.string), id="Literal with int value, converted to string datatype"),
        pytest.param(Literal(42, datatype=XSD.integer), "literal", None, Literal(42, datatype=XSD.integer), id="Literal with int value, no new datatype"),
        pytest.param(URIRef("http://example.com"), "unknown", None, URIRef("http://example.com"), id="Unknown object type with URIRef, unchanged"),
        pytest.param(123, "literal", None, 123, id="Wrong type input, no changes")
    ]
)
def test_convert_object(input: Any, object_type: str, datatype: URIRef | None, expected: Any) -> None:
    assert convert_object(input, object_type, datatype) == expected


if __name__ == "__main__":
    pytest.main()