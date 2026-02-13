import pytest
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import DCAT, DCTERMS, RDF
from cim_plugin import header
from cim_plugin import header
from cim_plugin.namespaces import MD
from cim_plugin.header import CIMMetadataHeader


# Unit tests .collect_profile
@pytest.mark.parametrize(
        "triples, expected",
        [
            pytest.param([(RDF.type, MD.FullModel), (URIRef("http://iec.ch/TC57/61970-552/ModelDescription/1#Model.profile"), Literal("http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0"))], "http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0", id="Fullmodel header, model profile"),
            pytest.param([(RDF.type, MD.FullModel), (URIRef("http://purl.org/dc/terms/conformsTo"), Literal("http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0"))], "http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0", id="Fullmodel header, dcterms profile"),
            pytest.param([(RDF.type, DCAT.Dataset), (URIRef("http://purl.org/dc/terms/conformsTo"), Literal("http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0"))], "http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0", id="Dcat header, dcterms profile"),
            pytest.param([(RDF.type, DCAT.Dataset), (URIRef("http://iec.ch/TC57/61970-552/ModelDescription/1#Model.profile"), Literal("http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0"))], "http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0", id="Dcat header, model profile"),
            pytest.param([(RDF.type, DCAT.Dataset), (URIRef(DCTERMS.identifier), Literal("Not a profile"))], None, id="Profile not present")
        ]
)
def test_collect_profile_success(triples: tuple, expected: str) -> None:
    header = CIMMetadataHeader.empty(URIRef("s1"))
    for predicate, obj in triples:
        header.add_triple(predicate, obj)
    
    result = header.collect_profile()
    assert result == expected

if __name__ == "__main__":
    pytest.main()