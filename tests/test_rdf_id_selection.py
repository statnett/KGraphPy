import pytest
from rdflib import URIRef
from cim_plugin.rdf_id_selection import find_rdf_id_or_about

# Unit tests find_rdf_id_or_about
@pytest.mark.parametrize(
    "profile, predicate, expected",
    [
        pytest.param("Different_profile", "cim:wrong_pred", "about", id="Neither match"),
        pytest.param("http://iec.ch/TC57/ns/CIM/Operation-EU/3.0", "cim:wrong_pred", "ID", id="Profile match"),
        pytest.param("http://iec.ch/TC57/ns/CIM/ShortCircuit-EU/3.0", "cim:PetersenCoil", "about", id="Both match"),
        pytest.param("https://ap-voc.cim4.eu/EquipmentReliability/2.3", "cim:DCGround", "about", id="Both match, item further down in the file"),
        pytest.param(None, "cim:wrong_pred", "about", id="Profile None"),
    ]
)
def test_rdf_id_or_about_various(profile: str, predicate: str|URIRef, expected: str) -> None:
    result = find_rdf_id_or_about(profile, predicate)
    assert result == expected

def test_rdf_id_or_about_predicatenone() -> None:
    profile = "http://iec.ch/TC57/ns/CIM/Operation-EU/3.0"
    predicate = None
    with pytest.raises(TypeError):
        # Pylance silenced to test wrong input type
        find_rdf_id_or_about(profile, predicate)    # type: ignore

if __name__ == "__main__":
    pytest.main()