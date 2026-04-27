import pytest
from rdflib import URIRef
from cim_plugin.rdf_id_selection import find_rdf_id_or_about

# Unit tests find_rdf_id_or_about
@pytest.mark.parametrize(
    "profiles, object_type, expected",
    [
        pytest.param(["Different_profile"], "cim:wrong_pred", "about", id="Neither match"),
        pytest.param(["http://iec.ch/TC57/ns/CIM/Operation-EU/3.0"], "cim:wrong_pred", "ID", id="Profile match"),
        pytest.param(["http://iec.ch/TC57/ns/CIM/ShortCircuit-EU/3.0"], "cim:PetersenCoil", "about", id="Both match"),
        pytest.param(["https://ap.cim4.eu/EquipmentReliability/2.3"], "cim:DCGround", "about", id="Both match, item further down in the file"),
        pytest.param(["https://ap-voc.cim4.eu/EquipmentReliability/2.3"], "cim:wrong_pred", "ID", id="Old profile name"),
        pytest.param(["https://ap-voc.cim4.eu/EquipmentReliability/2.3"], "cim:DCGround", "about", id="Old profile name with exception"),
        pytest.param(["http://iec.ch/TC57/ns/CIM/ShortCircuit-EU/3.0"], "cim:wrong_pred", "ID", id="Old profile name with empty exceptions list"),
        pytest.param(None, "cim:wrong_pred", "about", id="Profile None"),
        pytest.param(["other_profile", "also_different_profile"], "cim:wrong_pred", "about", id="Multiple profiles, none match"),
        pytest.param(["other_profile", "http://iec.ch/TC57/ns/CIM/Operation-EU/3.0"], "cim:wrong_pred", "ID", id="Multiple profiles, one matches"),
        pytest.param(["other_profile", "http://iec.ch/TC57/ns/CIM/ShortCircuit-EU/3.0"], "cim:PetersenCoil", "about", id="Multiple profiles, one matches with exception"),
        pytest.param(["http://iec.ch/TC57/ns/CIM/ShortCircuit-EU/3.0", "http://iec.ch/TC57/ns/CIM/Operation-EU/3.0"], "cim:wrong_pred", "ID", id="Multiple profiles, all matches"),
        # This one is tricky. One profile requires ID and the other about. The function is currently designed to return ID if any profile requires it, but this may need to be reconsidered.
        pytest.param(["http://iec.ch/TC57/ns/CIM/ShortCircuit-EU/3.0", "http://iec.ch/TC57/ns/CIM/Operation-EU/3.0"], "cim:PetersenCoil", "ID", id="Multiple profiles, all matches, one with exception"),
    ]
)
def test_rdf_id_or_about_various(profiles: list[str] | None, object_type: str|URIRef, expected: str) -> None:
    result = find_rdf_id_or_about(profiles, object_type)
    assert result == expected

def test_rdf_id_or_about_objectnone() -> None:
    profile = "http://iec.ch/TC57/ns/CIM/Operation-EU/3.0"
    object_type = None
    with pytest.raises(TypeError):
        # Pylance silenced to test wrong input type
        find_rdf_id_or_about([profile], object_type)    # type: ignore


def test_rdf_id_or_about_profilenone(caplog: pytest.LogCaptureFixture) -> None:
    result = find_rdf_id_or_about(None, "Any_predicate")
    assert result == "about"
    assert "No profile found. Defaults to 'about'." in caplog.text

if __name__ == "__main__":
    pytest.main()