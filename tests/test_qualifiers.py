import pytest
from cim_plugin.qualifiers import CIMQualifierStrategy, UnderscoreQualifier, URNQualifier, NamespaceQualifier, CIMQualifierResolver, uuid_namespace
from rdflib import URIRef

# Unit tests .matches
@pytest.mark.parametrize(
    "strategy,uri,expected",
    [
        pytest.param(UnderscoreQualifier(), "_1234", True, id="UnderscoreQualifier matched _"),
        pytest.param(UnderscoreQualifier(), "#_abcd", True, id="UnderscoreQualifier matched #_"),
        pytest.param(UnderscoreQualifier(), "urn:uuid:1234", False, id="UnderscoreQualifier unmatched"),

        pytest.param(URNQualifier(), "urn:uuid:abcd", True, id="URNQualifier, matched"),
        pytest.param(URNQualifier(), "_1234", False, id="URNQualifier, unmatched"),

        pytest.param(NamespaceQualifier(), f"{uuid_namespace}:abcd", True, id="NamespaceQualifier, matched"),
        pytest.param(NamespaceQualifier(), "_1234", False, id="NamespaceQualifier, unmatched"),
    ]
)
def test_qualifierstrategy_matches(strategy: CIMQualifierStrategy, uri: str, expected: bool) -> None:
    assert strategy.matches(uri) == expected


# Unit tests .extract_uuid
@pytest.mark.parametrize(
    "strategy,uri,expected_uuid",
    [
        pytest.param(UnderscoreQualifier(), "_1234", "1234", id="UnderscoreQualifier extract from _"),
        pytest.param(UnderscoreQualifier(), "#_abcd", "abcd", id="UnderscoreQualifier extract from #_"),
        pytest.param(URNQualifier(), "urn:uuid:abcd", "abcd", id="URNQualifier extract from urn:uuid:"),
        pytest.param(NamespaceQualifier(), f"{uuid_namespace}:xyz", "xyz", id="NamespaceQualifier extract from namespace:"),
        pytest.param(NamespaceQualifier(), f"{uuid_namespace}:xyz", "xyz", id="NamespaceQualifier extract from :"),
    ]
)
def test_qualifierstrategy_extract_uuid(strategy: CIMQualifierStrategy, uri: str, expected_uuid: str) -> None:
    assert strategy.extract_uuid(uri) == expected_uuid


# Unit tests .build_about and .build_resources
@pytest.mark.parametrize(
    "strategy,uuid,expected_about,expected_resource",
    [
        pytest.param(UnderscoreQualifier(), "1234", "_1234", "#_1234", id="UnderscoreQualifier build"),
        pytest.param(URNQualifier(), "abcd", "urn:uuid:abcd", "urn:uuid:abcd", id="URNQualifier build"),
        pytest.param(NamespaceQualifier(), "xyz", f"{uuid_namespace}:xyz", f"{uuid_namespace}:xyz", id="NamespaceQualifier build"),
    ]
)
def test_qualifierstrategy_build(strategy: CIMQualifierStrategy, uuid: str, expected_about: str, expected_resource: str) -> None:
    assert strategy.build_special(uuid) == expected_about
    assert strategy.build_default(uuid) == expected_resource


# Unit tests CIMQualifierResolver.convert_about
@pytest.mark.parametrize(
    "input_uri,expected_uuid",
    [
        pytest.param("_1234", "1234", id="With underscore"),
        pytest.param("urn:uuid:abcd", "abcd", id="With urn"),
        pytest.param(f"{uuid_namespace}:xyz", "xyz", id="With namespace"),
        pytest.param("weird", "weird", id="Fallback"),
    ]
)
def test_resolver_underscore_convert_about(input_uri: str, expected_uuid: str) -> None:
    resolver = CIMQualifierResolver(UnderscoreQualifier())
    assert resolver.convert_to_special_qualifier(URIRef(input_uri)) == f"_{expected_uuid}"
    assert resolver.convert_to_default_qualifier(URIRef(input_uri)) == f"#_{expected_uuid}"


@pytest.mark.parametrize(
    "input_uri,expected_uuid",
    [
        pytest.param("_1234", "1234", id="With underscore"),
        pytest.param("urn:uuid:abcd", "abcd", id="With urn"),
        pytest.param(f"{uuid_namespace}:xyz", "xyz", id="With namespace"),
        pytest.param("weird", "weird", id="Fallback"),
    ]
)
def test_resolver_urn_convert_about(input_uri: str, expected_uuid: str) -> None:
    resolver = CIMQualifierResolver(URNQualifier())
    assert resolver.convert_to_special_qualifier(URIRef(input_uri)) == f"urn:uuid:{expected_uuid}"


@pytest.mark.parametrize(
    "input_uri,expected_uuid",
    [
        pytest.param("_1234", "1234", id="With underscore"),
        pytest.param("urn:uuid:abcd", "abcd", id="With urn"),
        pytest.param(f"{uuid_namespace}:xyz", "xyz", id="With namespace"),
        pytest.param("weird", "weird", id="Fallback"),
    ]
)
def test_resolver_namespace_convert_about(input_uri: str, expected_uuid: str) -> None:
    resolver = CIMQualifierResolver(NamespaceQualifier())
    assert resolver.convert_to_special_qualifier(URIRef(input_uri)) == f"{uuid_namespace}:{expected_uuid}"



if __name__ == "__main__":
    pytest.main()