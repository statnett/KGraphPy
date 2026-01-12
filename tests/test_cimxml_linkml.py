import pytest
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model.meta import SchemaDefinition, Prefix
from typing import Callable, Any, Optional, Dict
from cim_plugin.cimxml import update_namespace_in_model


@pytest.fixture
def make_schemaview() -> Callable[..., SchemaView]:
    """
    Lager et ekte SchemaView basert på et ekte SchemaDefinition.
    """
    def _factory(prefixes=None) -> SchemaView:
        schema = SchemaDefinition(
            id="test",
            name="test",
            prefixes=prefixes,
        )
        return SchemaView(schema)

    return _factory


# Unit tests update_namespace_in_model

def test_update_namespace_in_model_noschema(make_schemaview: Callable[..., SchemaView]) -> None:
    sv = make_schemaview()
    sv.schema = None

    with pytest.raises(ValueError):
        update_namespace_in_model(sv, "ex", "http://new/")


@pytest.mark.parametrize(
    "prefixes,expected",
    [
        pytest.param(
            {"ex": Prefix(prefix_prefix="ex", prefix_reference="http://old/")},
            "http://updated/",
            id="Prefix exists and is updated"
        ),
        pytest.param(
            {"other": Prefix(prefix_prefix="other", prefix_reference="http://old/")},
            "http://old/",
            id="Prefix not in schema, no update"
        ),
        pytest.param(
            {"ex": Prefix(prefix_prefix="ex", prefix_reference="")},
            "http://updated/",
            id="Prefix exist, but prefix_reference is empty -> Updated"
        ),
        pytest.param(
            {"ex": {"prefix_reference": "http://old/"}},
            "http://updated/",
            id="Prefix is not a prefix object -> Updated"
        ),
        pytest.param(
            {"ex": "http://old/"},
            "http://updated/",
            id="Prefix is a simple dict -> Updated"
        ),
        pytest.param(
            {},
            "",
            id="Prefix is empty -> Not updated"
        ),
    ],
)
def test_update_namespace_in_model_prefixes(make_schemaview: Callable[..., SchemaView], prefixes: dict[str, Any], expected: str) -> None:
    sv = make_schemaview(prefixes)

    update_namespace_in_model(sv, "ex", "http://updated/")

    # Using type: ignore to cancel pylance issues with linkML prefixes
    if "ex" in prefixes:
        assert sv.schema.prefixes["ex"].prefix_reference == expected    # type: ignore
    elif "other" in prefixes:
        assert sv.schema.prefixes["other"].prefix_reference == expected # type: ignore
    else:
        assert len(prefixes) == 0


def test_update_namespace_in_model_reinitializing_check(make_schemaview: Callable[..., SchemaView], monkeypatch):
    sv = make_schemaview()

    called = False

    def fake_init(self, schema):
        nonlocal called
        called = True

    monkeypatch.setattr(SchemaView, "__init__", fake_init)

    update_namespace_in_model(sv, "ex", "http://updated/")

    assert called is True   # Checking that schemaview.__init__(schema) is called


def test_update_namespace_in_model_prefixesisnone(make_schemaview: Callable[..., SchemaView]) -> None: 
    sv = make_schemaview(None) 
    update_namespace_in_model(sv, "ex", "http://updated/") 
    assert sv.schema.prefixes == {} # type: ignore


def test_update_namespace_in_model_missingprefixesattribute(monkeypatch: pytest.MonkeyPatch, make_schemaview: Callable[..., SchemaView]) -> None: 
    sv = make_schemaview({"ex": Prefix(prefix_prefix="ex", prefix_reference="http://old/")}) 
    del sv.schema.prefixes  # type: ignore
    update_namespace_in_model(sv, "ex", "http://updated/") 
    assert not hasattr(sv.schema, "prefixes")

def test_update_namespace_in_model_reinitraisesexception(monkeypatch: pytest.MonkeyPatch, make_schemaview: Callable[..., SchemaView]) -> None: 
    sv = make_schemaview({"ex": Prefix(prefix_prefix="ex", prefix_reference="http://old/")}) 
    
    def fake_init(self, schema): 
        raise RuntimeError("boom") 
    
    monkeypatch.setattr(SchemaView, "__init__", fake_init) 
    with pytest.raises(RuntimeError): 
        update_namespace_in_model(sv, "ex", "http://updated/")

def test_update_namespace_in_model_nonstringinput(make_schemaview: Callable[..., SchemaView]) -> None: 
    prefixes = {"ex": Prefix(prefix_prefix="ex", prefix_reference="http://old/")} 
    sv = make_schemaview(prefixes) 
    # Using type: ignore to test wrong datatype input without pylance complaining.
    update_namespace_in_model(sv, "ex", 12345)  # type: ignore
    sv.schema.prefixes["ex"].prefix_reference == 12345  # type: ignore


def test_update_namespace_in_model_prefixesislist(make_schemaview: Callable[..., SchemaView]) -> None: 
    prefixes = [ {"ex": {"prefix_reference": "http://old/"}} ] 
    sv = make_schemaview(prefixes) 
    update_namespace_in_model(sv, "ex", "http://updated/") 
    assert sv.schema.prefixes == {"ex": Prefix(prefix_prefix="ex", prefix_reference="http://updated/")} # type: ignore

def test_update_namespace_in_model_schemawrongtype(make_schemaview: Callable[..., SchemaView]) -> None: 
    sv = make_schemaview({"ex": Prefix(prefix_prefix="ex", prefix_reference="http://old/")}) 
    # Using type: ignore to test wrong datatype input without pylance complaining.
    sv.schema = "not-a-schema-object"   # type: ignore
    with pytest.raises(Exception): 
        update_namespace_in_model(sv, "ex", "http://updated/")

if __name__ == "__main__":
    pytest.main()