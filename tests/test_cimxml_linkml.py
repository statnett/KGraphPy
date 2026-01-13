import pytest
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model.meta import SchemaDefinition, Prefix
from typing import Callable, Any, Optional, Dict
from cim_plugin.cimxml import _get_current_namespace_from_model, update_namespace_in_model
import copy

# There are numerous type: ignore in this file. Where not otherwise stated, this is to silence 
# pylance about schemaview.schema.prefixes, which is caused by linkML having too wide type hints.

@pytest.fixture
def make_schemaview() -> Callable[..., SchemaView]:
    """Make a sample SchemaView."""
    def _factory(prefixes=None) -> SchemaView:
        schema = SchemaDefinition(
            id="test",
            name="test",
            prefixes=prefixes,
        )
        return SchemaView(schema)

    return _factory

# Unit tests for _get_current_namespace_from_model
@pytest.mark.parametrize(
    "prefixes,prefix,expected",
    [
        pytest.param(
            [{"ex": Prefix(prefix_prefix="ex", prefix_reference="http://example.org/")}],
            "ex",
            "http://example.org/",
            id="Prefix found"
        ),
        pytest.param(
            [
                {"foo": Prefix(prefix_prefix="foo", prefix_reference="http://foo.org/")},
                {"bar": Prefix(prefix_prefix="bar", prefix_reference="http://bar.org/")},
            ],
            "bar",
            "http://bar.org/",
            id="Multiple prefixes registered"
        ),
        pytest.param(
            [
                {"foo": Prefix(prefix_prefix="foo", prefix_reference="http://foo.org/")},
                {"bar": Prefix(prefix_prefix="bar", prefix_reference="http://bar.org/")},
            ],
            "missing",
            None,
            id="No prefix found"
        ),
    ]
)
def test_get_current_namespace_from_model_basic(make_schemaview: Callable[..., SchemaView], prefixes: list[dict], prefix: str, expected: str|None) -> None:
    sv = make_schemaview(prefixes)
    result = _get_current_namespace_from_model(sv, prefix)
    assert result == expected


def test_get_current_namespace_from_model_missingschema(make_schemaview: Callable[..., SchemaView]) -> None:
    sv = make_schemaview()
    sv.schema = None

    with pytest.raises(ValueError) as exc_info:
        _get_current_namespace_from_model(sv, "ex")

    assert "Schemaview not found or schemaview is missing schema." in str(exc_info.value)


def test_get_current_namespace_from_model_schemahasnamespaces(make_schemaview: Callable[..., SchemaView]) -> None:
    # The field "namespaces" is deprecated from linkML. 
    # This test shows that a schemaview that contain namespaces is outdated and cannot be handled.
    sv = make_schemaview(prefixes=[{"foo": Prefix(prefix_prefix="foo", prefix_reference="http://foo.org/")}])
    # Ignoring pylance for testing wrong input
    sv.schema.namespaces = {}   # type: ignore

    with pytest.raises(ValueError) as exc_info:
        _get_current_namespace_from_model(sv, "ex")

    assert "The attribute 'namespaces' found in schema. This schemaview is outdated." in str(exc_info.value)


@pytest.mark.parametrize(
    "prefixes,prefix,expected",
    [
        pytest.param({}, "ex", None, id="Empty prefixes"),
        pytest.param(None, "ex", None, id="None prefixes"),
        pytest.param(None, None, None, id="Prefix is None"),
        pytest.param({}, 1, None, id="Prefix is int"),
    ]
)
def test__get_current_namespace_from_model_edgecases(make_schemaview: Callable[..., SchemaView], prefixes: dict|None|int, prefix: str|None, expected: str|None) -> None:
    sv = make_schemaview(prefixes=prefixes)

    # Ignoring pylance for testing wrong input
    result = _get_current_namespace_from_model(sv, prefix)   # type: ignore
    assert result == expected


def test_get_current_namespace_from_model_nomutationofschemaview(make_schemaview: Callable[..., SchemaView]) -> None:
    prefixes = [{"foo": Prefix(prefix_prefix="foo", prefix_reference="http://foo.org/")},
                {"bar": Prefix(prefix_prefix="bar", prefix_reference="http://bar.org/")}]
    sv = make_schemaview(prefixes=prefixes)
    schema_before_id = id(sv.schema)
    prefixes_before = copy.deepcopy(sv.schema.prefixes) # type: ignore
    name_before = copy.deepcopy(getattr(sv.schema, "name", None))

    result = _get_current_namespace_from_model(sv, "foo")

    assert result == "http://foo.org/"

    # Checking that nothing has been changed in the schemaview
    assert id(sv.schema) == schema_before_id
    assert sv.schema.prefixes == prefixes_before    # type: ignore
    assert getattr(sv.schema, "name", None) == name_before


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


def test_update_namespace_in_model_missingprefixesattribute(make_schemaview: Callable[..., SchemaView]) -> None: 
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