import pytest
from unittest.mock import patch, MagicMock, call
from cim_plugin.cimxml_parser import CIMXMLParser, LiteralCastingError
from pytest import LogCaptureFixture
from typing import Callable
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model.meta import TypeDefinition, ClassDefinition, SlotDefinition
from rdflib import URIRef, Graph, Literal, BNode
from tests.fixtures import make_schemaview, cimxmlinstance_w_prefixes, make_cimxmlparser
import logging

logger = logging.getLogger("cimxml_logger")


# Unit tests .ensure_correct_namespace_model

@patch("cim_plugin.cimxml_parser.update_namespace_in_model")
@patch("cim_plugin.cimxml_parser._get_current_namespace_from_model")
def test_ensure_correct_namespace_model_noschemaview(mock_get: MagicMock, mock_update: MagicMock) -> None:
    obj = CIMXMLParser()
    obj.schemaview = None

    with pytest.raises(ValueError, match="Schemaview not found"):
        obj.ensure_correct_namespace_model("ex", "new_ns")
    
    mock_get.assert_not_called()
    mock_update.assert_not_called()


@pytest.mark.parametrize(
    "current, new_ns, update",
    [
        pytest.param("old_ns", "new_ns", True, id="Current namespace not correct -> update"),
        pytest.param("same_ns", "same_ns", False, id="Current namespace correct -> no update"),
        pytest.param("", "new_ns", True, id="Current namespace empty -> update"),
        pytest.param(" ", "new_ns", True, id="Current namespace whitespace -> update"),
        pytest.param(123, "new_ns", True, id="Current namespace numeric -> update"),
        pytest.param({}, "new_ns", True, id="Current namespace a dict -> update"),
        pytest.param([], "new_ns", True, id="Current namespace a list -> update"),
        pytest.param(" new_ns ", "new_ns", True, id="Current namespace same except with whitespace -> update"),
    ]
)
@patch("cim_plugin.cimxml_parser.update_namespace_in_model")
@patch("cim_plugin.cimxml_parser._get_current_namespace_from_model")
def test_ensure_correct_namespace_model_namespacehandling(mock_get: MagicMock, mock_update: MagicMock, cimxmlinstance_w_prefixes: CIMXMLParser, current: str, new_ns: str, update: bool, caplog: LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    mock_get.return_value = current

    cimxmlinstance_w_prefixes.ensure_correct_namespace_model("ex", new_ns)

    mock_get.assert_called_once()
    if update:
        mock_update.assert_called_once_with(cimxmlinstance_w_prefixes.schemaview, "ex", new_ns)
        assert f"Wrong namespace detected for ex in model. Correcting to {new_ns}." in caplog.text
    else:
        mock_update.assert_not_called()
        assert "Model has correct namespace for ex." in caplog.text

@patch("cim_plugin.cimxml_parser.update_namespace_in_model")
@patch("cim_plugin.cimxml_parser._get_current_namespace_from_model")
def test_ensure_correct_namespace_model_prefixnotfound(mock_get: MagicMock, mock_update: MagicMock, cimxmlinstance_w_prefixes: CIMXMLParser) -> None:
    mock_get.return_value = None

    with pytest.raises(ValueError, match="Prefix ex not found"):
        cimxmlinstance_w_prefixes.ensure_correct_namespace_model("ex", "new_ns")
    
    mock_get.assert_called_once()
    mock_update.assert_not_called()

@patch("cim_plugin.cimxml_parser.update_namespace_in_model")
@patch("cim_plugin.cimxml_parser._get_current_namespace_from_model")
def test_ensure_correct_namespace_model_multipleprefixes(mock_get: MagicMock, mock_update: MagicMock, caplog: LogCaptureFixture, make_schemaview: Callable[..., SchemaView]) -> None:
    caplog.set_level("INFO")
    mock_get.return_value = "www.bar.org"

    obj = CIMXMLParser()
    obj.schemaview = make_schemaview(prefixes=[{"ex": {"prefix_prefix": "ex", "prefix_reference": "www.example.org"}},
                                               {"foo": {"prefix_prefix": "foo", "prefix_reference": "www.bar.org"}}])
    obj.ensure_correct_namespace_model("foo", "www.foo.org")

    mock_get.assert_called_once()
    mock_update.assert_called_once_with(obj.schemaview, "foo", "www.foo.org")
    assert "Wrong namespace detected for foo in model. Correcting to www.foo.org." in caplog.text


@patch("cim_plugin.cimxml_parser.update_namespace_in_model")
@patch("cim_plugin.cimxml_parser._get_current_namespace_from_model")
def test_ensure_correct_namespace_model_errorfromcall(mock_get: MagicMock, mock_update: MagicMock, caplog: LogCaptureFixture, make_schemaview: Callable[..., SchemaView]) -> None:
    caplog.set_level("INFO")
    mock_get.return_value = "www.bar.org"
    mock_update.side_effect = ValueError

    obj = CIMXMLParser()
    obj.schemaview = make_schemaview(prefixes=[{"ex": {"prefix_prefix": "ex", "prefix_reference": "www.example.org"}},
                                               {"foo": {"prefix_prefix": "foo", "prefix_reference": "www.bar.org"}}])
    with pytest.raises(ValueError):
        obj.ensure_correct_namespace_model("foo", "www.foo.org")

    mock_get.assert_called_once()
    mock_update.assert_called_once_with(obj.schemaview, "foo", "www.foo.org")
    assert "Wrong namespace detected for foo in model. Correcting to www.foo.org." in caplog.text


# Unit tests .patch_missing_datatypes_in_model

@pytest.mark.parametrize(
    "schema_path, schemaview",
    [
        (None, None),
        ("schema.yaml", None),
        ("schema.yaml", MagicMock(schema=None)),
    ]
)
@patch("cim_plugin.cimxml_parser.patch_integer_ranges")
def test_patch_missing_datatypes_in_model_prerequisitesmissing(mock_patch: MagicMock, schema_path: str|None, schemaview: MagicMock|None) -> None:
    obj = CIMXMLParser()
    obj.schema_path = schema_path
    obj.schemaview = schemaview

    obj.patch_missing_datatypes_in_model()
    mock_patch.assert_not_called()

@patch("cim_plugin.cimxml_parser.patch_integer_ranges")
def test_patch_missing_datatypes_in_model_integeralreadypresent(mock_patch: MagicMock, make_schemaview: Callable[..., SchemaView], make_cimxmlparser: Callable[..., CIMXMLParser]) -> None:
    sv = make_schemaview(types={"integer": TypeDefinition(name="integer")})
    obj = make_cimxmlparser(schemaview=sv)
    
    obj.patch_missing_datatypes_in_model()
    mock_patch.assert_not_called()

    # Ensure no modification occurred
    assert sv.schema is not None
    assert sv.schema.types is not None
    assert "integer" in sv.schema.types

@patch("cim_plugin.cimxml_parser.patch_integer_ranges")
def test_patch_missing_datatypes_in_model_addinteger(mock_patch: MagicMock, make_schemaview: Callable[..., SchemaView], make_cimxmlparser: Callable[..., CIMXMLParser]) -> None:
    sv = make_schemaview(types={})
    obj = make_cimxmlparser(schemaview = sv)

    obj.patch_missing_datatypes_in_model()

    assert sv.schema is not None
    assert isinstance(sv.schema.types, dict)
    assert "integer" in sv.schema.types
    t = sv.schema.types["integer"]
    assert isinstance(t, TypeDefinition)
    assert t.base == "int"
    assert t.uri == "http://www.w3.org/2001/XMLSchema#integer"
    mock_patch.assert_called_once_with(sv, "schema.yaml")

@patch("cim_plugin.cimxml_parser.patch_integer_ranges")
def test_patch_missing_datatypes_in_model_addinteger_setmodifiedcalled(mock_patch: MagicMock, make_schemaview: Callable[..., SchemaView], make_cimxmlparser: Callable[..., CIMXMLParser], monkeypatch: pytest.MonkeyPatch) -> None:
    sv = make_schemaview(types={})
    obj = make_cimxmlparser(schemaview = sv)
    mock_set_called = False
    def _fake_set_modified(self):
        nonlocal mock_set_called
        mock_set_called = True

    monkeypatch.setattr(SchemaView, "set_modified", _fake_set_modified)

    obj.patch_missing_datatypes_in_model()

    assert sv.schema is not None
    assert isinstance(sv.schema.types, dict)
    assert "integer" in sv.schema.types
    mock_patch.assert_called_once_with(sv, "schema.yaml")
    assert mock_set_called == True

@patch("cim_plugin.cimxml_parser.patch_integer_ranges")
def test_patch_missing_datatypes_in_model_typedefinitioncallcheck(mock_patch: MagicMock, make_schemaview: Callable[..., SchemaView], make_cimxmlparser: Callable[..., CIMXMLParser]) -> None:
    """Optional: verify TypeDefinition is constructed with correct args."""
    sv = make_schemaview(types={})
    obj = make_cimxmlparser(schemaview=sv)

    with patch("cim_plugin.cimxml_parser.TypeDefinition") as TD:
        obj.patch_missing_datatypes_in_model()

        TD.assert_called_once_with(name="integer", base="int", uri="http://www.w3.org/2001/XMLSchema#integer")


@patch("cim_plugin.cimxml_parser.patch_integer_ranges")
def test_patch_missing_datatypes_in_model_errorfromcalledfunksjon(mock_patch: MagicMock, make_schemaview: Callable[..., SchemaView], make_cimxmlparser: Callable[..., CIMXMLParser], caplog: LogCaptureFixture) -> None:
    mock_patch.side_effect = ValueError("slot not found in schemaview")
    sv = make_schemaview(types={})
    obj = make_cimxmlparser(schemaview=sv)

    with pytest.raises(ValueError):
        obj.patch_missing_datatypes_in_model()

    assert "slot not found in schemaview" in caplog.text


@pytest.mark.parametrize(
    "types",
    [
        (None),
        ("not_a_dict"),
        ([]),
    ]
)
@patch("cim_plugin.cimxml_parser.patch_integer_ranges")
def test_patch_missing_datatypes_in_model_invalidtypes(mock_patch: MagicMock, types: str|list|None, make_schemaview: Callable[..., SchemaView], make_cimxmlparser: Callable[..., CIMXMLParser]) -> None:
    # linkML always turns schemaview.schema.types into a dict. This test documents that. See below.
    sv = make_schemaview(types=types)
    obj = make_cimxmlparser(schemaview=sv)

    obj.patch_missing_datatypes_in_model()
    mock_patch.assert_called_once()
    assert sv.schema is not None
    assert isinstance(sv.schema.types, dict)


@patch("cim_plugin.cimxml_parser.patch_integer_ranges")
def test_patch_missing_datatypes_in_model_invalidtypes_forcedtoraise(mock_patch: MagicMock, make_schemaview: Callable[..., SchemaView], make_cimxmlparser: Callable[..., CIMXMLParser]) -> None:
    # linkML always turns schemaview.schema.types into a dict. This test forces it None to test what happends if linkML fails to do that.
    sv = make_schemaview(types={})
    # Pylance silenced to test invalid datatype
    sv.schema.types = None  # Forces the types to be invalid    # type: ignore
    obj = make_cimxmlparser(schemaview=sv)

    with pytest.raises(TypeError, match="Expected types to be dict, got <class 'NoneType'>"):
        obj.patch_missing_datatypes_in_model()
    mock_patch.assert_not_called()
    assert sv.schema is not None
    assert sv.schema.types is None


@patch("cim_plugin.cimxml_parser.patch_integer_ranges")
def test_patch_missing_datatypes_in_model_integerpresentbutwrong(mock_patch: MagicMock, make_schemaview: Callable[..., SchemaView], make_cimxmlparser: Callable[..., CIMXMLParser]) -> None:
    # Pylance ignored to test wrong datatype
    sv = make_schemaview(types={"integer": TypeDefinition(name="integer", base=123)})   # type: ignore
    obj = make_cimxmlparser(schemaview=sv)
    
    obj.patch_missing_datatypes_in_model()
    mock_patch.assert_not_called()  # No modification, because integer already exists even though type is wrong
    assert sv.schema is not None
    assert sv.schema.types is not None
    assert "integer" in sv.schema.types


@patch("cim_plugin.cimxml_parser.patch_integer_ranges")
def test_patch_missing_datatypes_in_model_missingtypes(mock_patch: MagicMock, make_schemaview: Callable[..., SchemaView], make_cimxmlparser: Callable[..., CIMXMLParser]) -> None:
    sv = make_schemaview(types=None)
    del sv.schema.types # type: ignore
    obj = make_cimxmlparser(schemaview=sv)
    
    with pytest.raises(AttributeError):
        obj.patch_missing_datatypes_in_model()
    mock_patch.assert_not_called()
    assert sv.schema is not None
    assert "types" not in sv.schema


@patch("cim_plugin.cimxml_parser.patch_integer_ranges")
def test_patch_missing_datatypes_in_model_idempotence(mock_patch: MagicMock, make_schemaview: Callable[..., SchemaView], make_cimxmlparser: Callable[..., CIMXMLParser]) -> None:
    sv = make_schemaview(types={})
    obj = make_cimxmlparser(schemaview = sv)

    assert sv.schema is not None
    assert isinstance(sv.schema.types, dict)

    obj.patch_missing_datatypes_in_model()

    assert "integer" in sv.schema.types
    assert mock_patch.call_count == 1

    obj.patch_missing_datatypes_in_model()
    assert mock_patch.call_count == 1   # patch_integer_ranges is not called second run of the function because now integer is present


# Unit tests .normalize_rdf_ids

@patch("cim_plugin.cimxml_parser._clean_uri")
@patch("cim_plugin.cimxml_parser.detect_uri_collisions")
def test_normalize_rdf_ids_stops_on_collision(mock_detect: MagicMock, mock_clean: MagicMock) -> None:
    mock_detect.side_effect = ValueError("collision!")

    parser = CIMXMLParser()
    g = Graph()
    g.add((URIRef("s"), URIRef("p"), URIRef("o")))
    g.add((URIRef('www.something.com#_ab'), URIRef("p"), URIRef("o")))

    with pytest.raises(ValueError):
        parser.normalize_rdf_ids(g)

    mock_detect.assert_called_once_with(g, {"ab"})
    mock_clean.assert_not_called()


@pytest.mark.parametrize(
        "s, o, calls",
        [
            pytest.param(URIRef("a"), Literal("b"), [call(URIRef('a'), {}, set())], id="Only first call"),
            pytest.param(URIRef("s#_a"), Literal("b"), [call(URIRef('s#_a'), {}, set("a"))], id="One call with id_set"),
            pytest.param(URIRef("s#a"), URIRef("b"), [call(URIRef('s#a'), {}, set("a")), call(URIRef('b'), {}, set("a"))], id="Subject no _"),
            pytest.param(URIRef("a"), URIRef("b"), [call(URIRef('a'), {}, set()), call(URIRef('b'), {}, set())], id="Both called"),
            pytest.param(URIRef("s#_a"), URIRef("b"), [call(URIRef('s#_a'), {}, set("a")), call(URIRef('b'), {}, set("a"))], id="Both called, with id_set"),
            pytest.param(BNode("x"), Literal("b"),[], id="Subject is BNode → no calls"),
            pytest.param(BNode("x"), URIRef("b"),[call(URIRef("b"), {}, set())], id="Subject is BNode → object cleaned"),
            pytest.param(URIRef("a"), BNode("x"), [call(URIRef("a"), {}, set())], id="Object is BNode → only subject cleaned"),
            pytest.param(URIRef("http://ex.com/foo"), Literal("b"), [call(URIRef("http://ex.com/foo"), {}, set())], id="Subject without fragment still cleaned"),
            pytest.param(URIRef("s#_a"), URIRef("http://ex.com#not_id"), [call(URIRef("s#_a"), {}, {"a"}), call(URIRef("http://ex.com#not_id"), {}, {"a"})], id="Object fragment not in id_set → not cleaned, but _clean_uri is called.")
        ]
)
@patch("cim_plugin.cimxml_parser._clean_uri")
@patch("cim_plugin.cimxml_parser.detect_uri_collisions")
def test_normalize_rdf_ids_callscleanuri(mock_detect: MagicMock, mock_clean: MagicMock, s: URIRef|BNode, o: Literal|URIRef|BNode, calls: list) -> None:
    mock_detect.return_value = None
    mock_clean.side_effect = lambda uri, *_: URIRef("urn:uuid:test")

    parser = CIMXMLParser()
    g = Graph()
    g.add((s, URIRef("p"), o))

    parser.normalize_rdf_ids(g)

    mock_detect.assert_called_once()
    assert mock_clean.mock_calls == calls


@patch("cim_plugin.cimxml_parser._clean_uri")
@patch("cim_plugin.cimxml_parser.detect_uri_collisions")
def test_normalize_rdf_ids_mutatesgraph(mock_detect: MagicMock, mock_clean: MagicMock) -> None:
    mock_detect.return_value = None

    mock_clean.side_effect = [
        URIRef("urn:uuid:a"),  # new_s
        URIRef("urn:uuid:b"),  # new_o
    ]

    parser = CIMXMLParser()
    g = Graph()
    s = URIRef("http://ex.com#_a")
    p = URIRef("p")
    o = URIRef("http://ex.com#_b")
    g.add((s, p, o))

    parser.normalize_rdf_ids(g)

    triples = list(g)
    assert len(triples) == 1
    new_s, new_p, new_o = triples[0]
    assert new_s == URIRef("urn:uuid:a")
    assert new_o == URIRef("urn:uuid:b")
    assert new_p == p


def test_normalize_rdf_ids_withoutmocking() -> None:
    parser = CIMXMLParser()
    g = Graph()

    # Two triples sharing the same subject
    s = URIRef("http://ex.com#_a")
    p = URIRef("p")
    o1 = URIRef("http://ex.com#_b")
    o2 = URIRef("http://ex.com#_c")

    g.add((s, p, o1))
    g.add((s, p, o2))

    parser.normalize_rdf_ids(g)

    # If the bug exists, one triple will not be updated correctly
    assert all(str(tr[0]) == "urn:uuid:a" for tr in g)
    assert len(list(g)) == 2


@patch("cim_plugin.cimxml_parser._clean_uri")
@patch("cim_plugin.cimxml_parser.detect_uri_collisions")
def test_normalize_rdf_ids_reusesurimap(mock_detect: MagicMock, mock_clean: MagicMock) -> None:
    mock_detect.return_value = None

    mock_clean.side_effect = lambda uri, *_: URIRef("urn:uuid:test")

    parser = CIMXMLParser()
    g = Graph()
    s = URIRef("http://ex.com#_a")
    o = URIRef("http://ex.com#_a")
    p = URIRef("p")
    g.add((s, p, o))
    g.add((s, p, o))  # Same triple again

    parser.normalize_rdf_ids(g)
    assert mock_clean.call_count == 2 # _clean_uri should only be called once each for s and o


@patch("cim_plugin.cimxml_parser._clean_uri")
@patch("cim_plugin.cimxml_parser.detect_uri_collisions")
def test_normalize_rdf_ids_emptygraph(mock_detect: MagicMock, mock_clean: MagicMock) -> None:
    mock_detect.return_value = None

    parser = CIMXMLParser()
    g = Graph()

    parser.normalize_rdf_ids(g)

    mock_clean.assert_not_called()
    mock_detect.assert_called_once()


class DummySlot:
    def __init__(self, name, range):
        self.name = name
        self.range = range

# Unit tests .enrich_literal_datatypes

@pytest.mark.parametrize("schemaview, slot_index", [
    pytest.param(None, {"p": DummySlot("p", "string")}, id="Schemaview missing"),
    pytest.param("dummy", None, id="slot_index missing"),
])
@patch("cim_plugin.cimxml_parser.resolve_datatype_from_slot")
@patch("cim_plugin.cimxml_parser.create_typed_literal")
def test_enrich_literal_datatypes_missingprerequisites(mock_create: MagicMock, mock_resolve: MagicMock, schemaview: SchemaView|None, slot_index: dict|None, caplog: LogCaptureFixture) -> None:
    g = Graph()
    s, p, o = URIRef("s"), URIRef("p"), Literal("x")
    g.add((s, p, o))

    inst = CIMXMLParser()
    inst.schemaview = schemaview
    inst.slot_index = slot_index

    result = inst.enrich_literal_datatypes(g)

    assert list(result) == [(s, p, o)]
    assert "Missing schemaview or slot_index. Enriching not possible." in caplog.text
    mock_resolve.assert_not_called()
    mock_create.assert_not_called()


@patch("cim_plugin.cimxml_parser.resolve_datatype_from_slot", return_value=None)   # If slot.range is None, this function will return None
@patch("cim_plugin.cimxml_parser.create_typed_literal")
def test_enrich_literal_datatypes_slotrangenone(mock_create: MagicMock, mock_resolve: MagicMock, cimxmlinstance_w_prefixes: CIMXMLParser, caplog: LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    g = Graph()
    s, p, o = URIRef("s"), URIRef("p"), Literal("x")
    g.add((s, p, o))
    
    inst = cimxmlinstance_w_prefixes
    inst.slot_index = {"p": DummySlot("p", None)}
    
    result = inst.enrich_literal_datatypes(g)

    mock_resolve.assert_called_once()
    mock_create.assert_not_called()
    assert list(result) == [(s, p, o)]
    assert "No datatype found for range: None, for p" in caplog.text


@pytest.mark.parametrize("object, resolved", [
    pytest.param(Literal("x"), True, id="Literal without datatype"),
    pytest.param(Literal("x", datatype=URIRef('http://www.w3.org/2001/XMLSchema#string')), False, id="Literal with datatype"),
    pytest.param(Literal("x", lang="en"), False, id="Literal with language. Datatype is implicitly string."),
    pytest.param(BNode("x"), False, id="Blank node."),
    pytest.param(URIRef("x"), False, id="URI object")
])
@patch("cim_plugin.cimxml_parser.resolve_datatype_from_slot")
@patch("cim_plugin.cimxml_parser.create_typed_literal")
def test_enrich_literal_datatypes_objecthandling(mock_create: MagicMock, mock_resolve: MagicMock, object: Literal|BNode|URIRef, resolved: bool, cimxmlinstance_w_prefixes: CIMXMLParser) -> None:
    mock_resolve.return_value = "xsd:string"
    mock_create.return_value = object
    g = Graph()
    s, p, o = URIRef("s"), URIRef("p"), object
    g.add((s, p, o))

    inst = cimxmlinstance_w_prefixes
    inst.slot_index = {"p": DummySlot("p", "string")}

    result = inst.enrich_literal_datatypes(g)
    if resolved:
        mock_resolve.assert_called_once()
        mock_create.assert_called_once()
        assert len(list(result)) == 1   # Size of graph is not affected
    else:
        mock_resolve.assert_not_called()
        mock_create.assert_not_called()
        assert len(list(result)) == 1   # Size of graph is not affected

@patch("cim_plugin.cimxml_parser.resolve_datatype_from_slot")
@patch("cim_plugin.cimxml_parser.create_typed_literal")
def test_enrich_literal_datatypes_predicatenotfound(mock_create: MagicMock, mock_resolve: MagicMock, cimxmlinstance_w_prefixes: CIMXMLParser, caplog: LogCaptureFixture) -> None:
    logger.setLevel("INFO")
    g = Graph()
    s, p, o = URIRef("s"), URIRef("first_unknown"), Literal("42")
    s2, p2, o2 = URIRef("s2"), URIRef("also_unknown"), Literal("42")
    g.add((s, p, o))
    g.add((s2, p2, o2))

    inst = cimxmlinstance_w_prefixes
    inst.slot_index = {"p": DummySlot("p", "string")}

    result = inst.enrich_literal_datatypes(g)
    assert sorted(result) == sorted([(s, p, o), (s2, p2, o2)])
    assert "also_unknown" in caplog.text
    assert "first_unknown" in caplog.text
    mock_resolve.assert_not_called()
    mock_create.assert_not_called()

@patch("cim_plugin.cimxml_parser.resolve_datatype_from_slot", return_value=None)
@patch("cim_plugin.cimxml_parser.create_typed_literal")
def test_enrich_literal_datatypes_nodatatyperesolved(mock_create: MagicMock, mock_resolve: MagicMock, cimxmlinstance_w_prefixes: CIMXMLParser, caplog: LogCaptureFixture) -> None:
    logger.setLevel("INFO")
    g = Graph()
    s, p, o = URIRef("s"), URIRef("p"), Literal("hello")
    g.add((s, p, o))

    inst = cimxmlinstance_w_prefixes
    inst.slot_index = {"p": DummySlot("p", "string")}

    result = inst.enrich_literal_datatypes(g)

    assert list(result) == [(s, p, o)]
    mock_resolve.assert_called_once_with(inst.schemaview, inst.slot_index["p"])
    mock_create.assert_not_called()
    assert "No datatype found for range: string, for p" in caplog.text


@patch("cim_plugin.cimxml_parser.resolve_datatype_from_slot", return_value=URIRef("xsd:string"))
@patch("cim_plugin.cimxml_parser.create_typed_literal")
def test_enrich_literal_datatypes_successfulenrichment(mock_create: MagicMock, mock_resolve: MagicMock, cimxmlinstance_w_prefixes: CIMXMLParser, caplog: LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    g = Graph()
    s, p, o = URIRef("s"), URIRef("p"), Literal("hello")
    g.add((s, p, o))

    new_lit = Literal("hello", datatype=URIRef("xsd:string"))
    mock_create.return_value = new_lit

    inst = cimxmlinstance_w_prefixes
    inst.slot_index = {"p": DummySlot("p", "string")}

    result = inst.enrich_literal_datatypes(g)

    triples = list(result)
    assert len(triples) == 1
    assert triples[0] == (s, p, new_lit)

    mock_resolve.assert_called_once()
    mock_create.assert_called_once_with("hello", URIRef("xsd:string"), inst.schemaview)
    assert "Enriching done. Added datatypes to 1 triples." in caplog.text


@patch("cim_plugin.cimxml_parser.resolve_datatype_from_slot", return_value=URIRef("xsd:int"))
@patch("cim_plugin.cimxml_parser.create_typed_literal", side_effect=LiteralCastingError("bad cast"))
def test_enrich_literal_datatypes_castingerror(mock_create: MagicMock, mock_resolve: MagicMock, cimxmlinstance_w_prefixes: CIMXMLParser, caplog: LogCaptureFixture) -> None:
    g = Graph()
    s, p, o = URIRef("s"), URIRef("p"), Literal("not_an_int")
    s2, p2, o2 = URIRef("s2"), URIRef("p"), Literal("also_not_int")
    g.add((s, p, o))
    g.add((s2, p2, o2))

    inst = cimxmlinstance_w_prefixes
    inst.slot_index = {"p": DummySlot("p", "integer")}

    result = inst.enrich_literal_datatypes(g)

    assert len(list(result)) == 2
    assert (s, p, o) in list(result)
    assert (s2, p2, o2) in list(result)
    assert mock_resolve.call_count == 2
    assert mock_create.call_count == 2
    assert any("Error casting" in rec.message for rec in caplog.records)
    assert "Error casting also_not_int for s2, p: bad cast\n" in caplog.text
    assert "Error casting not_an_int for s, p: bad cast\n"  in caplog.text


def test_enrich_literal_datatypes_integrated(make_schemaview: Callable[..., SchemaView], caplog: LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    classes = {"A": ClassDefinition(name="A", attributes={"c1": SlotDefinition(name="c1", range="string"),
                                                          "c2": SlotDefinition(name="c2", range="Custom")})}
    types = {"Custom": TypeDefinition(name="Custom", base="integer", uri="xsd:integer")}
    prefixes = {"ex": {"prefix_prefix": "ex", "prefix_reference": "www.example.org"}}
    sv = make_schemaview(classes=classes, types=types, prefixes=prefixes)
    inst = CIMXMLParser()
    inst.schemaview = sv
    inst.slot_index = {"c1": DummySlot("c1", "string"), "c2": DummySlot("c2", "Custom")}
    g = Graph()
    g.add((URIRef("s"), URIRef("c1"), Literal("hello")))
    g.add((URIRef("s"), URIRef("c1"), Literal("hei", lang="no")))
    g.add((URIRef("s"), URIRef("c2"), Literal("1")))
    g.add((URIRef("s"), URIRef("d"), URIRef("not-a-literal")))
    result = inst.enrich_literal_datatypes(g)
    assert (URIRef("s"), URIRef("c1"), Literal("hello", datatype=URIRef('http://www.w3.org/2001/XMLSchema#string'))) in list(result)
    assert (URIRef("s"), URIRef("c1"), Literal("hei", lang="no")) in list(result)   # Literals with language tag is not given datatype
    assert (URIRef("s"), URIRef("c2"), Literal(1, datatype=URIRef('http://www.w3.org/2001/XMLSchema#integer'))) in list(result)
    assert (URIRef("s"), URIRef("d"), URIRef("not-a-literal")) in list(result)
    assert "Enriching done. Added datatypes to 2 triples." in caplog.text


def test_enrich_literal_datatypes_noupdates(make_schemaview: Callable[..., SchemaView], caplog: LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    classes = {"A": ClassDefinition(name="A", attributes={"c1": SlotDefinition(name="c1", range="string"),
                                                          "c2": SlotDefinition(name="c2", range="Custom")})}
    types = {"Custom": TypeDefinition(name="Custom", base="integer", uri="xsd:integer")}
    prefixes = {"ex": {"prefix_prefix": "ex", "prefix_reference": "www.example.org"}}
    sv = make_schemaview(classes=classes, types=types, prefixes=prefixes)
    inst = CIMXMLParser()
    inst.schemaview = sv
    inst.slot_index = {"c1": DummySlot("c1", "string"), "c2": DummySlot("c2", "Custom")}
    g = Graph()
    g.add((URIRef("s"), URIRef("d"), URIRef("not-a-literal")))
    result = inst.enrich_literal_datatypes(g)
    assert (URIRef("s"), URIRef("d"), URIRef("not-a-literal")) in list(result)
    assert "Enriching done. Added datatypes to 0 triples." in caplog.text



if __name__ == "__main__":
    pytest.main()