import pytest
from unittest.mock import patch, MagicMock, call
from cim_plugin.cimxml import CIMXMLParser
from tests.test_cimxml_linkml import make_schemaview
from pytest import LogCaptureFixture
from typing import Callable
from linkml_runtime import SchemaView
from rdflib import URIRef, Graph, Literal, BNode

@pytest.fixture
def instance(make_schemaview):
    """Create an instance with a real SchemaView."""
    obj = CIMXMLParser()
    obj.schemaview = make_schemaview(prefixes={"ex": {"prefix_prefix": "ex", "prefix_reference": "www.example.org"}})
    return obj

# Unit tests .ensure_correct_namespace_model

@patch("cim_plugin.cimxml.update_namespace_in_model")
@patch("cim_plugin.cimxml._get_current_namespace_from_model")
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
@patch("cim_plugin.cimxml.update_namespace_in_model")
@patch("cim_plugin.cimxml._get_current_namespace_from_model")
def testensure_correct_namespace_model_namespacehandling(mock_get: MagicMock, mock_update: MagicMock, instance: CIMXMLParser, current: str, new_ns: str, update: bool, caplog: LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    mock_get.return_value = current

    instance.ensure_correct_namespace_model("ex", new_ns)

    mock_get.assert_called_once()
    if update:
        mock_update.assert_called_once_with(instance.schemaview, "ex", new_ns)
        assert f"Wrong namespace detected for ex in model. Correcting to {new_ns}." in caplog.text
    else:
        mock_update.assert_not_called()
        assert "Model has correct namespace for ex." in caplog.text

@patch("cim_plugin.cimxml.update_namespace_in_model")
@patch("cim_plugin.cimxml._get_current_namespace_from_model")
def test_ensure_correct_namespace_model_prefixnotfound(mock_get: MagicMock, mock_update: MagicMock, instance: CIMXMLParser) -> None:
    mock_get.return_value = None

    with pytest.raises(ValueError, match="Prefix ex not found"):
        instance.ensure_correct_namespace_model("ex", "new_ns")
    
    mock_get.assert_called_once()
    mock_update.assert_not_called()

@patch("cim_plugin.cimxml.update_namespace_in_model")
@patch("cim_plugin.cimxml._get_current_namespace_from_model")
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


@patch("cim_plugin.cimxml.update_namespace_in_model")
@patch("cim_plugin.cimxml._get_current_namespace_from_model")
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


# Unit tests .normalize_rdf_ids

@patch("cim_plugin.cimxml._clean_uri")
@patch("cim_plugin.cimxml.detect_uri_collisions")
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
@patch("cim_plugin.cimxml._clean_uri")
@patch("cim_plugin.cimxml.detect_uri_collisions")
def test_normalize_rdf_ids_callscleanuri(mock_detect: MagicMock, mock_clean: MagicMock, s: URIRef|BNode, o: Literal|URIRef|BNode, calls: list) -> None:
    mock_detect.return_value = None
    mock_clean.side_effect = lambda uri, *_: URIRef("urn:uuid:test")

    parser = CIMXMLParser()
    g = Graph()
    g.add((s, URIRef("p"), o))

    parser.normalize_rdf_ids(g)

    mock_detect.assert_called_once()
    assert mock_clean.mock_calls == calls


@patch("cim_plugin.cimxml._clean_uri")
@patch("cim_plugin.cimxml.detect_uri_collisions")
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


@patch("cim_plugin.cimxml._clean_uri")
@patch("cim_plugin.cimxml.detect_uri_collisions")
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


@patch("cim_plugin.cimxml._clean_uri")
@patch("cim_plugin.cimxml.detect_uri_collisions")
def test_normalize_rdf_ids_emptygraph(mock_detect: MagicMock, mock_clean: MagicMock) -> None:
    mock_detect.return_value = None

    parser = CIMXMLParser()
    g = Graph()

    parser.normalize_rdf_ids(g)

    mock_clean.assert_not_called()
    mock_detect.assert_called_once()


if __name__ == "__main__":
    pytest.main()