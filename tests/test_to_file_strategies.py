from multiprocessing import context

import pytest
from unittest.mock import MagicMock, Mock, patch
from typing import Any, Callable
from tests.fixtures import make_schemaview, make_cimgraph
from cim_plugin.graph import CIMGraph
from cim_plugin.processor import CIMProcessor
from cim_plugin.jsonld_utilities import DEFAULT_CONTEXT_LINK
from rdflib import Literal, URIRef
from rdflib.namespace import RDF, DCAT
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model.meta import SlotDefinition, ClassDefinition

from cim_plugin.to_file_strategies import SerializationStrategy, TrigStrategy, CIMXMLStrategy, JSONLDStrategy, _select_strategy, _validate_options

# Unit tests TrigStrategy
def test_trigstrategy_noschemapath() -> None:
    processor = Mock()
    processor.graph = Mock(metadata_header=None)
    processor.schema = None

    strategy = TrigStrategy("out.trig", enrich_datatypes=False, schema_path=None)
    strategy.serialize(processor)
    processor.set_schema.assert_not_called()
    processor.graph.serialize.assert_called_once_with("out.trig", format="cimtrig") 

def test_trigstrategy_setschema() -> None:
    processor = Mock()
    processor.graph = Mock(metadata_header=None)
    processor.schema = "fake_schema"

    strategy = TrigStrategy("out.trig", enrich_datatypes=True, schema_path="dummy.yaml")
    strategy.serialize(processor)

    processor.set_schema.assert_called_once_with("dummy.yaml")
    processor.enrich_literal_datatypes.assert_called_once()
    processor.graph.serialize.assert_called_once_with("out.trig", format="cimtrig") 

def test_trigstrategy_setschemaenrichfalse() -> None:
    # Technically this is pointless as you don't need schema if you are not doing enriching.
    # But the file will be made regardless. No error, no warning.
    processor = Mock()
    processor.graph = Mock(metadata_header=None)
    processor.schema = None

    strategy = TrigStrategy("out.trig", enrich_datatypes=False, schema_path="dummy.yaml")
    strategy.serialize(processor)

    processor.set_schema.assert_called_once_with("dummy.yaml")
    processor.enrich_literal_datatypes.assert_not_called()
    processor.graph.serialize.assert_called_once_with("out.trig", format="cimtrig") 


def test_trigstrategy_noenrichwithschema():
    processor = Mock()
    processor.graph = Mock(metadata_header=None)
    processor.schema = "fake_schema"

    strategy = TrigStrategy("out.trig", enrich_datatypes=False)
    strategy.serialize(processor)

    processor.enrich_literal_datatypes.assert_not_called()
    processor.graph.serialize.assert_called_once_with("out.trig", format="cimtrig") 

@pytest.mark.parametrize("schema", ["fake_schema", None])
def test_trigstrategy_enrichesdatatypes(schema: str|None, caplog: pytest.LogCaptureFixture) -> None:
    processor = Mock()
    processor.graph = Mock(metadata_header=None)
    processor.schema = schema

    strategy = TrigStrategy("out.trig", enrich_datatypes=True)
    strategy.serialize(processor)

    if schema:
        processor.enrich_literal_datatypes.assert_called_once()
        assert "Cannot enrich datatypes" not in caplog.text
    else:
        assert "Cannot enrich datatypes" in caplog.text
        processor.enrich_literal_datatypes.assert_not_called()

    processor.graph.serialize.assert_called_once_with("out.trig", format="cimtrig") 

@pytest.mark.parametrize("header", ["fake_header", None])
def test_trigstrategy_mergeheader(header: str|None) -> None:
    processor = Mock()
    processor.header = header
    processor.schema = None

    strategy = TrigStrategy("out.trig")
    strategy.serialize(processor)

    if header:
        processor.merge_header.assert_called_once()
    else:
        processor.merge_header.assert_not_called()

    processor.graph.serialize.assert_called_once_with("out.trig", format="cimtrig") 

# Unit tests CIMXMLStrategy
@pytest.mark.parametrize("header", ["fake_header", None])
def test_cimxmlstrategy_header(header: str|None, caplog: pytest.LogCaptureFixture) -> None:
    processor = Mock()
    processor.header = header
    
    strategy = CIMXMLStrategy("out.xml")
    strategy.serialize(processor)

    processor.merge_header.assert_not_called()
    
    if header:
        assert "Serializing without an extracted header may create a corrupt CIMXML file." not in caplog.text
    else:
        assert "Serializing without an extracted header may create a corrupt CIMXML file." in caplog.text

    processor.graph.serialize.assert_called_once_with("out.xml", format="cimxml", qualifier=None) 


def test_cimxmlstrategy_qualifier() -> None:
    processor = Mock()
    processor.graph = Mock(metadata_header="fake_header")
    
    strategy = CIMXMLStrategy("out.xml", qualifier="urn")
    strategy.serialize(processor)

    processor.graph.serialize.assert_called_once_with("out.xml", format="cimxml", qualifier="urn") 


# Unit tests JSONLDStrategy
@pytest.mark.parametrize("input_context", [None, "context.json"])
@patch("cim_plugin.to_file_strategies.open")
@patch("cim_plugin.to_file_strategies.reorder_jsonld")
def test_jsonldstrategy_context(mock_reorder: MagicMock, mock_open: MagicMock, input_context: str|None) -> None:
    processor = Mock()
    processor.header = Mock()
    header_subject = "h1"
    processor.header.subject = header_subject
    processor.graph.serialize = Mock(return_value="serialized graph")
    
    strategy = JSONLDStrategy("out.json", context=input_context)
    strategy.enrich_datatypes = Mock()
    
    strategy.serialize(processor)

    context = input_context or DEFAULT_CONTEXT_LINK

    assert strategy.context == context
    processor.merge_header.assert_called_once()
    strategy.enrich_datatypes.assert_called_once_with(processor)
    processor.graph.serialize.assert_called_once_with(format="json-ld", context=context, auto_compact=True)
    mock_reorder.assert_called_once_with("serialized graph", priority_subject=header_subject)
    mock_open.assert_called_once_with("out.json", "w", encoding="utf-8")
    handle = mock_open.return_value.__enter__.return_value
    handle.write.assert_called_once_with(mock_reorder.return_value)


@patch("cim_plugin.to_file_strategies.open")
@patch("cim_plugin.to_file_strategies.reorder_jsonld")
def test_jsonldstrategy_noheader(mock_reorder: MagicMock, mock_open: MagicMock) -> None:
    processor = Mock()
    processor.header = None
    processor.graph.serialize = Mock(return_value="serialized graph")
    
    strategy = JSONLDStrategy("out.json")
    strategy.enrich_datatypes = Mock()
    
    strategy.serialize(processor)

    processor.merge_header.assert_not_called()
    strategy.enrich_datatypes.assert_called_once_with(processor)
    processor.graph.serialize.assert_called_once_with(format="json-ld", context=DEFAULT_CONTEXT_LINK, auto_compact=True)
    mock_reorder.assert_called_once_with("serialized graph", priority_subject=None)
    mock_open.assert_called_once_with("out.json", "w", encoding="utf-8")
    handle = mock_open.return_value.__enter__.return_value
    handle.write.assert_called_once_with(mock_reorder.return_value)

@pytest.mark.parametrize(
    "context, load_return",
    [
        pytest.param("http://example.com/context", {"@context": {"some": "context"}}, id="Context as URL"),
        pytest.param({"@context": {"some": "context"}}, None, id="Context as dict"),
        pytest.param(1, TypeError, id="Context as invalid type")
    ]
)
@patch("cim_plugin.to_file_strategies.enrich_graph_datatypes")
@patch("cim_plugin.to_file_strategies.extract_datatype_map")
@patch("cim_plugin.to_file_strategies.load_json_from_url")
def test_jsonldstrategy_enrichdatatypes(mock_load: MagicMock, mock_extract: MagicMock, mock_enrich: MagicMock, context: str|dict, load_return: dict|None|Exception) -> None:
    g = CIMGraph()
    processor = CIMProcessor(g)
    mock_load.return_value = load_return
    mock_extract.return_value = {"some": "dict"}

    strategy = JSONLDStrategy("out.json", context=context)
    if isinstance(load_return, type) and issubclass(load_return, Exception):
        with pytest.raises(load_return):
            strategy.enrich_datatypes(processor)
    else:
        strategy.enrich_datatypes(processor)

        if load_return:
            mock_load.assert_called_once_with(context)
        else:
            mock_load.assert_not_called()

        mock_extract.assert_called_once_with({"@context": {"some": "context"}})
        mock_enrich.assert_called_once_with(g, {"some": "dict"})
        assert context == strategy.context  # The context has not been modified


# Unit tests _select_strategy
@pytest.mark.parametrize(
    "format, options, expected, error",
    [
        pytest.param("trig", {}, TrigStrategy, False, id="Trig, no options"),
        pytest.param("trig", {"enrich_datatypes": True, "schema_path": "dummy.yaml"}, TrigStrategy, False, id="Trig, allowed options"),
        pytest.param("trig", {"qualifier": "urn"}, TrigStrategy, True, id="Trig, invalid options"),
        pytest.param("cimxml", {}, CIMXMLStrategy, False, id="CIMXML, no options"),
        pytest.param("cimxml", {"qualifier": "urn"}, CIMXMLStrategy, False, id="CIMXML, allowed options"),
        pytest.param("cimxml", {"enrich_datatypes": True}, CIMXMLStrategy, True, id="CIMXML, invalid options"),
        pytest.param("jsonld", {}, JSONLDStrategy, False, id="JSON-LD, no options"),
        pytest.param("jsonld", {"context": "http://example.com/context"}, JSONLDStrategy, False, id="JSON-LD, allowed options"),
        pytest.param("jsonld", {"enrich_datatypes": True}, JSONLDStrategy, True, id="JSON-LD, invalid options")
    ]
)
def test_select_strategy_various(format: str, options: dict[str, Any], expected: SerializationStrategy, error: bool) -> None:
    file_path="dummy"
    if not error:
        result = _select_strategy(format=format, file_path=file_path, options=options)
        assert type(result) == expected
    else:
        with pytest.raises(ValueError) as exc:
            _select_strategy(format=format, file_path=file_path, options=options)
        
        options_set = set(options)
        assert f"Options {options_set} are not valid for this format. " in str(exc.value)

def test_select_strategy_unknownformat() -> None:
    with pytest.raises(ValueError) as exc:
        _select_strategy("turtle", "dummy.ttl", {})
    
    assert "Unknown format: turtle" in str(exc.value)


def test_select_strategy_passes_options():
    s = _select_strategy("trig", "file.trig", {"schema_path": "x", "enrich_datatypes": True})
    assert isinstance(s, TrigStrategy)
    assert s.schema_path == "x"
    assert s.enrich_datatypes is True

# Unit tests _validate_options
@pytest.mark.parametrize(
        "options, allowed, error",
        [
            pytest.param({}, set(), False, id="No options, none allowed"),
            pytest.param({"one": 1}, set(), True, id="One option, none allowed"),
            pytest.param({}, {"one"}, False, id="No options, one allowed"),
            pytest.param({"one": 1}, {"one"}, False, id="One option"),
            pytest.param({"one": 1, "two": 2}, {"one", "two"}, False, id="Multiple options"),
            pytest.param({"one": 1, "two": 2}, {"one"}, True, id="Multiple options, one not allowed"),
            pytest.param({"one": 1, "two": 2}, {"three"}, True, id="Multiple options, none allowed")
        ]
)
def test_validate_options_various(options: dict[str, Any], allowed: set[str], error: bool) -> None:
    if error:
        with pytest.raises(ValueError) as exc:
            _validate_options(options=options, allowed=allowed)
            assert "not valid for this format." in str(exc.value)
    else:
        _validate_options(options=options, allowed=allowed)


# Integration tests
def test_trigstrategy_integration(make_cimgraph: CIMGraph, make_schemaview: Callable[..., SchemaView]) -> None:
    classes = {"A": ClassDefinition(name="A", attributes={"http://www.example.com/c1": SlotDefinition(name="http://www.example.com/c1", range="integer")})}
    prefixes = {"ex": {"prefix_prefix": "ex", "prefix_reference": "http://www.example.com/"}}
    sv = make_schemaview(classes=classes, prefixes=prefixes)
    
    g = make_cimgraph
    g.add((URIRef("http://www.example.com/s1"), URIRef("http://www.example.com/c1"), Literal("1")))
    pr = CIMProcessor(g)
    pr.schema = sv
    pr.slot_index = {"http://www.example.com/c1": SlotDefinition(name="c1", range="integer")}
    pr.graph.serialize = Mock()

    strategy = TrigStrategy(file_path="dummy.trig", schema_path=None, enrich_datatypes=True)
    strategy.serialize(pr)
    
    assert (URIRef("http://example.com/header"), RDF.type, DCAT.Dataset) in pr.graph
    assert (URIRef("http://www.example.com/s1"), URIRef("http://www.example.com/c1"), Literal(1, datatype=URIRef('http://www.w3.org/2001/XMLSchema#integer'))) in pr.graph
    pr.graph.serialize.assert_called_once_with("dummy.trig", format="cimtrig")

if __name__ == "__main__":
    pytest.main()
