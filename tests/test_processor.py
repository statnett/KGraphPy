from unittest import result

import pytest
from unittest.mock import call, patch, MagicMock
from rdflib import Namespace, URIRef, Literal, BNode, Graph
from cim_plugin.header import CIMMetadataHeader
from cim_plugin.graph import CIMGraph
from cim_plugin.namespaces import MD, DCAT_EXT
from cim_plugin.exceptions import LiteralCastingError
from rdflib.namespace import RDF
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model.meta import SlotDefinition, ClassDefinition, TypeDefinition, Prefix
from tests.fixtures import make_schemaview, make_slot_index, make_cimgraph
from typing import Callable
import logging

from cim_plugin.processor import CIMProcessor, merge_namespace_managers, replace_namespace, _make_header_graph_for_conversion

logger = logging.getLogger('cimxml_logger')

# Unit tests .identifier
def test_identifier_immutability() -> None:
    g = CIMGraph(identifier="graph1")
    pr = CIMProcessor(g)
    assert pr.identifier == URIRef("graph1")
    with pytest.raises(AttributeError):
        # Pylance silenced to test an invalid action
        pr.identifier = URIRef("graph2")    # type: ignore


def test_identifier_updates() -> None:
    g = CIMGraph(identifier="graph1")
    pr = CIMProcessor(g)

    pr.graph = CIMGraph(identifier="graph2")
    assert pr.identifier == URIRef("graph2")

# Unit tests .replace_header
@patch("cim_plugin.processor.merge_namespace_managers")
def test_replace_header_inputnone(mock_merge: MagicMock) -> None:
    header1 = CIMMetadataHeader.empty(URIRef("h1"))
    g = CIMGraph()
    g.metadata_header = header1
    
    pr = CIMProcessor(g)
    assert pr.graph.metadata_header
    assert pr.graph.metadata_header.subject == URIRef("h1")

    pr.replace_header(None)
    assert pr.graph.metadata_header is None
    mock_merge.assert_not_called()

def test_replace_header_basic() -> None:
    header1 = CIMMetadataHeader.empty(URIRef("h1"))
    header2 = CIMMetadataHeader.empty(URIRef("h2"))
    g = CIMGraph()
    g.metadata_header = header1
    
    pr = CIMProcessor(g)
    assert pr.graph.metadata_header
    assert pr.graph.metadata_header.subject == URIRef("h1")

    pr.replace_header(header2)
    
    assert pr.graph.metadata_header.subject == URIRef("h2")
    assert pr.graph.metadata_header is header2

@patch("cim_plugin.processor.merge_namespace_managers")
def test_replace_header_checkcalled(mock_merge: MagicMock) -> None:
    header1 = CIMMetadataHeader.empty(URIRef("h1"))
    header2 = CIMMetadataHeader.empty(URIRef("h2"))
    g = CIMGraph()
    g.metadata_header = header1
    
    pr = CIMProcessor(g)
    assert pr.graph.metadata_header
    assert pr.graph.metadata_header.subject == URIRef("h1")

    pr.replace_header(header2)
    
    assert pr.graph.metadata_header.subject == URIRef("h2")
    mock_merge.assert_called_once_with(pr.graph.namespace_manager, header2.graph.namespace_manager)

@pytest.mark.parametrize(
        "prefix, namespace, collision",
        [
            pytest.param("ex", "https://example.com/", False, id="Same prefix in new header as old"),
            pytest.param("new", "https://new.com/", False, id="Different prefix in new header"),
            pytest.param("ex", "https://new.com/", False, id="Same prefix, different namespace in new header"),
            pytest.param("foo", "www.bar.org/", False, id="Same prefix, namespace as data"),
            pytest.param("foo", "www.foo.org/", True, id="Same prefix, new namespace then data"),
            pytest.param("foo", "https//oldname.com/", True, id="New namespace, but namespace already exist"),
            pytest.param("foo", "www.bar.org/foo", True, id="Overlapped namespaces"),
        ]
)
def test_replace_header_namespaces(prefix: str, namespace: str, collision: bool) -> None:
    # Compares new and old namespaces
    header1 = CIMMetadataHeader.empty(URIRef("h1"))
    header1.graph.bind("ex", "https://example.com/")
    header2 = CIMMetadataHeader.empty(URIRef("h2"))
    header2.graph.bind(prefix, namespace)
    g = CIMGraph()
    g.bind("old", "https//oldname.com/")
    g.bind("foo", "www.bar.org/")
    g.metadata_header = header1
    
    pr = CIMProcessor(g)    
    pr.replace_header(header2)

    assert pr.graph.metadata_header
    header_nm = pr.graph.metadata_header.graph.namespace_manager.store
    data_nm = pr.graph.namespace_manager.store

    # The data namespaces are never changed, just added to
    assert data_nm.namespace("foo") == URIRef("www.bar.org/")
    assert data_nm.namespace("old") == URIRef("https//oldname.com/")

    if not collision:
        assert header_nm.namespace(prefix) == URIRef(namespace)
        assert data_nm.namespace(prefix) == header_nm.namespace(prefix)
    else:
        assert header_nm.namespace(prefix) == URIRef(namespace) # The header keeps its old namespace
        assert data_nm.namespace(prefix) != header_nm.namespace(prefix) # The header and data namespaces are now different
        assert f"{data_nm.namespace(prefix)} overwrites {namespace} for {prefix}"



def test_replace_header_nooverrideofoldprefix() -> None:
    header1 = CIMMetadataHeader.empty(URIRef("h1"))
    header1.graph.bind("new", "https//oldname.com/")
    g = CIMGraph()
    g.bind("ex", "https://example.com")
    g.bind("old", "https//oldname.com/")
    
    pr = CIMProcessor(g)    
    pr.replace_header(header1)
    assert pr.graph.metadata_header
    header_nm = pr.graph.metadata_header.graph.namespace_manager.store
    data_nm = pr.graph.namespace_manager.store
    
    assert header_nm.namespace("new") == data_nm.namespace("old")   # Header namespace_manager keeps its own prefix
    assert data_nm.namespace("old") == URIRef("https//oldname.com/")
    assert "new" not in list(data_nm.namespaces())


@patch("cim_plugin.processor.merge_namespace_managers")
def test_replace_header_inputwrongtype(mock_merge: MagicMock) -> None:
    header1 = CIMMetadataHeader.empty(URIRef("h1"))
    g = CIMGraph()
    g.metadata_header = header1
    
    pr = CIMProcessor(g)
    
    with pytest.raises(TypeError) as exc:
        # Pylance silenced to test wrong input type.
        pr.replace_header("header") # type: ignore
    
    assert "The new header must be of type CIMMetadataHeader." in str(exc.value)
    mock_merge.assert_not_called()


# Unit tests .extract_header
@patch("cim_plugin.processor.create_header_attribute")
def test_extract_header_calls(mock_create: MagicMock) -> None:
    header = CIMMetadataHeader.empty(URIRef("h1"))
    mock_create.return_value = header
    g = CIMGraph()
    pr = CIMProcessor(g)
    
    pr.extract_header()

    mock_create.assert_called_once_with(g)
    assert pr.graph.metadata_header
    assert pr.graph.metadata_header is header
    assert pr.graph.metadata_header.subject == URIRef("h1")


@patch("cim_plugin.processor.create_header_attribute")
def test_extract_header_createexception(mock_create: MagicMock) -> None:
    mock_create.side_effect = TypeError
    g = CIMGraph()
    pr = CIMProcessor(g)
    
    with pytest.raises(TypeError):
        pr.extract_header()

    mock_create.assert_called_once_with(g)
    assert pr.graph.metadata_header is None


def test_extract_header_basic() -> None:
    g = CIMGraph()
    g.add((URIRef("h1"), RDF.type, DCAT_EXT.Dataset))
    g.add((URIRef("s1"), URIRef("p1"), URIRef("o")))

    pr = CIMProcessor(g)
    
    pr.extract_header()

    assert pr.graph.metadata_header
    assert pr.graph.metadata_header.subject == URIRef("h1")
    assert (URIRef("h1"), RDF.type, DCAT_EXT.Dataset) in pr.graph.metadata_header.triples
    assert len(pr.graph) == 1
    assert (URIRef("s1"), URIRef("p1"), URIRef("o")) in pr.graph


def test_extract_header_headeralready(caplog: pytest.LogCaptureFixture) -> None:
    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty(URIRef("h1"))
    g.metadata_header.add_triple(RDF.type, DCAT_EXT.Dataset)
    g.add((URIRef("s1"), URIRef("p1"), URIRef("o")))

    pr = CIMProcessor(g)
    
    pr.extract_header()

    assert pr.graph.metadata_header
    assert pr.graph.metadata_header.subject == URIRef("h1") # Header remains unchanged
    assert (URIRef("h1"), RDF.type, DCAT_EXT.Dataset) in pr.graph.metadata_header.triples
    assert len(pr.graph) == 1
    assert (URIRef("s1"), URIRef("p1"), URIRef("o")) in pr.graph
    assert caplog.records[0].levelname == "ERROR"
    assert caplog.records[0].message == "Metadata header already exist. Use .replace_header instead."


def test_extract_header_multiplecalls(caplog: pytest.LogCaptureFixture) -> None:
    g = CIMGraph()
    g.add((URIRef("h1"), RDF.type, DCAT_EXT.Dataset))
    g.add((URIRef("s1"), URIRef("p1"), URIRef("o")))

    pr = CIMProcessor(g)
    
    pr.extract_header()
    pr.extract_header()

    assert pr.graph.metadata_header
    assert pr.graph.metadata_header.subject == URIRef("h1")
    assert (URIRef("h1"), RDF.type, DCAT_EXT.Dataset) in pr.graph.metadata_header.triples
    assert len(pr.graph) == 1
    assert (URIRef("s1"), URIRef("p1"), URIRef("o")) in pr.graph
    assert "Metadata header already exist. Use .replace_header instead." in caplog.text


def test_extract_header_largergraphs() -> None:
    g = CIMGraph()
    g.add((URIRef("h1"), RDF.type, DCAT_EXT.Dataset))
    g.add((URIRef("h1"), DCAT_EXT.keyword, Literal("header")))
    g.add((URIRef("s1"), URIRef("p1"), URIRef("o")))
    g.add((URIRef("s2"), URIRef("p2"), URIRef("o2")))

    pr = CIMProcessor(g)
    
    pr.extract_header()

    assert pr.graph.metadata_header
    assert pr.graph.metadata_header.subject == URIRef("h1")
    header_triples = pr.graph.metadata_header.triples
    assert len(header_triples) == 2
    assert (URIRef("h1"), DCAT_EXT.keyword, Literal("header")) in header_triples
    assert (URIRef("h1"), RDF.type, DCAT_EXT.Dataset) in header_triples
    assert len(pr.graph) == 2
    assert (URIRef("s1"), URIRef("p1"), URIRef("o")) in pr.graph
    assert (URIRef("s2"), URIRef("p2"), URIRef("o2")) in pr.graph


def test_extract_header_bnodes() -> None:
    g = CIMGraph()
    g.add((URIRef("h1"), RDF.type, DCAT_EXT.Dataset))
    g.add((URIRef("s1"), URIRef("p1"), URIRef("o")))
    b1 = BNode() 
    b2 = BNode() 
    g.add((URIRef("h1"), URIRef("urn:p:1"), b1)) 
    g.add((b1, URIRef("urn:p:2"), b2)) 
    g.add((b2, URIRef("urn:p:3"), Literal("value"))) 

    pr = CIMProcessor(g)
    
    pr.extract_header()

    assert pr.graph.metadata_header
    assert pr.graph.metadata_header.subject == URIRef("h1")
    assert (URIRef("h1"), RDF.type, DCAT_EXT.Dataset) in pr.graph.metadata_header.triples
    assert (URIRef("h1"), URIRef("urn:p:3"), Literal("value")) in pr.graph.metadata_header.triples
    assert len(pr.graph) == 1
    assert (URIRef("s1"), URIRef("p1"), URIRef("o")) in pr.graph

# Unit tests .convert_header
@pytest.mark.parametrize("header", [None, CIMMetadataHeader.empty(URIRef("h1"))])
@patch("cim_plugin.processor._make_header_graph_for_conversion")
def test_convert_header_headernone(mock_make: MagicMock, header: CIMMetadataHeader | None, caplog: pytest.LogCaptureFixture) -> None:
    g = CIMGraph()
    g.metadata_header = header
    pr = CIMProcessor(g)
    
    pr.convert_header()

    mock_make.assert_not_called()
    assert pr.graph.metadata_header == header   # No changes to header
    assert "No metadata header found for conversion." in caplog.text

@patch("cim_plugin.processor.convert_triple")
@patch("cim_plugin.processor._make_header_graph_for_conversion")
def test_convert_header_typetriplenotconverted(mock_make: MagicMock, mock_convert: MagicMock, make_cimgraph: CIMGraph, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    g = make_cimgraph   # The header contains the type triple, but no other triples
    assert g.metadata_header
    header_subject = g.metadata_header.subject
    new_graph = Graph()
    new_graph.add((header_subject, RDF.type, MD.FullModel))
    mock_make.return_value = ("md_fullmodel", new_graph)
    pr = CIMProcessor(g)
    
    pr.convert_header()

    mock_make.assert_called_once()
    mock_convert.assert_not_called()  # The type triple is not converted
    assert pr.graph.metadata_header
    assert len(pr.graph.metadata_header.triples) == 1
    assert (header_subject, RDF.type, MD.FullModel) in pr.graph.metadata_header.triples
    assert "Converted 0 header triples to md_fullmodel." in caplog.text
    assert "triples could not be converted and was not included in the new header" not in caplog.text  # No unconverted triples


@patch("cim_plugin.processor.convert_triple")
@patch("cim_plugin.processor._make_header_graph_for_conversion")
def test_convert_header_success(mock_make: MagicMock, mock_convert: MagicMock, make_cimgraph: CIMGraph, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    g = make_cimgraph
    assert g.metadata_header
    g.metadata_header.add_triple(DCAT_EXT.keyword, Literal("test"))
    g.metadata_header.add_triple(DCAT_EXT.version, Literal("unconverted_description"))
    header_subject = g.metadata_header.subject
    pr = CIMProcessor(g)
    new_graph = Graph()
    new_graph.add((header_subject, RDF.type, MD.FullModel))
    mock_make.return_value = ("md_fullmodel", new_graph)
    
    def convert_side_effect(triple, target_format=None):
        s, p, o = triple
        if p == DCAT_EXT.keyword:
            return (s, MD.Model.description, Literal("converted_keyword"))
        else:
            return None

    mock_convert.side_effect = convert_side_effect    
    
    pr.convert_header()

    assert pr.graph.metadata_header
    assert pr.graph.metadata_header.subject == header_subject
    mock_make.assert_called_once()
    mock_convert.assert_has_calls([call((header_subject, DCAT_EXT.keyword, Literal("test")), target_format="md_fullmodel"),
                                   call((header_subject, DCAT_EXT.version, Literal("unconverted_description")), target_format="md_fullmodel")], any_order=True)

    assert len(pr.graph.metadata_header.triples) == 2
    assert (header_subject, RDF.type, MD.FullModel) in pr.graph.metadata_header.triples
    assert (header_subject, MD.Model.description, Literal("converted_keyword")) in pr.graph.metadata_header.triples
    assert "Converted 1 header triples to md_fullmodel." in caplog.text
    assert "1 triples could not be converted and was not included in the new header" in caplog.text
    assert "Literal('unconverted_description')" in caplog.text  # The DCAT_CIM.version triple is not converted and should be logged

@patch("cim_plugin.processor.convert_triple")
@patch("cim_plugin.processor._make_header_graph_for_conversion")
def test_convert_header_makeheadererror(mock_make: MagicMock, mock_convert: MagicMock, make_cimgraph: CIMGraph, caplog: pytest.LogCaptureFixture) -> None:
    g = make_cimgraph
    pr = CIMProcessor(g)
    mock_make.side_effect = ValueError("Conversion not possible.")
    
    with pytest.raises(ValueError):
        pr.convert_header()

    mock_make.assert_called_once()
    mock_convert.assert_not_called()
    assert pr.graph.metadata_header is g.metadata_header  # No changes to header


@patch("cim_plugin.processor.convert_triple")
@patch("cim_plugin.processor._make_header_graph_for_conversion")
def test_convert_header_converttripleerror(mock_make: MagicMock, mock_convert: MagicMock, make_cimgraph: CIMGraph, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    g = make_cimgraph
    assert g.metadata_header
    g.metadata_header.add_triple(DCAT_EXT.keyword, Literal("test"))
    header_subject = g.metadata_header.subject
    pr = CIMProcessor(g)
    new_graph = Graph()
    new_graph.add((header_subject, RDF.type, MD.FullModel))
    mock_make.return_value = ("md_fullmodel", new_graph)
    mock_convert.side_effect = ValueError("Unknown target format")

    with pytest.raises(ValueError):
        pr.convert_header()

    mock_make.assert_called_once()
    mock_convert.assert_called_once_with((header_subject, DCAT_EXT.keyword, Literal("test")), target_format="md_fullmodel")
    assert pr.graph.metadata_header is g.metadata_header  # No changes to header
    

# Unit tests .merge_header
@patch("cim_plugin.processor.merge_namespace_managers")
def test_merge_header_headernone(mock_merge: MagicMock) -> None:
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), Literal("o")))
    g.metadata_header = None

    pr = CIMProcessor(g)
    pr.merge_header()

    assert len(pr.graph) == 1
    assert (URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), Literal("o")) in pr.graph
    mock_merge.assert_not_called()

def test_merge_header_basic() -> None:
    header = CIMMetadataHeader.empty(URIRef("h1"))
    header.add_triple(RDF.type, DCAT_EXT.Dataset)
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), Literal("o")))
    g.metadata_header = header

    pr = CIMProcessor(g)
    pr.merge_header()

    assert (URIRef("h1"), RDF.type, DCAT_EXT.Dataset) in pr.graph
    assert len(pr.graph) == 2

@patch("cim_plugin.processor.merge_namespace_managers")
def test_merge_header_mergecalled(mock_merge: MagicMock) -> None:
    header = CIMMetadataHeader.empty(URIRef("h1"))
    header.graph.bind("foo", "www.bar.org/")
    header.graph.bind("md", MD)
    header.add_triple(RDF.type, MD.FullModel)
    header.add_triple(URIRef("www.bar.org/p"), Literal("oh"))
    g = CIMGraph()
    g.bind("md", MD)
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), Literal("o")))
    g.metadata_header = header

    pr = CIMProcessor(g)
    pr.merge_header()

    assert pr.graph.metadata_header
    assert (URIRef("h1"), RDF.type, MD.FullModel) in pr.graph
    assert (URIRef("h1"), URIRef("www.bar.org/p"), Literal("oh")) in pr.graph
    assert len(pr.graph) == 3
    ns = pr.graph.namespace_manager.store
    assert ("md", URIRef('http://iec.ch/TC57/61970-552/ModelDescription/1#')) in list(ns.namespaces())
    # The namespaces are not merged when merge_namespace_managers is mocked
    assert ("foo", URIRef("www.bar.org/")) not in list(ns.namespaces())
    mock_merge.assert_called_once_with(pr.graph.namespace_manager, pr.graph.metadata_header.graph.namespace_manager)

def test_merge_header_namespaces() -> None:
    header = CIMMetadataHeader.empty(URIRef("h1"))
    header.graph.bind("foo", "www.bar.org/")
    header.graph.bind("md", MD)
    header.add_triple(RDF.type, MD.FullModel)
    header.add_triple(URIRef("www.bar.org/p"), Literal("oh"))
    g = CIMGraph()
    g.bind("md", MD)
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), Literal("o")))
    g.metadata_header = header # Using this line to add the header to the graph

    pr = CIMProcessor(g)
    pr.merge_header()

    assert (URIRef("h1"), RDF.type, MD.FullModel) in pr.graph
    assert (URIRef("h1"), URIRef("www.bar.org/p"), Literal("oh")) in pr.graph
    assert len(pr.graph) == 3
    ns = pr.graph.namespace_manager.store
    assert ("md", URIRef('http://iec.ch/TC57/61970-552/ModelDescription/1#')) in list(ns.namespaces())
    # Header namespaces automatically transferred
    assert ("foo", URIRef("www.bar.org/")) in list(ns.namespaces())

def test_merge_header_usingreplaceheader() -> None:
    # This test gives same result as the one above. 
    # merge_namespace_managers ensures namespaces will not have duplicates, even though it is run twice when .replace_header has been used.
    header = CIMMetadataHeader.empty(URIRef("h1"))
    header.graph.bind("foo", "www.bar.org/")
    header.graph.bind("md", MD)
    header.add_triple(RDF.type, MD.FullModel)
    header.add_triple(URIRef("www.bar.org/p"), Literal("oh"))
    g = CIMGraph()
    g.bind("md", MD)
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), Literal("o")))
    # g.metadata_header = header    # Using .replace_header instead
    ns_before = list(g.namespace_manager.store.namespaces())

    pr = CIMProcessor(g)
    pr.replace_header(header)
    pr.merge_header()

    assert (URIRef("h1"), RDF.type, MD.FullModel) in pr.graph
    assert (URIRef("h1"), URIRef("www.bar.org/p"), Literal("oh")) in pr.graph
    assert len(pr.graph) == 3
    ns_after = pr.graph.namespace_manager.store
    assert ("foo", URIRef("www.bar.org/")) not in ns_before
    assert len(list(ns_after.namespaces())) == len(ns_before) + 1 # Only one namespace has been added
    assert ("md", URIRef('http://iec.ch/TC57/61970-552/ModelDescription/1#')) in list(ns_after.namespaces())
    assert ("foo", URIRef("www.bar.org/")) in list(ns_after.namespaces())


def test_merge_header_idempotency() -> None:
    header = CIMMetadataHeader.empty(URIRef("h1"))
    header.graph.bind("md", MD)
    header.add_triple(RDF.type, MD.FullModel)
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), Literal("o")))
    g.metadata_header = header
    ns_before = list(g.namespace_manager.store.namespaces())

    pr = CIMProcessor(g)
    pr.merge_header()
    pr.merge_header()

    assert (URIRef("h1"), RDF.type, MD.FullModel) in pr.graph
    assert len(pr.graph) == 2   # No duplicate triples
    ns_after = pr.graph.namespace_manager.store
    assert len(list(ns_after.namespaces())) == len(ns_before) + 1 # Only one namespace has been added
    assert ("md", URIRef('http://iec.ch/TC57/61970-552/ModelDescription/1#')) in list(ns_after.namespaces())


def test_merge_header_noduplicatetriples() -> None:
    header = CIMMetadataHeader.empty(URIRef("h1"))
    header.graph.bind("md", MD)
    header.add_triple(RDF.type, MD.FullModel)
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.bind("md", MD)
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), Literal("o")))
    g.add((URIRef("h1"), RDF.type, MD.FullModel))   # Header triple already in graph
    g.metadata_header = header
    ns_before = list(g.namespace_manager.store.namespaces())

    pr = CIMProcessor(g)
    pr.merge_header()

    assert (URIRef("h1"), RDF.type, MD.FullModel) in pr.graph
    assert len(pr.graph) == 2   # No duplicate triples
    ns_after = pr.graph.namespace_manager.store
    assert len(list(ns_after.namespaces())) == len(ns_before) # No new namespace has been added


def test_merge_header_namespacecollision(caplog: pytest.LogCaptureFixture) -> None:
    header = CIMMetadataHeader.empty(URIRef("h1"))
    header.graph.bind("ex", "www.new.com/")
    header.add_triple(URIRef("www.new.com/p1"), Literal("oh"))
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), Literal("o")))
    g.metadata_header = header

    pr = CIMProcessor(g)
    pr.merge_header()

    # The triples uses the header namespace for ex, not the graph namespace
    assert (URIRef("h1"), URIRef("www.new.com/p1"), Literal("oh")) in pr.graph
    assert pr.graph.namespace_manager.store.namespace("ex") == URIRef("https://example.com/")
    assert pr.graph.metadata_header
    assert pr.graph.metadata_header.graph.namespace_manager.store.namespace("ex") == URIRef("www.new.com/")
    assert len(pr.graph) == 2
    assert caplog.records[0].levelname == "ERROR"
    assert "Namespace for 'ex' differs between graphs (https://example.com/ vs www.new.com/). https://example.com/ is kept." in caplog.text


def test_merge_header_emptyheader() -> None:
    header = CIMMetadataHeader.empty(URIRef("h1"))
    header.graph.bind("md", MD)
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), Literal("o")))
    g.metadata_header = header

    pr = CIMProcessor(g)
    pr.merge_header()

    assert len(pr.graph) == 1   # No added triples
    ns_after = pr.graph.namespace_manager.store
    assert ns_after.namespace("md") == URIRef('http://iec.ch/TC57/61970-552/ModelDescription/1#')

# Unit tests .namespaces_different_from_model
def test_namespaces_different_from_model_noschema() -> None:
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    pr = CIMProcessor(g)
    with pytest.raises(AttributeError) as exc:
        pr.namespaces_different_from_model()
    
    assert "No schema detected. Import a linkML schema using .set_schema()" in str(exc.value)


def test_namespaces_different_from_model_emptygraph(make_schemaview: Callable[..., SchemaView]) -> None:
    schema = make_schemaview()
    g = CIMGraph()
    pr = CIMProcessor(g)
    pr.schema = schema
    result = pr.namespaces_different_from_model()
    assert result == None

def test_namespaces_different_from_model_emptyschema(make_schemaview: Callable[..., SchemaView]) -> None:
    schema = make_schemaview(prefixes={})
    g = CIMGraph()
    g.bind("ex", "https://example.com")
    pr = CIMProcessor(g)
    pr.schema = schema
    result = pr.namespaces_different_from_model()
    assert result == None


@pytest.mark.parametrize(
        "graph_prefix, schema_ns, graph_ns, expected",
        [
            pytest.param("ex", "https://example.com/", "https://example.com/", None, id="No differences"),
            pytest.param("foo", "", "https://bar.com/", None, id="Prefix not shared"), # foo is not in schema and ex is not in graph
            pytest.param("ex", "https://example.com/", "https://new.com", {("ex", "https://new.com", "https://example.com/")}, id="Different namespace"),
            pytest.param("ex", "https://example.com/", "https://example.com/ ", {("ex", "https://example.com/ ", "https://example.com/")}, id="Whitespace in graph"),
            # pytest.param("ex", "https://example.com/ ", "https://example.com/", {("ex", "https://example.com/", "https://example.com/")}, id="Whitespace in schema"), # linkML does not allow whitespace in namespaces
            pytest.param("ex", "https://example.com/", "https://example.com", {("ex", "https://example.com", "https://example.com/")}, id="Missing /"),
            pytest.param("ex", "https://example.com/", "https://EXAMPLE.com/", {("ex", "https://EXAMPLE.com/", "https://example.com/")}, id="Uppercase"),
            pytest.param("ex", "https://example.com#", "https://example.com", {("ex", "https://example.com", "https://example.com#")}, id="Missing #"),
        ]
)
def test_namespaces_different_from_model_various(graph_prefix: str, schema_ns: str, graph_ns: str, expected: set|None, make_schemaview: Callable[..., SchemaView]) -> None:
    schema = make_schemaview(prefixes={"ex": schema_ns})
    g = CIMGraph()
    g.bind(graph_prefix, graph_ns)
    pr = CIMProcessor(g)
    pr.schema = schema
    result = pr.namespaces_different_from_model()
    assert result == expected


def test_namespaces_different_from_model_header(make_schemaview: Callable[..., SchemaView]) -> None:
    # Namespaces in the header is also checked with the schema
    schema = make_schemaview(prefixes={"ex": "https://example.com/", "foo": "www.bar.com#"})
    header = CIMMetadataHeader.empty(URIRef("h1"))
    header.add_triple(URIRef("www.bar.com/ph"), Literal("oh"))
    header.graph.bind("foo", "www.bar.com/")
    g = CIMGraph()
    g.bind("ex", "https://example.com")
    g.metadata_header = header
    pr = CIMProcessor(g)
    pr.schema = schema
    result = pr.namespaces_different_from_model()
    assert result == {("foo", "www.bar.com/", "www.bar.com#"), ("ex", "https://example.com", "https://example.com/")}

def test_namespaces_different_from_model_emptyheader(make_schemaview: Callable[..., SchemaView]) -> None:
    schema = make_schemaview(prefixes={"ex": "https://example.com/"})
    header = CIMMetadataHeader.empty(URIRef("h1"))
    g = CIMGraph()
    g.bind("ex", "https://example.com")
    g.metadata_header = header
    pr = CIMProcessor(g)
    pr.schema = schema
    result = pr.namespaces_different_from_model()
    assert result == {("ex", "https://example.com", "https://example.com/")}


@pytest.mark.parametrize(
        "header_ns, graph_ns, expected_result",
        [
            pytest.param("www.bar.com/", "www.bar.com#", {("foo", "www.bar.com#", "www.bar.com/")}, id="Graph wrong, header correct"),
            pytest.param("www.bar.com#", "www.bar.com/", {("foo", "www.bar.com#", "www.bar.com/")}, id="Graph correct, header wrong"),
            pytest.param("www.bar.com#", "www.bar.com#", {("foo", "www.bar.com#", "www.bar.com/")}, id="Both wrong, same issue"),
            pytest.param("www.bar.com#", "www.bar.com", {("foo", "www.bar.com#", "www.bar.com/"), ("foo", "www.bar.com", "www.bar.com/")}, id="Both wrong, different issues")
        ]
)
def test_namespaces_different_from_model_headervsgraph(header_ns: str, graph_ns: str, expected_result: set[tuple[str, URIRef]], make_schemaview: Callable[..., SchemaView]) -> None:
    schema = make_schemaview(prefixes={"foo": "www.bar.com/"})
    header = CIMMetadataHeader.empty(URIRef("h1"))
    header.add_triple(URIRef("www.bar.com/ph"), Literal("oh"))
    header.graph.bind("foo", header_ns)
    g = CIMGraph()
    g.bind("foo", graph_ns)
    g.metadata_header = header
    pr = CIMProcessor(g)
    pr.schema = schema
    result = pr.namespaces_different_from_model()
    assert result == expected_result


def test_namespaces_different_from_model_multiplebindings(make_schemaview: Callable[..., SchemaView]) -> None:
    # Namespaces in the header is also checked with the schema
    schema = make_schemaview(prefixes={"ex": "https://example.com/"})
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.bind("ex", "https://new.com", replace=True)
    pr = CIMProcessor(g)
    pr.schema = schema
    result = pr.namespaces_different_from_model()
    assert result == {("ex", "https://new.com", "https://example.com/")}

def test_namespaces_different_from_model_schemaprefixesalist(make_schemaview: Callable[..., SchemaView]) -> None:
    # The namespaces in the SchemaView is in the form of a list
    schema = make_schemaview(prefixes=[Prefix(prefix_prefix="ex", prefix_reference="https://example.com/"), Prefix(prefix_prefix="foo", prefix_reference="www.bar.com#")])
    header = CIMMetadataHeader.empty(URIRef("h1"))
    header.add_triple(URIRef("www.bar.com/ph"), Literal("oh"))
    header.graph.bind("foo", "www.bar.com/")
    g = CIMGraph()
    g.bind("ex", "https://example.com")
    g.metadata_header = header
    pr = CIMProcessor(g)
    pr.schema = schema
    result = pr.namespaces_different_from_model()
    assert result == {("foo", "www.bar.com/", "www.bar.com#"), ("ex", "https://example.com", "https://example.com/")}

# Unit tests .update_namespace
@pytest.mark.parametrize(
    "prefix, new_namespace",
    [
        pytest.param("ex", "www.new.com/", id="New namespace as string"),
        pytest.param("ex", URIRef("www.new.com/"), id="New namespace as URIRef"),
        pytest.param("ex", " www.new.com/ ", id="Whitespace"),
        pytest.param("ex", "www.new.com#", id="With #"),
        pytest.param("ex", "www.new.com", id="Missing /"),
        pytest.param("ex", "https://exAMPLE.com/", id="Same as old, but with uppercase"),
    ]
)
def test_update_namespace_various(prefix: str, new_namespace: str|URIRef) -> None:
    header = CIMMetadataHeader.empty(URIRef("https://example.com/h1"))
    header.graph.bind("ex", "https://example.com/")
    header.add_triple(URIRef("https://example.com/ph"), URIRef("https://example.com/oh"))
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), URIRef("https://example.com/o1")))
    g.metadata_header = header
    
    pr = CIMProcessor(g)
    pr.update_namespace(prefix=prefix, namespace=new_namespace)

    new_namespace = str(new_namespace).strip()
    assert pr.graph.metadata_header
    assert pr.graph.namespace_manager.store.namespace(prefix) == URIRef(new_namespace)
    assert pr.graph.metadata_header.graph.namespace_manager.store.namespace(prefix) == URIRef(new_namespace)
    assert (URIRef(f"{new_namespace}s1"), URIRef(f"{new_namespace}p1"), URIRef(f"{new_namespace}o1")) in pr.graph
    assert (URIRef(f"{new_namespace}h1"), URIRef(f"{new_namespace}ph"), URIRef(f"{new_namespace}oh")) in pr.graph.metadata_header.graph


@pytest.mark.parametrize(
    "prefix, new_namespace",
    [
        pytest.param("ex", "https://example.com/", id="Same as old"),
        pytest.param("new", "www.new.com/", id="New prefix"),
        pytest.param("EX", "https://example.com/", id="Uppercase prefix, seen as new"),
        pytest.param(None, "www.new.com/", id="None prefix"),
        pytest.param(" ", "www.new.com/", id="Empty prefix"),
    ]
)
@patch("cim_plugin.processor.update_namespace_in_triples")
def test_update_namespace_nochanges(mock_update: MagicMock, prefix: str, new_namespace: str|URIRef) -> None:
    header = CIMMetadataHeader.empty(URIRef("https://example.com/h1"))
    header.graph.bind("ex", "https://example.com/")
    header.add_triple(URIRef("https://example.com/ph"), URIRef("https://example.com/oh"))
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), URIRef("https://example.com/o1")))
    g.metadata_header = header
    
    pr = CIMProcessor(g)
    pr.update_namespace(prefix=prefix, namespace=new_namespace)

    assert pr.graph.metadata_header
    data_ns = pr.graph.namespace_manager.store
    header_ns = pr.graph.metadata_header.graph.namespace_manager.store

    assert data_ns.namespace("ex") == URIRef("https://example.com/")
    assert header_ns.namespace("ex") == URIRef("https://example.com/")
    mock_update.assert_not_called()
    
    if prefix != "ex":
        assert data_ns.namespace(prefix) == None
        assert header_ns.namespace(prefix) == None


@pytest.mark.parametrize(
    "prefix, new_namespace, which_changed",
    [
        pytest.param("ex", "www.new.com/", "graph", id="Graph changed"),
        pytest.param("foo", "www.new.com/", "header", id="Header changed"),
    ]
)
def test_update_namespace_onlyonechanged(prefix: str, new_namespace: str|URIRef, which_changed: str) -> None:
    header = CIMMetadataHeader.empty(URIRef("https://bar.com/h1"))
    header.graph.bind("foo", "https://bar.com/")
    header.add_triple(URIRef("https://bar.com/ph"), URIRef("https://bar.com/oh"))
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), URIRef("https://example.com/o1")))
    g.metadata_header = header
    
    pr = CIMProcessor(g)
    pr.update_namespace(prefix=prefix, namespace=new_namespace)

    new_namespace = str(new_namespace).strip()
    assert pr.graph.metadata_header

    data_ns = pr.graph.namespace_manager.store
    header_ns = pr.graph.metadata_header.graph.namespace_manager.store

    if which_changed == "graph":
        assert data_ns.namespace(prefix) == URIRef(new_namespace)
        assert (URIRef(f"{new_namespace}s1"), URIRef(f"{new_namespace}p1"), URIRef(f"{new_namespace}o1")) in pr.graph
        assert header_ns.namespace("foo") == URIRef("https://bar.com/")
        assert (URIRef("https://bar.com/h1"), URIRef("https://bar.com/ph"), URIRef("https://bar.com/oh")) in pr.graph.metadata_header.graph
    elif which_changed == "header":
        assert pr.graph.metadata_header.graph.namespace_manager.store.namespace(prefix) == URIRef(new_namespace)
        assert (URIRef(f"{new_namespace}h1"), URIRef(f"{new_namespace}ph"), URIRef(f"{new_namespace}oh")) in pr.graph.metadata_header.graph
        assert data_ns.namespace("ex") == URIRef("https://example.com/")
        assert (URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), URIRef("https://example.com/o1")) in pr.graph


def test_update_namespace_graphfixed() -> None:
    header = CIMMetadataHeader.empty(URIRef("https://example.com/h1"))
    header.graph.bind("ex", "https://example.com/")
    header.add_triple(URIRef("https://example.com/ph"), URIRef("https://example.com/oh"))
    g = CIMGraph()
    g.bind("ex", "https://ex.com/")
    g.add((URIRef("https://ex.com/s1"), URIRef("https://ex.com/p1"), URIRef("https://ex.com/o1")))
    g.metadata_header = header
    
    pr = CIMProcessor(g)
    pr.update_namespace(prefix="ex", namespace="https://example.com/")

    assert pr.graph.metadata_header
    assert pr.graph.namespace_manager.store.namespace("ex") == URIRef("https://example.com/")
    assert pr.graph.metadata_header.graph.namespace_manager.store.namespace("ex") == URIRef("https://example.com/")
    assert (URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), URIRef("https://example.com/o1")) in pr.graph
    assert (URIRef("https://example.com/h1"), URIRef("https://example.com/ph"), URIRef("https://example.com/oh")) in pr.graph.metadata_header.graph


@patch("cim_plugin.processor.update_namespace_in_triples")
def test_update_namespace_emptystringnamespace(mock_update) -> None:
    header = CIMMetadataHeader.empty(URIRef("https://bar.com/h1"))
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), URIRef("https://example.com/o1")))
    g.metadata_header = header
    
    pr = CIMProcessor(g)
    
    with pytest.raises(ValueError) as exc:
        pr.update_namespace(prefix="ex", namespace=" ")
    
    assert "Namespace cannot be empty." in str(exc.value)
    mock_update.assert_not_called()


@patch("cim_plugin.processor.update_namespace_in_triples")
def test_update_namespace_namespacenone(mock_update) -> None:
    header = CIMMetadataHeader.empty(URIRef("https://bar.com/h1"))
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), URIRef("https://example.com/o1")))
    g.metadata_header = header
    
    pr = CIMProcessor(g)
    
    with pytest.raises(ValueError) as exc:
        # Pylance silenced to allow wrong input type
        pr.update_namespace(prefix="ex", namespace=None)    # type: ignore
    
    assert "Namespace cannot be empty." in str(exc.value)
    mock_update.assert_not_called()


@patch("cim_plugin.processor.update_namespace_in_triples")
def test_update_namespace_noheader(mock_update: MagicMock) -> None:
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), URIRef("https://example.com/o1")))
    g.metadata_header = None
    
    pr = CIMProcessor(g)
    pr.update_namespace(prefix="ex", namespace="www.new.com/")

    assert pr.graph.metadata_header is None
    assert pr.graph.namespace_manager.store.namespace("ex") == URIRef("www.new.com/")
    assert mock_update.call_count == 1


@patch("cim_plugin.processor.update_namespace_in_triples")
def test_update_namespace_emptyheader(mock_update: MagicMock) -> None:
    header = CIMMetadataHeader.empty(URIRef("h1"))
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), URIRef("https://example.com/o1")))
    g.add((URIRef("https://example.com/s2"), URIRef("https://example.com/p2"), URIRef("https://example.com/o2")))
    g.metadata_header = header
    
    pr = CIMProcessor(g)    
    pr.update_namespace(prefix="ex", namespace="www.new.com/")
    
    assert pr.graph.metadata_header is header
    assert pr.graph.namespace_manager.store.namespace("ex") == URIRef("www.new.com/")
    assert mock_update.call_count == 1


def test_update_namespace_multipletriples() -> None:
    header = CIMMetadataHeader.empty(URIRef("h1"))
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), URIRef("https://example.com/o1")))
    g.add((URIRef("https://example.com/s2"), URIRef("https://example.com/p2"), URIRef("https://example.com/o2")))
    g.metadata_header = header
    
    pr = CIMProcessor(g)    
    pr.update_namespace(prefix="ex", namespace="www.new.com/")
    
    assert pr.graph.metadata_header is header
    assert pr.graph.namespace_manager.store.namespace("ex") == URIRef("www.new.com/")
    assert len(pr.graph) == 2
    assert (URIRef("www.new.com/s1"), URIRef("www.new.com/p1"), URIRef("www.new.com/o1")) in pr.graph
    assert (URIRef("www.new.com/s2"), URIRef("www.new.com/p2"), URIRef("www.new.com/o2")) in pr.graph

# Unit tests .enrich_literal_datatypes

@pytest.mark.parametrize("schemaview, slot_dict", [
    pytest.param(None, [{"p": "string"}], id="Schemaview missing"),
    pytest.param("dummy", None, id="slot_index missing"),
])
@patch("cim_plugin.processor.replace_namespace")
@patch("cim_plugin.processor.resolve_datatype_from_slot")
@patch("cim_plugin.processor.create_typed_literal")
def test_enrich_literal_datatypes_missingprerequisites(mock_create: MagicMock, mock_resolve: MagicMock, mock_replace: MagicMock, schemaview: SchemaView|None, slot_dict: list[dict]|None, make_slot_index: Callable[..., dict], caplog: pytest.LogCaptureFixture) -> None:
    g = CIMGraph()
    s, p, o = URIRef("s"), URIRef("p"), Literal("x")
    g.add((s, p, o))

    slot_index = make_slot_index(slot_dict) if slot_dict else None

    inst = CIMProcessor(g)
    inst.schema = schemaview
    inst.slot_index = slot_index

    inst.enrich_literal_datatypes(allow_different_namespaces=True)

    assert list(inst.graph) == [(s, p, o)]
    assert "Missing schemaview or slot_index. Enriching not possible." in caplog.text
    mock_resolve.assert_not_called()
    mock_create.assert_not_called()
    mock_replace.assert_not_called()


@patch("cim_plugin.processor.resolve_datatype_from_slot", return_value=None)   # If slot.range is None, this function will return None
@patch("cim_plugin.processor.create_typed_literal")
def test_enrich_literal_datatypes_slotrangenone(mock_create: MagicMock, mock_resolve: MagicMock, make_slot_index: Callable[..., dict], make_schemaview: Callable[..., SchemaView], caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    g = CIMGraph()
    s, p, o = URIRef("s"), URIRef("p"), Literal("x")
    g.add((s, p, o))
    
    inst = CIMProcessor(g)
    inst.schema = make_schemaview()
    inst.slot_index = make_slot_index([{"p": None}])
    
    inst.enrich_literal_datatypes()

    mock_resolve.assert_called_once()
    mock_create.assert_not_called()
    assert list(inst.graph) == [(s, p, o)]
    assert "No datatype found for range: None, for p" in caplog.text


@pytest.mark.parametrize("object, resolved", [
    pytest.param(Literal("x"), True, id="Literal without datatype"),
    pytest.param(Literal("x", datatype=URIRef('http://www.w3.org/2001/XMLSchema#string')), False, id="Literal with datatype"),
    pytest.param(Literal("x", lang="en"), False, id="Literal with language. Datatype is implicitly string."),
    pytest.param(BNode("x"), False, id="Blank node."),
    pytest.param(URIRef("x"), False, id="URI object")
])
@patch("cim_plugin.processor.resolve_datatype_from_slot")
@patch("cim_plugin.processor.create_typed_literal")
def test_enrich_literal_datatypes_objecthandling(mock_create: MagicMock, mock_resolve: MagicMock, object: Literal|BNode|URIRef, resolved: bool, make_slot_index: Callable[..., dict], make_schemaview: Callable[..., SchemaView]) -> None:
    mock_resolve.return_value = "xsd:string"
    mock_create.return_value = object
    g = CIMGraph()
    s, p, o = URIRef("s"), URIRef("p"), object
    g.add((s, p, o))

    inst = CIMProcessor(g)
    inst.schema = make_schemaview()
    inst.slot_index = make_slot_index([{"p": "string"}])

    inst.enrich_literal_datatypes()
    if resolved:
        mock_resolve.assert_called_once()
        mock_create.assert_called_once()
        assert len(list(inst.graph)) == 1   # Size of graph is not affected
    else:
        mock_resolve.assert_not_called()
        mock_create.assert_not_called()
        assert len(list(inst.graph)) == 1   # Size of graph is not affected

@patch("cim_plugin.processor.resolve_datatype_from_slot")
@patch("cim_plugin.processor.create_typed_literal")
def test_enrich_literal_datatypes_predicatenotfound(mock_create: MagicMock, mock_resolve: MagicMock, make_slot_index: Callable[..., dict], make_schemaview: Callable[..., SchemaView], caplog: pytest.LogCaptureFixture) -> None:
    logger.setLevel("INFO")
    g = CIMGraph()
    s, p, o = URIRef("s"), URIRef("first_unknown"), Literal("42")
    s2, p2, o2 = URIRef("s2"), URIRef("also_unknown"), Literal("42")
    g.add((s, p, o))
    g.add((s2, p2, o2))

    inst = CIMProcessor(g)
    inst.schema = make_schemaview()
    inst.slot_index = make_slot_index([{"p": "string"}])

    inst.enrich_literal_datatypes()
    assert sorted(inst.graph) == sorted([(s, p, o), (s2, p2, o2)])
    assert "also_unknown" in caplog.text
    assert "first_unknown" in caplog.text
    mock_resolve.assert_not_called()
    mock_create.assert_not_called()

@patch("cim_plugin.processor.resolve_datatype_from_slot", return_value=None)
@patch("cim_plugin.processor.create_typed_literal")
def test_enrich_literal_datatypes_nodatatyperesolved(mock_create: MagicMock, mock_resolve: MagicMock, make_slot_index: Callable[..., dict], make_schemaview: Callable[..., SchemaView], caplog: pytest.LogCaptureFixture) -> None:
    logger.setLevel("INFO")
    g = CIMGraph()
    s, p, o = URIRef("s"), URIRef("p"), Literal("hello")
    g.add((s, p, o))

    inst = CIMProcessor(g)
    inst.schema = make_schemaview()
    inst.slot_index = make_slot_index([{"p": "string"}])

    inst.enrich_literal_datatypes()

    assert list(inst.graph) == [(s, p, o)]
    mock_resolve.assert_called_once_with(inst.schema, inst.slot_index["p"])
    mock_create.assert_not_called()
    assert "No datatype found for range: string, for p" in caplog.text


@patch("cim_plugin.processor.resolve_datatype_from_slot", return_value=URIRef("xsd:string"))
@patch("cim_plugin.processor.create_typed_literal")
def test_enrich_literal_datatypes_successfulenrichment(mock_create: MagicMock, mock_resolve: MagicMock, make_slot_index: Callable[..., dict], make_schemaview: Callable[..., SchemaView], caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    g = CIMGraph()
    s, p, o = URIRef("s"), URIRef("p"), Literal("hello")
    g.add((s, p, o))

    new_lit = Literal("hello", datatype=URIRef("xsd:string"))
    mock_create.return_value = new_lit

    inst = CIMProcessor(g)
    inst.schema = make_schemaview()
    inst.slot_index = make_slot_index([{"p": "string"}])

    inst.enrich_literal_datatypes()

    triples = list(inst.graph)
    assert len(triples) == 1
    assert triples[0] == (s, p, new_lit)

    mock_resolve.assert_called_once()
    mock_create.assert_called_once_with("hello", URIRef("xsd:string"), inst.schema)
    assert "Enriching done. Added datatypes to 1 triples." in caplog.text


@patch("cim_plugin.processor.resolve_datatype_from_slot", return_value=URIRef("xsd:int"))
@patch("cim_plugin.processor.create_typed_literal", side_effect=LiteralCastingError("bad cast"))
def test_enrich_literal_datatypes_castingerror(mock_create: MagicMock, mock_resolve: MagicMock, make_slot_index: Callable[..., dict], make_schemaview: Callable[..., SchemaView], caplog: pytest.LogCaptureFixture) -> None:
    g = CIMGraph()
    s, p, o = URIRef("s"), URIRef("p"), Literal("not_an_int")
    s2, p2, o2 = URIRef("s2"), URIRef("p"), Literal("also_not_int")
    g.add((s, p, o))
    g.add((s2, p2, o2))

    inst = CIMProcessor(g)
    inst.schema = make_schemaview()
    inst.slot_index = make_slot_index([{"p": "string"}])

    inst.enrich_literal_datatypes()
    
    result = inst.graph
    assert len(list(result)) == 2
    assert (s, p, o) in list(result)
    assert (s2, p2, o2) in list(result)
    assert mock_resolve.call_count == 2
    assert mock_create.call_count == 2
    assert any("Error casting" in rec.message for rec in caplog.records)
    assert "Error casting also_not_int for s2, p: bad cast\n" in caplog.text
    assert "Error casting not_an_int for s, p: bad cast\n"  in caplog.text


@pytest.mark.parametrize("allow, returned", 
        [
            pytest.param(False, None, id="Different namespaces not allowed"), 
            pytest.param(True, None, id="Different namespaces allowed, none found"),
            pytest.param(True, {("ex", "example.com", "example.org")}, id="Different namespaces allowed, some found"),
        ]
)
@patch("cim_plugin.processor.replace_namespace", return_value="p")
@patch("cim_plugin.processor.resolve_datatype_from_slot", return_value=URIRef("xsd:string"))
@patch("cim_plugin.processor.create_typed_literal")
def test_enrich_literal_datatypes_allownamespaces(mock_create: MagicMock, mock_resolve: MagicMock, mock_replace: MagicMock, allow: bool, returned: set[tuple[str, str, str]]|None, make_slot_index: Callable[..., dict], make_schemaview: Callable[..., SchemaView]) -> None:
    g = CIMGraph()
    s, p, o = URIRef("s"), URIRef("p"), Literal("hello")
    g.add((s, p, o))

    new_lit = Literal("hello", datatype=URIRef("xsd:string"))
    mock_create.return_value = new_lit

    inst = CIMProcessor(g)
    inst.schema = make_schemaview()
    inst.slot_index = make_slot_index([{"p": "string"}])
    inst.namespaces_different_from_model = MagicMock(return_value = returned)

    inst.enrich_literal_datatypes(allow_different_namespaces=allow)

    triples = list(inst.graph)
    assert len(triples) == 1
    assert triples[0] == (s, p, new_lit)

    mock_resolve.assert_called_once()
    mock_create.assert_called_once_with("hello", URIRef("xsd:string"), inst.schema)

    if allow:
        inst.namespaces_different_from_model.assert_called_once()
        if returned:
            mock_replace.assert_called()
        else:
            mock_replace.assert_not_called()
    else:
        inst.namespaces_different_from_model.assert_not_called()
        mock_replace.assert_not_called()


def test_enrich_literal_datatypes_integrated(make_schemaview: Callable[..., SchemaView], caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    classes = {"A": ClassDefinition(name="A", attributes={"http://www.example.com/c1": SlotDefinition(name="http://www.example.com/c1", range="string"),
                                                          "http://www.example.com/c2": SlotDefinition(name="http://www.example.com/c2", range="Custom")})}
    types = {"Custom": TypeDefinition(name="Custom", base="integer", uri="xsd:integer")}
    prefixes = {"ex": {"prefix_prefix": "ex", "prefix_reference": "http://www.example.com/"}}
    sv = make_schemaview(classes=classes, types=types, prefixes=prefixes)
    
    g = CIMGraph()
    g.bind("ex", "www.example.com/")
    g.add((URIRef("s"), URIRef("http://www.example.com/c1"), Literal("hello")))
    g.add((URIRef("s"), URIRef("http://www.example.com/c1"), Literal("hei", lang="no")))
    g.add((URIRef("s"), URIRef("http://www.example.com/c2"), Literal("1")))
    g.add((URIRef("s"), URIRef("d"), URIRef("not-a-literal")))

    inst = CIMProcessor(g)
    inst.schema = sv
    inst.slot_index = {"http://www.example.com/c1": SlotDefinition(name="c1", range="string"), "http://www.example.com/c2": SlotDefinition(name="c2", range="Custom")}
    
    inst.enrich_literal_datatypes(allow_different_namespaces=False)

    result = inst.graph
    assert (URIRef("s"), URIRef("http://www.example.com/c1"), Literal("hello", datatype=URIRef('http://www.w3.org/2001/XMLSchema#string'))) in list(result)
    assert (URIRef("s"), URIRef("http://www.example.com/c1"), Literal("hei", lang="no")) in list(result)   # Literals with language tag is not given datatype
    assert (URIRef("s"), URIRef("http://www.example.com/c2"), Literal(1, datatype=URIRef('http://www.w3.org/2001/XMLSchema#integer'))) in list(result)
    assert (URIRef("s"), URIRef("d"), URIRef("not-a-literal")) in list(result)
    assert "Enriching done. Added datatypes to 2 triples." in caplog.text


def test_enrich_literal_datatypes_integrateddifferentnamespaces(make_schemaview: Callable[..., SchemaView], caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    classes = {"A": ClassDefinition(name="A", attributes={"http://www.example.com/c1": SlotDefinition(name="http://www.example.com/c1", range="string"),
                                                          "http://www.example.com/c2": SlotDefinition(name="http://www.example.com/c2", range="Custom")})}
    types = {"Custom": TypeDefinition(name="Custom", base="integer", uri="xsd:integer")}
    prefixes = {"ex": {"prefix_prefix": "ex", "prefix_reference": "http://www.example.com/"}}   # Schema namespace is slightly different from graph namespace
    sv = make_schemaview(classes=classes, types=types, prefixes=prefixes)
    
    g = CIMGraph()
    g.bind("ex", "www.example.org/")
    g.add((URIRef("s"), URIRef("www.example.org/c1"), Literal("hello")))
    g.add((URIRef("s"), URIRef("www.example.org/c1"), Literal("hei", lang="no")))
    g.add((URIRef("s"), URIRef("www.example.org/c2"), Literal("1")))
    g.add((URIRef("s"), URIRef("d"), URIRef("not-a-literal")))

    inst = CIMProcessor(g)
    inst.schema = sv
    
    inst.slot_index = {"http://www.example.com/c1": SlotDefinition(name="c1", range="string"), "http://www.example.com/c2": SlotDefinition(name="c2", range="Custom")}
    
    inst.enrich_literal_datatypes(allow_different_namespaces=True)
    
    result = inst.graph

    assert (URIRef("s"), URIRef("www.example.org/c1"), Literal("hello", datatype=URIRef('http://www.w3.org/2001/XMLSchema#string'))) in list(result)
    assert (URIRef("s"), URIRef("www.example.org/c1"), Literal("hei", lang="no")) in list(result)   # Literals with language tag is not given datatype
    assert (URIRef("s"), URIRef("www.example.org/c2"), Literal(1, datatype=URIRef('http://www.w3.org/2001/XMLSchema#integer'))) in list(result)
    assert (URIRef("s"), URIRef("d"), URIRef("not-a-literal")) in list(result)
    assert "Enriching done. Added datatypes to 2 triples." in caplog.text


def test_enrich_literal_datatypes_integrateddifferentnamespacesoverlap(make_schemaview: Callable[..., SchemaView], caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    classes = {"A": ClassDefinition(name="A", attributes={"http://www.example.com/c1": SlotDefinition(name="http://www.example.com/c1", range="string"),
                                                          "http://www.example.org/bar/c2": SlotDefinition(name="http://www.example.org/bar/c2", range="Custom")})}
    types = {"Custom": TypeDefinition(name="Custom", base="integer", uri="xsd:integer")}
    prefixes = {"ex": {"prefix_prefix": "ex", "prefix_reference": "http://www.example.com/"},
                "foo": {"prefix_prefix": "foo", "prefix_reference": "http://www.example.org/bar/"}}   # This one is the same as graph
    sv = make_schemaview(classes=classes, types=types, prefixes=prefixes)
    
    g = CIMGraph()
    g.bind("ex", "www.example.org/")
    g.bind("foo", "www.example.org/bar/")    # Overlap between this namespace and ex
    g.add((URIRef("s"), URIRef("www.example.org/c1"), Literal("hello")))
    g.add((URIRef("s"), URIRef("www.example.org/c1"), Literal("hei", lang="no")))
    g.add((URIRef("s"), URIRef("www.example.org/bar/c2"), Literal("1")))
    g.add((URIRef("s"), URIRef("d"), URIRef("not-a-literal")))

    inst = CIMProcessor(g)
    inst.schema = sv
    
    inst.slot_index = {"http://www.example.com/c1": SlotDefinition(name="c1", range="string"), "http://www.example.org/bar/c2": SlotDefinition(name="c2", range="Custom")}
    
    inst.enrich_literal_datatypes(allow_different_namespaces=True)
    
    result = inst.graph
    print(list(result))

    assert (URIRef("s"), URIRef("www.example.org/c1"), Literal("hello", datatype=URIRef('http://www.w3.org/2001/XMLSchema#string'))) in list(result)
    assert (URIRef("s"), URIRef("www.example.org/c1"), Literal("hei", lang="no")) in list(result)
    assert (URIRef("s"), URIRef("d"), URIRef("not-a-literal")) in list(result)
    assert (URIRef("s"), URIRef("www.example.org/bar/c2"), Literal(1, datatype=URIRef('http://www.w3.org/2001/XMLSchema#integer'))) in list(result)
    assert "Enriching done. Added datatypes to 2 triples." in caplog.text
    


def test_enrich_literal_datatypes_noupdates(make_slot_index: Callable[..., dict], make_schemaview: Callable[..., SchemaView], caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    classes = {"A": ClassDefinition(name="A", attributes={"c1": SlotDefinition(name="c1", range="string"),
                                                          "c2": SlotDefinition(name="c2", range="Custom")})}
    types = {"Custom": TypeDefinition(name="Custom", base="integer", uri="xsd:integer")}
    prefixes = {"ex": {"prefix_prefix": "ex", "prefix_reference": "www.example.org"}}
    sv = make_schemaview(classes=classes, types=types, prefixes=prefixes)
    g = CIMGraph()
    g.add((URIRef("s"), URIRef("d"), URIRef("not-a-literal")))
    inst = CIMProcessor(g)
    inst.schema = sv
    inst.slot_index = make_slot_index([{"c1": "string"}, {"c2": "Custom"}])
    
    inst.enrich_literal_datatypes()

    assert (URIRef("s"), URIRef("d"), URIRef("not-a-literal")) in list(inst.graph)
    assert "Enriching done. Added datatypes to 0 triples." in caplog.text


# Unit tests merge_namespace_managers
def test_merge_namespace_managers_nodiffs() -> None:
    # No new namespaces, but the rdflib defaults are there
    maing = CIMGraph()
    otherg = CIMGraph()
    main_old = list(maing.namespace_manager.store.namespaces())
    other_old = list(otherg.namespace_manager.store.namespaces())

    merge_namespace_managers(maing.namespace_manager, otherg.namespace_manager)

    main_new = list(maing.namespace_manager.store.namespaces())
    other_new = list(otherg.namespace_manager.store.namespaces())

    assert main_old == main_new
    assert other_old == other_new
    assert other_new == main_new


def test_merge_namespace_managers_newnamespace() -> None:
    maing = CIMGraph()
    otherg = CIMGraph()
    otherg.bind("ex", "https://example.com")
    main_old = list(maing.namespace_manager.store.namespaces())
    other_old = list(otherg.namespace_manager.store.namespaces())

    merge_namespace_managers(maing.namespace_manager, otherg.namespace_manager)

    main_new = list(maing.namespace_manager.store.namespaces())
    other_new = list(otherg.namespace_manager.store.namespaces())

    assert len(main_new) == len(main_old) + 1
    assert ("ex", URIRef("https://example.com")) in main_new
    assert other_old == other_new
    assert other_new == main_new

def test_merge_namespace_managers_extrainmain() -> None:
    # No new namespaces, but the rdflib defaults are there
    maing = CIMGraph()
    otherg = CIMGraph()
    maing.bind("ex", "https://example.com")
    main_old = list(maing.namespace_manager.store.namespaces())
    other_old = list(otherg.namespace_manager.store.namespaces())

    merge_namespace_managers(maing.namespace_manager, otherg.namespace_manager)

    main_new = list(maing.namespace_manager.store.namespaces())
    other_new = list(otherg.namespace_manager.store.namespaces())

    assert main_new == main_old
    assert ("ex", URIRef("https://example.com")) in main_new
    assert ("ex", URIRef("https://example.com")) not in other_new
    assert other_old == other_new
    assert len(other_new) + 1 == len(main_new)


@pytest.mark.parametrize(
        "prefix, namespace, collision",
        [
            pytest.param("new", "https://new.com/", False, id="New namespace"),
            pytest.param("foo", "www.bar.org/", False, id="Same prefix, namespace as data"),
            pytest.param("foo", "www.foo.org/", True, id="Same prefix, new namespace then data"),
            pytest.param("new", "https//oldname.com/", True, id="New namespace, but namespace already exist"),
            pytest.param("foo", "www.bar.org/foo", True, id="Overlapped namespaces"),
            pytest.param("foo", "www.bar.org", True, id="Missing /. Treated as collision.")
        ]
)
def test_merge_namespace_managers_various(prefix: str, namespace: str, collision: bool) -> None:
    otherg = CIMGraph()
    otherg.bind("ex", "https://example.com/")
    otherg.bind(prefix, namespace)
    otherg.add((URIRef("https://example.com/s1"), URIRef(f"{namespace}p1"), Literal("o")))
    maing = CIMGraph()
    maing.bind("old", "https//oldname.com/")
    maing.bind("foo", "www.bar.org/")
    maing.add((URIRef("https//oldname.com/s2"), URIRef("www.bar.org/p2"), Literal("o")))
    
    merge_namespace_managers(maing.namespace_manager, otherg.namespace_manager)

    other_nm = otherg.namespace_manager.store
    main_nm = maing.namespace_manager.store

    # The namespaces of maing are never changed, just added to
    assert main_nm.namespace("foo") == URIRef("www.bar.org/")
    assert main_nm.namespace("old") == URIRef("https//oldname.com/")
    assert main_nm.namespace("ex") == URIRef("https://example.com/")
    
    # Triples do not change
    assert (URIRef("https//oldname.com/s2"), URIRef("www.bar.org/p2"), Literal("o")) in maing
    assert (URIRef("https://example.com/s1"), URIRef(f"{namespace}p1"), Literal("o")) in otherg

    if not collision:
        assert other_nm.namespace(prefix) == URIRef(namespace)
        assert main_nm.namespace(prefix) == other_nm.namespace(prefix)
    else:
        assert other_nm.namespace(prefix) == URIRef(namespace) # The otherg keeps its old namespace
        assert main_nm.namespace(prefix) != other_nm.namespace(prefix) # The otherg and maing namespaces remains different
        

def test_merge_namespace_managers_errorlogged(caplog: pytest.LogCaptureFixture) -> None:
    otherg = CIMGraph()
    otherg.bind("foo", "https://foo.com/")
    maing = CIMGraph()
    maing.bind("foo", "www.bar.org/")

    merge_namespace_managers(maing.namespace_manager, otherg.namespace_manager)

    other_nm = otherg.namespace_manager.store
    main_nm = maing.namespace_manager.store
    assert other_nm.namespace("foo") == URIRef("https://foo.com/")
    assert main_nm.namespace("foo") == URIRef("www.bar.org/")
    assert caplog.records[0].levelname == "ERROR"
    assert f"Namespace for 'foo' differs between graphs (www.bar.org/ vs https://foo.com/). www.bar.org/ is kept." in caplog.text


# Unit tests replace_namespace
@pytest.mark.parametrize(
    "predicate, diffs, expected",
    [
        pytest.param("https://example.com/p", {("ex", "https://example.com/", "https://example.org/")}, "https://example.org/p", id="Simple substitution"),
        pytest.param("https://example.com/p%20", {("ex", "https://example.com/", "https://example.org/")}, "https://example.org/p%20", id="Unusual predicate"),
        pytest.param("https://example.com/p", {("ex", "https://example.com/", "https://example.com")}, "https://example.comp", id="Removing /"),
        pytest.param("https://example.com/p", {("ex", "https://example.com/", "https://example.com#")}, "https://example.com#p", id="From / to #"),
        pytest.param("https://example.com/foo/p", {("ex", "https://example.com/", "https://example.org/")}, "https://example.com/foo/p", id="Overlap"),
        pytest.param(
            "https://example.com/foo/p", 
            {("ex", "https://example.com/", "https://example.org/"), ("foo", "https://example.com/foo/", "https://example.com/bar/")}, 
            "https://example.com/bar/p", 
            id="Overlap, both have changes"
        ),
        pytest.param("https://example.com/p", {("foo", "https://example.com/foo/", "https://example.com/bar/")}, "https://example.com/p", id="Not in set, no change"),
        pytest.param("https://example.com/p", {}, "https://example.com/p", id="Empty set, no change"),
        pytest.param("https://example.com/", {("ex", "https://example.com/", "https://example.org/")}, "https://example.org/", id="Empty predicate"),
        pytest.param("https://example.com/p", {("ex", "https://example.com/", "https://example.com/ ")}, "https://example.com/ p", id="Whitespace in new"),
        pytest.param("https://example.com/ p", {("ex", "https://example.com/ ", "https://example.org/")}, "https://example.com/ p", id="Whitespace in old"),
        pytest.param("https://example.com#p", {("ex", "https://example.com/", "https://example.org/")}, "https://example.com#p", id="Mismatch between namespace manager and set, no change"),
        pytest.param("https://example.com/p", {("exa", "https://example.com/", "https://example.org/")}, "https://example.com/p", id="Different prefix, same namespace, no change"),
        pytest.param("ex:p", {("exa", "https://example.com/", "https://example.org/")}, "ex:p", id="Predicate not a full uri, no change"),
        pytest.param(
            "https://example.com/p", 
            {("ex", "https://example.com/", "https://example.org/"), ("exa", "https://example.com/", "https://wrong/")}, 
            "https://example.org/p", 
            id="Prefix overlap, both are present" # This is not allowed in the namespace manager, but could occur in the set of differences
        ),
    ]
)
def test_replace_namespace_various(predicate: str, diffs: set[tuple[str, str, str]], expected: str) -> None:
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.bind("foo", "https://example.com/foo/")
    g.add((URIRef("s1"), URIRef(predicate), Literal("o")))
    result = replace_namespace(predicate=predicate, graph=g, replacements=diffs)
    assert result == expected

    # The graph is unchanged
    assert g.namespace_manager.store.namespace("ex") == URIRef("https://example.com/")
    assert (URIRef("s1"), URIRef(predicate), Literal("o")) in g


def test_replace_namespace_invaliduri(caplog: pytest.LogCaptureFixture) -> None:
    g = CIMGraph()
    g.bind("ex", "https://example.com/ ")   # Whitespace in uri raises a ValueError caught by the function
    g.bind("foo", "https://example.com/foo/")
    predicate = "https://example.com/ p"
    g.add((URIRef("s1"), URIRef(predicate), Literal("o")))
    result = replace_namespace(predicate=predicate, graph=g, replacements={("ex", "https://example.com/ ", "https://example.org/")})
    assert result == predicate
    assert f"Error in compute_qname for {predicate}: " in caplog.text


def test_replace_namespace_emptygraph() -> None:
    g = CIMGraph()
    predicate = "https://example.com/p"
    result = replace_namespace(predicate=predicate, graph=g, replacements={("ex", "https://example.com/", "https://example.org/")})
    assert result == predicate  # When namespace is not in the namespaces manager, the predicate is returned unchanged


def test_replace_namespace_duplicatesinreplacements() -> None:
    # Documents what happends if the function is given a replacements set with duplicated prefix, old_ns.
    # This will never happen if the replacements set is built by CIMProcess.namespaces_different_from_model 
    # because rdflib does not allow duplicate namespaces. 
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    diffs = {("ex", "https://example.com/", "https://first.com/"), ("ex", "https://example.com/", "https://second.com/")}
    predicate = "https://example.com/p"
    g.add((URIRef("s1"), URIRef(predicate), Literal("o")))
    result = replace_namespace(predicate=predicate, graph=g, replacements=diffs)
    # Sets are not ordered, so sometimes it gives one result and sometimes the other
    assert result in {"https://first.com/p", "https://second.com/p"}

# Unit tests .validate_header
@patch("cim_plugin.processor.validate_header")
def test_validate_header_noheader(mock_validate: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    g = CIMGraph()
    pr = CIMProcessor(g)

    pr.validate_header()

    mock_validate.assert_not_called()
    assert "No metadata header found. Validation not possible." in caplog.text

@patch("cim_plugin.processor.validate_header")
def test_validate_header_headerpresent(mock_validate: MagicMock, make_cimgraph: CIMGraph, caplog: pytest.LogCaptureFixture) -> None:
    g = make_cimgraph
    pr = CIMProcessor(g)

    pr.validate_header()

    mock_validate.assert_called_once_with(g.metadata_header, format="cimxml")

@patch("cim_plugin.processor.validate_header")
def test_validate_header_trigformat(mock_validate: MagicMock, make_cimgraph: CIMGraph, caplog: pytest.LogCaptureFixture) -> None:
    g = make_cimgraph
    pr = CIMProcessor(g)

    pr.validate_header(format="trig")

    mock_validate.assert_called_once_with(g.metadata_header, format="trig")

# Unit tests ._build_copy_for_serialization
def test_build_copy_for_serialization_emptygraph() -> None:
    g = CIMGraph()
    pr = CIMProcessor(g)

    result = pr._build_copy_for_serialization()

    assert result.graph.identifier is pr.graph.identifier
    assert result.graph is not pr.graph
    assert len(result.graph) == 0
    assert result.schema is None
    assert result.slot_index is None
    assert result.graph.metadata_header is None


def test_build_copy_for_serialization_basic(make_schemaview: Callable[..., SchemaView], make_cimgraph: CIMGraph) -> None:
    g = make_cimgraph
    g.add((URIRef("http://foo.org/ns#s1"), URIRef("http://foo.org/ns#s2"), Literal("o")))
    sv = make_schemaview()
    pr = CIMProcessor(g)
    pr.schema = sv
    pr.slot_index = {"mocked": {"one": 1, "two": 2}}

    result = pr._build_copy_for_serialization()

    assert result.identifier == URIRef("graph1")
    assert result.graph is not pr.graph
    assert result.graph.metadata_header is pr.graph.metadata_header
    assert result.schema is pr.schema
    assert result.slot_index is not pr.slot_index
    assert len(result.graph) == 1
    assert result.graph.metadata_header
    assert set(result.graph.namespace_manager.namespaces()) == set(pr.graph.namespace_manager.namespaces())
    assert result.graph.namespace_manager.store.namespace("foo") == URIRef("http://foo.org/ns#")
    assert result.graph.metadata_header.graph.namespace_manager.store.namespace("ex") == URIRef("http://example.com/")
    assert (URIRef("http://example.com/header"), RDF.type, DCAT_EXT.Dataset) in result.graph.metadata_header.graph
    assert (URIRef("http://foo.org/ns#s1"), URIRef("http://foo.org/ns#s2"), Literal("o")) in result.graph
    assert result.slot_index == {"mocked": {"one": 1, "two": 2}}


def test_build_copy_for_serialization_mutability(make_schemaview: Callable[..., SchemaView], make_cimgraph: CIMGraph) -> None:
    g = make_cimgraph
    g.add((URIRef("http://foo.org/ns#s1"), URIRef("http://foo.org/ns#s2"), Literal("o")))
    sv = make_schemaview()
    pr = CIMProcessor(g)
    pr.schema = sv
    pr.slot_index = {"mocked": {"one": 1, "two": 2}}

    result = pr._build_copy_for_serialization()
    
    assert result.slot_index
    result.slot_index["mocked"]["one"] = 999
    assert pr.slot_index["mocked"]["one"] == 1
    assert result.slot_index["mocked"]["one"] == 999

    assert result.graph.metadata_header
    assert pr.graph.metadata_header
    result.graph.metadata_header.add_triple(RDF.type, DCAT_EXT.Distribution)
    assert (None, None, DCAT_EXT.Distribution) in pr.graph.metadata_header.graph

    result.graph.add((URIRef("x"), URIRef("y"), URIRef("z")))
    assert (URIRef("x"), URIRef("y"), URIRef("z")) not in pr.graph

    assert result.schema
    result.schema.add_slot(SlotDefinition("new_slot", "string"))
    assert pr.schema and pr.schema.schema and isinstance(pr.schema.schema.slots, dict)
    assert pr.schema.schema.slots["new_slot"] == SlotDefinition("new_slot", "string")
    
# Unit tests _make_header_graph_for_conversion
def test_make_header_graph_for_conversion_unknowntype() -> None:
    header = CIMMetadataHeader.empty(subject=URIRef("h1"), metadata_objects=[URIRef("custom_type")])
    header.add_triple(RDF.type, URIRef("custom_type"))
    with pytest.raises(ValueError, match=f"Unknown header type: {header.header_type}. Conversion not possible."):
        _make_header_graph_for_conversion(header)

@pytest.mark.parametrize("header", [None, CIMMetadataHeader.empty(subject=URIRef("h1"))])
def test_make_header_graph_for_conversion_emptyheader(header: CIMMetadataHeader | None) -> None:
    with pytest.raises((ValueError, AttributeError)) as exc:    # Exceptions carried forward from header.header_type
        # Pylance silenced to test wrong input
        _make_header_graph_for_conversion(header)   # type: ignore

    msg = str(exc.value)
    assert ( "No header type found in header." in msg
            or "'NoneType' object has no attribute 'header_type'" in msg)


def test_make_header_graph_for_conversion_fullmodel() -> None:
    header = CIMMetadataHeader.empty(subject=URIRef("h1"))
    header.add_triple(RDF.type, MD.FullModel)
    target_format, graph = _make_header_graph_for_conversion(header)

    assert target_format == "dcat_dataset"
    assert len(graph) == 1
    assert (URIRef("h1"), RDF.type, DCAT_EXT.Dataset) in graph
    assert graph.namespace_manager.store.namespace("dcat") == URIRef("http://www.w3.org/ns/dcat#") # Dcat is rdflib default

def test_make_header_graph_for_conversion_dcatdataset() -> None:
    header = CIMMetadataHeader.empty(subject=URIRef("h1"))
    header.add_triple(RDF.type, DCAT_EXT.Dataset)
    target_format, graph = _make_header_graph_for_conversion(header)

    assert target_format == "md_fullmodel"
    assert len(graph) == 1
    assert (URIRef("h1"), RDF.type, MD.FullModel) in graph
    assert graph.namespace_manager.store.namespace("md") == URIRef("http://iec.ch/TC57/61970-552/ModelDescription/1#")


def test_make_header_graph_for_conversion_extrardftype() -> None:
    header = CIMMetadataHeader.empty(subject=URIRef("h1"))
    header.add_triple(RDF.type, MD.FullModel)
    header.add_triple(RDF.type, URIRef("custom_type"))
    target_format, graph = _make_header_graph_for_conversion(header)

    assert target_format == "dcat_dataset"
    assert len(graph) == 1
    assert (URIRef("h1"), RDF.type, DCAT_EXT.Dataset) in graph
    assert (URIRef("h1"), RDF.type, URIRef("custom_type")) not in graph

def test_make_header_graph_for_conversion_ambiguoustype() -> None:
    header = CIMMetadataHeader.empty(subject=URIRef("h1"))
    header.add_triple(RDF.type, MD.FullModel)
    header.add_triple(RDF.type, DCAT_EXT.Dataset)
    with pytest.raises(ValueError, match='Multiple header types found in header.'): # ValueError carried forward from header.header_type.
        _make_header_graph_for_conversion(header)


def test_make_header_graph_for_conversion_notype() -> None:
    header = CIMMetadataHeader.empty(subject=URIRef("h1"))
    header.add_triple(DCAT_EXT.keyword, Literal("value"))
    with pytest.raises(ValueError, match='No header type found in header.'): # ValueError carried forward from header.header_type.
        _make_header_graph_for_conversion(header)

def test_make_header_graph_for_conversion_headerunchanged() -> None:
    header = CIMMetadataHeader.empty(subject=URIRef("h1"))
    header.add_triple(RDF.type, MD.FullModel)
    target_format, graph = _make_header_graph_for_conversion(header)
    header.add_triple(RDF.type, URIRef("custom_type"))

    assert len(graph) == 1
    assert (URIRef("h1"), RDF.type, DCAT_EXT.Dataset) in graph
    assert (URIRef("h1"), RDF.type, URIRef("custom_type")) not in graph # Changes to header do not affect the graph returned.
    assert (URIRef("h1"), RDF.type, URIRef("custom_type")) in header.graph


    
if __name__ == "__main__":
    pytest.main()