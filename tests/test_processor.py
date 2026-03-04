import pytest
from unittest.mock import patch, MagicMock
from rdflib import URIRef, Literal
from cim_plugin.processor import CIMProcessor
from cim_plugin.header import CIMMetadataHeader
from cim_plugin.graph import CIMGraph
from cim_plugin.namespaces import MD
from rdflib.namespace import DCAT, RDF


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
    g.add((URIRef("h1"), RDF.type, DCAT.Dataset))
    g.add((URIRef("s1"), URIRef("p1"), URIRef("o")))

    pr = CIMProcessor(g)
    
    pr.extract_header()

    assert pr.graph.metadata_header
    assert pr.graph.metadata_header.subject == URIRef("h1")
    assert (URIRef("h1"), RDF.type, DCAT.Dataset) in pr.graph.metadata_header.triples
    assert len(pr.graph) == 1
    assert (URIRef("s1"), URIRef("p1"), URIRef("o")) in pr.graph


def test_extract_header_headeralready(caplog: pytest.LogCaptureFixture) -> None:
    g = CIMGraph()
    g.metadata_header = CIMMetadataHeader.empty(URIRef("h1"))
    g.metadata_header.add_triple(RDF.type, DCAT.Dataset)
    g.add((URIRef("s1"), URIRef("p1"), URIRef("o")))

    pr = CIMProcessor(g)
    
    pr.extract_header()

    assert pr.graph.metadata_header
    assert pr.graph.metadata_header.subject == URIRef("h1") # Header remains unchanged
    assert (URIRef("h1"), RDF.type, DCAT.Dataset) in pr.graph.metadata_header.triples
    assert len(pr.graph) == 1
    assert (URIRef("s1"), URIRef("p1"), URIRef("o")) in pr.graph
    assert caplog.records[0].levelname == "ERROR"
    assert caplog.records[0].message == "Metadata header already exist. Use .replace_header instead."


def test_extract_header_multiplecalls(caplog: pytest.LogCaptureFixture) -> None:
    g = CIMGraph()
    g.add((URIRef("h1"), RDF.type, DCAT.Dataset))
    g.add((URIRef("s1"), URIRef("p1"), URIRef("o")))

    pr = CIMProcessor(g)
    
    pr.extract_header()
    pr.extract_header()

    assert pr.graph.metadata_header
    assert pr.graph.metadata_header.subject == URIRef("h1")
    assert (URIRef("h1"), RDF.type, DCAT.Dataset) in pr.graph.metadata_header.triples
    assert len(pr.graph) == 1
    assert (URIRef("s1"), URIRef("p1"), URIRef("o")) in pr.graph
    assert "Metadata header already exist. Use .replace_header instead." in caplog.text


def test_extract_header_largergraphs() -> None:
    g = CIMGraph()
    g.add((URIRef("h1"), RDF.type, DCAT.Dataset))
    g.add((URIRef("h1"), DCAT.keyword, Literal("header")))
    g.add((URIRef("s1"), URIRef("p1"), URIRef("o")))
    g.add((URIRef("s2"), URIRef("p2"), URIRef("o2")))

    pr = CIMProcessor(g)
    
    pr.extract_header()

    assert pr.graph.metadata_header
    assert pr.graph.metadata_header.subject == URIRef("h1")
    header_triples = pr.graph.metadata_header.triples
    assert len(header_triples) == 2
    assert (URIRef("h1"), DCAT.keyword, Literal("header")) in header_triples
    assert (URIRef("h1"), RDF.type, DCAT.Dataset) in header_triples
    assert len(pr.graph) == 2
    assert (URIRef("s1"), URIRef("p1"), URIRef("o")) in pr.graph
    assert (URIRef("s2"), URIRef("p2"), URIRef("o2")) in pr.graph


if __name__ == "__main__":
    pytest.main()