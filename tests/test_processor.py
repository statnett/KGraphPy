from sys import prefix

import pytest
from unittest.mock import patch, MagicMock
from rdflib import URIRef, Literal, BNode
from cim_plugin.processor import CIMProcessor, merge_namespace_managers
from cim_plugin.header import CIMMetadataHeader
from cim_plugin.graph import CIMGraph
from cim_plugin.namespaces import MD
from rdflib.namespace import DCAT, RDF

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


def test_extract_header_bnodes() -> None:
    g = CIMGraph()
    g.add((URIRef("h1"), RDF.type, DCAT.Dataset))
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
    assert (URIRef("h1"), RDF.type, DCAT.Dataset) in pr.graph.metadata_header.triples
    assert (URIRef("h1"), URIRef("urn:p:3"), Literal("value")) in pr.graph.metadata_header.triples
    assert len(pr.graph) == 1
    assert (URIRef("s1"), URIRef("p1"), URIRef("o")) in pr.graph


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


if __name__ == "__main__":
    pytest.main()