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
    header.add_triple(RDF.type, DCAT.Dataset)
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), Literal("o")))
    g.metadata_header = header

    pr = CIMProcessor(g)
    pr.merge_header()

    assert (URIRef("h1"), RDF.type, DCAT.Dataset) in pr.graph
    assert len(pr.graph) == 2

@patch("cim_plugin.processor.merge_namespace_managers")
def test_merge_header_mergecalled(mock_merge: MagicMock) -> None:
    header = CIMMetadataHeader.empty(URIRef("h1"))
    header.graph.bind("foo", "www.bar.org/")
    header.graph.bind("md", MD)
    header.add_triple(RDF.type, MD.Fullmodel)
    header.add_triple(URIRef("www.bar.org/p"), Literal("oh"))
    g = CIMGraph()
    g.bind("md", MD)
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), Literal("o")))
    g.metadata_header = header

    pr = CIMProcessor(g)
    pr.merge_header()

    assert pr.graph.metadata_header
    assert (URIRef("h1"), RDF.type, MD.Fullmodel) in pr.graph
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
    header.add_triple(RDF.type, MD.Fullmodel)
    header.add_triple(URIRef("www.bar.org/p"), Literal("oh"))
    g = CIMGraph()
    g.bind("md", MD)
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), Literal("o")))
    g.metadata_header = header # Using this line to add the header to the graph

    pr = CIMProcessor(g)
    pr.merge_header()

    assert (URIRef("h1"), RDF.type, MD.Fullmodel) in pr.graph
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
    header.add_triple(RDF.type, MD.Fullmodel)
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

    assert (URIRef("h1"), RDF.type, MD.Fullmodel) in pr.graph
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
    header.add_triple(RDF.type, MD.Fullmodel)
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), Literal("o")))
    g.metadata_header = header
    ns_before = list(g.namespace_manager.store.namespaces())

    pr = CIMProcessor(g)
    pr.merge_header()
    pr.merge_header()

    assert (URIRef("h1"), RDF.type, MD.Fullmodel) in pr.graph
    assert len(pr.graph) == 2   # No duplicate triples
    ns_after = pr.graph.namespace_manager.store
    assert len(list(ns_after.namespaces())) == len(ns_before) + 1 # Only one namespace has been added
    assert ("md", URIRef('http://iec.ch/TC57/61970-552/ModelDescription/1#')) in list(ns_after.namespaces())


def test_merge_header_noduplicatetriples() -> None:
    header = CIMMetadataHeader.empty(URIRef("h1"))
    header.graph.bind("md", MD)
    header.add_triple(RDF.type, MD.Fullmodel)
    g = CIMGraph()
    g.bind("ex", "https://example.com/")
    g.bind("md", MD)
    g.add((URIRef("https://example.com/s1"), URIRef("https://example.com/p1"), Literal("o")))
    g.add((URIRef("h1"), RDF.type, MD.Fullmodel))   # Header triple already in graph
    g.metadata_header = header
    ns_before = list(g.namespace_manager.store.namespaces())

    pr = CIMProcessor(g)
    pr.merge_header()

    assert (URIRef("h1"), RDF.type, MD.Fullmodel) in pr.graph
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