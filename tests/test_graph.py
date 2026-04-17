import pytest
from rdflib import URIRef, Literal, Namespace
from rdflib.namespace import XSD
from cim_plugin.processor import CIMProcessor
from typing import Any
from cim_plugin.graph import CIMGraph


def test_cimgraph_provenance() -> None:
    g = CIMGraph()
    pr = CIMProcessor(g, provenance_description="Initial load")

    g.add((URIRef("s1"), URIRef("p1"), Literal("o")))
    assert (URIRef("s1"), URIRef("p1"), Literal("o")) in pr.graph

    g.remove((URIRef("s1"), URIRef("p1"), Literal("o")))
    assert (URIRef("s1"), URIRef("p1"), Literal("o")) not in pr.graph

    g.add((URIRef("s1"), URIRef("p1"), Literal("o", datatype=XSD.string)))
    assert (URIRef("s1"), URIRef("p1"), Literal("o", datatype=XSD.string)) in pr.graph

    assert pr.provenance
    entries = pr.provenance.entries
    assert len(entries) == 4
    assert entries[0]["step_name"] == "load_graph"
    assert entries[1]["step_name"] == "add_triple"
    assert entries[1]["description"] == "Added triple (rdflib.term.URIRef('s1'), rdflib.term.URIRef('p1'), rdflib.term.Literal('o'))."
    assert entries[2]["step_name"] == "remove_triple"
    assert entries[2]["description"] == "Removed triple (rdflib.term.URIRef('s1'), rdflib.term.URIRef('p1'), rdflib.term.Literal('o'))."
    assert entries[3]["description"] == "Added triple (rdflib.term.URIRef('s1'), rdflib.term.URIRef('p1'), rdflib.term.Literal('o', datatype=rdflib.term.URIRef('http://www.w3.org/2001/XMLSchema#string')))."

def test_cimgraph_noprovenance() -> None:
    g = CIMGraph()

    g.bind("ex", "http://example.org/")
    g.add((URIRef("s1"), URIRef("p1"), Literal("o")))
    assert (URIRef("s1"), URIRef("p1"), Literal("o")) in g

    g.remove((URIRef("s1"), URIRef("p1"), Literal("o")))
    assert (URIRef("s1"), URIRef("p1"), Literal("o")) not in g

    assert g.namespace_manager.store.namespace("ex") == URIRef("http://example.org/")
    assert g._provenance is None

def test_cimgraph_chaining() -> None:
    g = CIMGraph()
    pr = CIMProcessor(g, provenance_description="Initial load")
    t1 = (URIRef("s1"), URIRef("p1"), Literal("o"))
    t2 = (URIRef("s2"), URIRef("p2"), Literal("o2"))

    g.add(t1).add(t2).remove(t1)

    assert pr.provenance
    assert len(pr.provenance.entries) == 4
    assert pr.provenance.entries[1]["description"] == "Added triple (rdflib.term.URIRef('s1'), rdflib.term.URIRef('p1'), rdflib.term.Literal('o'))."
    assert pr.provenance.entries[2]["description"] == "Added triple (rdflib.term.URIRef('s2'), rdflib.term.URIRef('p2'), rdflib.term.Literal('o2'))."
    assert pr.provenance.entries[3]["description"] == "Removed triple (rdflib.term.URIRef('s1'), rdflib.term.URIRef('p1'), rdflib.term.Literal('o'))."

def test_remove_multiple() -> None:
    g = CIMGraph()
    pr = CIMProcessor(g, provenance_description="Initial load")
    t1 = (URIRef("s1"), URIRef("p"), Literal("o"))
    t2 = (URIRef("s2"), URIRef("p"), Literal("o2"))
    g.add(t1).add(t2)

    g.remove((None, URIRef("p"), None))

    assert len(g) == 0
    assert pr.provenance
    assert len(pr.provenance.entries) == 4
    # When remove is used this way only one provenance entry is created even if multiple triples are removed.
    assert pr.provenance.entries[3]["description"] == "Removed triple (None, rdflib.term.URIRef('p'), None)."
    
def test_bind_override() -> None:
    g = CIMGraph()
    
    g.bind("ex", "http://example.org/")
    g.bind("ex2", "http://example.org/", override=False)
    assert g.namespace_manager.store.namespace("ex") == URIRef("http://example.org/")
    assert g.namespace_manager.store.namespace("ex2") == None


def test_bind_replace() -> None:
    g = CIMGraph()
    
    g.bind("ex", "http://example.org/")
    g.bind("ex", "http://new.org/", replace=True)
    assert g.namespace_manager.store.namespace("ex") == URIRef("http://new.org/")

@pytest.mark.parametrize("namespace", ["http://example.org/", Namespace("http://example.org/"), URIRef("http://example.org/")])
def test_bind_provenance(namespace: Any) -> None:
    g = CIMGraph()
    pr = CIMProcessor(g, provenance_description="Initial load")

    g.bind("ex", namespace)
    assert pr.graph.namespace_manager.store.namespace("ex") == URIRef("http://example.org/")

    assert pr.provenance
    entries = pr.provenance.entries
    assert len(entries) == 2
    assert entries[0]["step_name"] == "load_graph"
    assert entries[1]["step_name"] == "bind_namespace"
    assert entries[1]["description"] == "Bound namespace http://example.org/ to prefix ex."

def test_set_triple() -> None:
    g = CIMGraph()
    pr = CIMProcessor(g, provenance_description="Initial load")

    g.add((URIRef("s1"), URIRef("p1"), Literal("o")))
    assert (URIRef("s1"), URIRef("p1"), Literal("o")) in pr.graph

    g.set((URIRef("s1"), URIRef("p1"), Literal("o2")))
    assert (URIRef("s1"), URIRef("p1"), Literal("o2")) in pr.graph
    assert (URIRef("s1"), URIRef("p1"), Literal("o")) not in pr.graph

    assert pr.provenance
    entries = pr.provenance.entries
    assert len(entries) == 3
    assert entries[0]["step_name"] == "load_graph"
    assert entries[1]["step_name"] == "add_triple"
    assert entries[1]["description"] == "Added triple (rdflib.term.URIRef('s1'), rdflib.term.URIRef('p1'), rdflib.term.Literal('o'))."
    assert entries[2]["step_name"] == "set_triple"
    assert entries[2]["description"] == "Set triple (rdflib.term.URIRef('s1'), rdflib.term.URIRef('p1'), rdflib.term.Literal('o2'))."
    substeps = entries[2]["sub_steps"]
    assert len(substeps) == 2
    assert substeps[0]["step_name"] == "remove_triple"
    assert substeps[0]["description"] == "Removed triple (rdflib.term.URIRef('s1'), rdflib.term.URIRef('p1'), None)."
    assert substeps[1]["step_name"] == "add_triple"
    assert substeps[1]["description"] == "Added triple (rdflib.term.URIRef('s1'), rdflib.term.URIRef('p1'), rdflib.term.Literal('o2'))."
    
def test_iadd() -> None:
    g = CIMGraph(identifier=URIRef("self_graph"))
    pr = CIMProcessor(g, provenance_description="Initial load")
    otherg = CIMGraph(identifier=URIRef("other_graph"))
    otherg.add((URIRef("s1"), URIRef("p1"), Literal("o")))
    g += otherg
        
    assert (URIRef("s1"), URIRef("p1"), Literal("o")) in pr.graph
    assert pr.provenance and pr._provenance
    entries = pr.provenance.entries
    assert len(entries) == 2
    assert entries[1]["step_name"] == "merging_graphs"
    assert entries[1]["description"] == "Graph other_graph merged into self_graph."
    assert pr._provenance._entries[1].sub_steps == []   # addN is used instead of g.add when merging, so no substeps should be recorded

def test_isub() -> None:
    g = CIMGraph(identifier=URIRef("self_graph"))
    pr = CIMProcessor(g, provenance_description="Initial load")
    otherg = CIMGraph(identifier=URIRef("other_graph"))
    otherg.add((URIRef("s1"), URIRef("p1"), Literal("o")))
    g += otherg
    g -= otherg
        
    assert (URIRef("s1"), URIRef("p1"), Literal("o")) not in pr.graph
    assert pr.provenance and pr._provenance
    entries = pr.provenance.entries
    assert len(entries) == 3
    assert entries[2]["step_name"] == "removing_subgraph_from_graph"
    assert entries[2]["description"] == "Triples in graph other_graph removed from self_graph."
    substeps = pr.provenance.entries[2]["sub_steps"]
    assert len(substeps) == 1
    assert substeps[0]["step_name"] == "remove_triple"
    assert substeps[0]["description"] == "Removed triple (rdflib.term.URIRef('s1'), rdflib.term.URIRef('p1'), rdflib.term.Literal('o'))."

if __name__ == "__main__":
    pytest.main()