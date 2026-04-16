import pytest
from rdflib import URIRef, Literal
from cim_plugin.processor import CIMProcessor
from cim_plugin.graph import CIMGraph

def test_cimgraph_provenance() -> None:
    g = CIMGraph()
    pr = CIMProcessor(g, provenance_description="Initial load")

    g.add((URIRef("s1"), URIRef("p1"), Literal("o")))
    assert (URIRef("s1"), URIRef("p1"), Literal("o")) in pr.graph

    g.remove((URIRef("s1"), URIRef("p1"), Literal("o")))
    assert (URIRef("s1"), URIRef("p1"), Literal("o")) not in pr.graph

    assert pr.provenance
    entries = pr.provenance.entries
    assert len(entries) == 3
    assert entries[0]["step_name"] == "load_graph"
    assert entries[1]["step_name"] == "add_triple"
    assert entries[1]["description"] == "Added triple (rdflib.term.URIRef('s1'), rdflib.term.URIRef('p1'), rdflib.term.Literal('o'))"
    assert entries[2]["step_name"] == "remove_triple"
    assert entries[2]["description"] == "Removed triple (rdflib.term.URIRef('s1'), rdflib.term.URIRef('p1'), rdflib.term.Literal('o'))"

if __name__ == "__main__":
    pytest.main()