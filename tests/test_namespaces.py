import pytest
from rdflib import URIRef, Namespace, Graph, Literal
from rdflib.namespace import RDF
from cim_plugin.namespaces import collect_specific_namespaces


# Unit tests collect_specific_namespaces
@pytest.mark.parametrize(
        "triples, expected",
        [
            pytest.param([], {}, id="Empty triples"),
            pytest.param([(URIRef("http://example.com/s1"), URIRef("p1"), URIRef("o1"))], {"ex": URIRef("http://example.com/")}, id="Subject collected"),
            pytest.param([(URIRef("s1"), URIRef("http://example.com/p1"), URIRef("o1"))], {"ex": URIRef("http://example.com/")}, id="Predicate collected"),
            pytest.param([(URIRef("s1"), URIRef("p1"), URIRef("http://example.com/o1"))], {"ex": URIRef("http://example.com/")}, id="Object collected"),
            pytest.param([(URIRef("http://example.com/s1"), URIRef("http://foo.org/ns#/p1"), URIRef("http://bar.org/o1"))], 
                         {"ex": URIRef("http://example.com/"), "foo": URIRef("http://foo.org/ns#"), "bar": URIRef("http://bar.org/")}, id="Namespaces in all"),
            pytest.param([(URIRef("http://notpresent.com/s1"), URIRef("p1"), URIRef("o1"))], {}, id="Namespace not in namespace_manager"),
            pytest.param([(URIRef("http://example.com/s1"), URIRef("p1"), URIRef("o1")), (URIRef("http://example.com/s2"), URIRef("http://bar.org/p1"), URIRef("o1"))], 
                         {"ex": URIRef("http://example.com/"), "bar": URIRef("http://bar.org/")}, id="Multiple triples"),
            pytest.param([(URIRef("http://bar.org/foo/s1"), URIRef("p1"), URIRef("http://bar.org/o1"))], 
                         {"bf": URIRef("http://bar.org/foo/"), "bar": URIRef("http://bar.org/")}, id="Overlapping namespaces"),
            pytest.param([(URIRef("http://bar.org/foo/s1"), URIRef("p1"), URIRef("o1"))], {"bf": URIRef("http://bar.org/foo/")}, id="Overlapping namespaces, only longest present"),
            pytest.param([(URIRef("s1"), URIRef("p1"), Literal("http://example.com/o1"))], {}, id="Literal not collected"),
            pytest.param([(URIRef("www.noslash.coms1"), URIRef("p1"), URIRef("o1"))], {"no": URIRef("www.noslash.com")}, id="Odd namespace"),
            pytest.param([(URIRef("http://example.com/%s1"), URIRef("p1"), URIRef("o1"))], {"ex": URIRef("http://example.com/")}, id="Symbols in name"),
            pytest.param([(URIRef("http://example.com/"), URIRef("p1"), URIRef("o1"))], {"ex": URIRef("http://example.com/")}, id="Namespace is the entire uri"),
        ]
)
def test_collect_specific_namespaces_basic(triples: list[tuple], expected: dict[str, URIRef]) ->None:
    g = Graph()
    g.bind("ex", Namespace("http://example.com/"))
    g.bind("foo", Namespace("http://foo.org/ns#"))
    g.bind("bar", Namespace("http://bar.org/"))
    g.bind("bf", Namespace("http://bar.org/foo/"))
    g.bind("no", Namespace("www.noslash.com"))
    nm = g.namespace_manager

    result = collect_specific_namespaces(triples, nm)
    assert result == expected

def test_collect_specific_namespaces_emptymanager() -> None:
    g = Graph()
    # No namespaces bound to graph
    g.add((URIRef("s1"), RDF.type, Literal("o")))

    result = collect_specific_namespaces(list(g.triples((None, None, None))), g.namespace_manager)
    assert "rdf" in result.keys() # rdf is a default namespace in rdflib
    assert len(result) == 1

def test_collect_specific_namespaces_emptygraf() -> None:
    g = Graph()

    result = collect_specific_namespaces(list(g.triples((None, None, None))), g.namespace_manager)
    assert len(result) == 0


if __name__ == "__main__":
    pytest.main()