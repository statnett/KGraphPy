import pytest
from rdflib import URIRef, Namespace, Graph, Literal, BNode
from rdflib.namespace import RDF
from cim_plugin.namespaces import collect_specific_namespaces, update_namespace_in_triples


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


# Unit tests update_namespace_in_triples
@pytest.mark.parametrize(
    "triple, old_ns, new_ns, expected",
    [
        pytest.param(
            (URIRef("http://old.com/a"), URIRef("x"), URIRef("y")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("http://new.com/a"), URIRef("x"), URIRef("y")),
            id="Subject update"
        ),
        pytest.param(
            (URIRef("s"), URIRef("http://old.com/p"), URIRef("o")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("s"), URIRef("http://new.com/p"), URIRef("o")),
            id="Predicate update"
        ),
        pytest.param(
            (URIRef("s"), URIRef("p"), URIRef("http://old.com/o")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("s"), URIRef("p"), URIRef("http://new.com/o")),
            id="Object update"
        ),
        pytest.param(
            (URIRef("s"), URIRef("p"), URIRef("o")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("s"), URIRef("p"), URIRef("o")),
            id="Simple no update"
        ),
        pytest.param(
            (URIRef("http://unrecognized.com/s"), URIRef("http://unrecognized.com/p"), URIRef("http://unrecognized.com/o")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("http://unrecognized.com/s"), URIRef("http://unrecognized.com/p"), URIRef("http://unrecognized.com/o")),
            id="No update, all uris."
        ),
        pytest.param(
            (BNode("s"), URIRef("http://old.com/p"), URIRef("o")),
            "http://old.com/",
            "http://new.com/",
            (BNode("s"), URIRef("http://new.com/p"), URIRef("o")),
            id="Subject is BNode"
        ),
        pytest.param(
            (URIRef("s"), URIRef("http://old.com/p"), BNode("o")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("s"), URIRef("http://new.com/p"), BNode("o")),
            id="Object is BNode"
        ),
        pytest.param(
            (URIRef("s"), URIRef("http://old.com/p"), Literal("o")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("s"), URIRef("http://new.com/p"), Literal("o")),
            id="Object is Literal"
        ),
        pytest.param(
            (URIRef("http://old.comX/a"), URIRef("x"), URIRef("y")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("http://old.comX/a"), URIRef("x"), URIRef("y")),
            id="Partial match -> No update"
        ),
        # This may cause problems!!
        pytest.param(
            (URIRef("http://old.com/a/http://old.com/b"), URIRef("x"), URIRef("y")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("http://new.com/a/http://new.com/b"), URIRef("x"), URIRef("y")),
            id="Multiple occurence of same uri"
        ),
        pytest.param(
            (URIRef("http://old.complex/foo/a"), URIRef("x"), URIRef("y")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("http://old.complex/foo/a"), URIRef("x"), URIRef("y")),
            id="Uri is part of another uri -> No update"
        ),
        pytest.param(
            (URIRef("http://old.com/a"), URIRef("x"), URIRef("y")),
            "http://old.com/",
            "",
            (URIRef("a"), URIRef("x"), URIRef("y")),
            id="New namespace empty"
        ),
    ]
)
def test_update_namespace_in_triples_singletriple(triple: tuple, old_ns: str, new_ns: str, expected: tuple) -> None:
    g = Graph()
    g.add(triple)

    update_namespace_in_triples(g, old_ns, new_ns)
    for s, p, o in g:
        print(s, p, o)
    assert len(g) == 1
    assert expected in g


def test_update_namespace_in_triples_mixedtriples() -> None:
    old_ns = "http://old.com/"
    new_ns = "http://new.com/"

    g = Graph()
    t1 = (URIRef("http://old.com/a"), URIRef("p"), URIRef("o"))
    t2 = (URIRef("s"), URIRef("http://old.com/p"), URIRef("o"))
    t3 = (URIRef("s"), URIRef("p"), URIRef("http://old.com/o"))
    t4 = (URIRef("s"), URIRef("p"), URIRef("o"))  # unchanged

    for t in (t1, t2, t3, t4):
        g.add(t)

    update_namespace_in_triples(g, old_ns, new_ns)

    expected = {
        (URIRef("http://new.com/a"), URIRef("p"), URIRef("o")),
        (URIRef("s"), URIRef("http://new.com/p"), URIRef("o")),
        (URIRef("s"), URIRef("p"), URIRef("http://new.com/o")),
        t4,
    }

    assert len(g) == 4
    for triple in expected:
        assert triple in g


def test_update_namespace_in_triples_multiplereplacements() -> None:
    g = Graph()
    old_ns = "http://old.com/"
    new_ns = "http://new.com/"

    triple = (URIRef("http://old.com/s"), URIRef("http://old.com/p"), URIRef("http://old.com/o"),)
    g.add(triple)

    update_namespace_in_triples(g, old_ns, new_ns)

    expected = (URIRef("http://new.com/s"), URIRef("http://new.com/p"), URIRef("http://new.com/o"),)

    assert len(g) == 1
    assert expected in g
    assert triple not in g


def test_update_namespace_in_triples_duplicatesafterreplacement() -> None:
    # This test shows what happends if replacements leads to two uris being made identical
    g = Graph()
    old_ns = "http://old.com/"
    new_ns = "http://new.com/"

    t1 = (URIRef("http://old.com/a"), URIRef("p"), URIRef("o"))
    t2 = (URIRef("http://new.com/a"), URIRef("p"), URIRef("o"))
    g.add(t1)
    g.add(t2)

    assert len(g) == 2  # Number of triples before replacement

    update_namespace_in_triples(g, old_ns, new_ns)

    expected = (URIRef("http://new.com/a"), URIRef("p"), URIRef("o"))

    assert len(g) == 1  # Number of triples after replacement
    assert expected in g


def test_update_namespace_in_triples_emptygraph() -> None:
    g = Graph()

    update_namespace_in_triples(g, "http://old.com/", "http://new.com/")

    assert len(g) == 0


def test_update_namespace_in_triples() -> None:
    g = Graph()
    g.add((URIRef("a"), URIRef("x"), URIRef("y")))
    assert len(g) == 1

    with pytest.raises(ValueError, match="old_namespace cannot be an empty string"):
        update_namespace_in_triples(g, "", "http://new.com/")

if __name__ == "__main__":
    pytest.main()