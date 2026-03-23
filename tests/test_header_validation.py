
import pytest
from rdflib import Graph, Node, URIRef
from cim_plugin.header_validation import _remove_invalid_triples


# Unit tests _remove_invalid_triples
@pytest.mark.parametrize(
    "triples, predicates, obj, expected", 
    [
        pytest.param([], None, None, [], id="Empty graph"),
        pytest.param(
            [(URIRef("s1"), URIRef("p1"), URIRef("o1"))], 
            None, None, 
            [(URIRef("s1"), URIRef("p1"), URIRef("o1"))], 
            id="No predicates or objects"
        ),
        pytest.param(
            [(URIRef("s1"), URIRef("p1"), URIRef("o1")), (URIRef("s2"), URIRef("p2"), URIRef("o2"))], 
            URIRef("p1"), None, 
            [(URIRef("s2"), URIRef("p2"), URIRef("o2"))], 
            id="Remove by one predicate"
        ),
        pytest.param(
            [(URIRef("s1"), URIRef("p1"), URIRef("o1")), (URIRef("s2"), URIRef("p2"), URIRef("o2"))], 
            [URIRef("p1"), URIRef("p2")], None, 
            [], 
            id="Remove by multiple predicates"
        ),
        pytest.param(
            [(URIRef("s1"), URIRef("p1"), URIRef("o1")), (URIRef("s2"), URIRef("p2"), URIRef("o2"))], 
            None, URIRef("o1"), 
            [(URIRef("s2"), URIRef("p2"), URIRef("o2"))], 
            id="Remove by one object"
        ),
        pytest.param(
            [(URIRef("s1"), URIRef("p1"), URIRef("o1")), (URIRef("s2"), URIRef("p2"), URIRef("o2"))], 
            None, [URIRef("o1"), URIRef("o2")], 
            [], 
            id="Remove by multiple objects"
        ),
        pytest.param(
            [(URIRef("s1"), URIRef("p1"), URIRef("o1")), (URIRef("s2"), URIRef("p2"), URIRef("o2")), (URIRef("s3"), URIRef("p3"), URIRef("o3"))], 
            [URIRef("p1"), URIRef("p3")], [URIRef("o2")], 
            [], 
            id="Remove by multiple predicates and objects"
        ),
        pytest.param(
            [(URIRef("s1"), URIRef("p1"), URIRef("o1")), (URIRef("s2"), URIRef("p2"), URIRef("o2")), (URIRef("s3"), URIRef("p3"), URIRef("o3"))], 
            [URIRef("p1")], [URIRef("o2")], 
            [(URIRef("s3"), URIRef("p3"), URIRef("o3"))], 
            id="Remove by one predicate and one object"
        ),
        pytest.param(
            [(URIRef("s1"), URIRef("p1"), URIRef("o1"))], 
            [URIRef("p1")], [URIRef("o1")], 
            [], 
            id="Remove by one predicate and one object in the same triple"
        ),
        pytest.param(
            [(URIRef("s1"), URIRef("p1"), URIRef("o1")), (URIRef("s2"), URIRef("p1"), URIRef("o2"))],
            URIRef("p1"), None, 
            [],
            id="Remove multiple triples by same input."
        ),
        pytest.param(
            [(URIRef("s1"), URIRef("p1"), URIRef("o1")), (URIRef("s2"), URIRef("p2"), URIRef("o2"))],
            [URIRef("p1"), URIRef("p1")], None, 
            [(URIRef("s2"), URIRef("p2"), URIRef("o2"))],
            id="Duplicate input."
        ),
        pytest.param(
            [(URIRef("s1"), URIRef("p1"), URIRef("o1"))], 
            URIRef("p2"), URIRef("o2"), 
            [(URIRef("s1"), URIRef("p1"), URIRef("o1"))], 
            id="No matching input"
        ),
        pytest.param(
            [(URIRef("s1"), URIRef("p1"), URIRef("o1"))], 
            [], [], 
            [(URIRef("s1"), URIRef("p1"), URIRef("o1"))], 
            id="Empty lists input"
        ),
    ]
)
def test_remove_invalid_triples_various(triples: list[tuple[Node, Node, Node]], predicates: URIRef|list[URIRef], obj: URIRef|list[URIRef], expected: list[tuple[Node, Node, Node]]) -> None:
    graph = Graph()
    for s, p, o in triples:
        graph.add((s, p, o))
    
    _remove_invalid_triples(graph, predicates=predicates, obj=obj)
    
    assert set(graph) == set(expected)


def test_remove_invalid_triples_logging(caplog: pytest.LogCaptureFixture) -> None:
    graph = Graph()
    s, p, o = URIRef("s1"), URIRef("p1"), URIRef("o1")
    graph.add((s, p, o))
    
    with caplog.at_level("ERROR"):
        _remove_invalid_triples(graph, predicates=p)
    
    assert f"Invalid triple detected in header, removing: ({s}, {p}, {o})" in caplog.text

if __name__ == "__main__":
    pytest.main()