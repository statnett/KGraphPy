import pytest
from rdflib import URIRef
from cim_plugin.jsonld_utilities import reorder_jsonld, sort_subjects, sort_predicates

# Unit tests sort_subjects
@pytest.mark.parametrize(
    "priority_subject, expected_order", 
    [
        pytest.param(None, ["a", "b", "c"], id="No priority"),
        pytest.param("a", ["a", "b", "c"], id="Priority a"),
        pytest.param("d", ["a", "b", "c"], id="Priority not in list"),
        pytest.param(URIRef("c"), ["c", "a", "b"], id="Priority URIRef"),
        pytest.param("C", ["a", "b", "c"], id="Priority different case than nodes, treated as not in list"),
    ]
)
def test_sort_subjects_priority(priority_subject, expected_order) -> None:
    nodes = [
        {"@id": "b", "name": "Node B"},
        {"@id": "a", "name": "Node A"},
        {"@id": "c", "name": "Node C"}
    ]
    sorted_nodes = sort_subjects(nodes, priority_subject=priority_subject)
    assert [n["@id"] for n in sorted_nodes] == expected_order


def test_sort_subjects_missingid() -> None:
    nodes = [
        {"name": "no id"},
        {"@id": "b"},
        {"name": "also no id"},
        {"@id": "a"},
    ]
    sorted_nodes = sort_subjects(nodes)
    assert [n.get("@id") for n in sorted_nodes] == [None, None, "a", "b"] # Missing @id should are treated as empty string and come first in alphabetical order.
    assert [n.get("name") for n in sorted_nodes] == ["no id", "also no id", None, None] # Nodes without @id are still included, but not sorted.

def test_sort_subjects_duplicateids() -> None:
    nodes = [
        {"@id": "a", "x": 2},
        {"@id": "a", "x": 1},
    ]
    sorted_nodes = sort_subjects(nodes)
    assert [n["x"] for n in sorted_nodes] == [2, 1] # If there are duplicate @id, their order is preserved (stable sort).

def test_sort_subjects_priorityblanknode() -> None:
    # Blank nodes are treated the same as any other node.
    nodes = [
        {"@id": "b"},
        {"@id": "_:b1"},
        {"@id": "a"},
    ]
    sorted_nodes = sort_subjects(nodes, priority_subject="_:b1")
    assert [n["@id"] for n in sorted_nodes] == ["_:b1", "a", "b"]


# Unit tests sort_predicates
def test_sort_predicates_basic() -> None:
    node = {
        "@type": "Example",
        "z": 3,
        "a": 1,
        "@id": "node1",
        "m": 2,
        "z": 2,
        "@alpha": "random value" # Not standard in JSON-LD formats. Included to show that all keys are included even when not valid for the intended usage of the function.
    }
    sorted_node = sort_predicates(node)
    assert list(sorted_node.keys()) == ["@id", "@type", "@alpha", "a", "m", "z"] # @id and @type first, then invalid @ keys, then predicates in alphabetical order.
    for k, v in node.items():
        assert sorted_node[k] == v # Values are preserved.

    assert sorted_node["z"] == 2 # Duplicates are removed and the last value is kept.
    # This is not an issue because predicate duplicates with different objects have the values in lists, ex. {"dct:issued": ["2020-01-01", "2021-01-01"]}


def test_sort_predicates_emptynode() -> None:
    node = {}
    sorted_node = sort_predicates(node)
    assert sorted_node == {} # Return empty node without error.

@pytest.mark.parametrize(
    "node, expected_order",
    [
        pytest.param({"@id": "1", "@type": "T", "a": 1, "b": 2}, ["@id", "@type", "a", "b"], id="Already sorted"),
        pytest.param({"@z": 1, "@a": 2}, ["@a", "@z"], id="Only @ keys"),
        pytest.param({"a": 1, "c": 2, "b": 3}, ["a", "b", "c"], id="Only predicate keys, no @ keys"),
    ]
)
def test_sort_predicates_various(node: dict, expected_order: list[str]) -> None:
    sorted_node = sort_predicates(node)
    assert list(sorted_node.keys()) == expected_order

def test_sort_predicates_valuespreserved() -> None:
    node = {
        "@id": "x",
        "b": [1, 2],
        "a": {"nested": True},
    }
    sorted_node = sort_predicates(node)
    assert sorted_node["a"] == {"nested": True}
    assert sorted_node["b"] == [1, 2]

if __name__ == "__main__":
    pytest.main()