import pytest
from unittest.mock import patch, MagicMock
import json
from rdflib import URIRef
from typing import Any
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
        {"@id": "_:a"}, # Blank nodes are sorted before other nodes in alphanumeric order
    ]
    sorted_nodes = sort_subjects(nodes, priority_subject="_:b1")
    assert [n["@id"] for n in sorted_nodes] == ["_:b1", "_:a", "a", "b"]


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

# Unit test reorder_jsonld
def test_reorder_jsonld_basic() -> None:
    raw = """
    {
      "@context": {"link": "http://example.com/context", "@vocab": "http://example.com/vocab#"},
      "@graph": [
        {"@id": "b", "name": "B"},
        {"@id": "a", "name": "A", "extra": "value"}
      ]
    }
    """

    result = reorder_jsonld(raw)

    data = json.loads(result)
    assert data["@context"] == {"link": "http://example.com/context", "@vocab": "http://example.com/vocab#"} # The non-graph part of the JSON-LD should be preserved.
    assert data["@graph"][0]["@id"] == "a"
    assert data["@graph"][1]["@id"] == "b"
    assert list(data["@graph"][0].keys()) == ["@id", "extra", "name"] # Checking correct order of the keys.

@pytest.mark.parametrize(
    "raw, expected_result",
    [
        pytest.param("""{"random": [{"@id": "b", "name": "B"}, {"@id": "a", "name": "A"}]}""",
                     {"random": [{"@id": "b", "name": "B"}, {"@id": "a", "name": "A"}]},
                        id="No @graph, should not sort"
        ),
        pytest.param("""[{"@id": "b", "name": "B"}, {"@id": "a", "name": "A"}]""",
                        [{"@id": "a", "name": "A"}, {"@id": "b", "name": "B"}],
                        id="List, output a sorted list"
        ),
        pytest.param("""{"@graph": [{"@id": "_:b", "name": "B"}, {"@id": "a", "name": "A"}]}""",
                    {"@graph": [{"@id": "_:b", "name": "B"}, {"@id": "a", "name": "A"}]},
                    id="Blank nodes, the blank nodes ids comes first"
        ),
        pytest.param("""{}""", {}, id="Empty dict input"),
        pytest.param("""[]""", [], id="Empty list input"),
        pytest.param('{"@graph": "not a list"}', {"@graph": "not a list"}, id="@graph is not a list, return unsorted"),
        pytest.param('{"@graph": [{"@id": "x", "tags": ["b", "a", "c"]}]}', {"@graph": [{"@id": "x", "tags": ["b", "a", "c"]}]}, id="Nodes with arrays. The array values are not sorted."),
        pytest.param('{"@graph": [{"@id": "x", "address": {"street": "any", "country": "Norway"}}]}', 
                     {"@graph": [{"@id": "x", "address": {"street": "any", "country": "Norway"}}]}, 
                     id="Nested dicts. The nested dict values are not sorted (no recursive sorting)."),
        
        # NB! The tests below does not actually test predicate sorting (because dicts are unordered). But the sort_predicates tests does.
        pytest.param("""{"@graph": [{"year": "b", "name": "B"}, {"year": "a", "name": "A"}]}""",
                    {"@graph": [{"name": "B", "year": "b"}, {"name": "A", "year": "a"}]},
                    id="No @id in nodes, predicates sorted alphanumerically" 
        ),
        pytest.param("""{"@graph": [{"address": "b", "@type": "B"}, {"address": "a", "@type": "A"}]}""",
                    {"@graph": [{"@type": "B", "address": "b"}, {"@type": "A", "address": "a"}]},
                    id="No @id but @type is there, predicates sorted alphanumerically" 
        ),

    ]
)
def test_reorder_jsonld_unusualinputs(raw: str, expected_result: Any) -> None:
    result = reorder_jsonld(raw)

    data = json.loads(result)
    assert data == expected_result

@patch("cim_plugin.jsonld_utilities.sort_predicates", side_effect=lambda x: x)
@patch("cim_plugin.jsonld_utilities.sort_subjects", return_value=[{"x": 1}])
def test_reorder_jsonld_calls_helpers(mock_subj: MagicMock, mock_pred: MagicMock) -> None:
    
    raw = '{"@graph": [{"x": 2}]}'
    reorder_jsonld(raw, priority_subject="some subject")

    mock_subj.assert_called_once_with([{"x": 2}], "some subject")
    mock_pred.assert_called_once()


def test_reorder_jsonld_notjsoninput() -> None:    
    with pytest.raises(json.decoder.JSONDecodeError):
        reorder_jsonld("not jsone")


def test_reorder_jsonld_idempotency() -> None:
    raw = """
    {
      "@context": {"link": "http://example.com/context", "@vocab": "http://example.com/vocab#"},
      "@graph": [
        {"@id": "b", "name": "B"},
        {"@id": "a", "name": "A", "extra": "value"}
      ]
    }
    """

    result1 = reorder_jsonld(raw)
    result2 = reorder_jsonld(result1)

    assert result1 == result2 # Reordering an already reordered JSON-LD should not change it.


if __name__ == "__main__":
    pytest.main()