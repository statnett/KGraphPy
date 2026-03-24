import pytest
from unittest.mock import patch, MagicMock
from rdflib import Graph, Literal, Node, URIRef
from rdflib.namespace import XSD
from typing import Any
import datetime

from cim_plugin.header_validation import _remove_invalid_triples, _fix_datetime_format_in_triples, _fix_datetime_format


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


# Unit tests _fix_datetime_format
@pytest.mark.parametrize(
    "input, expected, comment",
    [
        pytest.param(Literal("2025-02-14T00:00:00+00:00"), Literal("2025-02-14T00:00:00+00:00", datatype=XSD.dateTime), "no casting", id="UTC with +00:00"),
        pytest.param(Literal("2025-02-14T01:00:00+01:00"), Literal("2025-02-14T01:00:00+01:00", datatype=XSD.dateTime), "no casting", id="Non-UTC timezone"),
        pytest.param(Literal("2025-02-14T01:00:00Z"), Literal("2025-02-14T01:00:00Z", datatype=XSD.dateTime), "no casting", id="Already in correct format"),
        pytest.param(Literal("2025-02-14"),  Literal("2025-02-14T00:00:00+00:00", datatype=XSD.dateTime), "cast", id="Date only, time added"),
        pytest.param(Literal("Not a date"), "Not a date", "cast error", id="Invalid date string, unchanged"),
        pytest.param(URIRef("o"), URIRef("o"), "literal error", id="Input is URI, unchanged"),
        pytest.param(Literal("2025-02-14T00:00:00"), Literal("2025-02-14T00:00:00",datatype=XSD.dateTime), "no casting", id="Missing timezone, unchanged"),
        pytest.param(Literal("2025-02-14T00:00:00+0000"), Literal("2025-02-14T00:00:00+0000",datatype=XSD.dateTime), "no casting", id="Timezone without colon, unchanged"),
        pytest.param("2025-02-14T00:00:00+00:00", "2025-02-14T00:00:00+00:00", "literal error", id="Input is not a Literal, unchanged"),
        pytest.param(Literal("2025-02-14T00:00:00+00:00", datatype=XSD.dateTime), Literal("2025-02-14T00:00:00+00:00", datatype=XSD.dateTime), "no casting", id="Input literal with datatype"),
        pytest.param(Literal("2025-02-14", datatype=XSD.date),  Literal("2025-02-14T00:00:00+00:00", datatype=XSD.dateTime), "cast", id="Date with date datatype"),
        pytest.param(Literal(datetime.datetime(2025, 2, 14, tzinfo=datetime.timezone.utc)), Literal("2025-02-14T00:00:00+00:00", datatype=XSD.dateTime), "cast", id="Datetime with timezone as input"),
        pytest.param(Literal("2025-13-99T99:99:99Z"), Literal("2025-13-99T99:99:99Z", datatype=XSD.dateTime), "no casting", id="Invalid date and time string, unchanged"),
        pytest.param(Literal("2025-02-14t00:00:00Z"), Literal("2025-02-14t00:00:00Z", datatype=XSD.dateTime), "no casting", id="Lowercase 't' in datetime string, casted"),
        pytest.param(Literal("2025-02-14T00:00:00.123456Z"), Literal("2025-02-14T00:00:00.123456Z", datatype=XSD.dateTime), "no casting", id="Datetime string with microseconds, unchanged"),
        pytest.param(Literal("2025-02-14", lang="en"), Literal("2025-02-14T00:00:00+00:00", datatype=XSD.dateTime), "cast", id="Literal with language tag, unchanged"),
        pytest.param(Literal("2025-02-14 00:00:00Z"), Literal("2025-02-14 00:00:00Z", datatype=XSD.dateTime), "no casting", id="Datetime string with space instead of 'T', casted"),
        pytest.param(Literal("20250214T000000Z"), Literal("20250214T000000Z", datatype=XSD.dateTime), "no casting", id="Datetime string with T and no :")
    ]
)
def test_fix_datetime_format(input: Any, expected: Any, comment: str|None, caplog: pytest.LogCaptureFixture) -> None:
    result = _fix_datetime_format(input)
    print(result)
    if comment == "cast":
        assert isinstance(result, Literal)
        assert result == expected
        assert result.datatype == XSD.dateTime
    elif comment == "no casting":
        assert isinstance(result, Literal)
        assert result == expected
        assert result.datatype == XSD.dateTime
    elif comment == "cast error":
        assert isinstance(result, Literal)
        assert result.value == expected
        assert result.datatype == input.datatype
        assert "Failed to correct datetime format for literal" in caplog.text
    elif comment == "literal error":
        assert "Expected a Literal for datetime correction" in caplog.text
        assert result == input
        

@pytest.mark.parametrize(
    "input, calls",
    [
        pytest.param("2025-02-14", True, id="Date string should call cast"),
        pytest.param("2025-02-14T00:00:00+00:00", False, id="Already correct format should not call cast"),
        pytest.param("2025-02-14t00:00:00Z", False, id="Lowercase 't' cast not called"),
    ]
)
@patch("cim_plugin.header_validation.cast_datetime_utc")
def test_fix_datetime_format_calls(mock_cast: MagicMock, input: str, calls: bool) -> None:
    o = Literal(input)
    return_value = Literal("2025-02-14T00:00:00+00:00", datatype=XSD.dateTime)
    mock_cast.return_value = return_value
    fixed_object = _fix_datetime_format(o)
    
    if calls:
        mock_cast.assert_called_once_with(o)
        assert fixed_object == return_value
    else:
        mock_cast.assert_not_called()
        assert fixed_object == return_value


@patch("cim_plugin.header_validation.cast_datetime_utc")
def test_fix_datetime_format_castingerror(mock_cast: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    o = Literal("not a date")
    mock_cast.side_effect = ValueError("Invalid date format")

    fixed_object = _fix_datetime_format(o)
    mock_cast.assert_called_once_with(o)
    assert fixed_object == o
    assert "Failed to correct datetime format for literal" in caplog.text

    
if __name__ == "__main__":
    pytest.main()