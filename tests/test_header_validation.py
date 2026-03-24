import pytest
from unittest.mock import call, patch, MagicMock
from rdflib import Graph, Literal, Node, URIRef
from rdflib.namespace import XSD, DCAT, DCTERMS
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

# Unit tests _fix_datetime_format_in_triples
@pytest.mark.parametrize(
    "predicates, object_returned",
    [
        pytest.param([DCAT.endDate], None, id="Predicate match, object None, triple not changed"),
        pytest.param([DCAT.endDate], Literal("old"), id="Predicate match, object same value, triple not changed"),
        pytest.param([DCAT.comment], Literal("old"), id="Predicate does not match, triple not changed"),
        pytest.param([DCAT.endDate], Literal("new"), id="DCAT.endDate, object new value, triple updated"),
        pytest.param([DCAT.startDate], Literal("new"), id="DCAT.startDate, object new value, triple updated"),
        pytest.param([DCTERMS.issued], Literal("new"), id="DCTERMS.issued, object new value, triple updated"),
        pytest.param([DCAT.endDate, DCAT.startDate], Literal("new"), id="Multiple predicates match, triples updated"),
        pytest.param([DCAT.startDate, DCAT.comment], Literal("new"), id="Multiple predicates, only one match, one triple updated"),
    ]
)
@patch("cim_plugin.header_validation._fix_datetime_format")
def test_fix_datetime_format_in_triples_basic(mock_fix: MagicMock, predicates: list[URIRef], object_returned: Any, caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    s = URIRef("s")
    old_o = Literal("old")
    g.add((s, URIRef("p1"), URIRef("o")))
    for p in predicates:
        g.add((s, p, old_o))
    mock_fix.return_value = object_returned
    _fix_datetime_format_in_triples(g)

    assert (s, URIRef("p1"), URIRef("o")) in g

    for p in predicates:    
        if not object_returned:
            assert f"Found None for {p}. Expected a datetime." in caplog.text
            assert (s, p, old_o) in g
        elif object_returned == old_o:
            assert (s, p, old_o) in g
        else:
            if p in {DCAT.endDate, DCAT.startDate, DCTERMS.issued}:
                assert (s, p, object_returned) in g
                assert f"Corrected date format for predicate {p}: from {old_o} to {object_returned}." in caplog.text
            else:
                assert (s, p, old_o) in g


@pytest.mark.parametrize(
    "predicate",
    [
        pytest.param(DCAT.endDate, id="DCAT.endDate"),
        pytest.param(DCAT.startDate, id="DCAT.startDate"),
        pytest.param(DCTERMS.issued, id="DCTERMS.issued"),
        pytest.param(DCAT.comment, id="DCAT.comment"),
    ]
)
@patch("cim_plugin.header_validation._fix_datetime_format")
def test_fix_datetime_format_in_triples_calls(mock_fix: MagicMock, predicate: URIRef) -> None:
    g = Graph()
    s = URIRef("s")
    old_o = Literal("old")
    g.add((s, predicate, old_o))
    mock_fix.return_value = Literal("new")
    predicates = [DCAT.endDate, DCAT.startDate, DCTERMS.issued]
    _fix_datetime_format_in_triples(g)
    if predicate in predicates:
        mock_fix.assert_called_once_with(old_o)
    else:
        mock_fix.assert_not_called()


@patch("cim_plugin.header_validation._fix_datetime_format")
def test_fix_datetime_format_in_triples_logcounts(mock_fix: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    s = URIRef("s")
    old_1 = Literal("old1")
    old_2 = Literal("old2")
    g.add((s, DCAT.endDate, old_1))
    g.add((s, DCAT.startDate, old_2))
    mock_fix.return_value = Literal("new")
    _fix_datetime_format_in_triples(g)
    
    mock_fix.assert_has_calls([call(old_1), call(old_2)], any_order=True)
    
    assert len(g) == 2
    assert (s, DCAT.endDate, Literal("new")) in g
    assert (s, DCAT.startDate, Literal("new")) in g
    assert caplog.text.count("Corrected date format for predicate") == 2


@patch("cim_plugin.header_validation._fix_datetime_format")
def test_fix_datetime_format_in_triples_idempotency(mock_fix: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    s = URIRef("s")
    old_1 = Literal("old1")
    g.add((s, DCAT.endDate, old_1))
    mock_fix.return_value = Literal("new")

    _fix_datetime_format_in_triples(g)
    _fix_datetime_format_in_triples(g)
    
    # Second call should not change anything, as the triple is already in correct format. 
    mock_fix.assert_has_calls([call(old_1), call(Literal("new"))], any_order=False)
    
    assert len(g) == 1
    assert (s, DCAT.endDate, Literal("new")) in g
    assert caplog.text.count("Corrected date format for predicate") == 1


@patch("cim_plugin.header_validation._fix_datetime_format")
def test_fix_datetime_format_in_triples_duplicates(mock_fix: MagicMock) -> None:
    g = Graph()
    s = URIRef("s")
    old_1 = Literal("old1")
    old_2 = Literal("old2")
    g.add((s, DCAT.endDate, old_1))
    g.add((s, DCAT.endDate, old_2))
    mock_fix.return_value = Literal("new")
    _fix_datetime_format_in_triples(g)
    
    mock_fix.assert_has_calls([call(old_1), call(old_2)], any_order=True)
    # Both triples gets the same new object, which makes them duplicates (removed by rdflib).
    assert len(g) == 1
    assert (s, DCAT.endDate, Literal("new")) in g

@patch("cim_plugin.header_validation._fix_datetime_format")
def test_fix_datetime_format_in_triples_emptygraph(mock_fix: MagicMock) -> None:
    g = Graph()
    _fix_datetime_format_in_triples(g)
    
    mock_fix.assert_not_called()
    assert len(g) == 0


@patch("cim_plugin.header_validation._fix_datetime_format")
def test_fix_datetime_format_in_triples_exception(mock_fix: MagicMock) -> None:
    g = Graph()
    s = URIRef("s")
    old_1 = Literal("old1")
    g.add((s, DCAT.startDate, old_1))
    g.add((s, DCAT.endDate, old_1))
    mock_fix.side_effect = [Literal("New"), TypeError("Simulated error")]

    with pytest.raises(TypeError, match="Simulated error"):
        _fix_datetime_format_in_triples(g)
    
        assert len(g) == 2
        # Nothing is changed when exception is raised. However, changes before the exception are not rolled back.
        assert (s, DCAT.startDate, Literal("New")) in g
        assert (s, DCAT.endDate, old_1) in g    

def test_fix_datetime_format_in_triples_objecturi(caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    s = URIRef("s")
    old_1 = URIRef("old1")
    g.add((s, DCAT.endDate, old_1))
    
    _fix_datetime_format_in_triples(g)
    
    # When the object is not a Literal, it is kept as is.
    assert len(g) == 1
    assert (s, DCAT.endDate, old_1) in g
    assert "Expected a Literal for datetime correction, got: rdflib.term.URIRef('old1')" in caplog.text


# Separate test
"""
One for this:
pytest.param([DCAT.startDate, DCAT.startDate], Literal("new"), id="Duplicate predicates match, triples updated"),

"""

if __name__ == "__main__":
    pytest.main()