import pytest
from unittest.mock import ANY, call, patch, MagicMock
from rdflib import BNode, Graph, Literal, Node, URIRef
from rdflib.namespace import XSD, DCAT, DCTERMS, RDF
from cim_plugin.header import CIMMetadataHeader
from cim_plugin.namespaces import JSONLD, RDFG, MD
from typing import Any
import datetime

from cim_plugin.header_validation import (
    # _check_dcterms_issued_count, 
    _check_trig_rdfg_graph,
    _correct_triple_representation_by_predicate,
    _make_bnode_triple_for_given_predicate,
    _remove_cimxml_rdfg_graph,
    _remove_cimxml_rdfg_graph, 
    _remove_invalid_triples, 
    _fix_datetime_format_in_triples, 
    _fix_datetime_format,
    _fix_cimxml_period_of_time_format,
    _fix_trig_period_of_time_format,
    validate_header
)


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

def test_remove_invalid_triples_idempotency(caplog: pytest.LogCaptureFixture) -> None:
    # Check that if the function is run multiple times, it does not remove more triples than necessary.
    g = Graph()
    identifier = URIRef("id1")
    g.add((identifier, DCTERMS.conformsTo, Literal("whatever")))
    g.add((identifier, RDF.type, RDFG.Graph))

    _remove_invalid_triples(g, predicates=RDF.type, obj=RDFG.Graph)
    _remove_invalid_triples(g, predicates=RDF.type, obj=RDFG.Graph)

    assert caplog.text.count("Invalid triple detected in header, removing:") == 1    
    assert len(g) == 1
    assert (identifier, RDF.type, RDFG.Graph) not in g
    assert (identifier, DCTERMS.conformsTo, Literal("whatever")) in g

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
        pytest.param([DCAT.keyword], Literal("old"), id="Predicate does not match, triple not changed"),
        pytest.param([DCAT.endDate], Literal("new"), id="DCAT.endDate, object new value, triple updated"),
        pytest.param([DCAT.startDate], Literal("new"), id="DCAT.startDate, object new value, triple updated"),
        pytest.param([DCTERMS.issued], Literal("new"), id="DCTERMS.issued, object new value, triple updated"),
        pytest.param([DCAT.endDate, DCAT.startDate], Literal("new"), id="Multiple predicates match, triples updated"),
        pytest.param([DCAT.startDate, DCAT.keyword], Literal("new"), id="Multiple predicates, only one match, one triple updated"),
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
        pytest.param(DCAT.keyword, id="DCAT.keyword"),
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


# Unit tests _check_dcterms_issued_count
# Function replaced by _correct_triple_representation_by_predicate
# @pytest.mark.parametrize(
#     "identifier, triples, expected_count",
#     [
#         pytest.param(URIRef("id1"), [], 0, id="No triples"),
#         pytest.param(URIRef("id1"), [(URIRef("id1"), URIRef("p"), Literal("o"))], 0, id="No dcterms:issued triple"),
#         pytest.param(URIRef("id1"), [(URIRef("s"), URIRef("p"), Literal("o"))], 0, id="No dcterms:issued triple, different identifier"),
#         pytest.param(URIRef("id1"), [(URIRef("id1"), DCTERMS.issued, Literal("2020-01-01"))], 1, id="Has issued triple, same identifier"),
#         pytest.param(URIRef("id1"), [(URIRef("s"), DCTERMS.issued, Literal("2020-01-01"))], 1, id="Has issued triple, different identifier"),
#         pytest.param(URIRef("id1"), [(URIRef("id1"), URIRef("p"), Literal("o")), (URIRef("id1"), DCTERMS.issued, Literal("2020-01-01"))], 1, id="Multiple triples, one dcterms:issued triple"),
#         pytest.param(URIRef("id1"), [(URIRef("s"), DCTERMS.issued, Literal("2020-01-01")), (URIRef("id1"), DCTERMS.issued, Literal("2020-01-01"))], 2, id="Multiple issued triples, different identifiers"),
#         pytest.param(URIRef("id1"), [(URIRef("id1"), DCTERMS.issued, Literal("2020-01-01")), (URIRef("id1"), DCTERMS.issued, Literal("2020-01-02"))], 2, id="Multiple issued triples, same identifiers, not duplicates"),
#         pytest.param(URIRef("id1"), [(URIRef("id1"), DCTERMS.issued, URIRef("2020-01-01"))], 1, id="Has issued triple, URIRef object"),
#         pytest.param(URIRef("id1"), [(BNode(), DCTERMS.issued, Literal("2020-01-01"))], 1, id="Has issued triple, blank node identifier")
#     ]
# )
# def test_check_dcterms_issued_count_various(identifier: URIRef, triples: list[tuple[Node, Node, Node]], expected_count: int, caplog: pytest.LogCaptureFixture) -> None:
#     g = Graph()
#     for s, p, o in triples:
#         g.add((s, p, o))

#     _check_dcterms_issued_count(g, identifier)

#     if expected_count == 0:
#         assert "Missing required dcterms:issued triple. Creating dummy triple without date." in caplog.text
#         assert (identifier, DCTERMS.issued, Literal("unknown")) in g
#         assert len(g) == len(triples) + 1
#     if expected_count == 1:
#         assert "Missing required dcterms:issued triple. Creating dummy triple without date." not in caplog.text
#         assert "All but one should be removed." not in caplog.text
#         assert (identifier, DCTERMS.issued, Literal("unknown")) not in g
#         assert len(g) == len(triples)
#     if expected_count > 1:
#         assert "All but one should be removed." in caplog.text
#         assert len(g) == len(triples) 
#         assert (identifier, DCTERMS.issued, Literal("unknown")) not in g


# def test_check_dcterms_issued_count_duplicates(caplog: pytest.LogCaptureFixture) -> None:
#     # Triple duplicates are removed by rdflib, so only one dcterms:issued triple remains.
#     g = Graph()
#     identifier = URIRef("id1")
#     g.add((identifier, DCTERMS.issued, Literal("2020-01-01")))
#     g.add((identifier, DCTERMS.issued, Literal("2020-01-01")))

#     _check_dcterms_issued_count(g, identifier)

#     assert "All but one should be removed." not in caplog.text
#     assert "Missing required dcterms:issued triple. Creating dummy triple without date." not in caplog.text
#     assert len(g) == 1


# def test_check_dcterms_issued_count_onlyonedummy(caplog: pytest.LogCaptureFixture) -> None:
#     # Check that a new dummy triple is not created if the function is run multiple times.
#     g = Graph()
#     identifier = URIRef("id1")
#     g.add((identifier, DCTERMS.conformsTo, Literal("whatever")))

#     _check_dcterms_issued_count(g, identifier)
#     _check_dcterms_issued_count(g, identifier)

#     assert "All but one should be removed." not in caplog.text
#     assert caplog.text.count("Creating dummy triple without date.") == 1
#     assert len(g) == 2
#     assert (identifier, DCTERMS.issued, Literal("unknown")) in g
#     assert (identifier, DCTERMS.conformsTo, Literal("whatever")) in g


# Unit tests _make_bnode_triple_for_given_predicate
def test_make_bnode_triple_for_given_predicate_emptygraph(caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    bnode = BNode()
    _make_bnode_triple_for_given_predicate(g, bnode, DCAT.endDate)

    assert len(g) == 1
    assert (bnode, DCAT.endDate, Literal("unknown")) in g
    assert f"Missing required {DCAT.endDate} triple. Creating dummy triple with no date." in caplog.text


@pytest.mark.parametrize(
    "triples, expected_triples, triples_found",
    [
        pytest.param([], [Literal("unknown")], 0, id="No triple found"),
        pytest.param([(URIRef("s"), Literal("2025-02-14"))], [Literal("2025-02-14")], 1, id="One triple found"),
        pytest.param([(URIRef("s"), Literal("2025-02-14")), (URIRef("s"), Literal("2025-02-31"))], [Literal("2025-02-14"), Literal("2025-02-31")], 2, id="Two triples found"),
        pytest.param([(URIRef("s1"), Literal("2025-02-14")), (URIRef("s2"), Literal("2025-02-31"))], [Literal("2025-02-14"), Literal("2025-02-31")], 2, id="Two triples found with different subjects"),    # The invalid date is kept as is
        pytest.param([(URIRef("s1"), Literal("2025-02-14")), (URIRef("s2"), Literal("2025-02-14"))], [Literal("2025-02-14")], 1, id="Two triples, different subjects, same object"),    # They are made duplicates which rdflib removes
        pytest.param([(BNode("s"), Literal("2025-02-14"))], [Literal("2025-02-14")], 1, id="BNode triple found"),
        pytest.param([(URIRef("s"), URIRef("2025-02-14"))], [URIRef("2025-02-14")], 1, id="One triple found with uri as object"),   # The type of object is irrelevant
    ]
)
def test_make_bnode_triple_for_given_predicate_various(triples: list[tuple[Node, Node]], expected_triples: list[Node], triples_found: int, caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    g.add((URIRef("s"), URIRef("p"), Literal("o")))
    for s, o in triples:
        g.add((s, DCAT.endDate, o))
    original_len = len(g)
    bnode = BNode()
    _make_bnode_triple_for_given_predicate(g, bnode, DCAT.endDate)

    assert (URIRef("s"), URIRef("p"), Literal("o")) in g
    for o in expected_triples:
        assert (bnode, DCAT.endDate, o) in g
    
    if triples_found == 0:
        assert f"Missing required {DCAT.endDate} triple. Creating dummy triple with no date." in caplog.text
        assert len(g) == original_len + 1  # Original triple + new triple
    elif triples_found >= 1:
        for s, o in triples:
            assert (s, DCAT.endDate, o) not in g
    
    if triples_found > 1:    
        assert f"Multiple {DCAT.endDate} triples. All but one should be removed." in caplog.text
    

def test_make_bnode_triple_for_given_predicate_idempotency(caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    g.add((URIRef("s"), URIRef("p"), Literal("o")))
    bnode = BNode()
    _make_bnode_triple_for_given_predicate(g, bnode, DCAT.endDate)
    _make_bnode_triple_for_given_predicate(g, bnode, DCAT.endDate)

    assert len(g) == 2
    assert (URIRef("s"), URIRef("p"), Literal("o")) in g
    assert (bnode, DCAT.endDate, Literal("unknown")) in g
    assert caplog.text.count(f"Missing required {DCAT.endDate} triple. Creating dummy triple with no date.") == 1


def test_make_bnode_triple_for_given_predicate_subjectissamebnode() -> None:
    # Same bnode already a subject of a triple with the predicate. Documenting that no infinite loop or error occurs.
    g = Graph()
    g.add((URIRef("s"), URIRef("p"), Literal("o")))
    bnode = BNode()
    g.add((bnode, DCAT.endDate, Literal("2025-02-14")))

    _make_bnode_triple_for_given_predicate(g, bnode, DCAT.endDate)

    assert len(g) == 2
    assert (URIRef("s"), URIRef("p"), Literal("o")) in g
    assert (bnode, DCAT.endDate, Literal("2025-02-14")) in g
    


# Unit tests _check_trig_rdfg_graph
@pytest.mark.parametrize(
    "identifier, triples, expected_count",
    [
        pytest.param(URIRef("id1"), [], 0, id="Empty graph"),
        pytest.param(URIRef("id1"), [(URIRef("s"), URIRef("p"), Literal("o"))], 0, id="Graph without rdfg:graph triple"),
        pytest.param(URIRef("id1"), [(URIRef("s"), RDF.type, RDFG.Graph)], 0, id="Graph with one rdfg:graph triple, wrong subject"),
        pytest.param(URIRef("id1"), [(URIRef("id1"), RDF.type, RDFG.Graph)], 1, id="Graph with one rdfg:graph triple, correct subject"),
        pytest.param(URIRef("id1"), [(URIRef("id1"), RDF.type, RDFG.Graph), (URIRef("s"), RDF.type, RDFG.Graph)], 1, id="Graph with multiple rdfg:graph triple, only one correct subject"),
        pytest.param(URIRef("id1"), [(URIRef("id1"), RDF.type, RDFG.Graph), (URIRef("id1"), RDF.type, DCAT.Dataset)], 1, id="Graph with multiple rdf:type triples. Only one rdfg:Graph."),
    ]
)
def test_check_trig_rdfg_graph_various(identifier: URIRef, triples: list[tuple[Node, Node, Node]], expected_count: int, caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    for s, p, o in triples:
        g.add((s, p, o))

    _check_trig_rdfg_graph(g, identifier)

    if expected_count == 0:
        assert "Missing required rdf:type rdfg:Graph triple for Trig header. Adding it." in caplog.text
        assert (identifier, RDF.type, RDFG.Graph) in g
        assert len(g) == len(triples) + 1
    if expected_count == 1:
        assert "Missing required rdf:type rdfg:Graph triple for Trig header. Adding it." not in caplog.text
        assert (identifier, RDF.type, RDFG.Graph) in g
        assert len(g) == len(triples)

def test_check_trig_rdfg_graph_duplicates(caplog: pytest.LogCaptureFixture) -> None:
    # Triple duplicates are removed by rdflib, so only one rdfg:Graph triple remains.
    g = Graph()
    identifier = URIRef("id1")
    g.add((identifier, RDF.type, RDFG.Graph))
    g.add((identifier, RDF.type, RDFG.Graph))

    _check_trig_rdfg_graph(g, identifier)

    assert "Missing required rdf:type rdfg:Graph triple for Trig header. Adding it." not in caplog.text
    assert len(g) == 1

def test_check_trig_rdfg_graph_onlyoneadded(caplog: pytest.LogCaptureFixture) -> None:
    # Check that a new rdfg:Graph triple is not created if the function is run multiple times.
    g = Graph()
    identifier = URIRef("id1")
    g.add((identifier, DCTERMS.conformsTo, Literal("whatever")))

    _check_trig_rdfg_graph(g, identifier)
    _check_trig_rdfg_graph(g, identifier)

    assert caplog.text.count("Missing required rdf:type rdfg:Graph triple for Trig header. Adding it.") == 1
    assert len(g) == 2
    assert (identifier, RDF.type, RDFG.Graph) in g
    assert (identifier, DCTERMS.conformsTo, Literal("whatever")) in g


def test_check_trig_rdfg_graph_identifierbnode(caplog: pytest.LogCaptureFixture) -> None:
    # Check that the function works even if identifier is a blank node. Edge case.
    g = Graph()
    identifier = BNode()
    g.add((identifier, DCTERMS.conformsTo, Literal("whatever")))

    # Silencing pylance to check wrong input datatype.
    _check_trig_rdfg_graph(g, identifier)   # type: ignore

    assert "Missing required rdf:type rdfg:Graph triple for Trig header. Adding it." in caplog.text
    assert len(g) == 2
    assert (identifier, RDF.type, RDFG.Graph) in g
    assert (identifier, DCTERMS.conformsTo, Literal("whatever")) in g


# Unit tests _remove_cimxml_rdfg_graph
@pytest.mark.parametrize(
    "triples, expected_count",
    [
        pytest.param([], 0, id="Empty graph"),
        pytest.param([(URIRef("s"), URIRef("p"), Literal("o"))], 0, id="Graph without rdfg:graph triple"),
        pytest.param([(URIRef("s"), RDF.type, RDFG.Graph)], 1, id="Graph with one rdfg:graph triple"),
        pytest.param([(URIRef("s1"), RDF.type, RDFG.Graph), (URIRef("s2"), RDF.type, RDFG.Graph)], 2, id="Graph with two rdfg:graph triple"),
        pytest.param([(URIRef("id1"), RDF.type, RDFG.Graph), (URIRef("id1"), RDF.type, DCAT.Dataset)], 1, id="Graph with multiple rdf:type triples. Only one rdfg:Graph."),
    ]
)
def test_remove_cimxml_rdfg_graph_various(triples: list[tuple[Node, Node, Node]], expected_count: int, caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    for s, p, o in triples:
        g.add((s, p, o))

    _remove_cimxml_rdfg_graph(g)

    if expected_count == 0:
        assert len(g) == len(triples)
        assert "Invalid rdf:type rdfg:Graph triple detected in CIMXML header, removing it." not in caplog.text
        for triple in triples:  # Check that other triples are not removed.
            assert triple in g
    elif expected_count > 0:
        assert len(g) == len(triples) - expected_count
        assert caplog.text.count("Invalid rdf:type rdfg:Graph triple detected in CIMXML header, removing it.") == 1


def test_remove_cimxml_rdfg_graph_idempotency(caplog: pytest.LogCaptureFixture) -> None:
    # Check that if the function is run multiple times, it does not remove more triples than necessary.
    g = Graph()
    identifier = URIRef("id1")
    g.add((identifier, DCTERMS.conformsTo, Literal("whatever")))
    g.add((identifier, RDF.type, RDFG.Graph))

    _remove_cimxml_rdfg_graph(g)
    _remove_cimxml_rdfg_graph(g)

    assert caplog.text.count("Invalid rdf:type rdfg:Graph triple detected in CIMXML header, removing it.") == 1    
    assert len(g) == 1
    assert (identifier, RDF.type, RDFG.Graph) not in g
    assert (identifier, DCTERMS.conformsTo, Literal("whatever")) in g


# Unit tests _fix_cimxml_period_of_time_format
@pytest.mark.parametrize(
    "occurences",
    [
        pytest.param(0, id="No periodOfTime triple"),
        pytest.param(1, id="One periodOfTime triple"),
        pytest.param(2, id="Two periodOfTime triples"),
    ]
)
def test_fix_cimxml_period_of_time_format_periodoftime(occurences: int) -> None:
    g = Graph()
    id = URIRef("id1")
    g.add((id, DCTERMS.conformsTo, Literal("whatever")))
    g.add((id, DCAT.endDate, Literal("2025-02-14T00:00:00+00:00")))
    g.add((id, DCAT.startDate, Literal("2025-02-01T00:00:00+00:00")))
    for i in range(occurences):
        g.add((URIRef(f"s{i}"), RDF.type, DCTERMS.PeriodOfTime))

    _fix_cimxml_period_of_time_format(g, id)

    assert (id, DCTERMS.conformsTo, Literal("whatever")) in g
    assert (id, DCAT.endDate, Literal("2025-02-14T00:00:00+00:00")) in g
    assert (id, DCAT.startDate, Literal("2025-02-01T00:00:00+00:00")) in g
    assert len(g) == 3
    for i in range(occurences):
        assert (URIRef(f"s{i}"), RDF.type, DCTERMS.PeriodOfTime) not in g


@patch("cim_plugin.header_validation._correct_triple_representation_by_predicate")
def test_fix_cimxml_period_of_time_format_emptygraph(mock_correct: MagicMock) -> None:
    g = Graph()
    id = URIRef("id1")

    _fix_cimxml_period_of_time_format(g, id)

    mock_correct.assert_has_calls([call(g, DCAT.endDate, id), call(g, DCAT.startDate, id)])
    assert len(g) == 0


@patch("cim_plugin.header_validation._correct_triple_representation_by_predicate")
def test_fix_cimxml_period_of_time_format_blanknodes(mock_correct: MagicMock) -> None:
    g = Graph()
    id = BNode("id1")
    other_bnode = BNode("other")
    g.add((id, RDF.type, DCTERMS.PeriodOfTime))
    g.add((other_bnode, RDF.type, DCAT.Dataset))

    # Pylance silenced to test wrong input datatype.
    _fix_cimxml_period_of_time_format(g, id)    # type: ignore

    mock_correct.assert_has_calls([call(g, DCAT.endDate, id), call(g, DCAT.startDate, id)])
    assert len(g) == 1
    assert (other_bnode, RDF.type, DCAT.Dataset) in g
    assert (id, RDF.type, DCTERMS.PeriodOfTime) not in g

# Unit tests _fix_trig_period_of_time_format
@patch("cim_plugin.header_validation._make_bnode_triple_for_given_predicate")
def test__fix_trig_period_of_time_format_emptygraph(mock_make_bnode: MagicMock) -> None:
    g = Graph()
    id = URIRef("id1")

    _fix_trig_period_of_time_format(g, id)

    assert len(g) == 2

    temporal_triple = list(g.triples((id, DCTERMS.temporal, None)))
    assert len(temporal_triple) == 1
    assert temporal_triple[0] in g
    bnode = temporal_triple[0][2]
    assert (bnode, RDF.type, DCTERMS.PeriodOfTime) in g
    mock_make_bnode.assert_has_calls([call(g, bnode, DCAT.endDate), call(g, bnode, DCAT.startDate)], any_order=True)

@pytest.mark.parametrize(
    "occurences",
    [
        pytest.param(0, id="No temporal triple"),
        pytest.param(1, id="One temporal triple"),
        pytest.param(2, id="Two temporal triples"),
    ]
)
@patch("cim_plugin.header_validation._make_bnode_triple_for_given_predicate")
def test__fix_trig_period_of_time_format_temporalremoval(mock_make_bnode: MagicMock, occurences: int) -> None:
    g = Graph()
    id = URIRef("id1")
    for i in range(occurences):
        g.add((URIRef(f"s{i}"), DCTERMS.temporal, URIRef(f"o{i}")))

    _fix_trig_period_of_time_format(g, id)

    assert len(g) == 2

    temporal_triple = list(g.triples((id, DCTERMS.temporal, None)))
    assert len(temporal_triple) == 1
    assert temporal_triple[0] in g
    bnode = temporal_triple[0][2]
    assert (bnode, RDF.type, DCTERMS.PeriodOfTime) in g
    mock_make_bnode.assert_has_calls([call(g, bnode, DCAT.endDate), call(g, bnode, DCAT.startDate)], any_order=True)
    for i in range(occurences):
        assert (URIRef(f"s{i}"), DCTERMS.temporal, URIRef(f"o{i}")) not in g


@pytest.mark.parametrize(
    "occurences",
    [
        pytest.param(0, id="No temporal triple"),
        pytest.param(1, id="One temporal triple"),
        pytest.param(2, id="Two temporal triples"),
    ]
)
@patch("cim_plugin.header_validation._make_bnode_triple_for_given_predicate")
def test__fix_trig_period_of_time_format_periodoftimeremoval(mock_make_bnode: MagicMock, occurences: int) -> None:
    g = Graph()
    id = URIRef("id1")
    for i in range(occurences):
        g.add((URIRef(f"s{i}"), RDF.type, DCTERMS.PeriodOfTime))

    _fix_trig_period_of_time_format(g, id)

    assert len(g) == 2

    temporal_triple = list(g.triples((id, DCTERMS.temporal, None)))
    assert len(temporal_triple) == 1
    assert temporal_triple[0] in g
    bnode = temporal_triple[0][2]
    assert (bnode, RDF.type, DCTERMS.PeriodOfTime) in g
    mock_make_bnode.assert_has_calls([call(g, bnode, DCAT.endDate), call(g, bnode, DCAT.startDate)], any_order=True)
    for i in range(occurences):
        assert (URIRef(f"s{i}"), RDF.type, DCTERMS.PeriodOfTime) not in g


def test__fix_trig_period_of_time_format_temporalgroupexistswrongid() -> None:
    # Documents that if a temporal node group exists under wrong identifier, it is removed and the information moved to a new temporal node group with a new blank node.
    # This is to ensure that the temporal node group contains all the required information under the correct identifier.
    g = Graph()
    id = URIRef("id2")
    first_bnode = BNode("first")
    g.add((URIRef("id1"), DCTERMS.temporal, first_bnode))
    g.add((first_bnode, RDF.type, DCTERMS.PeriodOfTime))
    g.add((first_bnode, DCAT.endDate, Literal("2025-02-14T00:00:00+00:00")))
    g.add((first_bnode, DCAT.startDate, Literal("2025-02-01T00:00:00+00:00")))
    g.add((URIRef("id1"), RDF.type, DCAT.Dataset))  # Triple that should not be removed.

    _fix_trig_period_of_time_format(g, id)

    assert len(g) == 5
    assert (URIRef("id1"), RDF.type, DCAT.Dataset) in g # Other triples are unchanged
    temporal_triple = list(g.triples((id, DCTERMS.temporal, None)))
    assert temporal_triple[0] in g
    bnode = temporal_triple[0][2]
    assert (bnode, RDF.type, DCTERMS.PeriodOfTime) in g
    assert (bnode, DCAT.endDate, Literal("2025-02-14T00:00:00+00:00")) in g
    assert (bnode, DCAT.startDate, Literal("2025-02-01T00:00:00+00:00")) in g
    assert (URIRef("id1"), DCTERMS.temporal, first_bnode) not in g
    assert (first_bnode, RDF.type, DCTERMS.PeriodOfTime) not in g
    assert (first_bnode, DCAT.endDate, Literal("2025-02-14T00:00:00+00:00")) not in g
    assert (first_bnode, DCAT.startDate, Literal("2025-02-01T00:00:00+00:00")) not in g


def test__fix_trig_period_of_time_format_temporalgroupexistscorrectid() -> None:
    # Documents that even if a temporal node group exists under correct identifier, it is still removed and the information moved to a new temporal node group with a new blank node. 
    # This is to ensure that the temporal node group contains all the required information under the correct identifier, and that any incorrect formatting is fixed.
    g = Graph()
    id = URIRef("id1")
    first_bnode = BNode("first")
    g.add((URIRef("id1"), DCTERMS.temporal, first_bnode))
    g.add((first_bnode, RDF.type, DCTERMS.PeriodOfTime))
    g.add((first_bnode, DCAT.endDate, Literal("2025-02-14T00:00:00+00:00")))
    g.add((first_bnode, DCAT.startDate, Literal("2025-02-01T00:00:00+00:00")))
    g.add((URIRef("id1"), RDF.type, DCAT.Dataset))  # Triple that should not be removed.

    _fix_trig_period_of_time_format(g, id)

    assert len(g) == 5
    assert (URIRef("id1"), RDF.type, DCAT.Dataset) in g # Other triples are unchanged
    temporal_triple = list(g.triples((id, DCTERMS.temporal, None)))
    assert temporal_triple[0] in g
    bnode = temporal_triple[0][2]
    assert (bnode, RDF.type, DCTERMS.PeriodOfTime) in g
    assert (bnode, DCAT.endDate, Literal("2025-02-14T00:00:00+00:00")) in g
    assert (bnode, DCAT.startDate, Literal("2025-02-01T00:00:00+00:00")) in g
    assert (URIRef("id1"), DCTERMS.temporal, first_bnode) not in g
    assert (first_bnode, RDF.type, DCTERMS.PeriodOfTime) not in g
    assert (first_bnode, DCAT.endDate, Literal("2025-02-14T00:00:00+00:00")) not in g
    assert (first_bnode, DCAT.startDate, Literal("2025-02-01T00:00:00+00:00")) not in g


def test__fix_trig_period_of_time_format_multipletemporalgroupexists() -> None:
    # Documents that when multiple temporal node groups exist, the information from all of them are moved to a new temporal node group with a new blank node. 
    g = Graph()
    id = URIRef("id1")
    first_bnode = BNode("first")
    second_bnode = BNode("second")
    g.add((URIRef("id1"), DCTERMS.temporal, first_bnode))
    g.add((URIRef("id1"), DCTERMS.temporal, second_bnode))
    g.add((first_bnode, RDF.type, DCTERMS.PeriodOfTime))
    g.add((first_bnode, DCAT.endDate, Literal("2025-02-14T00:00:00+00:00")))
    g.add((first_bnode, DCAT.startDate, Literal("2025-02-01T00:00:00+00:00")))
    g.add((URIRef("id1"), RDF.type, DCAT.Dataset))  # Triple that should not be removed.
    g.add((second_bnode, RDF.type, DCTERMS.PeriodOfTime))
    g.add((second_bnode, DCAT.endDate, Literal("2025-03-14")))

    _fix_trig_period_of_time_format(g, id)

    assert len(g) == 6
    assert (URIRef("id1"), RDF.type, DCAT.Dataset) in g # Other triples are unchanged
    temporal_triple = list(g.triples((id, DCTERMS.temporal, None)))
    assert temporal_triple[0] in g
    bnode = temporal_triple[0][2]
    # The triples that should remain
    assert (bnode, RDF.type, DCTERMS.PeriodOfTime) in g
    assert (bnode, DCAT.endDate, Literal("2025-02-14T00:00:00+00:00")) in g
    assert (bnode, DCAT.endDate, Literal("2025-03-14")) in g
    assert (bnode, DCAT.startDate, Literal("2025-02-01T00:00:00+00:00")) in g
    # The triples that are removed
    assert (URIRef("id1"), DCTERMS.temporal, first_bnode) not in g
    assert (first_bnode, RDF.type, DCTERMS.PeriodOfTime) not in g
    assert (first_bnode, DCAT.endDate, Literal("2025-02-14T00:00:00+00:00")) not in g
    assert (first_bnode, DCAT.startDate, Literal("2025-02-01T00:00:00+00:00")) not in g
    assert (URIRef("id1"), DCTERMS.temporal, second_bnode) not in g
    assert (second_bnode, RDF.type, DCTERMS.PeriodOfTime) not in g
    assert (second_bnode, DCAT.endDate, Literal("2025-03-14")) not in g
    

# Unit tests _correct_triple_representation_by_predicate
@pytest.mark.parametrize(
    "identifier, triples, expected_count",
    [
        pytest.param(URIRef("id1"), [], 0, id="No triples"),
        pytest.param(URIRef("id1"), [(URIRef("id1"), URIRef("p"), Literal("o"))], 0, id="No dcterms:issued triple"),
        pytest.param(URIRef("id1"), [(URIRef("s"), URIRef("p"), Literal("o"))], 0, id="No dcterms:issued triple, different identifier"),
        pytest.param(URIRef("id1"), [(URIRef("id1"), DCTERMS.issued, Literal("2020-01-01"))], 1, id="Has issued triple, same identifier"),
        pytest.param(URIRef("id1"), [(URIRef("s"), DCTERMS.issued, Literal("2020-01-01"))], 1, id="Has issued triple, different identifier"),
        pytest.param(URIRef("id1"), [(URIRef("id1"), URIRef("p"), Literal("o")), (URIRef("id1"), DCTERMS.issued, Literal("2020-01-01"))], 1, id="Multiple triples, one dcterms:issued triple"),
        pytest.param(URIRef("id1"), [(URIRef("s"), DCTERMS.issued, Literal("2020-01-01")), (URIRef("id1"), DCTERMS.issued, Literal("2020-01-01"))], 2, id="Multiple issued triples, different identifiers"),
        pytest.param(URIRef("id1"), [(URIRef("id1"), DCTERMS.issued, Literal("2020-01-01")), (URIRef("id1"), DCTERMS.issued, Literal("2020-01-02"))], 2, id="Multiple issued triples, same identifiers, not duplicates"),
        pytest.param(URIRef("id1"), [(URIRef("id1"), DCTERMS.issued, URIRef("2020-01-01"))], 1, id="Has issued triple, URIRef object"),
        pytest.param(URIRef("id1"), [(BNode(), DCTERMS.issued, Literal("2020-01-01"))], 1, id="Has issued triple, blank node identifier")
    ]
)
def test_correct_triple_representation_by_predicate_various(identifier: URIRef, triples: list[tuple[Node, Node, Node]], expected_count: int, caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    for s, p, o in triples:
        g.add((s, p, o))

    _correct_triple_representation_by_predicate(g, DCTERMS.issued, identifier)

    if expected_count == 0:
        assert "Missing required http://purl.org/dc/terms/issued triple. Creating dummy triple without date." in caplog.text
        assert (identifier, DCTERMS.issued, Literal("unknown")) in g
        assert len(g) == len(triples) + 1
    if expected_count == 1:
        assert "Missing required http://purl.org/dc/terms/issued triple. Creating dummy triple without date." not in caplog.text
        assert "All but one should be removed." not in caplog.text
        assert (identifier, DCTERMS.issued, Literal("unknown")) not in g
        assert len(g) == len(triples)
    if expected_count > 1:
        assert "All but one should be removed." in caplog.text
        assert len(g) == len(triples) 
        assert (identifier, DCTERMS.issued, Literal("unknown")) not in g


def test_correct_triple_representation_by_predicate_duplicates(caplog: pytest.LogCaptureFixture) -> None:
    # Triple duplicates are removed by rdflib, so only one dcterms:issued triple remains.
    g = Graph()
    identifier = URIRef("id1")
    g.add((identifier, DCAT.endDate, Literal("2020-01-01")))
    g.add((identifier, DCAT.endDate, Literal("2020-01-01")))

    _correct_triple_representation_by_predicate(g, DCAT.endDate, identifier)

    assert "All but one should be removed." not in caplog.text
    assert f"Missing required {DCAT.endDate} triple. Creating dummy triple without date." not in caplog.text
    assert len(g) == 1


def test_correct_triple_representation_by_predicate_onlyonedummy(caplog: pytest.LogCaptureFixture) -> None:
    # Check that a new dummy triple is not created if the function is run multiple times.
    g = Graph()
    identifier = URIRef("id1")
    g.add((identifier, DCTERMS.conformsTo, Literal("whatever")))

    _correct_triple_representation_by_predicate(g, DCAT.startDate, identifier)
    _correct_triple_representation_by_predicate(g, DCAT.startDate, identifier)

    assert "All but one should be removed." not in caplog.text
    assert caplog.text.count("Creating dummy triple without date.") == 1
    assert len(g) == 2
    assert (identifier, DCAT.startDate, Literal("unknown")) in g
    assert (identifier, DCTERMS.conformsTo, Literal("whatever")) in g


# Unit tests validate_header
@pytest.mark.parametrize(
    "format",
    [
        pytest.param("trig", id="Trig format"),
        pytest.param("cimxml", id="CIMXML format"),
        pytest.param("jsonld", id="JSON-LD format"),
        pytest.param("unknown", id="Unknown format"),
        pytest.param("", id="Empty format"),
        pytest.param(None, id="None format"),
        pytest.param("TRIG", id="Uppercase format"),
        pytest.param("CiMxMl", id="Mixed case format"),
        pytest.param(" trig ", id="Format with whitespace"),
    ]
)
@patch("cim_plugin.header_validation._fix_trig_period_of_time_format")
@patch("cim_plugin.header_validation._check_trig_rdfg_graph")
@patch("cim_plugin.header_validation._remove_cimxml_rdfg_graph")
@patch("cim_plugin.header_validation._fix_cimxml_period_of_time_format")
@patch("cim_plugin.header_validation._correct_triple_representation_by_predicate")
@patch("cim_plugin.header_validation._fix_datetime_format_in_triples")
@patch("cim_plugin.header_validation._remove_invalid_triples")
def test_validate_header_calls(mock_remove: MagicMock, mock_fix_datetime: MagicMock, mock_correct: MagicMock, mock_fix_period_xml: MagicMock, mock_remove_rdfgraph: MagicMock, mock_check_rdfgraph: MagicMock, mock_fix_period_trig: MagicMock, format: str, caplog: pytest.LogCaptureFixture) -> None:
    header = CIMMetadataHeader.empty(URIRef("id1"))
    header.add_triple(RDF.type, DCAT.Dataset)
    validate_header(header=header, format=format)

    mock_remove.assert_called_once_with(header.graph, predicates=[DCAT.distribution, JSONLD.base], obj=DCAT.Distribution)
    mock_fix_datetime.assert_called_once_with(header.graph)
    mock_correct.assert_called_once_with(header.graph, DCTERMS.issued, URIRef("id1"))

    format = format.lower().strip() if format is not None else "cimxml"
    if format == "cimxml":
        mock_fix_period_xml.assert_called_once_with(header.graph, URIRef("id1"))
        mock_remove_rdfgraph.assert_called_once_with(header.graph)
        mock_check_rdfgraph.assert_not_called()
        mock_fix_period_trig.assert_not_called()
    elif format == "trig":
        mock_fix_period_trig.assert_called_once_with(header.graph, URIRef("id1"))        
        mock_remove_rdfgraph.assert_not_called()
        mock_check_rdfgraph.assert_called_once_with(header.graph, URIRef("id1"))
        mock_fix_period_xml.assert_not_called()
    elif format == "jsonld":
        assert "Header validation for JSON-LD format is not implemented yet." in caplog.text
    else:
        assert f"Unknown format specified for header validation: {format}. No validation performed." in caplog.text

@patch("cim_plugin.header_validation._fix_trig_period_of_time_format")
@patch("cim_plugin.header_validation._check_trig_rdfg_graph")
@patch("cim_plugin.header_validation._remove_cimxml_rdfg_graph")
@patch("cim_plugin.header_validation._fix_cimxml_period_of_time_format")
@patch("cim_plugin.header_validation._correct_triple_representation_by_predicate")
@patch("cim_plugin.header_validation._fix_datetime_format_in_triples")
@patch("cim_plugin.header_validation._remove_invalid_triples")
def test_validate_header_nordftype(mock_remove: MagicMock, mock_fix_datetime: MagicMock, mock_correct: MagicMock, mock_fix_period_cimxml: MagicMock, mock_remove_rdfgraph: MagicMock, mock_check_rdfgraph: MagicMock, mock_fix_period_trig: MagicMock) -> None:
    # If there are no rdf.type triples the header is invalid and ValueError is raised.
    header = CIMMetadataHeader.empty(URIRef("id1"))
    header.add_triple(DCAT.distribution, URIRef("dist1"))
    
    with pytest.raises(ValueError, match="No header type found in header."):
        validate_header(header=header, format="cimxml")

    mock_remove.assert_not_called()
    mock_fix_datetime.assert_not_called()
    mock_correct.assert_not_called()
    mock_fix_period_cimxml.assert_not_called()
    mock_remove_rdfgraph.assert_not_called()
    mock_check_rdfgraph.assert_not_called()
    mock_fix_period_trig.assert_not_called()

@patch("cim_plugin.header_validation._check_trig_rdfg_graph")
@patch("cim_plugin.header_validation._remove_cimxml_rdfg_graph")
@patch("cim_plugin.header_validation._fix_cimxml_period_of_time_format")
@patch("cim_plugin.header_validation._fix_trig_period_of_time_format")
@patch("cim_plugin.header_validation._correct_triple_representation_by_predicate")
@patch("cim_plugin.header_validation._fix_datetime_format_in_triples")
@patch("cim_plugin.header_validation._remove_invalid_triples")
def test_validate_header_fullmodelheader(mock_remove: MagicMock, mock_fix_datetime: MagicMock, mock_correct: MagicMock, mock_fix_period_cimxml: MagicMock, mock_fix_period_trig: MagicMock, mock_remove_rdfgraph: MagicMock, mock_check_rdfgraph: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    header = CIMMetadataHeader.empty(URIRef("id1"))
    header.add_triple(RDF.type, MD.FullModel)
    validate_header(header=header, format="cimxml")

    mock_remove.assert_not_called()
    mock_fix_datetime.assert_not_called()
    mock_correct.assert_not_called()
    mock_fix_period_cimxml.assert_not_called()
    mock_fix_period_trig.assert_not_called()
    mock_remove_rdfgraph.assert_not_called()
    mock_check_rdfgraph.assert_not_called()
    assert "Validation for MD.FullModel header is not implemented yet. No validation performed for this header type." in caplog.text


@patch("cim_plugin.header_validation._check_trig_rdfg_graph")
@patch("cim_plugin.header_validation._remove_cimxml_rdfg_graph")
@patch("cim_plugin.header_validation._fix_cimxml_period_of_time_format")
@patch("cim_plugin.header_validation._fix_trig_period_of_time_format")
@patch("cim_plugin.header_validation._correct_triple_representation_by_predicate")
@patch("cim_plugin.header_validation._fix_datetime_format_in_triples")
@patch("cim_plugin.header_validation._remove_invalid_triples")
def test_validate_header_multiplerdftypes(mock_remove: MagicMock, mock_fix_datetime: MagicMock, mock_correct: MagicMock, mock_fix_period_cimxml: MagicMock, mock_fix_period_trig: MagicMock, mock_remove_rdfgraph: MagicMock, mock_check_rdfgraph: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    # Header should never have two header_type triples.
    header = CIMMetadataHeader.empty(URIRef("id1"))
    header.add_triple(RDF.type, MD.FullModel)
    header.add_triple(RDF.type, DCAT.Dataset)
    with pytest.raises(ValueError, match="Multiple header types found in header."):
        validate_header(header=header, format="cimxml")

    mock_remove.assert_not_called()
    mock_fix_datetime.assert_not_called()
    mock_correct.assert_not_called()
    mock_fix_period_cimxml.assert_not_called()
    mock_fix_period_trig.assert_not_called()
    mock_remove_rdfgraph.assert_not_called()
    mock_check_rdfgraph.assert_not_called()


@patch("cim_plugin.header_validation._check_trig_rdfg_graph")
@patch("cim_plugin.header_validation._remove_cimxml_rdfg_graph")
@patch("cim_plugin.header_validation._fix_cimxml_period_of_time_format")
@patch("cim_plugin.header_validation._fix_trig_period_of_time_format")
@patch("cim_plugin.header_validation._correct_triple_representation_by_predicate")
@patch("cim_plugin.header_validation._fix_datetime_format_in_triples")
@patch("cim_plugin.header_validation._remove_invalid_triples")
def test_validate_header_unknownheader(mock_remove: MagicMock, mock_fix_datetime: MagicMock, mock_correct: MagicMock, mock_fix_period_cimxml: MagicMock, mock_fix_period_trig: MagicMock, mock_remove_rdfgraph: MagicMock, mock_check_rdfgraph: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    header = CIMMetadataHeader.empty(URIRef("id1"), metadata_objects=[URIRef("www.custom.org/header")])
    header.add_triple(RDF.type, URIRef("www.custom.org/header"))
    validate_header(header=header, format="cimxml")

    mock_remove.assert_not_called()
    mock_fix_datetime.assert_not_called()
    mock_correct.assert_not_called()
    mock_fix_period_cimxml.assert_not_called()
    mock_fix_period_trig.assert_not_called()
    mock_remove_rdfgraph.assert_not_called()
    mock_check_rdfgraph.assert_not_called()
    assert "Unknown header type: www.custom.org/header. No validation performed." in caplog.text


@pytest.mark.parametrize("header", [None, CIMMetadataHeader.empty(URIRef("id1"))])
@patch("cim_plugin.header_validation._check_trig_rdfg_graph")
@patch("cim_plugin.header_validation._remove_cimxml_rdfg_graph")
@patch("cim_plugin.header_validation._fix_cimxml_period_of_time_format")
@patch("cim_plugin.header_validation._fix_trig_period_of_time_format")
@patch("cim_plugin.header_validation._correct_triple_representation_by_predicate")
@patch("cim_plugin.header_validation._fix_datetime_format_in_triples")
@patch("cim_plugin.header_validation._remove_invalid_triples")
def test_validate_header_emptyheader(mock_remove: MagicMock, mock_fix_datetime: MagicMock, mock_correct: MagicMock, mock_fix_period_cimxml: MagicMock, mock_fix_period_trig: MagicMock, mock_remove_rdfgraph: MagicMock, mock_check_rdfgraph: MagicMock, header: Any, caplog: pytest.LogCaptureFixture) -> None:
    header = header
    validate_header(header=header, format="cimxml")

    mock_remove.assert_not_called()
    mock_fix_datetime.assert_not_called()
    mock_correct.assert_not_called()
    mock_fix_period_cimxml.assert_not_called()
    mock_fix_period_trig.assert_not_called()
    mock_remove_rdfgraph.assert_not_called()
    mock_check_rdfgraph.assert_not_called()
    assert "Header graph is empty. No validation performed." in caplog.text

# Integration tests validate_header
def test_validate_header_integration_cimxml(caplog: pytest.LogCaptureFixture) -> None:
    id = URIRef("id1")
    header = CIMMetadataHeader.empty(id)
    header.add_triple(RDF.type, DCAT.Dataset)
    header.add_triple(RDF.type, DCAT.Distribution)
    header.add_triple(DCAT.distribution, URIRef("dist1"))
    header.add_triple(JSONLD.base, URIRef("base1"))
    header.add_triple(DCAT.endDate, Literal("2025-02-14 00:00:00Z"))
    header.add_triple(DCTERMS.issued, Literal("2020-01-01"))
    header.add_triple(RDF.type, RDFG.Graph)
    header.add_triple(RDF.type, DCTERMS.PeriodOfTime)

    validate_header(header=header, format="cimxml")

    expected_triples = {(id, RDF.type, DCAT.Dataset), 
                        (id, DCTERMS.issued, Literal("2020-01-01T00:00:00+00:00", datatype=XSD.dateTime)), 
                        (id, DCAT.endDate, Literal("2025-02-14T00:00:00Z", datatype=XSD.dateTime)), 
                        (id, DCAT.startDate, Literal("unknown"))}
    
    assert set(header.graph) == expected_triples
    assert f"Invalid triple detected in header, removing: ({id}, {RDF.type}, {DCAT.Distribution})" in caplog.text
    assert f"Invalid triple detected in header, removing: ({id}, {JSONLD.base}, {URIRef('base1')})" in caplog.text
    assert f"Invalid triple detected in header, removing: ({id}, {DCAT.distribution}, {URIRef('dist1')})" in caplog.text
    assert "Invalid rdf:type rdfg:Graph triple detected in CIMXML header, removing it." in caplog.text
    assert f"Corrected date format for predicate {DCTERMS.issued}: from {Literal('2020-01-01')} to {Literal('2020-01-01T00:00:00+00:00', datatype=XSD.dateTime)}." in caplog.text
    assert f"Missing required {DCAT.startDate} triple. Creating dummy triple without date." in caplog.text


def test_validate_header_integration_trig(caplog: pytest.LogCaptureFixture) -> None:
    id = URIRef("id1")
    header = CIMMetadataHeader.empty(id)
    header.add_triple(RDF.type, DCAT.Dataset)
    header.add_triple(RDF.type, DCAT.Distribution)
    header.add_triple(DCAT.distribution, URIRef("dist1"))
    header.add_triple(JSONLD.base, URIRef("base1"))
    header.add_triple(DCAT.endDate, Literal("2025-02-14 00:00:00Z"))
    header.add_triple(DCTERMS.issued, Literal("2020-01-01"))
    header.add_triple(RDF.type, DCTERMS.PeriodOfTime)

    validate_header(header=header, format="trig")

    expected_triples = {(id, RDF.type, DCAT.Dataset), 
                        (id, DCTERMS.issued, Literal("2020-01-01T00:00:00+00:00", datatype=XSD.dateTime)),
                        (id, RDF.type, RDFG.Graph)}

    # Expected triples present
    for triples in expected_triples:
        assert triples in header.graph

    # Expected blank node triples present
    temporal_bnodes = list(header.graph.objects(id, DCTERMS.temporal))
    assert len(temporal_bnodes) == 1
    bnode = temporal_bnodes[0]
    assert (bnode, RDF.type, DCTERMS.PeriodOfTime) in header.graph
    assert (bnode, DCAT.endDate, Literal("2025-02-14T00:00:00Z", datatype=XSD.dateTime)) in header.graph
    assert (bnode, DCAT.startDate, Literal("unknown")) in header.graph

    # Log messages
    assert f"Invalid triple detected in header, removing: ({id}, {RDF.type}, {DCAT.Distribution})" in caplog.text
    assert f"Invalid triple detected in header, removing: ({id}, {JSONLD.base}, {URIRef('base1')})" in caplog.text
    assert f"Invalid triple detected in header, removing: ({id}, {DCAT.distribution}, {URIRef('dist1')})" in caplog.text
    assert f"Corrected date format for predicate {DCTERMS.issued}: from {Literal('2020-01-01')} to {Literal('2020-01-01T00:00:00+00:00', datatype=XSD.dateTime)}." in caplog.text
    assert f"Missing required {DCAT.startDate} triple. Creating dummy triple with no date." in caplog.text
    assert "Missing required rdf:type rdfg:Graph triple for Trig header. Adding it." in caplog.text

    

if __name__ == "__main__":
    pytest.main()