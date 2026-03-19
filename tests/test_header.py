from typing import Iterable, Callable, Any
import pytest
from unittest.mock import patch, MagicMock
from rdflib import Graph, URIRef, Literal, BNode, Node
from rdflib.namespace import DCAT, DCTERMS, RDF
from cim_plugin.namespaces import MD
from cim_plugin.header import CIMMetadataHeader, create_header_attribute
from tests.fixtures import build_graph_with_blank_header, make_graph, fake_parse_factory
import logging

logger = logging.getLogger('cimxml_logger')


# Unit tests .__init__
def test_init_generatesuuidsubject() -> None:
    h = CIMMetadataHeader()
    assert isinstance(h.subject, URIRef)
    assert str(h.subject).startswith("urn:uuid:")


def test_init_providedsubject() -> None:
    s = URIRef("urn:test")
    h = CIMMetadataHeader(subject=s)
    assert h.subject == s


def test_init_graphnone() -> None:
    h = CIMMetadataHeader(graph=None)
    assert len(h.graph) == 0    # An empty graph is made


def test_init_graphinput() -> None:
    g = Graph()
    g.add((URIRef("a"), URIRef("b"), Literal("c")))
    h = CIMMetadataHeader(graph=g)
    assert h.graph == g


def test_init_metadataobjectsdefaults() -> None:
    h = CIMMetadataHeader(metadata_objects=None)
    assert h.metadata_objects == CIMMetadataHeader.DEFAULT_METADATA_OBJECTS
    assert h.metadata_objects == {MD.FullModel, DCAT.Dataset}


def test_init_metadataobjectsoverride() -> None:
    objs = {URIRef("urn:meta")}
    h = CIMMetadataHeader(metadata_objects=objs)
    assert h.metadata_objects == objs
    

def test_init_reachablenodesdefault() -> None:
    h = CIMMetadataHeader()
    assert h.reachable_nodes == set()


def test_init_reachablenodesoverride() -> None:
    nodes: set[Node] = {URIRef("urn:x")}
    h = CIMMetadataHeader(reachable_nodes=nodes)
    assert h.reachable_nodes is nodes


def test_init_profilepredicatesdefault() -> None:
    h = CIMMetadataHeader()
    assert h.profile_predicates == CIMMetadataHeader.DEFAULT_PROFILE_PREDICATES


def test_init_profilepredicatesoverride() -> None:
    preds = {URIRef("urn:test:profile")}
    h = CIMMetadataHeader(profile_predicates=preds)
    assert h.profile_predicates is preds


@patch.object(CIMMetadataHeader, "collect_profile")
def test_init_profilenone(mock_collect: MagicMock) -> None:
    mock_collect.return_value = "profile"
    h = CIMMetadataHeader(profile=None)
    assert h.profile == "profile"
    mock_collect.assert_called_once()


def test_init_profileoverride() -> None:
    h = CIMMetadataHeader(profile="manual")
    assert h.profile == "manual"


@patch.object(CIMMetadataHeader, "collect_profile")
def test_init_profilepredicatesusedincollect(mock_collect: MagicMock) -> None:
    preds = {URIRef("urn:test:profile")}
    h = CIMMetadataHeader(profile=None, profile_predicates=preds)
    mock_collect.assert_called_once()
    assert h.profile_predicates is preds



# Unit tests .from_graph
@patch("cim_plugin.header.collect_specific_namespaces")
@patch.object(CIMMetadataHeader, "_collect_header_triples")
def test_from_graph_noheadertriples(mock_triples: MagicMock, mock_namespaces: MagicMock) -> None:
    g = Graph()
    # No rdf:type triples at all
    with pytest.raises(ValueError, match="No metadata header"):
        CIMMetadataHeader.from_graph(g)
    mock_triples.assert_not_called()
    mock_namespaces.assert_not_called()

@patch("cim_plugin.header.collect_specific_namespaces")
@patch.object(CIMMetadataHeader, "_collect_header_triples")
def test_from_graph_multipleheaders(mock_triples: MagicMock, mock_namespaces: MagicMock) -> None:
    g = Graph()
    g.add((URIRef("urn:h1"), RDF.type, MD.FullModel))
    g.add((URIRef("urn:h2"), RDF.type, DCAT.Dataset))

    with pytest.raises(ValueError, match="Multiple metadata headers"):
        CIMMetadataHeader.from_graph(g)
    mock_triples.assert_not_called()
    mock_namespaces.assert_not_called()


@patch("cim_plugin.header.collect_specific_namespaces")
@patch.object(CIMMetadataHeader, "_collect_header_triples")
def test_from_graph_onlyheadertriple(mock_triples: MagicMock, mock_namespaces: MagicMock) -> None:
    g = Graph()
    header = URIRef("urn:header")
    g.add((header, RDF.type, DCAT.Dataset))

    mock_triples.return_value = (header, [(header, RDF.type, DCAT.Dataset)], set(header))
    result = CIMMetadataHeader.from_graph(g)

    assert result.subject == header
    assert result.triples == [(header, RDF.type, DCAT.Dataset)]
    mock_triples.assert_called_once_with(g, header)
    mock_namespaces.assert_called_once_with([(header, RDF.type, DCAT.Dataset)], g.namespace_manager)


@patch("cim_plugin.header.collect_specific_namespaces")
@patch.object(CIMMetadataHeader, "_collect_header_triples")
def test_from_graph_namespaces(mock_triples: MagicMock, mock_namespaces: MagicMock) -> None:
    g = Graph()
    header = URIRef("urn:header")
    g.add((header, RDF.type, DCAT.Dataset))

    mock_triples.return_value = (header, [(header, RDF.type, DCAT.Dataset)], set(header))
    mock_namespaces.return_value = {"rdf": URIRef("rdf_test"), "dcat": URIRef("dcat_test")}
    result = CIMMetadataHeader.from_graph(g)

    assert result.subject == header
    assert result.triples == [(header, RDF.type, DCAT.Dataset)]
    ns = result.graph.namespace_manager.store
    assert len(list(ns.namespaces())) == 2
    assert ns.namespace("rdf") == URIRef("rdf_test")
    assert ns.namespace("dcat") == URIRef("dcat_test")
    mock_triples.assert_called_once_with(g, header)
    mock_namespaces.assert_called_once_with([(header, RDF.type, DCAT.Dataset)], g.namespace_manager)

@patch("cim_plugin.header.collect_specific_namespaces")
@patch.object(CIMMetadataHeader, "_collect_header_triples")
def test_from_graph_blankheaderrepair(mock_triples: MagicMock, mock_namespaces: MagicMock) -> None:
    g = Graph()
    header = BNode()
    repaired = URIRef("urn:uuid:fixed")

    g.add((header, RDF.type, DCAT.Dataset))

    mock_triples.return_value = (repaired, [(repaired, RDF.type, DCAT.Dataset)], set(repaired))
    mock_namespaces.return_value = {"rdf": URIRef("rdf_test"), "dcat": URIRef("dcat_test")}

    result = CIMMetadataHeader.from_graph(g)

    assert result.subject == repaired
    assert result.triples == [(repaired, RDF.type, DCAT.Dataset)]
    mock_triples.assert_called_once_with(g, header)
    mock_namespaces.assert_called_once_with([(repaired, RDF.type, DCAT.Dataset)], g.namespace_manager)


@patch("cim_plugin.header.collect_specific_namespaces")
@patch.object(CIMMetadataHeader, "_collect_header_triples")
def test_from_graph_metadataobjectsoverride(mock_triples: MagicMock, mock_namespaces: MagicMock) -> None:
    g = Graph()
    header = URIRef("urn:header")

    g.add((header, RDF.type, URIRef("urn:custom:Type")))

    mock_triples.return_value = (header, [], set())

    result = CIMMetadataHeader.from_graph(g, metadata_objects={URIRef("urn:custom:Type")})

    assert result.metadata_objects == {URIRef("urn:custom:Type")}


# Integration tests .from_graph
def test_from_graph_integration_dcatheader() -> None:
    g = Graph()
    header = URIRef("urn:header")
    g.add((header, RDF.type, DCAT.Dataset))
    g.add((header, URIRef("urn:p"), URIRef("urn:o")))
    reachable = {header}

    result = CIMMetadataHeader.from_graph(g)

    assert set(result.triples) == {(header, RDF.type, DCAT.Dataset), (header, URIRef("urn:p"), URIRef("urn:o"))}
    assert len(list(result.graph.namespace_manager.store.namespaces())) == 2
    assert result.reachable_nodes == reachable


def test_from_graph_integration_fullmodelheader() -> None:
    g = Graph()
    header = URIRef("urn:header")
    g.bind("md", MD)    # While dcat is in the default list, MD must be added
    g.add((header, RDF.type, MD.FullModel))
    g.add((header, URIRef("urn:p"), URIRef("urn:o")))
    reachable = {header}

    result = CIMMetadataHeader.from_graph(g)

    assert set(result.triples) == {(header, RDF.type, MD.FullModel), (header, URIRef("urn:p"), URIRef("urn:o"))}
    assert len(list(result.graph.namespace_manager.store.namespaces())) == 2
    assert result.reachable_nodes == reachable


def test_from_graph_integration_specialheadernotinput() -> None:
    g = Graph()
    header = URIRef("urn:header")
    g.add((header, RDF.type, URIRef("urn:meta:Header")))
    with pytest.raises(ValueError, match="No metadata header"):
        CIMMetadataHeader.from_graph(g)
    
    
def test_from_graph_integration_metadataobjectsoverride() -> None:
    g = Graph()
    header = URIRef("urn:header")
    g.add((header, RDF.type, URIRef("urn:meta:Header")))
    g.add((header, URIRef("urn:p"), URIRef("urn:o")))
    reachable = {header}

    result = CIMMetadataHeader.from_graph(g, metadata_objects=[URIRef("urn:meta:Header")])

    assert set(result.triples) == {(header, RDF.type, URIRef("urn:meta:Header")), (header, URIRef("urn:p"), URIRef("urn:o"))}
    assert len(list(result.graph.namespace_manager.store.namespaces())) == 1
    assert result.reachable_nodes == reachable


def test_from_graph_integration_blankheaderrepair(caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    header = BNode()
    repaired = URIRef("urn:uuid:fixed")

    g.add((header, RDF.type, DCAT.Dataset))
    g.add((header, DCTERMS.identifier, Literal("fixed")))

    result = CIMMetadataHeader.from_graph(g)

    assert result.subject == repaired
    assert (repaired, RDF.type, DCAT.Dataset) in result.graph
    assert len(list(result.graph.namespace_manager.store.namespaces())) == 3
    assert "Metadata header subject is a blank node" in caplog.text


def test_from_graph_integration_namespaces() -> None:
    g = Graph()
    header = URIRef("urn:uuid:fixed")
    g.bind("md", MD)
    g.add((header, RDF.type, DCAT.Dataset))
    g.add((header, DCTERMS.identifier, Literal("fixed")))
    g.add((header, MD.profile, Literal("profile1")))

    result = CIMMetadataHeader.from_graph(g)

    assert result.subject == header
    assert (header, RDF.type, DCAT.Dataset) in result.graph
    assert len(list(result.graph.namespace_manager.store.namespaces())) == 4
    for prefix, ns in result.graph.namespace_manager.store.namespaces():
        assert prefix in {"md", "rdf", "dcat", "dcterms"}

# Unit tests ._collect_header_triples

def test_collect_header_triples_reachability(build_graph_with_blank_header: tuple[Graph, BNode, set[BNode]]) -> None:
    # Checks that triples with blank subject nodes are reachable
    g, subject, expected_reachable = build_graph_with_blank_header

    final_subject, triples, reachable = CIMMetadataHeader._collect_header_triples(g, subject)

    assert reachable == expected_reachable


@patch.object(CIMMetadataHeader, "_repair_blank_header_subject")
def test_collect_header_triples_repairsblanksubject(mock_repair: MagicMock, build_graph_with_blank_header: tuple[Graph, BNode, set[BNode]]) -> None:
    g, subject, _ = build_graph_with_blank_header
    mock_repair.side_effect = lambda graph, blank: URIRef("urn:uuid:fixed")

    final_subject, triples, reachable = CIMMetadataHeader._collect_header_triples(g, subject)

    assert final_subject == URIRef("urn:uuid:fixed")
    mock_repair.assert_called_once()


@patch.object(CIMMetadataHeader, "_repair_blank_header_subject")
def test_collect_header_triples_urisubject(mock_repair: MagicMock) -> None:
    g = Graph()
    subject = URIRef("urn:header")
    g.add((subject, RDF.type, URIRef("urn:meta:Header")))

    final_subject, triples, reachable = CIMMetadataHeader._collect_header_triples(g, subject)

    assert final_subject == subject
    mock_repair.assert_not_called()


@patch.object(CIMMetadataHeader, "_repair_blank_header_subject")
def test_collect_header_triples_rewrites_blank_subjects(mock_repair: MagicMock, build_graph_with_blank_header: tuple[Graph, BNode, set[BNode]]) -> None:
    g, header, _ = build_graph_with_blank_header
    mock_repair.side_effect = lambda graph, blank: URIRef("urn:uuid:fixed")

    final_subject, triples, reachable = CIMMetadataHeader._collect_header_triples(g, header)

    # All subjects must now be URIRef("urn:uuid:fixed")
    for s, p, o in triples:
        assert isinstance(s, URIRef)
        assert s == URIRef("urn:uuid:fixed")
    mock_repair.assert_called_once()


@patch.object(CIMMetadataHeader, "_repair_blank_header_subject")
def test_collect_header_triples_predicatesandliterals(mock_repair: MagicMock):
    g = Graph()
    header = BNode()
    mock_repair.side_effect = lambda graph, blank: URIRef("urn:uuid:fixed")

    g.add((header, RDF.type, URIRef("urn:meta:Header")))
    g.add((header, URIRef("urn:p"), Literal("value")))

    final_subject, triples, reachable = CIMMetadataHeader._collect_header_triples(g, header)

    assert triples == [
        (URIRef("urn:uuid:fixed"), RDF.type, URIRef("urn:meta:Header")),
        (URIRef("urn:uuid:fixed"), URIRef("urn:p"), Literal("value"))
    ]


@patch.object(CIMMetadataHeader, "_repair_blank_header_subject")
def test_collect_header_triples_cycles(mock_repair: MagicMock) -> None:
    g = Graph()
    header = BNode()
    b1 = BNode()
    b2 = BNode()

    g.add((header, RDF.type, URIRef("urn:meta:Header")))
    g.add((header, URIRef("urn:p1"), b1))
    g.add((b1, URIRef("urn:p2"), b2))
    g.add((b2, URIRef("urn:p3"), b1))  # cycle

    mock_repair.return_value = URIRef("urn:uuid:fixed")

    final_subject, triples, reachable = CIMMetadataHeader._collect_header_triples(g, header)

    assert reachable == {header, b1, b2}
    # No infinite loops, no duplicates
    assert len(triples) == 1
    mock_repair.assert_called_once()


@pytest.mark.parametrize(
    "subject_is_blank, expected_subject",
    [
        pytest.param(True, URIRef("urn:uuid:fixed"), id="Blank subject repaired"),
        pytest.param(False, URIRef("urn:header"), id="URIRef subject kept"),
    ]
)
@patch.object(CIMMetadataHeader, "_repair_blank_header_subject")
def test_collect_header_triples_emptyheader(mock_repair: MagicMock, subject_is_blank: bool, expected_subject: URIRef):
    g = Graph()
    header = BNode() if subject_is_blank else URIRef("urn:header")

    mock_repair.return_value = URIRef("urn:uuid:fixed")

    g.add((header, RDF.type, URIRef("urn:meta:Header")))

    final_subject, triples, reachable = CIMMetadataHeader._collect_header_triples(g, header)

    assert final_subject == expected_subject
    assert reachable == {header}
    assert len(triples) == 1
    assert triples[0][1] == RDF.type
    if subject_is_blank:
        mock_repair.assert_called_once()
    else:
        mock_repair.assert_not_called()


@patch.object(CIMMetadataHeader, "_repair_blank_header_subject")
def test_collect_header_triples_branchingbfs(mock_repair: MagicMock) -> None:
    g = Graph()
    header = BNode()
    b1 = BNode()
    b2 = BNode()

    g.add((header, RDF.type, URIRef("urn:meta:Header")))
    g.add((header, URIRef("urn:p1"), b1))
    g.add((header, URIRef("urn:p2"), b2))
    g.add((b1, URIRef("urn:p3"), Literal("x")))
    g.add((b2, URIRef("urn:p4"), Literal("y")))

    mock_repair.return_value = URIRef("urn:uuid:fixed")

    final_subject, triples, reachable = CIMMetadataHeader._collect_header_triples(g, header)

    assert reachable == {header, b1, b2}
    assert len(triples) == 3
    for s, p, o in triples:
        assert not isinstance(o, BNode)
    mock_repair.assert_called_once()

@patch.object(CIMMetadataHeader, "_repair_blank_header_subject")
def test_collect_header_triples_ignoresunreachablenodes(mock_repair: MagicMock) -> None:
    g = Graph()
    header = BNode()
    reachable_b = BNode()
    unreachable_b = BNode()

    g.add((header, RDF.type, URIRef("urn:meta:Header")))
    g.add((header, URIRef("urn:p1"), reachable_b))
    g.add((reachable_b, URIRef("urn:p2"), Literal("ok")))

    # Unreachable subtree
    g.add((unreachable_b, URIRef("urn:p3"), Literal("ignored")))

    mock_repair.return_value = URIRef("urn:uuid:fixed")

    final_subject, triples, reachable = CIMMetadataHeader._collect_header_triples(g, header)

    assert unreachable_b not in reachable
    assert all(unreachable_b not in t for t in triples)


@pytest.mark.parametrize(
    "obj, subject_is_blank, expected_count",
    [
        pytest.param(Literal("x"), True, 2, id="Literal kept (blank subject)"),
        pytest.param(Literal("x"), False, 2, id="Literal kept (URIRef subject)"),
        pytest.param(URIRef("urn:external"), True, 2, id="URIRef kept (blank subject)"),
        pytest.param(URIRef("urn:external"), False, 2, id="URIRef kept (URIRef subject)"),
        pytest.param(BNode(), True, 1, id="Blank object skipped (blank subject)"),
        pytest.param(BNode(), False, 1, id="Blank object skipped (URIRef subject)"),
    ]
)
@patch.object(CIMMetadataHeader, "_repair_blank_header_subject")
def test_collect_header_triples_mixedobjects(mock_repair: MagicMock, obj: Node, subject_is_blank: bool, expected_count: int) -> None:
    g = Graph()
    header = BNode() if subject_is_blank else URIRef("urn:header")

    mock_repair.return_value = URIRef("urn:uuid:fixed")

    g.add((header, RDF.type, URIRef("urn:meta:Header")))
    g.add((header, URIRef("urn:p"), obj))

    final_subject, triples, reachable = CIMMetadataHeader._collect_header_triples(g, header)

    assert len(triples) == expected_count


# Unit tests .empty
def test_empty_basic() -> None:
    h = CIMMetadataHeader.empty()
    assert isinstance(h.subject, URIRef)
    assert h.triples == []
    assert h.reachable_nodes == set()
    assert h.metadata_objects == CIMMetadataHeader.DEFAULT_METADATA_OBJECTS
    assert h.profile_predicates == CIMMetadataHeader.DEFAULT_PROFILE_PREDICATES


def test_empty_subjectoverride() -> None:
    s = URIRef("urn:test")
    h = CIMMetadataHeader.empty(subject=s)
    assert h.subject == s


def test_empty_metadataobjectsoverride() -> None:
    objs = {URIRef("urn:meta")}
    h = CIMMetadataHeader.empty(metadata_objects=objs)
    assert h.metadata_objects == objs


def test_empty_profilepredicatesoverride() -> None:
    preds = {URIRef("urn:custom:profile")}
    h = CIMMetadataHeader.empty(profile_predicates=preds)
    assert h.profile_predicates is preds


@patch.object(CIMMetadataHeader, "collect_profile")
def test_empty_profile_override(mock_collect: MagicMock) -> None:
    h = CIMMetadataHeader.empty(profile="manual")
    assert h.profile == "manual"
    mock_collect.assert_not_called()


@patch.object(CIMMetadataHeader, "collect_profile")
def test_empty_profile_none_calls_collect(mock_collect: MagicMock) -> None:
    mock_collect.return_value = "auto"
    h = CIMMetadataHeader.empty(profile=None)
    assert h.profile == "auto"
    mock_collect.assert_called_once()

# Unit tests .from_manifest
@patch("cim_plugin.header.collect_specific_namespaces")
def test_from_manifest_wrongfiletype(mock_collect: MagicMock) -> None:
    with patch("cim_plugin.header.Graph.parse", side_effect=Exception) as mock_parse:
        with pytest.raises(Exception) as exc:
            CIMMetadataHeader.from_manifest("dummy.trig", "graph1")
            mock_parse.assert_called_once_with("dummy.trig", "graph1")
    
    mock_collect.assert_not_called()


@patch("cim_plugin.header.collect_specific_namespaces")
def test_from_manifest_emptyfile(mock_collect: MagicMock, fake_parse_factory: Callable) -> None:
    mock_graph = Graph()
    with patch("cim_plugin.header.Graph.parse", new=fake_parse_factory(mock_graph)) as mock_parse:
        with pytest.raises(ValueError) as exc:
            CIMMetadataHeader.from_manifest("dummy.xml", "graph1")
            mock_parse.assert_called_once_with("dummy.xml", "xml")
    
    mock_collect.assert_not_called()
    assert "No header triples matching graph identifier graph1 found in manifest file." in str(exc.value)
    

@pytest.mark.parametrize(
    "manifest_triples, subject, expected_triples, expect_error",
    [
        pytest.param(
            [
                (URIRef("graph1"), URIRef("http://example.org/p"), Literal("o")),
                (URIRef("graph2"), URIRef("http://example.org/p"), Literal("x")),
            ],
            URIRef("graph1"),
            {(URIRef("graph1"), URIRef("http://example.org/p"), Literal("o"))},
            False,
            id="URIRef subject, graph with multiple subjects."
        ),
        pytest.param(
            [
                (URIRef("graph1"), URIRef("http://example.org/p"), Literal("o")),
                (URIRef("graph2"), URIRef("http://example.org/p"), Literal("x")),
            ],
            "graph1",
            {(URIRef("graph1"), URIRef("http://example.org/p"), Literal("o"))},
            False,
            id="String subject, graph with multiple subjects."
        ),
                pytest.param(
            [
                (URIRef("graph1#section"), URIRef("http://example.org/p"), Literal("o")),
                (URIRef("graph1"), URIRef("http://example.org/p"), Literal("x")),
            ],
            URIRef("graph1#section"),
            {(URIRef("graph1#section"), URIRef("http://example.org/p"), Literal("o"))},
            False,
            id="URIRef subject with fragment"
        ),
        pytest.param(
            [
                (URIRef("graph1"), URIRef("http://example.org/p"), Literal("o")),
                (URIRef("graph2"), URIRef("http://example.org/p"), Literal("x")),
            ],
            URIRef("graph3"),
            set(),
            True,
            id="Subject not found, no matches."
        ),
        pytest.param(
            [
                (URIRef("graph1"), URIRef("http://example.org/p1"), Literal("o1")),
                (URIRef("graph1"), URIRef("http://example.org/p2"), Literal("o2")),
                (URIRef("graph2"), URIRef("http://example.org/p3"), Literal("o3")),
            ],
            URIRef("graph1"),
            {
                (URIRef("graph1"), URIRef("http://example.org/p1"), Literal("o1")),
                (URIRef("graph1"), URIRef("http://example.org/p2"), Literal("o2")),
            },
            False,
            id="Multiple triples for the same subject"
        ),
        pytest.param(
            [
                (Literal("weird"), URIRef("http://example.org/p"), Literal("o")),
            ],
            Literal("weird"),
            set(),
            True,
            id="Literal subject, no matches"
        ),
        pytest.param(
            [
                (BNode("b1"), URIRef("http://example.org/p"), Literal("o")),
                (URIRef("graph1"), URIRef("http://example.org/p"), Literal("x")),
            ],
            BNode("b1"),
            set(),
            True,
            id="Blank node subject, no matches."
        ),
        pytest.param(
            [
                (BNode("b1"), URIRef("http://example.org/p"), Literal("o")),
                (URIRef("graph1"), URIRef("http://example.org/p"), Literal("x")),
            ],
            "b1",
            set(),
            True,
            id="String subject, blank node in manifest, no matches."
        ),
        pytest.param(
            [
                (BNode("graph1"), URIRef("http://example.org/p"), Literal("o")),
                (URIRef("graph1"), URIRef("http://example.org/p"), Literal("x")),
                (Literal("graph1"), URIRef("http://example.org/p"), Literal("y")),
            ],
            "graph1",
            {(URIRef("graph1"), URIRef("http://example.org/p"), Literal("x"))},
            False,
            id="Mixed triples case, matches only the URI."
        ),
    ]
)
@patch("cim_plugin.header.collect_specific_namespaces")
def test_from_manifest_parametrized(
    mock_collect: MagicMock,
    make_graph: Callable[..., Graph],
    fake_parse_factory: Callable,
    manifest_triples: list[tuple[Node, Node, Node]],
    subject: Any,
    expected_triples: set[tuple[Node, Node, Node]],
    expect_error: bool,
) -> None:
    mock_collect.return_value = {"ex": "http://example.org/"}
    manifest = make_graph(manifest_triples)
    print(list(manifest))
    with patch("cim_plugin.header.Graph.parse", new=fake_parse_factory(manifest)):
        
        if expect_error:
            with pytest.raises(ValueError) as exc:
                CIMMetadataHeader.from_manifest("dummy.xml", subject)
                mock_collect.assert_not_called()
                assert "No header triples matching graph identifier" in str(exc.value)
            return

        header = CIMMetadataHeader.from_manifest("dummy.xml", subject)

        mock_collect.assert_called_once()
        assert set(header.graph) == expected_triples
        assert header.graph.namespace_manager.store.namespace("ex") == URIRef("http://example.org/")


def test_from_manifest_namespaces(fake_parse_factory: MagicMock) -> None:
    mock_graph = Graph()
    mock_graph.bind("ex", "https://example.org/")
    mock_graph.bind("foo", "www.bar.com/")
    mock_graph.bind("exa", "www.extra.com/")
    mock_graph.add((URIRef("graph1"), URIRef("https://example.org/p1"), Literal("o1")))
    mock_graph.add((URIRef("graph1"), URIRef("www.bar.com/p2"), Literal("o2")))
    mock_graph.add((URIRef("graph2"), URIRef("www.extra.com/p3"), Literal("o3")))
    
    
    with patch("cim_plugin.header.Graph.parse", new=fake_parse_factory(mock_graph)):
        header = CIMMetadataHeader.from_manifest("dummy.xml", "graph1")
    
    assert header.subject == URIRef("graph1")
    assert len(header.graph) == 2
    assert len(list(header.graph.namespace_manager.store.namespaces())) == 2
    assert (URIRef("graph1"), URIRef("https://example.org/p1"), Literal("o1")) in header.graph
    assert (URIRef("graph1"), URIRef("www.bar.com/p2"), Literal("o2")) in header.graph
    # The used namespaces are included, but not the one that was not used
    assert header.graph.namespace_manager.store.namespace("ex") == URIRef("https://example.org/")
    assert header.graph.namespace_manager.store.namespace("foo") == URIRef("www.bar.com/")
    assert header.graph.namespace_manager.store.namespace("exa") == None

def test_from_manifest_nonamespacemanager(fake_parse_factory: MagicMock) -> None:
    # This should not be possible, as there are always default namespaces when parsing.
    # Documents that everything still works if it ever does.
    mock_graph = Graph()
    mock_graph.add((URIRef("graph1"), URIRef("https://example.org/p1"), Literal("o1")))
    # Pylance silenced to test an edge case
    mock_graph.namespace_manager = None    # type: ignore
    
    with patch("cim_plugin.header.Graph.parse", new=fake_parse_factory(mock_graph)):
        header = CIMMetadataHeader.from_manifest("dummy.xml", "graph1")
    
    assert header.subject == URIRef("graph1")
    assert len(header.graph) == 1
    assert len(list(header.graph.namespace_manager.store.namespaces())) == 0
    assert (URIRef("graph1"), URIRef("https://example.org/p1"), Literal("o1")) in header.graph
    assert header.graph.namespace_manager.store.namespace("ex") == None

@patch("cim_plugin.header.collect_specific_namespaces")
def test_from_manifest_differentnamespacereturned(mock_collect: MagicMock, fake_parse_factory: MagicMock) -> None:
    # Documents what happends if collect_specific_namespaces return a different namespace then in the triples (but same prefix).
    # This should never happen because it only returns the namespaces used by the triples.
    mock_graph = Graph()
    mock_graph.bind("ex", "https://example.org/")
    mock_graph.add((URIRef("graph1"), URIRef("https://example.org/p1"), Literal("o1")))
    mock_collect.return_value = {"ex": "www.new.com/"}
    
    with patch("cim_plugin.header.Graph.parse", new=fake_parse_factory(mock_graph)):
        header = CIMMetadataHeader.from_manifest("dummy.xml", "graph1")
    
    assert (URIRef("graph1"), URIRef("https://example.org/p1"), Literal("o1")) in header.graph
    assert header.graph.namespace_manager.store.namespace("ex") == URIRef("www.new.com/")


# Unit tests ._repair_blank_header_subject
@pytest.mark.parametrize(
        "header_type", [MD.FullModel, DCAT.Dataset]
)
def test_repair_blank_header_subject_dctidentifier(header_type: Node, caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    b = BNode()
    g.add((b, RDF.type, header_type))
    g.add((b, DCTERMS.identifier, Literal("1234")))

    header = CIMMetadataHeader.from_graph(g)

    assert header.subject == URIRef("urn:uuid:1234")
    records = caplog.messages
    assert len(records) == 1
    assert "blank node" in records[0]

@pytest.mark.parametrize(
        "header_type", [MD.FullModel, DCAT.Dataset]
)
def test_repair_blank_header_subject_withoutidentifier(header_type: Node, caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    b = BNode()
    g.add((b, RDF.type, header_type))

    header = CIMMetadataHeader.from_graph(g)

    assert isinstance(header.subject, URIRef)
    assert header.subject.startswith("urn:uuid:")
    assert caplog.text.count("blank node") == 1
    assert "Random UUID generated" in caplog.text


def test_repair_blank_header_subject_dctidentifierisuri(caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    b = BNode()
    g.add((b, RDF.type, DCAT.Dataset))
    g.add((b, DCTERMS.identifier, URIRef("urn:foo")))

    header = CIMMetadataHeader.from_graph(g)

    assert header.subject != URIRef("urn:foo")
    records = caplog.messages
    assert len(records) == 2
    assert "blank node" in records[0]
    assert "Random UUID generated" in records[1]


def test_repair_blank_header_subject_multipledctidentifier(caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    b = BNode()
    g.add((b, RDF.type, DCAT.Dataset))
    g.add((b, DCTERMS.identifier, Literal("A")))
    g.add((b, DCTERMS.identifier, Literal("B")))

    header = CIMMetadataHeader.from_graph(g)

    assert header.subject == URIRef("urn:uuid:A")   # First encountered wins
    records = caplog.messages
    assert len(records) == 1
    assert "blank node" in records[0]

@pytest.mark.parametrize(
        "literal, expected", 
        [
            pytest.param("", None, id="Empty Literal. Cannot extract uuid."), 
            pytest.param("  ", "urn:uuid:  ", id="Two whitespaces. Able to extract."),
            pytest.param("not-a-uuid", "urn:uuid:not-a-uuid", id="Not a valid uuid. Able to extract.")
        ]
)
def test_repair_blank_header_subject_emptydctidentifier(literal: str, expected: str, caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    b = BNode()
    g.add((b, RDF.type, DCAT.Dataset))
    g.add((b, DCTERMS.identifier, Literal(literal)))

    header = CIMMetadataHeader.from_graph(g)

    records = caplog.messages
    assert "blank node" in records[0]
    
    if expected:
        assert header.subject == URIRef(expected)
    else:
        assert "Random UUID generated" in records[1]
        

@pytest.mark.parametrize(
        "header_type", [MD.FullModel, DCAT.Dataset]
)
@patch.object(CIMMetadataHeader, "_repair_blank_header_subject")
def test_repair_blank_header_subject_norepairneeded(mock_repair: MagicMock, header_type: Node) -> None:
    g = Graph()
    s = URIRef("urn:header")
    g.add((s, RDF.type, header_type))

    header = CIMMetadataHeader.from_graph(g)

    assert header.subject == s
    mock_repair.assert_not_called()


# Unit tests .header_type
@pytest.mark.parametrize(
        "triples, metadata_objects, expected",
        [
            pytest.param([(RDF.type, MD.FullModel)], None, MD.FullModel, id="Fullmodel header"),
            pytest.param([(RDF.type, DCAT.Dataset)], None, DCAT.Dataset, id="Dcat header"),
            pytest.param([(RDF.type, URIRef("www.custom.org/type"))], [URIRef("www.custom.org/type")], URIRef("www.custom.org/type"), id="Custom type header"),
            pytest.param([(RDF.type, MD.FullModel), (RDF.type, DCAT.Dataset)], None, MD.FullModel, id="Multiple header types -> First encountered wins"),
        ]
)
def test_header_type_success(triples: tuple, metadata_objects: Iterable|None, expected: str) -> None:
    header = CIMMetadataHeader.empty(URIRef("s1"), metadata_objects=metadata_objects)
    for predicate, obj in triples:
        header.add_triple(predicate, obj)
    
    result = header.header_type
    assert result == expected


def test_header_type_noheadertype() -> None:
    header = CIMMetadataHeader.empty(URIRef("s1"))
    header.add_triple(URIRef("www.example.org/p"), URIRef("www.example.org/o"))

    with pytest.raises(ValueError) as exc:
        header.header_type

    assert "No triple with rdf:type found in header." in str(exc.value)


def test_header_type_notriples() -> None:
    header = CIMMetadataHeader.empty(URIRef("s1"))

    with pytest.raises(ValueError) as exc:
        header.header_type

    assert "No triple with rdf:type found in header." in str(exc.value)

# Unit tests .collect_profile
@pytest.mark.parametrize(
        "triples, expected",
        [
            pytest.param([(RDF.type, MD.FullModel), (URIRef(MD["Model.profile"]), Literal("model"))], "model", id="Fullmodel header, model profile"),
            pytest.param([(RDF.type, MD.FullModel), (URIRef(DCTERMS.conformsTo), Literal("dcterms"))], "dcterms", id="Fullmodel header, dcterms profile"),
            pytest.param([(RDF.type, DCAT.Dataset), (URIRef(DCTERMS.conformsTo), Literal("dcterms"))], "dcterms", id="Dcat header, dcterms profile"),
            pytest.param([(RDF.type, DCAT.Dataset), (URIRef(MD["Model.profile"]), Literal("model"))], "model", id="Dcat header, model profile"),
            pytest.param([(RDF.type, DCAT.Dataset), (URIRef(DCTERMS.identifier), Literal("Not a profile"))], None, id="Profile not present"),
            pytest.param([(RDF.type, DCAT.Dataset), (URIRef(DCTERMS.conformsTo), URIRef("dcterms"))], "dcterms", id="URIRef"),
            pytest.param([(RDF.type, DCAT.Dataset), (URIRef(DCTERMS.conformsTo), BNode("Not a profile"))], None, id="BNode"),
            pytest.param([(RDF.type, DCAT.Dataset), (URIRef(DCTERMS.conformsTo), Literal(42))], "42", id="Integer literal"),
            pytest.param([(RDF.type, DCAT.Dataset), (URIRef(DCTERMS.conformsTo), Literal("dcterms", lang="en"))], "dcterms", id="Literal with language"),
        ]
)
def test_collect_profile_success(triples: list[tuple], expected: str) -> None:
    header = CIMMetadataHeader.empty(URIRef("s1"))
    for predicate, obj in triples:
        header.add_triple(predicate, obj)
    
    result = header.collect_profile()
    assert result == expected


def test_collect_profile_customprofile() -> None:
    header = CIMMetadataHeader.empty(URIRef("s1"), profile_predicates={URIRef("custom")})
    header.add_triple(RDF.type, MD.FullModel)
    header.add_triple(URIRef("custom"), Literal("model"))
    
    result = header.collect_profile()
    assert result == "model"


def test_collect_profile_multipleprofiles() -> None:
    header = CIMMetadataHeader.empty(URIRef("s1"))
    header.add_triple(RDF.type, DCAT.Dataset)
    header.add_triple(DCTERMS.conformsTo, Literal("model1"))
    header.add_triple(DCTERMS.conformsTo, Literal("model2"))
    
    result = header.collect_profile()
    assert result == "model1"   # First encountered wins


# Unit tests .set_subject
@pytest.mark.parametrize(
        "new_subject",
        [
            pytest.param("h2", id="Simple replace"),
            pytest.param("h1", id="New is same as old")
        ]
)
def test_set_subject_basic(new_subject: str) -> None:
    header = CIMMetadataHeader.empty(URIRef("h1"))
    header.add_triple(RDF.type, DCAT.Dataset)
    header.add_triple(DCTERMS.conformsTo, Literal("model1"))
    header.add_triple(DCTERMS.contributor, Literal("contributor1"))
    header.add_triple(DCTERMS.contributor, URIRef("h1"))
    
    header.set_subject(URIRef(new_subject))

    assert header.subject == URIRef(new_subject)
    assert len(header.triples) == 4
    assert (URIRef(new_subject), DCTERMS.contributor, URIRef("h1")) in header.triples
    for s, p, o in header.triples:
        assert s == URIRef(new_subject)


def test_set_subject_emptyheader() -> None:
    header = CIMMetadataHeader.empty(URIRef("h1"))

    header.set_subject(URIRef("h2"))

    assert header.subject == URIRef("h2")
    assert len(header.triples) == 0


def test_set_subject_multiplecalls() -> None:
    header = CIMMetadataHeader.empty(URIRef("h1"))
    header.add_triple(RDF.type, DCAT.Dataset)
    header.add_triple(DCTERMS.conformsTo, Literal("model1"))
    header.add_triple(DCTERMS.contributor, Literal("contributor1"))
    header.add_triple(DCTERMS.contributor, Literal("contributor2"))
    
    header.set_subject(URIRef("h2"))
    header.set_subject(URIRef("h3"))

    assert header.subject == URIRef("h3")
    assert len(header.triples) == 4
    for s, p, o in header.triples:
        assert s == URIRef("h3")


#Unit tests create_header_attribute
def test_create_header_attribute_fromgraph() -> None:
    graph = MagicMock()
    fake_header = MagicMock()

    with patch.object(CIMMetadataHeader, "from_graph", return_value=fake_header) as mock_from_graph, \
         patch.object(CIMMetadataHeader, "empty") as mock_empty, \
         patch.object(logger, "error") as mock_log:

        result = create_header_attribute(graph)

    assert result is fake_header
    mock_from_graph.assert_called_once_with(graph)
    mock_empty.assert_not_called()
    mock_log.assert_not_called()


def test_create_header_attribute_valueerror(caplog: pytest.LogCaptureFixture) -> None:
    graph = MagicMock()
    fake_empty_header = MagicMock()
    fake_empty_header.subject = "generated-id"

    with patch.object(CIMMetadataHeader, "from_graph", side_effect=ValueError("oops")) as mock_from_graph, \
         patch.object(CIMMetadataHeader, "empty", return_value=fake_empty_header) as mock_empty:
        result = create_header_attribute(graph)

    assert result is fake_empty_header
    mock_from_graph.assert_called_once_with(graph)
    mock_empty.assert_called_once()
    
    assert len(caplog.records) == 1 
    record = caplog.records[0] 
    assert record.levelname == "ERROR" 
    assert "oops" in record.message 
    assert "generated-id" in record.message


def test_create_header_attribute_otherexceptions() -> None:
    graph = Graph()

    with patch.object(CIMMetadataHeader, "from_graph", side_effect=TypeError("boom")):
        with pytest.raises(TypeError):
            create_header_attribute(graph)


if __name__ == "__main__":
    pytest.main()