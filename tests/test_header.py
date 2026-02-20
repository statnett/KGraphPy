import pytest
from unittest.mock import patch, MagicMock
from rdflib import Graph, URIRef, Literal, BNode, Node
from rdflib.namespace import DCAT, DCTERMS, RDF
from cim_plugin.namespaces import MD
from cim_plugin.header import CIMMetadataHeader
from tests.fixtures import build_graph_with_blank_header


# Unit tests .__init__
def test_init_generatesuuidsubject() -> None:
    h = CIMMetadataHeader()
    assert isinstance(h.subject, URIRef)
    assert str(h.subject).startswith("urn:uuid:")


def test_init_providedsubject() -> None:
    s = URIRef("urn:test")
    h = CIMMetadataHeader(subject=s)
    assert h.subject == s


def test_init_triplesnone() -> None:
    h = CIMMetadataHeader(triples=None)
    assert h.triples == []


def test_init_triplesinput() -> None:
    triples = [(URIRef("a"), URIRef("b"), Literal("c"))]
    h = CIMMetadataHeader(triples=triples)
    assert h.triples == triples
    assert h.triples is not triples  # must be a copy


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
    preds: set[Node] = {URIRef("urn:test:profile")}
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
    preds: set[Node] = {URIRef("urn:test:profile")}
    h = CIMMetadataHeader(profile=None, profile_predicates=preds)
    mock_collect.assert_called_once()
    assert h.profile_predicates is preds



# Unit tests .from_graph
@patch.object(CIMMetadataHeader, "_collect_header_triples")
def test_from_graph_noheadertriples(mock_collect: MagicMock) -> None:
    g = Graph()
    # No rdf:type triples at all
    with pytest.raises(ValueError, match="No metadata header"):
        CIMMetadataHeader.from_graph(g)
    mock_collect.assert_not_called()

@patch.object(CIMMetadataHeader, "_collect_header_triples")
def test_from_graph_multipleheaders(mock_collect: MagicMock) -> None:
    g = Graph()
    g.add((URIRef("urn:h1"), RDF.type, MD.FullModel))
    g.add((URIRef("urn:h2"), RDF.type, DCAT.Dataset))

    with pytest.raises(ValueError, match="Multiple metadata headers"):
        CIMMetadataHeader.from_graph(g)
    mock_collect.assert_not_called()


@patch.object(CIMMetadataHeader, "_collect_header_triples")
def test_from_graph_onlyheadertriple(mock_collect: MagicMock) -> None:
    g = Graph()
    header = URIRef("urn:header")
    g.add((header, RDF.type, DCAT.Dataset))

    mock_collect.return_value = (header, [(header, RDF.type, DCAT.Dataset)], set(header))

    result = CIMMetadataHeader.from_graph(g)

    assert result.subject == header
    assert result.triples == [(header, RDF.type, DCAT.Dataset)]
    mock_collect.assert_called_once_with(g, header)


@patch.object(CIMMetadataHeader, "_collect_header_triples")
def test_from_graph_blankheaderrepair(mock_collect: MagicMock) -> None:
    g = Graph()
    header = BNode()
    repaired = URIRef("urn:uuid:fixed")

    g.add((header, RDF.type, DCAT.Dataset))

    mock_collect.return_value = (repaired, [(repaired, RDF.type, DCAT.Dataset)], set(repaired))

    result = CIMMetadataHeader.from_graph(g)

    assert result.subject == repaired
    assert result.triples == [(repaired, RDF.type, DCAT.Dataset)]
    mock_collect.assert_called_once_with(g, header)


@patch.object(CIMMetadataHeader, "_collect_header_triples")
def test_from_graph_metadataobjectsoverride(mock_collect: MagicMock) -> None:
    g = Graph()
    header = URIRef("urn:header")

    g.add((header, RDF.type, URIRef("urn:custom:Type")))

    mock_collect.return_value = (header, [], set())

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

    assert result.triples == [(header, RDF.type, DCAT.Dataset), (header, URIRef("urn:p"), URIRef("urn:o"))]
    assert result.reachable_nodes == reachable


def test_from_graph_integration_fullmodelheader() -> None:
    g = Graph()
    header = URIRef("urn:header")
    g.add((header, RDF.type, MD.FullModel))
    g.add((header, URIRef("urn:p"), URIRef("urn:o")))
    reachable = {header}

    result = CIMMetadataHeader.from_graph(g)

    assert result.triples == [(header, RDF.type, MD.FullModel), (header, URIRef("urn:p"), URIRef("urn:o"))]
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

    assert result.triples == [(header, RDF.type, URIRef("urn:meta:Header")), (header, URIRef("urn:p"), URIRef("urn:o"))]
    assert result.reachable_nodes == reachable


def test_from_graph_integration_blankheaderrepair(caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    header = BNode()
    repaired = URIRef("urn:uuid:fixed")

    g.add((header, RDF.type, DCAT.Dataset))
    g.add((header, DCTERMS.identifier, Literal("fixed")))

    result = CIMMetadataHeader.from_graph(g)

    assert result.subject == repaired
    assert result.triples == [(repaired, RDF.type, DCAT.Dataset), (repaired, DCTERMS.identifier, Literal("fixed"))]
    assert "Metadata header subject is a blank node" in caplog.text

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
# Consider if tests with custom inputs should be added.
def test_empty_createsminimalheader() -> None:
    h = CIMMetadataHeader.empty()
    assert h.triples == []
    assert isinstance(h.subject, URIRef)

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


# Unit tests .collect_profile
@pytest.mark.parametrize(
        "triples, expected",
        [
            pytest.param([(RDF.type, MD.FullModel), (URIRef("http://iec.ch/TC57/61970-552/ModelDescription/1#Model.profile"), Literal("http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0"))], "http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0", id="Fullmodel header, model profile"),
            pytest.param([(RDF.type, MD.FullModel), (URIRef("http://purl.org/dc/terms/conformsTo"), Literal("http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0"))], "http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0", id="Fullmodel header, dcterms profile"),
            pytest.param([(RDF.type, DCAT.Dataset), (URIRef("http://purl.org/dc/terms/conformsTo"), Literal("http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0"))], "http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0", id="Dcat header, dcterms profile"),
            pytest.param([(RDF.type, DCAT.Dataset), (URIRef("http://iec.ch/TC57/61970-552/ModelDescription/1#Model.profile"), Literal("http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0"))], "http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0", id="Dcat header, model profile"),
            pytest.param([(RDF.type, DCAT.Dataset), (URIRef(DCTERMS.identifier), Literal("Not a profile"))], None, id="Profile not present")
        ]
)
def test_collect_profile_success(triples: tuple, expected: str) -> None:
    header = CIMMetadataHeader.empty(URIRef("s1"))
    for predicate, obj in triples:
        header.add_triple(predicate, obj)
    
    result = header.collect_profile()
    assert result == expected

if __name__ == "__main__":
    pytest.main()