import pytest
from unittest.mock import Mock, patch, MagicMock
import io
from rdflib import Graph, URIRef, Literal, Namespace, RDF
from cim_plugin.graph import CIMGraph
from cim_plugin.header import CIMMetadataHeader
from cim_plugin.processor import CIMProcessor
from cim_plugin.to_file_strategies import JSONLDStrategy
import time
from cim_plugin.namespaces import update_namespace_in_triples, DCAT_EXT


@pytest.fixture(scope="session")
def big_graph() -> Graph:
    g = CIMGraph()
    g.bind("ex", "http://example.org/")
    g.metadata_header = CIMMetadataHeader.empty(URIRef("h1"))
    g.metadata_header.add_triple(RDF.type, DCAT_EXT.Dataset)

    NS_SPECIAL = Namespace("http://example.org/special/")
    NS_OTHER = Namespace("http://example.org/other/")

    # Example: 20% of triples use the special namespace
    for i in range(5_000_000):
        if i % 100 < 20:
            subj = NS_SPECIAL[f"s{i}"]
        else:
            subj = NS_OTHER[f"s{i}"]

        g.add((
            subj,
            URIRef("http://example.org/p"),
            Literal(i),
        ))

    return g


@pytest.mark.skip(reason="No more testing needed unless optimizations are made to the function.")
def test_update_namespace(big_graph: Graph) -> None:
    start = time.perf_counter()
    update_namespace_in_triples(big_graph, old_namespace="http://example.org/special/", new_namespace="http://example.org/new/")
    duration = time.perf_counter() - start

    assert len(big_graph) == 5_000_000
    print(f"Namespace replacement: {duration:.2f} seconds for 20% of 5 million triples")

    # Results of repeated runs:
    # Namespace replacement: 125.20 seconds for 80% of 5 million triples
    # Namespace replacement: 91.55 seconds for 60% of 5 million triples
    # Namespace replacement: 61.28 seconds for 40% of 5 million triples
    # Namespace replacement: 33.88 seconds for 20% of 5 million triples
    # Summary: The time is approximately linear with the number of triples to update, 
    # with about 0.17 seconds per 100k triples updated (~30sec per 1 million).


@pytest.mark.skip(reason="No more testing needed unless optimizations are made to the function.")
def test_cimxmlserialize(big_graph: Graph) -> None:
    stream = io.BytesIO()

    start = time.perf_counter()
    big_graph.serialize(destination=stream, format="cimxml")
    duration = time.perf_counter() - start

    assert stream.getvalue()  # minimal sanity check
    print(f"CIMXML serialization: {duration:.2f} seconds for 5 million triples")

    # CIMXML serialization: 23.82 seconds for 5 million triples without header
    # CIMXML serialization: 24.60 seconds for 5 million triples with header


@pytest.mark.skip(reason="No more testing needed unless optimizations are made to the function.")
def test_cimtrigserialize(big_graph: Graph) -> None:
    stream = io.BytesIO()

    start = time.perf_counter()
    big_graph.serialize(destination=stream, format="cimtrig")
    duration = time.perf_counter() - start

    assert stream.getvalue()  # minimal sanity check
    print(f"CIMTRIG serialization: {duration:.2f} seconds for 5 million triples")

    # CIMTRIG serialization: 244.20 seconds for 5 million triples without header
    # CIMTRIG serialization: 231.84 seconds for 5 million triples with header


@pytest.mark.stress
def test_jsonldserialize(big_graph: CIMGraph) -> None:
    pr = CIMProcessor(big_graph)

    buffer = io.StringIO()

    fake_open = MagicMock()
    fake_open.return_value.__enter__.return_value = buffer
    fake_open.return_value.__exit__.return_value = False

    with patch("cim_plugin.to_file_strategies.open", fake_open, create=True):
        jstrategy = JSONLDStrategy("dummy_path.jsonld")

        start = time.perf_counter()
        jstrategy.serialize(pr)
        duration = time.perf_counter() - start

    output = buffer.getvalue()
    assert output.strip()  # minimal sanity check

    print(f"JSON-LD serialization: {duration:.2f} seconds for 5 million triples")

    # JSON-LD serialization: 146.94 seconds for 5 million triples with header
