import uuid
from rdflib import Graph, RDF, Namespace, Dataset, URIRef
import logging

logger = logging.getLogger('cimxml_logger')

MD = Namespace("http://iec.ch/TC57/61970-552/ModelDescription/1#") 
DCAT = Namespace("http://www.w3.org/ns/dcat#")


def get_graph_uuid(graph: Graph) -> uuid.UUID:
    """Get uuid from graph.
    
    Will find the uuid from MD.FullModel or DCAT.Dataset, in that order.
    Thid uuid can be used as a graph identifier.

    Parameters:
        graph (Graph): The graph to search for uuid.

    Raises:
        ValueError: If none of the metadata headers are found.

    Returns:
        uuid.UUID
    """
    for s in graph.subjects(RDF.type, MD.FullModel): 
        return _extract_uuid_from_urn(str(s)) 
    
    for s in graph.subjects(RDF.type, DCAT.Dataset): 
        return _extract_uuid_from_urn(str(s)) 
    
    raise ValueError("Did not find md:FullModel or dcat:Dataset in the graf.")


def _extract_uuid_from_urn(urn: str) -> uuid.UUID: 
    """Extract a uuid.UUID for a URI like 'urn:uuid:1234-...'.
    
    Parameters:
        urn (str): The uuid.

    Raises:
        ValueError: If urn does not start with 'urn:uuid:'.

    Returns:
        uuid.UUID: The urn without 'urn:uuid:'.
    """ 
    prefix = "urn:uuid:" 
    if not urn.lower().startswith(prefix):
        raise ValueError(f"Invalid model URI: {urn}") 
    
    return uuid.UUID(urn[len(prefix):])


def load_cimxml_graph(file_path: str, schema_path: str) -> tuple[uuid.UUID, Graph]: 
    g = Graph() 
    g.parse(file_path, format="cimxml", schema_path=schema_path) 
    # Your parser stores the UUID internally, or you can extract it here 
    uuid = get_graph_uuid(g) 
    return uuid, g


def combine_cimxml_to_dataset(files: list[str], schema_path: str) -> Dataset:
    ds = Dataset()

    for file_path in files:
        uuid, g = load_cimxml_graph(file_path, schema_path)

        # Create a named graph inside the dataset
        named = ds.graph(URIRef(f"urn:uuid:{uuid}"))

        # Copy namespace bindings from the graph into BOTH dataset and named graph
        for prefix, uri in g.namespace_manager.namespaces():
            ds.namespace_manager.bind(prefix, uri)
            named.namespace_manager.bind(prefix, uri)

        # Copy triples
        for t in g:
            named.add(t)

    # Optional: remove default graph if empty
    ds.default_context.remove((None, None, None))

    return ds

if __name__ == "__main__":
    print("utilities for cimxml parser")