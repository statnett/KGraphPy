import uuid
from rdflib import Graph, RDF, Namespace, Dataset, URIRef
from rdflib.exceptions import ParserError
import logging
from xml.sax import SAXParseException
from cim_plugin.exceptions import CIMXMLParseError

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


def load_cimxml_graph(file_path: str, schema_path: str|None = None) -> tuple[uuid.UUID, Graph]:
    """Load one CIMXML file to Graph and get graph uuid.
    
    Parameters:
        file_path (str): Path to CIMXML file.
        schema_path (str): Path to linkML file with the cim model

    Raises:
        CIMXMLParseError: If errors in the parsing process is raised.

    Returns:
        tuple[uuid.UUID, Graph]: The graph uuid and the Graph object.
    """
    try:
        g = Graph() 
        g.parse(file_path, format="cimxml", schema_path=schema_path) 
        uuid = get_graph_uuid(g) 
        return uuid, g
    except (FileNotFoundError, ParserError, SAXParseException, ValueError) as e:
        raise CIMXMLParseError(file_path, e) from e


def collect_cimxml_to_dataset(files: list[str], schema_path: str|None = None) -> Dataset:
    """Collect multiple CIMXML files into one Dataset object. 
    
    Namespaces will be normalised between graphs:
        - Same namespace with different prefixes: last prefix added keeps namespace, previous namespaces are set to None.
        - Same prefix with different namespaces: first namespace added is kept

    Parameters:
        files (list): Paths to files containing CIMXML graphs.
        schema_path (str): Path to the linkML file with the cim model.

    Returns:
        Dataset: All the graphs collected in a Dataset object.
    """
    ds = Dataset()

    for file_path in files:
        try:
            uuid, graph = load_cimxml_graph(file_path, schema_path)
        except (CIMXMLParseError) as e:
            logger.error(e)
            continue

        named = ds.graph(URIRef(f"urn:uuid:{uuid}"))

        for prefix, uri in graph.namespace_manager.namespaces():
            ds.namespace_manager.bind(prefix, uri)
            named.namespace_manager.bind(prefix, uri)

        for triple in graph:
            named.add(triple)

    return ds

if __name__ == "__main__":
    print("utilities for cimxml parser")