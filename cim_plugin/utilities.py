import uuid
from rdflib import Graph, Dataset, URIRef, Node
from rdflib.namespace import RDF, DCAT
from rdflib.exceptions import ParserError
import logging
from xml.sax import SAXParseException
from cim_plugin.exceptions import CIMXMLParseError
from cim_plugin.namespaces import MD
from cim_plugin.graph import CIMDataset, CIMGraph
from cim_plugin.header import CIMMetadataHeader

logger = logging.getLogger('cimxml_logger')


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


def load_cimxml_graph(file_path: str, schema_path: str|None = None) -> CIMGraph:
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
        g = CIMGraph() 
        g.parse(file_path, format="cimxml", schema_path=schema_path) 
        return g
    except (FileNotFoundError, ParserError, SAXParseException, ValueError) as e:
        raise CIMXMLParseError(file_path, e) from e


def collect_cimxml_to_dataset(files: list[str], schema_path: str|None = None) -> CIMDataset:
    """Collect multiple CIMXML files into one CIMDataset object. 
    
    Namespaces will be normalised between graphs:
        - Same namespace with different prefixes: last prefix added keeps namespace, previous namespaces are set to None.
        - Same prefix with different namespaces: first namespace added is kept

    Parameters:
        files (list): Paths to files containing CIMXML graphs.
        schema_path (str): Path to the linkML file with the cim model.

    Returns:
        CIMDataset: All the graphs collected in a CIMDataset object. Each graph consist of:
            - The parsed triples minus the header triples
            - A metadata header which contains the header triples
            - A namespace manager
    """
    ds = CIMDataset()

    for file_path in files:
        try:
            graph = load_cimxml_graph(file_path, schema_path)
        except (CIMXMLParseError) as e:
            logger.error(e)
            continue

        try:
            header = CIMMetadataHeader.from_graph(graph)
        except ValueError as e:
            header = CIMMetadataHeader.empty()
            logger.error(f"{e}: Metadata header cannot be extracted. Graph given random id {str(header.subject)}.")
            
        named = ds.graph(URIRef(header.subject))

        for prefix, uri in graph.namespace_manager.namespaces():
            ds.namespace_manager.bind(prefix, uri)
            named.namespace_manager.bind(prefix, uri)

        for triple in graph:
            if triple in header.triples:
                continue
            named.add(triple)
        
        named.metadata_header = header
        
    return ds

def extract_subjects_by_object_type(graph: Graph, object_type: list[URIRef]) -> list[Node]: 
    """Extract subjects with predicate rdf:type and matching specified objects.

    Parameters:
        graph (Graph): Target for subject extraction.
        object_type (list[URIRef]): The object URIRefs to match. 

    Returns:
        list[Node]: The subjects from the matching triples.
    """
    subject_list = []
    for s, p, o in graph.triples((None, RDF.type, None)): 
        if o in object_type: 
            subject_list.append(s) 
    return subject_list


def group_subjects_by_type(graph: Graph, skip_subjects: list[Node]=[]) -> dict[str, list[Node]]:
    """Group subjects with prediacte rdf:type by object.
    
    Parameters:
        graph (Graph): Target for grouping.
        skip_subjects (list[Node]): Optional list of subjects to skip grouping.

    Returns:
        dict[str, list[Node]]: All the objects with lists of subjects belonging to that type.
    """
    groups: dict[str, list[Node]] = {}

    nm = graph.namespace_manager

    for s in graph.subjects():
        if s in skip_subjects:
            continue

        t = next(graph.objects(s, RDF.type), None)
        if t is None:
            t_qname = "ErrorMissingType"
        else:
            t_qname = nm.normalizeUri(str(t))

        groups.setdefault(t_qname, []).append(s)

    return groups


if __name__ == "__main__":
    print("utilities for cimxml")