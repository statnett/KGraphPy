"""Various utility functions."""

import uuid
import re
from rdflib import Graph, URIRef, Node, Dataset
from rdflib.namespace import RDF
from rdflib.exceptions import ParserError
import logging
from xml.sax import SAXParseException
from cim_plugin.exceptions import CIMXMLParseError
from cim_plugin.graph import CIMDataset, CIMGraph
from cim_plugin.header import create_header_attribute
from cim_plugin.processor import CIMProcessor
from typing import Union, Iterable
from pathlib import Path

logger = logging.getLogger('cimxml_logger')


UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}"
)

def extract_uuid(uri: str) -> str | None:
    """Extract uuid from string using regex.
    
    Parameters:
        uri (str): The string to extract uuid from.

    Returns:
        str|None: The uuid if there is one, else None.
    """
    m = UUID_RE.search(uri)
    return m.group(0).lower() if m else None

# Used in the serializer. Should it be replaced by extract_uuid?
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

def load_cimxml_graph(file_path: str|Path) -> CIMGraph:
    """Load one CIMXML file to Graph.
    
    Parameters:
        file_path (str): Path to CIMXML file.

    Raises:
        CIMXMLParseError: If errors in the parsing process is raised.

    Returns:
        CIMGraph: The graph as a CIMGraph object.
    """
    try:
        g = CIMGraph() 
        g.parse(file_path, format="cimxml") 
        return g
    except (FileNotFoundError, ParserError, SAXParseException, ValueError) as e:
        raise CIMXMLParseError(file_path, e) from e


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


def load_graphs_from_trig(filepath: str|Path) -> list[CIMProcessor]:
    """Load graphs from trig file into individual CIMProcessor objects.
    
    Parameters:
        file_path (str|Path): Path to trig file.

    Returns:
        list[CIMProcessor]: List of CIMProcessor objects.
    """
    processors: list[CIMProcessor] = []
    
    ds = Dataset()
    ds.parse(filepath, format="trig")

    for ctx in ds.graphs():
        if "default" in ctx.identifier and len(ctx) == 0:
            continue
        cim = CIMGraph(identifier=ctx.identifier)
        for prefix, uri in ctx.namespace_manager.namespaces():
            cim.namespace_manager.bind(prefix, uri)
        cim += ctx

        processor = CIMProcessor(cim)
        processors.append(processor)

    return processors


def load_graphs_from_cimxml(files: Union[str, Path, Iterable[Union[str, Path]]]) -> list[CIMProcessor]:
    """Load graphs from one or more CIMXML files into a list of CIMProcessor objects.

    The graph identifiers is extracted from the RDF.type triple of each graph, or randomly generated as fallback.
    
    Parameters:
        files (Union[str, Path, Iterable[Union[str, Path]]]): A filepath or a list of filepaths.

    Returns:
        list[CIMProcessor]: The graphs from the files stored in CIMProcessor objects.
    """
    processors: list[CIMProcessor] = []
    
    if isinstance(files, (str, Path)):
        files = [files]

    for file_path in files:
        try:
            graph = load_cimxml_graph(file_path)
        except (CIMXMLParseError) as e:
            logger.error(e)
            continue

        header = create_header_attribute(graph)            
        cim = CIMGraph(identifier=URIRef(header.subject))
        for prefix, uri in graph.namespace_manager.namespaces():
            cim.namespace_manager.bind(prefix, uri)
        cim += graph

        processor = CIMProcessor(cim)
        processors.append(processor)

    return processors


# Replaced by load_graphs_from_cimxml
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
            graph = load_cimxml_graph(file_path) #, schema_path) # Used in the old version
        except (CIMXMLParseError) as e:
            logger.error(e)
            continue

        header = create_header_attribute(graph)            
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

# Not used anymore. Remove?
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


if __name__ == "__main__":
    print("Utilities for CIM.")