"""The class which handles metadata header triples and information about them."""

from pathlib import Path

from rdflib import Graph, Node, URIRef, RDF, BNode, Literal
from rdflib.namespace import DCTERMS, NamespaceManager
from cim_plugin.namespaces import MD, collect_specific_namespaces
from cim_plugin.header_conversion import convert_triple
from typing import Iterable, List, Tuple, Optional, Set
import logging
import uuid

logger = logging.getLogger("cimxml_logger")


class CIMMetadataHeader:
    """
    Represents the CIMXML metadata header extracted from a Graph.

    The header is defined as the subject that has rdf:type equal to one of the
    metadata object types (default: MD.FullModel, DCAT.Dataset).

    This class does NOT modify the graph. It simply extracts and stores the
    metadata triples so they can be inspected, edited, or serialized separately.
    """

    DEFAULT_METADATA_OBJECTS: Set[URIRef] = set()
    DEFAULT_PROFILE_PREDICATES: Set[URIRef] = {
        MD.Model.profile,
        DCTERMS.conformsTo,
    }

    def __init__(
        self,
        subject: Optional[URIRef] = None,
        graph: Optional[Graph] = None,
        metadata_objects: Optional[Iterable[URIRef]] = None,
        reachable_nodes: Optional[Set[Node]] = None,
        profile_predicates: Optional[Set[URIRef]] = None,
        profile: Optional[str] = None,
    ):
        if subject is None:
            subject = URIRef(f"urn:uuid:{uuid.uuid4()}")

        self.subject: URIRef = subject
        self.graph: Graph = graph if graph is not None else Graph()

        self.metadata_objects = (
            set(metadata_objects)
            if metadata_objects
            else set(self.DEFAULT_METADATA_OBJECTS)
        )

        self.reachable_nodes: Set[Node] = reachable_nodes or set()
        self.profile_predicates = profile_predicates or self.DEFAULT_PROFILE_PREDICATES

        self.profile: Optional[str] = profile or self.collect_profile()


    @classmethod
    def from_graph(
        cls, graph: Graph, metadata_objects: Optional[Iterable[URIRef]] = None
    ) -> "CIMMetadataHeader":
        """Extract the metadata header from a graph.

        Parameters:
            graph (Graph): The RDF graph containing CIMXML data.
            metadata_objects (Iterable[URIRef], optional): Override or extend the default metadata object types.

        Returns:
            CIMMetadataHeader: The extracted header.

        Raises:
            ValueError: If no header or multiple headers are found.
        """

        metadata_objects = (
            list(metadata_objects)
            if metadata_objects
            else list(cls.DEFAULT_METADATA_OBJECTS)
        )

        header_subjects = [
            s
            for (s, _, o) in graph.triples((None, RDF.type, None))
            if o in metadata_objects
        ]

        if not header_subjects:
            raise ValueError("No metadata header found in graph")

        if len(header_subjects) > 1:
            raise ValueError(f"Multiple metadata headers found: {header_subjects}")

        header_subject = header_subjects[0]

        final_subject, repaired_triples, reachable = cls._collect_header_triples(
            graph, header_subject
        )

        # Build a new graph for the header
        header_graph = Graph()
        for triple in repaired_triples:
            header_graph.add(triple)
        
        # Namespaces
        nm = NamespaceManager(header_graph, bind_namespaces="none")
        used_namespaces = collect_specific_namespaces(repaired_triples, graph.namespace_manager)
        for prefix, ns_uri in used_namespaces.items():
            nm.bind(prefix, ns_uri, override=True)
        
        header_graph.namespace_manager = nm

        return cls(
            subject=final_subject,
            graph=header_graph,
            metadata_objects=metadata_objects,
            reachable_nodes=reachable,
        )


    @classmethod
    def _collect_header_triples(
        cls, graph: Graph, header_subject: Node
    ) -> Tuple[URIRef, List[Tuple[Node, Node, Node]], Set[Node]]:
        """Collect all triples reachable from the header subject.

        If the subject is a blank node a repair will be attempted.
        See _repair_blank_header_subject for more information.

        Parameters:
            graph (Graph): The graph to collect the header triples from.
            header_subject (Node): A uri uuid which identifies the header subject.

        Returns:
            tuple[URIRef, list[tuple[Node, Node, Node]], set[Node]]: The repaired header subject, the triples and the blank nodes reachable from the header.
        """
        reachable: Set[Node] = set()
        queue: List[Node] = [header_subject]

        while queue:
            current = queue.pop()
            if current in reachable:
                continue

            reachable.add(current)

            for (_, _, obj) in graph.triples((current, None, None)):
                if isinstance(obj, BNode) and obj not in reachable:
                    queue.append(obj)

        collected: List[Tuple[Node, Node, Node]] = []
        for s in reachable:
            for triple in graph.triples((s, None, None)):
                collected.append(triple)

        if isinstance(header_subject, URIRef):
            final_subject = header_subject
        else:
            final_subject = cls._repair_blank_header_subject(graph, header_subject)

        repaired: List[Tuple[Node, Node, Node]] = []
        for (s, p, o) in collected:
            if isinstance(o, BNode):
                continue
            if isinstance(s, BNode):
                repaired.append((final_subject, p, o))
            else:
                repaired.append((s, p, o))

        return final_subject, repaired, reachable

    @classmethod
    def empty(
        cls,
        subject: Optional[URIRef] = None,
        metadata_objects: Optional[Iterable[URIRef]] = None,
        profile_predicates: Optional[Set[URIRef]] = None,
        profile: Optional[str] = None,
    ):
        """Creates an empty instance with optional attributes.
        
        Parameters:
            subject (URIRef): Subject used for all header triples. Should be a valid uuid.
            metadata_objects (URIRef): A custom rdf:type object. Default are md:FullModel and dcat:Dataset.
            profile_predicates (set[URIRef]): A custom predicate that holds the profile information. Default are md:Model.Profile and dcterms.conformsTo.
            profile (str): A custom profile.
        """
        g = Graph()
        return cls(
            subject=subject,
            graph=g,
            metadata_objects=metadata_objects,
            profile_predicates=profile_predicates,
            profile=profile,
        )
    

    @classmethod
    def from_manifest(cls, file_path: str|Path, graph_uri: URIRef|str, format: str="trig") -> "CIMMetadataHeader":
        """Creates a header from a manifest file.

        The manifest file can contain headers for multiple graphs. The correct header is found by the graph id.
        
        Parameters:
            file_path (str|Path): Path to manifest file. The file must be a valid RDF file.
            graph_uri (URIRef|str): The identifier of the graph.
            format (str): The format of the manifest file. Default is "trig".

        Raises:
            ValueError: If no header triples matching the graph_uri is found.

        Returns:
            CIMMetadataHeader: The new header.
        """
        graph_uri = URIRef(graph_uri)
        
        manifest_graph = Graph()
        manifest_graph.parse(source=file_path, format=format)
        
        header_graph = Graph()
        for triple in manifest_graph.triples((graph_uri, None, None)):
            header_graph.add(triple)
        
        if len(header_graph) == 0:
            raise ValueError(f"No header triples matching graph identifier {graph_uri} found in manifest file.")

        nm = NamespaceManager(header_graph, bind_namespaces="none")
        used_namespaces = collect_specific_namespaces(list(header_graph), manifest_graph.namespace_manager)
        for prefix, ns_uri in used_namespaces.items():
            nm.bind(prefix, ns_uri, override=True)
        
        header_graph.namespace_manager = nm

        return cls(subject=graph_uri, graph=header_graph)

    @staticmethod
    def _repair_blank_header_subject(graph: Graph, blank: Node) -> URIRef:
        """Repair header subject by turning it into a uuid uri.

        If DCTERMS.identifier is present, this is made the new header subject.
        With no other options, a random uuid4 will be generated.

        Parameters:
            graph (Graph): The graph to collect the identifier from.
            blank (Node): The subject blank node.

        Returns:
            URIRef: The new header subject uuid.
        """
        logger.error(
            f"Metadata header subject is a blank node ({blank}). "
            "Attempting to collect subject from dcterms:identifier."
        )

        for (_, _, identifier) in graph.triples((blank, DCTERMS.identifier, None)):
            if identifier and isinstance(identifier, Literal):
                return URIRef(f"urn:uuid:{identifier.toPython()}")

        new_id = uuid.uuid4()
        logger.error(
            f"No dcterms:identifier found for blank header subject. "
            f"Random UUID generated: {new_id}"
        )
        return URIRef(f"urn:uuid:{new_id}")


    @property
    def triples(self):
        """Convenient access to a list of all the triples in the header."""
        return list(self.graph.triples((None, None, None)))

    @property
    def header_type(self) -> Node:
        """The object node of the rdf:type triple.
        
        Raises:
            ValueError: If rdf:type is not found in any of the triples.

        Returns:
            Node: The object node.
        """
        headertypes = set()
        for (_, p, o) in self.graph.triples((self.subject, RDF.type, None)):
            if o in self.metadata_objects:
                headertypes.add(o)
        
        if len(headertypes) == 1:
            return headertypes.pop()
        elif len(headertypes) == 0:
            raise ValueError("No header type found in header.")
        else:
            raise ValueError("Multiple header types found in header.")


    def collect_profile(self) -> Optional[str]:
        """Collect the profile of a graph from the triple with predicate in self.profile_predicates.

        Returns:
            str: The profile or None if no profile is found.
        """
        for (_, p, o) in self.graph.triples((self.subject, None, None)):
            if p in self.profile_predicates:
                if isinstance(o, Literal):
                    return str(o.value)
                elif isinstance(o, URIRef):
                    return str(o)
        return None


    def set_subject(self, new_subject: URIRef) -> None:
        """Set a new subject uuid for the headers.
        
        All the triples will be rewritten with the new subject.

        Parameters:
        new_subject (URIRef): The new subject value. Should be a valid uuid.
        """
        old_subject = self.subject
        self.subject = new_subject

        new_graph = Graph()
        new_graph.namespace_manager = self.graph.namespace_manager

        for (s, p, o) in self.graph:
            if s == old_subject:
                new_graph.add((new_subject, p, o))
            else:
                new_graph.add((s, p, o))

        self.graph = new_graph


    def add_triple(self, predicate: Node, obj: Node):
        """Add a metadata triple with the fixed subject.
        Untested method.
        """
        self.graph.add((self.subject, predicate, obj))

    def remove_triple(self, predicate: Node, obj: Optional[Node] = None):
        """Remove metadata triples matching predicate (and optionally object).
        Untested method.
        """
        if obj is None:
            for (_, _, o) in list(self.graph.triples((self.subject, predicate, None))):
                self.graph.remove((self.subject, predicate, o))
        else:
            self.graph.remove((self.subject, predicate, obj))


    def iter_predicates(self):
        """Iterate through the predicates in the header.
        Unteste method. May be removed.
        """
        for (_, p, o) in self.graph.triples((self.subject, None, None)):
            yield p, o


def create_header_attribute(graph: Graph) -> CIMMetadataHeader:
    """Create a header from a graph.
    
    The header will be extracted from the graph. 
    If there are no header triples in the graph, an empty header object will be created with a random subject uuid.

    Parameters:
        graph (Graph or CIMGraph): The graph to extract the header from.

    Returns:
        CIMMetadataHeader: The header object.
    """
    try:
        header = CIMMetadataHeader.from_graph(graph)
    except ValueError as e:
        header = CIMMetadataHeader.empty()
        logger.error(f"{e}: Random id generated for graph: {str(header.subject)}")

    return header


if __name__ == "__main__":
    print("metadata header for cimxml graph")