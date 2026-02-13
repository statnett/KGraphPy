from rdflib import Graph, Node, URIRef, RDF, BNode, Literal
from rdflib.namespace import DCTERMS
from cim_plugin.namespaces import MD
from typing import Iterable, List, Tuple, Optional, Sequence, Set
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

    # Default metadata object types (can be extended by user)
    DEFAULT_METADATA_OBJECTS: List[URIRef] = []

    def __init__(
            self, 
            subject: Optional[URIRef] = None, 
            triples: Optional[Sequence[Tuple[Node, Node, Node]]] = None, 
            metadata_objects: Optional[Iterable[URIRef]] = None, 
            reachable_nodes: Optional[Set[Node]] = set(),
            profile: Optional[str] = None
    ):
        if subject is None:
            subject = URIRef(f"urn:uuid:{uuid.uuid4()}")

        self.subject: URIRef = subject
        self.triples: List[Tuple[Node, Node, Node]] = list(triples) if triples else []
        self.metadata_objects = list(metadata_objects) if metadata_objects else list(self.DEFAULT_METADATA_OBJECTS)
        self.reachable_nodes: Set[Node] = reachable_nodes if reachable_nodes else set()  # Blank nodes belonging to the header and therefore reachable through other header triples.
        self.profile: Optional[str] = profile or self.collect_profile()

    @classmethod
    def from_graph(cls, graph: Graph, metadata_objects: Optional[Iterable[URIRef]] = None) -> "CIMMetadataHeader":
        """
        Extract the metadata header from a graph.

        Parameters:
            graph (Graph): The RDF graph containing CIMXML data.
            metadata_objects (Iterable[URIRef], optional):
                Override or extend the default metadata object types.

        Returns:
            CIMMetadataHeader: The extracted header.

        Raises:
            ValueError: If no header or multiple headers are found.
        """
        metadata_objects = list(metadata_objects) if metadata_objects else list(cls.DEFAULT_METADATA_OBJECTS)

        header_subjects = [s for (s, _, o) in graph.triples((None, RDF.type, None)) if o in metadata_objects]

        if not header_subjects:
            raise ValueError("No metadata header found in graph")

        if len(header_subjects) > 1:
            raise ValueError(f"Multiple metadata headers found: {header_subjects}")

        header_subject = header_subjects[0]

        final_subject, repaired_triples, reachable = cls._collect_header_triples(graph, header_subject)
        return cls(final_subject, repaired_triples, metadata_objects, reachable)
    

    @classmethod
    def _collect_header_triples(cls, graph: Graph, header_subject: Node) -> Tuple[URIRef, List[Tuple[Node, Node, Node]], Set[Node]]:
        """
        Perform BFS reachability from the header subject, collect all reachable
        triples, and repair blank-node subjects.
        """

        # BFS reachability
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

        # Collect triples
        collected: List[Tuple[Node, Node, Node]] = []
        for s in reachable:
            for triple in graph.triples((s, None, None)):
                collected.append(triple)

        # Determine final subject URI
        if isinstance(header_subject, URIRef):
            final_subject = header_subject
        else:
            final_subject = cls._repair_blank_header_subject(graph, header_subject)

        # Rewrite blank-node subjects
        repaired: List[Tuple[Node, Node, Node]] = []
        for (s, p, o) in collected:
            if isinstance(s, BNode):
                repaired.append((final_subject, p, o))
            else:
                if isinstance(o, BNode):
                    continue

                repaired.append((s, p, o))

        return final_subject, repaired, reachable

    @classmethod
    def empty(cls, subject: Optional[URIRef] = None, metadata_objects: Iterable[URIRef]|None = None, profile: str|None = None):
        return cls(subject=subject, triples=[], metadata_objects=metadata_objects, profile=profile)


    @staticmethod
    def _repair_blank_header_subject(graph: Graph, blank: Node) -> URIRef:
        from rdflib.namespace import DCTERMS
        import uuid

        logger.warning(
            f"Metadata header subject is a blank node ({blank}). "
            "Attempting to repair."
        )

        # Try dct:identifier
        for (_, _, identifier) in graph.triples((blank, DCTERMS.identifier, None)):
            if identifier and isinstance(identifier, Literal): #identifier.toPython():
                return URIRef(f"urn:uuid:{identifier.toPython()}")

        # Fallback: generate UUID
        new_id = uuid.uuid4()
        logger.warning(
            f"No dct:identifier found for blank header subject. "
            f"Generated new UUID: {new_id}"
        )
        return URIRef(f"urn:uuid:{new_id}")

    @property
    def main_type(self) -> Node:
        for (_, p, o) in self.triples:
            if p == RDF.type and o in self.metadata_objects:
                return o
        raise ValueError("No metadata-object rdf:type found in header")


    def collect_profile(self) -> str|None:
        profile_predicates = [MD["Model.profile"], DCTERMS.conformsTo]
        for _, p, o in self.triples:
            if p in profile_predicates:
                if isinstance(o, Literal):
                    return o.value


    def set_subject(self, new_subject: URIRef):
        # Rewrite all triples that use the old subject
        old_subject = self.subject
        self.subject = new_subject

        new_triples = []
        for (s, p, o) in self.triples:
            if s == old_subject:
                new_triples.append((new_subject, p, o))
            else:
                new_triples.append((s, p, o))

        self.triples = new_triples


    def iter_predicates(self):
        """Yield (predicate, object) pairs for writing."""
        for _, p, o in self.triples:
            yield p, o


    def get_types(self) -> List[Node]:
        """Return all rdf:type values for the header subject."""
        return [o for (_, p, o) in self.triples if p == RDF.type]


    def add_triple(self, predicate: Node, obj: Node):
        """Add a metadata triple."""
        self.triples.append((self.subject, predicate, obj))


    def remove_triple(self, predicate: Node, obj: Optional[Node] = None):
        """Remove metadata triples matching predicate (and optionally object)."""
        self.triples = [
            (s, p, o)
            for (s, p, o) in self.triples
            if not (p == predicate and (obj is None or o == obj))
        ]


    def to_triples(self) -> List[Tuple[Node, Node, Node]]:
        """Return all metadata triples."""
        return list(self.triples)
                


if __name__ == "__main__":
    print("metadata header for cimxml graph")