"""Serializer for writing graphs to CIMXML files."""

from rdflib.serializer import Serializer
from rdflib.graph import Graph
from rdflib.term import URIRef, Literal, Node, BNode
from rdflib.namespace import RDF, DCAT
from xml.sax.saxutils import quoteattr, escape
import logging
from typing import IO, Any, Optional
from kgraphpy.utilities import _extract_uuid_from_urn, create_header_attribute
from kgraphpy.namespaces import MD, DCAT_EXT, collect_specific_namespaces
from kgraphpy.qualifiers import UnderscoreQualifier, URNQualifier, NamespaceQualifier, CIMQualifierResolver, is_uuid_qualified
from kgraphpy.header import CIMMetadataHeader
from kgraphpy.rdf_id_selection import find_rdf_id_or_about
from functools import lru_cache
from typing import Callable, cast

logger = logging.getLogger('cimxml_logger')


from rdflib.plugins.serializers.xmlwriter import ESCAPE_ENTITIES

METADATA_OBJECTS = [MD.FullModel, DCAT.Dataset]
QUALIFIER_MAP = {"underscore": UnderscoreQualifier, "urn": URNQualifier, "namespace": NamespaceQualifier}


class CIMXMLSerializer(Serializer):
    """CIMXML RDF graph serializer."""

    write: Callable[[str], int] | None = None
    qualifier_resolver: CIMQualifierResolver | None = None
    _used_namespaces: list[tuple[str, URIRef]] | None = None    # List of prefix, namespaces ordered alphabetically by prefix
    _namespace_lookup: list[tuple[str, str]] | None = None    # List of namespaces, prefix ordered longest-first by namespace string. Used for lookup when collecting namespaces.

    def __init__(self, store: Graph, **kwargs):
        super().__init__(store)

    def _init_qualifier_resolver(self, qualifier_name: str|None) -> None:
        """Initialize the qualifier resolver based on the provided qualifier name.
        
        Accepted names are "underscore", "urn", and "namespace". If None, defaults to "underscore".

        Parameters:
            qualifier_name (str|None): The name of the qualifier to use.
        """
        name = (qualifier_name or "underscore").lower()
        qualifier_cls = QUALIFIER_MAP.get(name)
        if qualifier_cls is None:
            raise ValueError(f"Unknown qualifier: {qualifier_name}")
        self.qualifier_resolver = CIMQualifierResolver(qualifier_cls())

    def _ensure_header(self) -> CIMMetadataHeader:
        """Ensure that the graph has a metadata header and return it."""
        header = getattr(self.store, "metadata_header", None)
        if header is None:
            header = create_header_attribute(self.store)
            setattr(self.store, "metadata_header", header)
        return header

    def _collect_used_namespaces(self) -> list[tuple[str, URIRef]]:
        """Collect namespaces used by the header and/or the data.
        
        The namespace is only collected if it is both registered in the namespace_manager 
        and present in the triples of either header or data.
        With namespace conflicts the namespace of the data is preserved.

        Returns:
            list[tuple[str, URIRef]]: Sorted list of tuples with prefix, namespace.
        """
        namespaces: dict[str, URIRef] = {}

        # --- Header namespaces ---
        header = getattr(self.store, "metadata_header", None)
        if header is not None:
            header_ns = collect_specific_namespaces(
                header.graph.triples((None, None, None)),
                header.graph.namespace_manager
            )
            namespaces.update(header_ns)

        # --- Data namespaces ---
        data_ns = collect_specific_namespaces(
            self.store.triples((None, None, None)),
            self.store.namespace_manager
        )
        namespaces.update(data_ns)

        return sorted(namespaces.items())

    def _build_subject_index(self, skip_subjects: set[URIRef]) -> dict[str, set[Node]]:
        """Build an index of subjects grouped by their rdf:type object, sorted by qname of the type.

        Subjects with missing or invalid rdf:type are grouped under the key "ErrorMissingType", to prevent loss of data.

        Parameters:
            skip_subjects (set[URIRef]): A set of subjects to skip when building the index.

        Returns:
            dict[str, set[Node]]: A dictionary where keys are qnames of rdf:type objects and values are sets of subjects with that rdf:type.
        """
        groups: dict[str, set[Node]] = {}

        nm = self.store.namespace_manager

        for s, _, t in self.store.triples((None, RDF.type, None)):
            if s in skip_subjects:
                continue

            t_qname = nm.normalizeUri(str(t)) if isinstance(t, URIRef) else str(t)
        
            groups.setdefault(t_qname, set()).add(s)

        all_subjects: set[Node] = set(self.store.subjects())
        typed_subjects: set[Node] = set().union(*groups.values()) if groups else set()
        missing: set[Node] = all_subjects - typed_subjects - skip_subjects
        if missing:
            groups["ErrorMissingType"] = missing

        return groups


    def serialize(self, stream: IO[bytes], base: Optional[str] = None, encoding: Optional[str] = None, **kwargs: Any) -> None:
        """Serialize graph to CIMXML format.
        
        Parameters:
            stream (IO[bytes]): The stream to serialize to.
            base (str): Not used. Inherited from rdflib.Serializer.
            encoding (str): The encoding used for the stream.
            qualifier (str): From **kwargs. Specifies the qualifier for uuids.
        """
        encoding = encoding or self.encoding
        self.write = write = lambda txt: stream.write(txt.encode(encoding, "replace"))
        
        header = self._ensure_header()
        
        qualifier_name = kwargs.pop("qualifier", None)
        self._init_qualifier_resolver(qualifier_name)
        
        write(f'<?xml version="1.0" encoding="{self.encoding}"?>\n')

        # Write xmlns:prefix="namespace" for all namespaces used
        # Namespaces not used will not be written
        write("<rdf:RDF\n")
        
        used_namespaces = self._collect_used_namespaces()
        self._used_namespaces = used_namespaces
        self._namespace_lookup = sorted([(str(ns), prefix) for prefix, ns in used_namespaces], key=lambda item: len(item[0]), reverse=True)

        for prefix, namespace in used_namespaces:
            if prefix:
                write(f'    xmlns:{prefix}="{namespace}"\n')
            else:
                write(f'    xmlns="{namespace}"\n')
        write("    >\n")

        self.write_header(header, depth=1)
        write("\n")

        # Sort by class and write triples by subject
        skip_subjects = {header.subject}
        groups = self._build_subject_index(skip_subjects)

        nm = self.store.namespace_manager
        sorted_types = sorted(groups.keys(), key=lambda t: nm.normalizeUri(str(t)))

        for t in sorted_types:
            for s in sorted(groups[t], key=_subject_sort_key):
                self.subject(s, depth=1)
        
        write("</rdf:RDF>\n")


    def write_header(self, header: CIMMetadataHeader, depth: int = 1) -> None:
        """Write the CIM metadata header in CIMXML format.
        
        Always uses URNQualifier for the subject uuid and any object uuids.
        
        Parameters:
            header (CIMMetadataHeader): The header with the triples to be written.
            depth (int): The size of indentation.
        """

        write = cast(Callable[[str], int], self.write)
        nm = self.store.namespace_manager
        indent = "  " * depth

        subject = header.subject
        try:
            types = header.header_type
            # dcat:Dataset is prioritized if multiple header types are present
            if DCAT_EXT.Dataset in types:
                subject_type = DCAT_EXT.Dataset
            else:
                subject_type = next(iter(types))
        except ValueError:
            logger.error(f"Header type missing. dcat:Dataset used as default.")
            subject_type = DCAT_EXT.Dataset

        # --- Temporarily override qualifier strategy ---
        assert self.qualifier_resolver is not None  # For type checker
        original_strategy = self.qualifier_resolver.output
        self.qualifier_resolver.output = URNQualifier()

        try:
            uri = quoteattr(self.qualifier_resolver.convert_to_special_qualifier(subject))
            subject_type_qname = nm.normalizeUri(str(subject_type))

            body_triples = [(p, o) for (_, p, o) in header.triples if not (p == RDF.type and o == subject_type)]
            body_triples.sort(key=lambda po: nm.normalizeUri(str(po[0])))

            write(f"{indent}<{subject_type_qname} rdf:about={uri}")
            
            if not body_triples:
                write("/>\n")
            else:
                write(">\n")
                for p, o in body_triples:
                    use_qualifier = is_uuid_qualified(self.qualifier_resolver, o)
                    self.predicate(p, o, depth + 1, use_qualifier=use_qualifier)

                write(f"{indent}</{subject_type_qname}>\n")

        finally:
            # --- Restore original qualifier strategy ---
            self.qualifier_resolver.output = original_strategy


    def subject(self, subject: Node, depth: int = 1) -> None:
        """Write subject with predicates and objects.
        
        Parameters:
            subject (Node): The subject to be written.
            depth (int): Indentation size.
        """
        nm = self.store.namespace_manager
        write = cast(Callable[[str], int], self.write)
        indent = "  " * depth
        
        header = self._ensure_header()
        
        # Dealing with malformed subjects
        if isinstance(subject, BNode) and subject in header.reachable_nodes:
            # Header blank nodes are dealt with by the header object
            return
        
        types = list(self.store.objects(subject, RDF.type))

        if not types:
            # logger.error(f"No rdf:type triple detected for {subject}.")
            self._write_untyped_subject(subject, depth)
            return
            
        # if len(types) > 1:
            # logger.error(f"Multiple rdf:type triples detected for {subject}.")
                
        subject_type = types[0] # In the triple this is the object, it specifies the rdf:type for the subject. If multiple, the first is arbitrarily chosen.
        # if not isinstance(subject_type, URIRef):
            # logger.error(f"The rdf:type object is not a uri: {subject_type}")
            
        # Shape and write the subject line
        rdf_keyword = find_rdf_id_or_about(header.profiles, str(subject_type))

        assert self.qualifier_resolver is not None  # For type checker
        if isinstance(subject, URIRef):
            if rdf_keyword == "ID":
                raw_uri = self.qualifier_resolver.convert_to_special_qualifier(subject)
            else:
                raw_uri = self.qualifier_resolver.convert_to_default_qualifier(subject)
        else:
            # logger.error(f"Subject is not a URIRef: {subject}")
            raw_uri = str(subject)

        uri = quoteattr(raw_uri)
        subject_type_qname = nm.normalizeUri(str(subject_type))

        write(f"{indent}<{subject_type_qname} rdf:{rdf_keyword}={uri}>\n")

        # Sort and write predicates and objects
        preds = [(p, o) for p, o in self.store.predicate_objects(subject) if not (p == RDF.type and o == subject_type)]
        preds.sort(key=lambda po: nm.normalizeUri(str(po[0])))

        for predicate, obj in preds:
            use_qualifier = is_uuid_qualified(self.qualifier_resolver, obj)
            self.predicate(predicate, obj, depth + 1, use_qualifier=use_qualifier)
        
        write(f"{indent}</{subject_type_qname}>\n")
  

    def predicate(self, predicate: Node, obj: Node, depth: int = 1, use_qualifier: bool = True) -> None:
        """Write predicate and object in CIMXML format.
        
        Parameters:
            predicate (Node): The predicate to be written.
            obj (Node): The object to be written.
            depth (int): Indentation size.
        """
        write = cast(Callable[[str], int], self.write)
        indent = "  " * depth

        qname = self._resolve_qname(str(predicate))

        # Write predicate and object
        if isinstance(obj, Literal):
            obj_text = escape(obj, ESCAPE_ENTITIES)
            write(f"{indent}<{qname}>{obj_text}</{qname}>\n")

        elif isinstance(obj, URIRef):
            if use_qualifier:
                assert self.qualifier_resolver is not None  # For type checker
                relativized_obj = quoteattr(self.qualifier_resolver.convert_to_default_qualifier(obj))
            else:
                relativized_obj = quoteattr(str(obj))

            write(f"{indent}<{qname} rdf:resource={relativized_obj}/>\n")

        else:
            # logger.error(f"Invalid object detected: {obj}.")
            write(f"{indent}<{qname}>{obj}</{qname}>\n")

    def _write_untyped_subject(self, subject: Node, depth: int) -> None:
        """Write subjects without rdf:type triple.

        The triples are written with an rdf:Description triple first, with all the predicates and objects listed below.
        
        Parameters:
            subject (Node): The untyped subject.
            depth (int): Size of indentation.

        Raises:
            AssertionError: If the qualifier resolver is not initialized.
        """
        write = cast(Callable[[str], int], self.write)
        indent = "  " * depth
    
        if any(self.store.objects(subject, RDF.type)):
            return

        assert self.qualifier_resolver is not None  # For type checker

        write(f"{indent}<rdf:Description rdf:about={quoteattr(str(subject))}>\n")

        # Write all predicates/objects so the triples are not lost.
        for p, o in self.store.predicate_objects(subject):
            use_qualifier = is_uuid_qualified(self.qualifier_resolver, o)
            self.predicate(p, o, depth + 1, use_qualifier=use_qualifier)

        write(f"{indent}</rdf:Description>\n")


    @lru_cache(maxsize=5000)
    def _resolve_qname(self, uri: str) -> str:
        """Resolve a URI to a QName using the namespaces collected in ._namespace_lookup.

        Parameters:
            uri (str): The URI to resolve.

        Returns:
            str: The resolved QName, or the original URI if no namespace matches.
        """
        if self._namespace_lookup:
            for ns_str, prefix in self._namespace_lookup:
                if uri.startswith(ns_str):
                    local_part = uri[len(ns_str):]
                    return f"{prefix}:{local_part}" if prefix else str(uri)
        return str(uri)
    

def _subject_sort_key(uri: Node) -> tuple[int, str]:
    """Create sort key for subject nodes.

    Valid uuid is sorted before the invalid.

    Parameters:
        uri (Node): The subject uri to sort.

    Returns:
        tuple[int, str]: The integer showing validity priority and the uuid.
    """
    s = str(uri)
    try:
        return (0, str(_extract_uuid_from_urn(s)))
    except ValueError:
        return (1, str(s))
        


if __name__ == "__main__":
    print("CIMXML serializer class")