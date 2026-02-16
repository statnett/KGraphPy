import uuid
from rdflib.serializer import Serializer
from rdflib.graph import Graph
from rdflib.term import URIRef, Identifier, Literal, Node, BNode
from rdflib.namespace import RDF, DCAT
from xml.sax.saxutils import quoteattr, escape
import logging
# from cim_plugin import header
from typing import IO, Any, Generator, Tuple, Dict, Optional
from cim_plugin.utilities import extract_subjects_by_object_type, group_subjects_by_type, _extract_uuid_from_urn, create_header_attribute
from cim_plugin.namespaces import MD
from cim_plugin.qualifiers import UnderscoreQualifier, URNQualifier, NamespaceQualifier, CIMQualifierResolver
from cim_plugin.header import CIMMetadataHeader
from cim_plugin.graph import CIMGraph
from cim_plugin.rdf_id_selection import find_rdf_id_or_about

logger = logging.getLogger('cimxml_logger')


from rdflib.plugins.serializers.xmlwriter import ESCAPE_ENTITIES

METADATA_OBJECTS = [MD.FullModel, DCAT.Dataset]
QUALIFIER_MAP = {"underscore": UnderscoreQualifier, "urn": URNQualifier, "namespace": NamespaceQualifier}


class CIMXMLSerializer(Serializer):
    """CIMXML RDF graph serializer."""

    def __init__(self, store: Graph, **kwargs):
        super(CIMXMLSerializer, self).__init__(store)

        self.__serialized: Dict[Node, int] = {}
        self._stream = None

    def _init_qualifier_resolver(self, qualifier_name: str|None) -> None:
        name = (qualifier_name or "underscore").lower()
        qualifier_cls = QUALIFIER_MAP.get(name)
        if qualifier_cls is None:
            raise ValueError(f"Unknown qualifier: {qualifier_name}")
        self.qualifier_resolver = CIMQualifierResolver(qualifier_cls())

    def _ensure_header(self) -> CIMMetadataHeader:
        header = getattr(self.store, "metadata_header", None)
        if header is None:
            header = create_header_attribute(self.store)
            setattr(self.store, "metadata_header", header)
        return header

    def _collect_used_namespaces(self) -> list[tuple[str, URIRef]]:
        nm = self.store.namespace_manager
        namespaces: dict[str, URIRef] = {}

        def add_uri(uri: URIRef|Node):
            # Skip URNs explicitly (optional but safe)
            if str(uri).startswith("urn:"):
                return
            try:
                prefix, ns, _ = nm.compute_qname_strict(str(uri))
                namespaces[prefix] = URIRef(ns)
            except ValueError:
                pass

        # --- 1. Header namespaces ---
        header = getattr(self.store, "metadata_header", None)
        if header is not None:
            add_uri(header.subject)
            add_uri(header.main_type)
            for _, p, o in header.triples:
                add_uri(p)
                if isinstance(o, URIRef):
                    add_uri(o)

        # --- 2. Data namespaces ---
        for s, p, o in self.store:
            if isinstance(s, URIRef):
                add_uri(s)
            add_uri(p)
            if isinstance(o, URIRef):
                add_uri(o)

        # Convert to sorted list
        return sorted(namespaces.items())

    def serialize(self, stream: IO[bytes], base: Optional[str] = None, encoding: Optional[str] = None, **kwargs: Any) -> None:
        self.__stream = stream
        header = self._ensure_header()
        
        qualifier_name = kwargs.pop("qualifier", None)
        self._init_qualifier_resolver(qualifier_name)
        encoding = encoding or self.encoding
        self.write = write = lambda uni: stream.write(uni.encode(encoding, "replace"))

        write('<?xml version="1.0" encoding="%s"?>\n' % self.encoding)

        # Write xmlns:prefix="namespace" for all namespaces used
        # Namespaces not used will not be written
        write("<rdf:RDF\n")

        bindings = self._collect_used_namespaces()
        
        for prefix, namespace in bindings:
            if prefix:
                write('    xmlns:%s="%s"\n' % (prefix, namespace))
            else:
                write('    xmlns="%s"\n' % namespace)
        write("    >\n")

        self.write_header(header, depth=1)
        write("\n")

        # Sort by class and write triples by subject
        groups = group_subjects_by_type(self.store, skip_subjects=[header.subject])
        sorted_types = sorted(groups.keys())

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

        write = self.write
        nm = self.store.namespace_manager
        indent = "  " * depth

        subject = header.subject
        subject_type = header.main_type

        # --- Temporarily override qualifier strategy ---
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
        if subject in self.__serialized:
            return
        
        self.__serialized[subject] = 1

        nm = self.store.namespace_manager
        write = self.write
        indent = "  " * depth
        header = self.store.metadata_header # pyright: ignore[reportAttributeAccessIssue]
        
        # Dealing with malformed subjects
        if not isinstance(subject, URIRef):
            if isinstance(subject, BNode) and subject in header.reachable_nodes:
                # Header blank nodes are dealt with by the header object
                return
            else:
                self._write_malformed_subject(subject, f"Subject is not a URIRef: {subject}", depth)
                return
        
        types = list(self.store.objects(subject, RDF.type))

        if len(types) == 0:
            self._write_malformed_subject(subject, f"No rdf:type found for {subject}", depth)
            return
        
        if len(types) > 1: 
            self._write_malformed_subject(subject, f"Multiple rdf:type values found for {subject}: {types}", depth ) 
            return
        
        subject_type = types[0] # In the triple this is the object, it specifies the rdf:type for the subject
        if not isinstance(subject_type, URIRef):
            self._write_malformed_subject(subject, f"The rdf:type object is not a uri: {subject_type}", depth)
            return

        # Shape and write the subject line
        rdf_keyword = find_rdf_id_or_about(header.profile, str(subject_type))

        if rdf_keyword == "ID":
            raw_uri = self.qualifier_resolver.convert_to_special_qualifier(subject)
        else:
            raw_uri = self.qualifier_resolver.convert_to_default_qualifier(subject)

        uri = quoteattr(raw_uri)
        subject_type_qname = nm.normalizeUri(str(subject_type))

        write(f"{indent}<{subject_type_qname} rdf:{rdf_keyword}={uri}>\n")

        # Sort and write predicates and objects
        preds = [(p, o) for p, o in self.store.predicate_objects(subject) if p != RDF.type]
        preds.sort(key=lambda po: nm.normalizeUri(str(po[0])))

        if (subject, None, None) in self.store:
            for predicate, obj in preds:
                self.predicate(predicate, obj, depth + 1)
        
        write(f"{indent}</{subject_type_qname}>\n")
                

    def predicate(self, predicate: Node, obj: Node, depth: int = 1, use_qualifier: bool = True) -> None:
        """Write predicate and object in CIMXML format.
        
        Parameters:
            predicate (Node): The predicate to be written.
            obj (Node): The object to be written.
            depth (int): Indentation size.
        """
        write = self.write
        indent = "  " * depth

        # Shape the predicate name to right format and deal with malformed predicates
        try:
            qname = self.store.namespace_manager.qname_strict(str(predicate))
        except (KeyError, ValueError):
            logger.error(f"Predicate {str(predicate)} not a valid predicate.")
            qname = f"MALFORMED_{str(predicate)}"

        # Write predicate and object
        if isinstance(obj, Literal):
            obj_text = escape(obj, ESCAPE_ENTITIES)
            write(f"{indent}<{qname}>{obj_text}</{qname}>\n")

        elif isinstance(obj, URIRef):
            if use_qualifier:
                relativized_obj = quoteattr(self.qualifier_resolver.convert_to_default_qualifier(obj))
            else:
                relativized_obj = quoteattr(str(obj))

            write(f"{indent}<{qname} rdf:resource={relativized_obj}/>\n")

        else:
            logger.error("Invalid object detected.")
            write(f"{indent}<{qname}>MALFORMED_{obj}</{qname}>\n")


    def _write_malformed_subject(self, subject: Node, message: str, depth: int) -> None:
        """Write triples with a malformed subject.

        - Marks subject as MALFORMED
        - Writes all predicates and object to the subject
        - Logs an error

        Parameters:
            subject (Node): The malformed subject.
            message (str): The message to write in the triple and send to log.
            depth (int): Size of indentation.
        """
        write = self.write
        indent = "  " * depth

        logger.error(message)

        # Open MALFORMED subject
        write(f"{indent}<MALFORMED rdf:about={quoteattr(str(subject))}>\n")
        write(f"{indent}  <message>{message}</message>\n")

        # Write all predicates/objects for debugging
        for p, o in self.store.predicate_objects(subject):
            self.predicate(p, o, depth + 1)

        # Close MALFORMED subject
        write(f"{indent}</MALFORMED>\n")


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

def is_uuid_qualified(resolver: CIMQualifierResolver, value: str|Node) -> bool:
    uri = str(value)
    return any(strategy.matches(uri) for strategy in resolver.strategies)

if __name__ == "__main__":
    print("Serializer class")