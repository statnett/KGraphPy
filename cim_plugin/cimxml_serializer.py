import uuid
from rdflib.serializer import Serializer
from rdflib.graph import Graph
from rdflib.term import URIRef, Identifier, Literal, Node
from rdflib.namespace import RDF, DCAT
from xml.sax.saxutils import quoteattr, escape
import logging
from typing import IO, Any, Generator, Tuple, Dict, Optional
from cim_plugin.utilities import extract_subjects_by_object_type, group_subjects_by_type, _extract_uuid_from_urn
from cim_plugin.namespaces import MD
from cim_plugin.qualifiers import UnderscoreQualifier, URNQualifier, NamespaceQualifier, CIMQualifierResolver

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

    def _init_qualifier_resolver(self, qualifier_name: str|None):
        name = (qualifier_name or "underscore").lower()
        qualifier_cls = QUALIFIER_MAP.get(name)
        if qualifier_cls is None:
            raise ValueError(f"Unknown qualifier: {qualifier_name}")
        self.qualifier_resolver = CIMQualifierResolver(qualifier_cls())

    def __bindings(self) -> Generator[Tuple[str, URIRef], None, None]:
        store = self.store
        nm = store.namespace_manager
        bindings: Dict[str, URIRef] = {}

        for predicate in set(store.predicates()):
            prefix, namespace, name = nm.compute_qname_strict(str(predicate))
            bindings[prefix] = URIRef(namespace)

        for prefix, namespace in bindings.items():
            yield prefix, namespace

    def serialize(self, stream: IO[bytes], base: Optional[str] = None, encoding: Optional[str] = None, **kwargs: Any) -> None:
        self.__stream = stream
        qualifier_name = kwargs.pop("qualifier", None)
        self._init_qualifier_resolver(qualifier_name)
        encoding = encoding or self.encoding
        self.write = write = lambda uni: stream.write(uni.encode(encoding, "replace"))

        write('<?xml version="1.0" encoding="%s"?>\n' % self.encoding)

        # Write xmlns:prefix="namespace" for all namespaces used
        # Namespaces not used will not be written
        write("<rdf:RDF\n")

        bindings = list(self.__bindings())
        bindings.sort()

        for prefix, namespace in bindings:
            if prefix:
                write('    xmlns:%s="%s"\n' % (prefix, namespace))
            else:
                write('    xmlns="%s"\n' % namespace)
        write("    >\n")

        # Write metadata header
        meta = extract_subjects_by_object_type(self.store, METADATA_OBJECTS)
        if len(meta) > 1:
            logger.error("Multiple metadata headers detected.")
        
        if meta:
            self.subject(meta[0], depth=1)
        else:
            logger.error("No metadata header detected.")

        write("\n")

        # Sort by class and write triples by subject
        groups = group_subjects_by_type(self.store, skip_subjects=meta)
        sorted_types = sorted(groups.keys())

        for t in sorted_types:
            for s in sorted(groups[t], key=_subject_sort_key):
                self.subject(s, depth=1)
        
        write("</rdf:RDF>\n")

    def subject(self, subject: Node, depth: int = 1) -> None:
        if subject in self.__serialized:
            return
        
        self.__serialized[subject] = 1

        nm = self.store.namespace_manager
        write = self.write
        indent = "  " * depth
        
        if not isinstance(subject, URIRef):
            logger.error(f"Invalid subject (not a URIRef): {subject}.")
            write(f"{indent}<Error rdf:about={subject}>\n") 
            write(f"{indent} <message>Subject is not a URIRef: {subject}</message>\n") 
            write(f"{indent}</Error>\n") 
            return
        
        uri = quoteattr(self.qualifier_resolver.convert_about(subject))
        subject_type = next(self.store.objects(subject, RDF.type), None)

        if subject_type is None:
            logger.error(f"No rdf:type found for {subject}.")
            subject_type_qname = "ErrorMissingType"
            
            write(f"{indent}<{subject_type_qname} rdf:about={uri}>\n") 
            write(f"{indent} <message>No rdf:type found for {subject}</message>\n") 
            write(f"{indent}</{subject_type_qname}>\n") 
            return
                               
        subject_type_qname = nm.normalizeUri(str(subject_type))

        # Write the triple subject with rdf:type
        write(f"{indent}<{subject_type_qname} rdf:about={uri}>\n")

        # Sort and write predicates and objects
        preds = [(p, o) for p, o in self.store.predicate_objects(subject) if p != RDF.type]
        preds.sort(key=lambda po: nm.normalizeUri(str(po[0])))

        if (subject, None, None) in self.store:
            for predicate, obj in preds:
                self.predicate(predicate, obj, depth + 1)
        
        write(f"{indent}</{subject_type_qname}>\n")
                

    def predicate(self, predicate: Node, obj: Node, depth: int = 1) -> None:
        """Write predicate and object in CIMXML format.
        
        Parameters:
            predicate (Node): The predicate to be written.
            obj (Node): The object to be written.
            depth (int): Indentation size.
        """
        write = self.write
        indent = "  " * depth

        try:
            qname = self.store.namespace_manager.qname_strict(str(predicate))
        except (KeyError, ValueError):
            logger.error(f"Predicate {str(predicate)} not a valid predicate.")
            qname = f"MALFORMED_{str(predicate)}"

        if isinstance(obj, Literal):
            obj_text = escape(obj, ESCAPE_ENTITIES)
            write(f"{indent}<{qname}>{obj_text}</{qname}>\n")

        elif isinstance(obj, URIRef):
            relativized_obj = quoteattr(self.qualifier_resolver.convert_resource(obj))
            write(f"{indent}<{qname} rdf:resource={relativized_obj}/>\n")

        else:
            logger.error("Invalid object detected.")
            write(f"{indent}<{qname}>INVALID OBJECT</{qname}>\n")


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
    print("Serializer class")