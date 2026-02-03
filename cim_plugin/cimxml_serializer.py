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

from rdflib.plugins.serializers.xmlwriter import ESCAPE_ENTITIES

METADATA_OBJECTS = [MD.FullModel, DCAT.Dataset]

logger = logging.getLogger('cimxml_logger')

class CIMXMLSerializer(Serializer):
    """CIMXML RDF graph serializer."""

    def __init__(self, store: Graph):
        super(CIMXMLSerializer, self).__init__(store)

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
        self.__serialized: Dict[Node, int] = {}
        encoding = self.encoding
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
            # for s in sorted(groups[t], key=lambda uri: str(uri)):
            for s in sorted(groups[t], key=_subject_sort_key):
                self.subject(s, depth=1)
        # Write triples by subject
        # for subject in self.store.subjects():
        #     self.subject(subject, 1)

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
        
        uri = quoteattr(self.relativize(subject))
        subject_type = next(self.store.objects(subject, RDF.type), None)

        if subject_type is None:
            logger.error("No rdf:type found for {subject}.")
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
            # for predicate, object in self.store.predicate_objects(subject):
                # if predicate != RDF.type:
                self.predicate(predicate, obj, depth + 1)
        
        write(f"{indent}</{subject_type_qname}>\n")
                

    def predicate(self, predicate: Node, obj: Node, depth: int = 1) -> None:
        write = self.write
        indent = "  " * depth
        qname = self.store.namespace_manager.qname_strict(str(predicate))

        if isinstance(obj, Literal):
            obj_text = escape(obj, ESCAPE_ENTITIES)
            write(f"{indent}<{qname}>{obj_text}</{qname}>\n")

        else:
            if isinstance(obj, URIRef):
                relativized_obj = quoteattr(self.relativize(obj))
                write(f"{indent}<{qname} rdf:resource={relativized_obj}/>\n")


def _subject_sort_key(uri: str) -> tuple[int, uuid.UUID|str]:
    """
    Sort CIM subjects by UUID extracted from their URN.
    Falls back to the full URI string if not a UUID URN.
    """
    s = str(uri)
    try:
        return (0, _extract_uuid_from_urn(s))
    except ValueError:
        # Non-UUID subjects go last, sorted by full URI
        return (1, s)

if __name__ == "__main__":
    print("Serializer class")