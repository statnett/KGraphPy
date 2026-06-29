"""Register the parser and serializer and set the default metadata object types."""

from rdflib.plugin import register
from rdflib.parser import Parser
from rdflib.serializer import Serializer
from kgraphpy.header import CIMMetadataHeader
from kgraphpy.namespaces import MD
from rdflib.namespace import DCAT

# Register plugins
register(
    "cimxml",          # formatname
    Parser,            # plugin-type
    "kgraphpy.cimxml_parser",          # module path
    "CIMXMLParser"     # name of class
)
register(
    "cimxml",
    Serializer,
    "kgraphpy.cimxml_serializer",
    "CIMXMLSerializer",
)
register(
    "cimtrig",
    Serializer,
    "kgraphpy.cimtrig_serializer",
    "CIMTrigSerializer",
)


# Set the default metadata object types for CIM
CIMMetadataHeader.DEFAULT_METADATA_OBJECTS = {
    MD.FullModel,
    DCAT.Dataset
}

