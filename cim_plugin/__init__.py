from rdflib.plugin import register
from rdflib.parser import Parser
from rdflib.serializer import Serializer
from cim_plugin.header import CIMMetadataHeader
from cim_plugin.namespaces import MD
from rdflib.namespace import DCAT

# Register plugins
register(
    "cimxml",          # formatname
    Parser,            # plugin-type
    "cim_plugin.cimxml_parser",          # module path
    "CIMXMLParser"     # name of class
)
register(
    "cimxml",
    Serializer,
    "cim_plugin.cimxml_serializer",
    "CIMXMLSerializer",
)


# Set the default metadata object types for CIM
CIMMetadataHeader.DEFAULT_METADATA_OBJECTS = {
    MD.FullModel,
    DCAT.Dataset
}

