from rdflib.plugin import register
from rdflib.parser import Parser
from rdflib.serializer import Serializer


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
