from rdflib.plugin import register
from rdflib.parser import Parser


register(
    "cimxml",          # formatname
    Parser,            # plugin-type
    "cim_plugin.cimxml_parser",          # module path
    "CIMXMLParser"     # name of class
)
