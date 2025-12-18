from rdflib.graph import Graph
from rdflib.plugin import register
from rdflib.parser import Parser
from rdflib import URIRef, Literal
import cim_plugin.cimxml

register(
    "cimxml",          # formatnavn du ønsker
    Parser,            # plugin-type
    "cim_plugin.cimxml",          # modulnavn (eller modulsti)
    "CIMXMLParser"     # klassenavn
)
from rdflib.plugin import plugins

def check_plugin_registered(name: str) -> None:
    print("Registrerte parser-plugins:") 
    for p in plugins(None, Parser): 
        if name in p.name:
            print(" -", p.name, "=>", p.module_path, p.class_name)

def main():
    file="./cimtest2.xml"
    linkmlfile = "../CoreEquipment.linkml.yaml"
    g = Graph()
    g.parse(file, format="cimxml", schema_path=linkmlfile)
    for s, p, o in g:
        if isinstance(o, Literal):
            print(f"Subject '{s}', predicate '{p}' and Object '{o}' with datatype: {o.datatype}")
        # print(s, "->", p, "->", o)
    # check_plugin_registered("cimxml")
    # print(list(g.triples((URIRef('http://iec.ch/TC57/CIM100#_b294e2e6-d1cd-2644-b079-27641ab2d844'), None, None))))


if __name__ == "__main__":
    main()
