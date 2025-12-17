from rdflib.graph import Graph
from rdflib.plugin import register, Parser
from rdflib import URIRef
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
    file="./cimtest1.xml"
    check_plugin_registered("cimxml")
    g = Graph()
    g.parse(file, format="cimxml")
    # for s, p, o in g:
        # print(s, "->", p, "->", o)
    print("postprocess:", list(g.triples((None, URIRef("urn:processed"), None))))

if __name__ == "__main__":
    main()
