from rdflib.graph import Graph, Dataset
from rdflib.plugin import register
from rdflib.parser import Parser
from rdflib import URIRef, Literal, XSD
from rdflib.compare import to_isomorphic, graph_diff
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


def rewrite_uri(graph, old_base, new_base):
    new_graph = Graph()

    for s, p, o in graph:
        if isinstance(s, URIRef) and s.startswith(old_base):
            s = URIRef(s.replace(old_base, new_base, 1))
        if isinstance(p, URIRef) and p.startswith(old_base):
            p = URIRef(p.replace(old_base, new_base, 1))
        if isinstance(o, URIRef) and o.startswith(old_base):
            o = URIRef(o.replace(old_base, new_base, 1))

        new_graph.add((s, p, o))

    return new_graph

def normalize_strings(g): 
    new = Graph() 
    for s, p, o in g: 
        if isinstance(o, Literal): 
            # Hvis literal er uten datatype → sett xsd:string 
            if o.datatype is None: 
                o = Literal(o, datatype=XSD.string) 
            # Hvis literal har xsd:string → behold # Hvis literal har annen datatype → behold 
        new.add((s, p, o)) 
    return new

def main():
    # file="../Nordic44/instances/Enterprise/cimxml/N44-ENT-Schneider_AC.xml"
    file="../Nordic44/instances/Grid/cimxml/Nordic44-HV_EQ.xml"
    linkmlfile = "../CoreEquipment.linkml.yaml"
    g = Graph()
    g.parse(file, format="cimxml", schema_path=linkmlfile)
    tfile = "../Nordic44/instances/Grid/trig/Nordic44-HV_EQ.trig"
    t = Dataset()
    t.parse(tfile, format="trig")
    tgraph = t.graph(URIRef('urn:uuid:e710212f-f6b2-8d4c-9dc0-365398d8b59c'))
    g_normalized = rewrite_uri(g, "http://iec.ch/TC57/CIM100#", "https://cim.ucaiug.io/ns#")
    g_normalized = rewrite_uri(g_normalized, "http://iec.ch/TC57/CIM100-EuropeanExtension/1/0#", "https://cim.ucaiug.io/ns/eu#")
    t_normalized = normalize_strings(tgraph)
    # g_normalized = normalize_strings(g_normalized)
    # count = 0
    # for s, p, o in g:
    #     print(s, "->", p, "->", o)
    #     if isinstance(o, Literal) and "integer" in o.datatype:
    #         print(s, "->", p, "->", o)
    #         print(o.datatype)

    #     count += 1
    #     if count == 10:
    #         break
    #         # print(f"Subject '{s}', predicate '{p}' and Object '{o}' with datatype: {o.datatype}")

    # check_plugin_registered("cimxml")
    g_test = g_normalized
    t_test = t_normalized
    # print(f"CIMXML: {len(g_test)}")
    # for item in list(g_test.triples((URIRef('urn:uuid:43e27a15-0192-4c01-bec3-413f770618c7'), None, None))):
    #     print(item)

    # print(f"Trig: {len(t_test)}")
    # for item in list(t_test.triples((URIRef('urn:uuid:43e27a15-0192-4c01-bec3-413f770618c7'), None, None))):
    #     print(item)

    isoC = to_isomorphic(g_test)
    isoT = to_isomorphic(t_test)
    print(isoC==isoT)
    in_both, in_cim, in_trig = graph_diff(g_test, t_test)
    print(f"In cim: {len(in_cim)}, in trig: {len(in_trig)}")
    print("cim:")
    count = 0
    for s, p, o in in_cim:
        # if not isinstance(o, Literal):
        print(s, p, o)
            # count += 1
            # if count == 5:
            #     break
    
    print("trig:")
    count = 0
    for s, p, o in in_trig:
        # if not isinstance(o, Literal):
        print(s, p, o)
            # count += 1
            # if count == 5:
            #     break

if __name__ == "__main__":
    main()
