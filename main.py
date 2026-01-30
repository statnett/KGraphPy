from rdflib.graph import Graph, Dataset
from rdflib import URIRef, Literal, XSD
from rdflib.compare import to_isomorphic, graph_diff
import cim_plugin
from rdflib.parser import Parser
import logging
from logging.config import dictConfig
from cim_plugin.log_config import LOG_CONFIG
from pathlib import Path
from cim_plugin.utilities import collect_cimxml_to_dataset

dictConfig(LOG_CONFIG)
logger = logging.getLogger('cimxml_logger')


def check_plugin_registered(name: str, type=Parser) -> None:
    from rdflib.plugin import plugins
    print("Registrered plugins:") 
    for p in plugins(None, type): 
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
            if o.datatype is None: 
                o = Literal(o, datatype=XSD.string) 
        new.add((s, p, o)) 
    return new

def main():
    # check_plugin_registered("cimxml", Parser)
    # file="../Nordic44/instances/Enterprise/cimxml/N44-ENT-Schneider_AC.xml"
    file="../Nordic44/instances/Grid/cimxml/Nordic44-HV_EQ.xml"
    file2="../Nordic44/instances/Grid/cimxml/Nordic44-HV_GL.xml"
    linkmlfile = "../CoreEquipment.linkml.yaml"
    ds = collect_cimxml_to_dataset([file, file2], linkmlfile)

    # tfile = "../Nordic44/instances/Grid/trig/Nordic44-HV_EQ.trig"
    # t = Dataset()
    # t.parse(tfile, format="trig")
    # tgraph = t.graph(URIRef('urn:uuid:e710212f-f6b2-8d4c-9dc0-365398d8b59c'))
    # t_normalized = normalize_strings(tgraph)
    g1 = list(ds.graphs())[0]
    count = 0
    for s, p, o in g1:
        print(s, "->", p, "->", o)
        if isinstance(o, Literal): # and "integer" in o.datatype:
            print(o.datatype)

        count += 1
        if count == 10:
            break

    # g_test = g
    # t_test = t_normalized
    # print(f"CIMXML: {len(g_test)}")
    # for item in list(g_test.triples((URIRef('urn:uuid:f1769d28-9aeb-11e5-91da-b8763fd99c5f'), URIRef('https://cim.ucaiug.io/ns#Equipment.normallyInService'), None))):
    #     print(item)

    # print(f"Trig: {len(t_test)}")
    # for item in list(t_test.triples((URIRef('urn:uuid:f1769d28-9aeb-11e5-91da-b8763fd99c5f'), URIRef('https://cim.ucaiug.io/ns#Equipment.normallyInService'), None))):
    #     print(item)

    # isoC = to_isomorphic(g_test)
    # isoT = to_isomorphic(t_test)
    # print(isoC==isoT)
    # in_both, in_cim, in_trig = graph_diff(g_test, t_test)
    # print(f"In cim: {len(in_cim)}, in trig: {len(in_trig)}")
    # # for pfx, ns in g.namespace_manager.namespaces():
    # #     print(pfx, ns)
    # print("cim:")
    # count = 0
    # for s, p, o in in_cim:
    #     if isinstance(o, Literal):
    #         print(s, p, o, o.datatype)
    #         count += 1
    #         if count == 5:
    #             break
    
    # print("trig:")
    # count = 0
    # for s, p, o in in_trig:
    #     if not isinstance(o, Literal):
    #         print(s, p, o)
    #         count += 1
    #         if count == 5:
    #             break




    # output_file = Path.cwd().parent / "Nordic44-HV_EQ_rdfxml.xml"
    # tgraph.serialize(destination=str(output_file), format="xml")

if __name__ == "__main__":
    main()
