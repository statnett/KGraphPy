import rdflib
from rdflib.graph import Graph, Dataset
from rdflib import URIRef, Literal, XSD, BNode
from rdflib.compare import to_isomorphic, graph_diff
import cim_plugin
import logging
from logging.config import dictConfig
from cim_plugin.log_config import LOG_CONFIG
from pathlib import Path
from cim_plugin.utilities import collect_cimxml_to_dataset, load_cimxml_graph, load_graphs_from_trig, load_graphs_from_cimxml
from cim_plugin.cimxml_serializer import CIMXMLSerializer

dictConfig(LOG_CONFIG)
logger = logging.getLogger('cimxml_logger')


def check_plugin_registered(name: str, plugin_type="Parser") -> None:
    from rdflib.plugin import plugins
    if plugin_type == "Parser":
        from rdflib.parser import Parser
        type = Parser
    elif plugin_type == "Serializer":
        from rdflib.serializer import Serializer
        type = Serializer
    else:
        raise ValueError(f"Plugin '{plugin_type}' not found")

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
    # check_plugin_registered("cimxml", "Serializer")
    # file2="../Nordic44/instances/Enterprise/cimxml/N44-ENT-Schneider_AC.xml"
    file="../Nordic44/instances/Grid/cimxml/Nordic44-HV_EQ.xml"
    # g = Graph()
    # g.parse(file, "xml")
    # file2="../Nordic44/instances/Grid/cimxml/Nordic44-HV_SSH.xml"
    # file3="../Nordic44/instances/NetworkCode/cimxml/N44-NC-HV_ER.xml"
    # linkmlfile = "../CoreEquipment.linkml.yaml"
    # ds = collect_cimxml_to_dataset([file, file2])
    # g3 = load_cimxml_graph(file3)
    g = load_graphs_from_cimxml([file])
    # for prefix, namespace in g3.namespace_manager.store.namespaces():
    #     print(prefix, namespace)
    tfile = "../Nordic44/instances/Grid/trig/Nordic44-HV_EQ.trig"
    t = load_graphs_from_trig(tfile)
    t1 = t[0]
    t1.extract_header()
    # print("Trig header")
    # for s, p, o in t1.graph.metadata_header.triples:
    #     print(s, p, o)
    # for g in t:
    #     print(g.graph.identifier)
    g1 = g[0]
    g1.extract_header()
    # print("cimxml header")
    # for s, p, o in g1.graph.metadata_header.triples:
    #     print(s, p, o)


    t1.replace_header(g1.graph.metadata_header)
    # print("Trig header after swap")
    # for s, p, o in t1.graph.metadata_header.triples:
    #     print(s, p, o)

    print("Trig namespaces")
    print(len(list(t1.graph.namespace_manager.store.namespaces())))
    print(t1.graph.namespace_manager.store.namespace("cim"))
    # for prefix, ns in t1.graph.namespace_manager.store.namespaces():
    #     print(prefix, ns)

    print("Trig header namespaces")
    print(len(list(t1.graph.metadata_header.graph.namespace_manager.store.namespaces())))
    print(t1.graph.metadata_header.graph.namespace_manager.store.namespace("cim"))
    # for prefix, ns in t1.graph.metadata_header.graph.namespace_manager.store.namespaces():
    #     print(prefix, ns)
    # counter = 0
    # for s, p, o in g1.graph:
    # #     if s == g.graph.identifier:
    #     print(s, p, o)
    #     counter += 1
    #     if counter == 10:
    #         break
    # print("header:")
    # for triple in g.graph.metadata_header.triples:
    #     print(triple)
    # t = Dataset()
    # t.parse(tfile, format="trig")
    # tgraph = t.graph(URIRef('urn:uuid:e710212f-f6b2-8d4c-9dc0-365398d8b59c'))
    # t_normalized = normalize_strings(tgraph)
    # g1 = ds.graph(URIRef('urn:uuid:e710212f-f6b2-8d4c-9dc0-365398d8b59c'))
    # g2 = ds.graph(URIRef('urn:uuid:1d08772d-c1d0-4c47-810d-b14908cd94f5'))
    # g3 = ds.graph(URIRef('urn:uuid:ebef4527-f0bc-4c59-8870-950af8ed9041'))
    # for g in ds.graphs():
        # print(g.identifier, type(g), getattr(g, "metadata_header", None))
    # for g in t.graphs():
    #     print(g.identifier)
    #     if "default" in g.identifier:
    #         print("This is default")
    #     counter = 0
    #     for s, p, o in g:
    #         if isinstance(o, Literal):
    #             print(s, p, o)
    #             counter += 1
    #             if counter == 2:
    #                 break

    # if g1.metadata_header:
    #     print(g1.metadata_header.triples)
    # g2 = ds.graph(URIRef('urn:uuid:ade44b65-0bfa-41e0-95c5-2ccb345a6fed'))
    # gs = CIMXMLSerializer(g1)
    # print(g1.identifier)
    # print(g2.identifier)
    # for g in list(t.graphs()):
    #     print(g.identifier)
    # for item in list(g3.triples((URIRef('urn:uuid:f1769a0e-9aeb-11e5-91da-b8763fd99c5f'), None, None))):
    #     print(item)
    # for item in lis
    # for s, p, o in tgraph:
    #     if isinstance(s, BNode):
    #         print(s, p, o)

    # g1 = tgraph
    # output_file = Path.cwd().parent / "cimxml_to_cimxm_grid_eq_parser_changed.xml"
    # g1.graph.serialize(destination=str(output_file), format="cimxml")

    # output_file2 = Path.cwd().parent / "cimxml_to_cimxml_grid_ssh_parser_changed.xml"
    # g2.serialize(destination=str(output_file2), format="cimxml")

    # output_file3 = Path.cwd().parent / "cimxml_to_cimxml_networkcode_er_parser_changed.xml"
    # g3.serialize(destination=str(output_file3), format="cimxml")


    output_file3 = Path.cwd().parent / "fromtrig_grid_eq_header_swapped.xml"
    t[0].graph.serialize(destination=str(output_file3), format="cimxml", qualifier="underscore")

    # print("Version:", rdflib.__version__) 
    # print("File:", rdflib.__file__)
    # count = 0
    # for s, p, o in g1:
    #     print(s, "->", p, "->", o)
    #     if isinstance(o, Literal): # and "integer" in o.datatype:
    #         print(o.datatype)

    #     count += 1
    #     if count == 10:
    #         break

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


    print(Path.cwd())
    # output_file = Path.cwd().parent / "Nordic44-HV_EQ_rdfxml.xml"
    # tgraph.serialize(destination=str(output_file), format="xml")

if __name__ == "__main__":
    main()
