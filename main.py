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
    # file2="../Nordic44/instances/Grid/cimxml/Nordic44-HV_SSH.xml"
    # file3="../Nordic44/instances/NetworkCode/cimxml/N44-NC-HV_ER.xml"
    linkmlfile = "../CoreEquipment.linkml.yaml"
    g = load_graphs_from_cimxml([file])
    g1 = g[0]
    # g1.extract_header()

    # tfile = "../Nordic44/instances/Grid/trig/Nordic44-HV_EQ.trig"
    # t = load_graphs_from_trig(tfile)
    # t1 = t[0]
    # t1.extract_header()
    
    # g1.set_schema(linkmlfile)
    g1.update_namespace("eu", "http://iec.ch/TC57/CIM100-European#")
    # diffs = g1.namespaces_different_from_model()
    # print(diffs)
    # g1.enrich_literal_datatypes(allow_different_namespaces=True)
    # t1.replace_header(g1.graph.metadata_header)
    counter = 0
    for s, p, o in g1.graph:
        if isinstance(o, Literal):
            print(s, p, o, o.datatype)
            counter += 1
            if counter == 5:
                break

    # print(g1.slot_index)
    # output_file = Path.cwd().parent / "cimxml_to_cimxm_grid_eq_parser_changed.xml"
    # g1.graph.serialize(destination=str(output_file), format="cimxml")

    # output_file2 = Path.cwd().parent / "cimxml_to_cimxml_grid_ssh_parser_changed.xml"
    # g2.serialize(destination=str(output_file2), format="cimxml")

    # output_file3 = Path.cwd().parent / "cimxml_to_cimxml_networkcode_er_parser_changed.xml"
    # g3.serialize(destination=str(output_file3), format="cimxml")

    # t1.update_namespace("cim", str(g1.graph.namespace_manager.store.namespace("cim")))
    # t1.update_namespace("eu", str(g1.graph.namespace_manager.store.namespace("eu")))

    # t1.set_schema(linkmlfile)
    # diffs = t1.namespaces_different_from_model()
    # print(diffs)
    # print(t1.schema.namespaces())

    # output_file3 = Path.cwd().parent / "fromtrig_grid_eq_header_swapped.xml"
    # t1.graph.serialize(destination=str(output_file3), format="cimxml", qualifier="underscore")

    # t1.merge_header()
    # output_file_trig = Path.cwd().parent / "fromxml_totrig_grid_eq.trig"
    # g1.graph.serialize(destination=str(output_file_trig), format="trig")
    

if __name__ == "__main__":
    main()
