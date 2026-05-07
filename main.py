from rdflib.graph import Graph
from rdflib import URIRef, Literal, XSD
import logging
from logging.config import dictConfig
from cim_plugin.log_config import LOG_CONFIG
from cim_plugin.utilities import load_graphs_from_trig, load_graphs_from_cimxml
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
    file="../Nordic44/instances/Grid/cimxml/Nordic44-HV_EQ.xml"
    # linkmlfile = "../CoreEquipment.linkml.yaml"
    g = load_graphs_from_cimxml([file])
    g1 = g[0]
    g1.extract_header()

    tfile = "../Nordic44/instances/Grid/trig/Nordic44-HV_EQ.trig"
    t = load_graphs_from_trig(tfile)
    t1 = t[0]
    t1.extract_header()
    
    # manifest_file = "../Nordic44/instances/Grid/cimxml/manifest.xml"
    # new_header = CIMMetadataHeader.from_manifest(manifest_file, g1.graph.metadata_header.subject)

    # counter = 0
    # for s, p, o in g1.graph:
    #     if isinstance(o, Literal):
    #         print(s, p, o, o.datatype)
    #         counter += 1
    #         if counter == 5:
    #             break

    # for triple in t1.header.graph:
    #       print(triple)

    # g1.replace_header(new_header)

    # output_file = Path.cwd().parent / "fromcimxml_grid_eq_corrected_header.xml"
    # g1.to_file(output_file, format="cimxml", qualifier="underscore")


if __name__ == "__main__":
    main()
