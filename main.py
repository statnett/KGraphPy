import rdflib
from pathlib import Path
from rdflib.graph import Graph, Dataset
from rdflib import URIRef, Literal, XSD, BNode
import cim_plugin   # If this is removed the cim parser and serializer will no longer work
import logging
from logging.config import dictConfig
from cim_plugin.log_config import LOG_CONFIG
from cim_plugin.utilities import collect_cimxml_to_dataset, load_cimxml_graph

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
    # Here follows example usage

    # These files are not included
    file="../Nordic44/instances/Grid/cimxml/Nordic44-HV_EQ.xml"
    file2="../Nordic44/instances/Grid/cimxml/Nordic44-HV_SSH.xml"
    file3="../Nordic44/instances/NetworkCode/cimxml/N44-NC-HV_ER.xml"
    linkmlfile = "../CoreEquipment.linkml.yaml"
    ds = collect_cimxml_to_dataset([file, file2], schema_path=linkmlfile)
    g3 = load_cimxml_graph(file3, schema_path=None)   # Has different namespace for cim, so must be loaded separetely from the others
    
    # Select individual graphs by name
    g1 = ds.graph(URIRef('urn:uuid:e710212f-f6b2-8d4c-9dc0-365398d8b59c'))
    g2 = ds.graph(URIRef('urn:uuid:1d08772d-c1d0-4c47-810d-b14908cd94f5'))
    for g in ds.graphs():
        print(g.identifier) # Find the names by printing the identifiers of the graphs

    counter = 0
    for s, p, o in g2:  # Show random triples
        if isinstance(o, Literal):
            print(s, p, o, o.datatype)
            counter += 1
            if counter == 5:
                break

    # How to look at the header
    if g1.metadata_header:
        print(g1.metadata_header.triples)

    # Examples of how to serialise to cimxml
    # output_file = Path.cwd().parent / "cimxml_to_cimxm_grid_eq_parser_changed.xml"
    # g1.serialize(destination=str(output_file), format="cimxml")

    # output_file2 = Path.cwd().parent / "cimxml_to_cimxml_grid_ssh_parser_changed.xml"
    # g2.serialize(destination=str(output_file2), format="cimxml")

    # output_file3 = Path.cwd().parent / "cimxml_to_cimxml_networkcode_er_parser_changed.xml"
    # g3.serialize(destination=str(output_file3), format="cimxml", qualifier="urn")



if __name__ == "__main__":
    main()
