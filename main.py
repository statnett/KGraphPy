import rdflib
from rdflib.graph import Graph, Dataset
from rdflib import URIRef, Literal, XSD, BNode
from rdflib.compare import to_isomorphic, graph_diff
from rdflib.namespace import DCTERMS, DCAT
import cim_plugin
import logging
from logging.config import dictConfig
from cim_plugin.log_config import LOG_CONFIG
from pathlib import Path
from cim_plugin.utilities import collect_cimxml_to_dataset, load_cimxml_graph, load_graphs_from_trig, load_graphs_from_cimxml
from cim_plugin.header import CIMMetadataHeader
from cim_plugin.namespaces import EU, CIM
from cim_plugin.enriching import cast_datetime_utc
from cim_plugin.header_validation import validate_header #correct_cimxml_header, correct_trig_header
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
    # linkmlfile = "../CoreEquipment.linkml.yaml"
    g = load_graphs_from_cimxml([file])
    g1 = g[0]
    g1.extract_header()

    tfile = "../Nordic44/instances/Grid/trig/Nordic44-HV_EQ.trig"
    t = load_graphs_from_trig(tfile)
    t1 = t[0]
    t1.extract_header()
    
    manifest_file = "../Nordic44/instances/Grid/cimxml/manifest.xml"
    new_header = CIMMetadataHeader.from_manifest(manifest_file, g1.graph.metadata_header.subject)
    # g1.replace_header(new_header)

    # g1.set_schema(linkmlfile)
    # g1.update_namespace("eu", EU)
    # g1.update_namespace("cim", CIM)

    # g1.enrich_literal_datatypes(allow_different_namespaces=True)
    # t1.replace_header(g1.graph.metadata_header)
    # counter = 0
    # for s, p, o in g1.graph:
    #     if isinstance(o, Literal):
    #         print(s, p, o, o.datatype)
    #         counter += 1
    #         if counter == 5:
    #             break
    # for s, p, o in new_header.graph.triples((new_header.subject, URIRef('http://purl.org/dc/terms/issued'), None)):
    #     if isinstance(o, Literal):
    #         print(type(o.toPython()))
    #         print(o.toPython())
    #         casted = cast_datetime_utc(o)
    #         print(casted)
    #         print(type(casted))

    # print("from trig:")
    # for s, p, o in t1.graph.metadata_header.graph.triples((new_header.subject, URIRef('http://purl.org/dc/terms/issued'), None)):
    #     if isinstance(o, Literal):
    #         print(o.toPython())
    #         print(type(o.toPython()))
    #         casted = cast_datetime_utc(o)
    #         print(casted)
    #         print(type(casted))
    # print(t1.graph.metadata_header.triples)

    # for triple in t1.graph.metadata_header.graph.triples((None, None, DCTERMS.PeriodOfTime)):
    #       print(triple)

    t1.graph.metadata_header.remove_triple(DCTERMS.issued, Literal('2025-02-14', datatype=XSD.date))
    # t1.graph.metadata_header.remove_triple(DCTERMS.issued, Literal('2025-02-14T00:00:00+00:00', datatype=XSD.dateTime))
    # for triple in t1.graph.metadata_header.graph.triples((None, DCTERMS.issued, None)):
    #       print(triple)

    # for triple in new_header.graph.triples((new_header.subject, DCAT.startDate, None)):
    #       print(triple)
    # cimxml_issues = validate_cimxml_header(new_header.triples)
    # trig_issues = validate_trig_header(t1.graph.metadata_header.triples)
    # print("CIMXML header issues:", cimxml_issues)
    # print("TRIG header issues:", len(trig_issues), trig_issues)
    g1.replace_header(new_header)
    g1.validate_header(format="cimxml")
    output_file = Path.cwd().parent / "fromcimxml_grid_eq_corrected_header.xml"
    g1.to_file(output_file, format="cimxml", qualifier="underscore")
    
    t1.validate_header(format="trig")
    output_file_trig = Path.cwd().parent / "fromtrig_grid_eq_corrected_header.trig"
    t1.to_file(output_file_trig, format="trig", enrich_datatypes=False)
    

if __name__ == "__main__":
    main()
