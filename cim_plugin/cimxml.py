from rdflib.plugin import register, Parser
from rdflib.plugins.parsers.rdfxml import RDFXMLParser
from rdflib import URIRef, Literal

class CIMXMLParser(Parser):
    name = "cimxml"
    format = "cimxml"

    def __init__(self):
        super().__init__()
        print("CIMXMLParser loaded")

    def parse(self, source, sink, **kwargs):
        print("CIMXMLParser.parse called")
        rdfxml = RDFXMLParser()
        rdfxml.parse(source, sink, **kwargs)
        self.post_process(sink)
        return sink
    
    def post_process(self, graph):
        print("Running post-process")
        graph.add((
            URIRef("urn:postprocess"),
            URIRef("urn:processed"),
            Literal(True)
        ))

if __name__ == "__main__":
    print("cimxml plugin for rdflib")

