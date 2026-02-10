from rdflib import Graph
from cim_plugin.header import CIMMetadataHeader

class CIMGraph(Graph):
    metadata_header: CIMMetadataHeader | None = None


if __name__ == "__main__":
    print("Graph subclass")