"""The graph and dataset classes used to handle the graph triples."""

# from linkml_runtime.utils.schemaview import SchemaView
from rdflib import Graph, Dataset
from cim_plugin.header import CIMMetadataHeader
# from cim_plugin.processor import CIMProcessor
# from typing import Optional

class CIMGraph(Graph):
    metadata_header: CIMMetadataHeader | None = None

    # @classmethod
    # def from_trig_format(cls, graph_path: str, schema_path: Optional[str] = None):
    #     """Unfinished method"""
    #     g = cls()
    #     g.parse(graph_path, format="trig")
    #     processor = CIMProcessor(g)
    #     processor.set_schema(schema_path)
    #     enrich = True if schema_path else False
    #     processor.process(enrich_datatypes=enrich)
    #     return g

    # def to_trig_format(self, file_path: str, enrich_datatypes: bool=False, schema_path: Optional[str] = None):
    #     """Unfinished method"""
    #     processor = CIMProcessor(self)
    #     if schema_path:
    #         processor.set_schema(schema_path)
    #     processor.prepare_for_serialization(enrich_datatypes=enrich_datatypes)
    #     return self.serialize(file_path, format="trig")


class CIMDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._graph_cache = {}

    def _wrap(self, g):
        ident = g.identifier

        # Return cached instance if it exists
        if ident in self._graph_cache:
            return self._graph_cache[ident]

        # Otherwise create a new CIMGraph wrapper
        new = CIMGraph(store=self.store, identifier=ident, base=g.base)
        new.namespace_manager = g.namespace_manager

        # Cache it
        self._graph_cache[ident] = new
        return new

    def graph(self, identifier=None, base=None, **kwargs):
        g = super().graph(identifier=identifier, base=base, **kwargs)
        return self._wrap(g)

    def graphs(self, triple=None):
        for g in super().graphs(triple=triple):
            yield self._wrap(g)

    def contexts(self, triple=None):
        for g in super().contexts(triple=triple):
            yield self._wrap(g)



if __name__ == "__main__":
    print("Graph subclass")