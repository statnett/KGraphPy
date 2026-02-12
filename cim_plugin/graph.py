from rdflib import Graph, Dataset
from cim_plugin.header import CIMMetadataHeader

class CIMGraph(Graph):
    metadata_header: CIMMetadataHeader | None = None


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