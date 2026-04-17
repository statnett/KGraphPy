"""The graph and dataset classes used to handle the graph triples."""

from typing import Iterable

from rdflib import Graph, Dataset
from rdflib.graph import _TripleType
from cim_plugin.provenance import Provenance, log_provenance
from cim_plugin.header import CIMMetadataHeader

class CIMGraph(Graph):
    _provenance: Provenance|None = None
    metadata_header: CIMMetadataHeader | None = None
    
    @log_provenance("add_triple", lambda self, triple: f"Added triple {triple}.")
    def add(self, triple) -> "CIMGraph":
        super().add(triple)
        self._provenance.mark_changed() if self._provenance else None
        return self
    
    @log_provenance("remove_triple", lambda self, triple: f"Removed triple {triple}.")
    def remove(self, triple) -> "CIMGraph":
        super().remove(triple)
        self._provenance.mark_changed() if self._provenance else None
        return self
    
    @log_provenance("bind_namespace", lambda self, prefix, namespace, **kwargs: f"Bound namespace {namespace} to prefix {prefix}.")
    def bind(self, prefix, namespace, override=True, replace=False) -> None:
        super().bind(prefix, namespace, override=override, replace=replace)
        self._provenance.mark_changed() if self._provenance else None

    @log_provenance("set_triple", lambda self, triple: f"Set triple {triple}.")
    def set(self, triple) -> "CIMGraph":
        super().set(triple)
        self._provenance.mark_changed() if self._provenance else None
        return self

    @log_provenance("merging_graphs", lambda self, other: f"Graph {other.identifier} merged into {self.identifier}.")
    def __iadd__(self, other: Iterable[_TripleType]) -> "CIMGraph":
        super().__iadd__(other)
        self._provenance.mark_changed() if self._provenance else None
        return self
    
    @log_provenance("removing_subgraph_from_graph", lambda self, other: f"Triples in graph {other.identifier} removed from {self.identifier}.")
    def __isub__(self, other: Iterable[_TripleType]) -> "CIMGraph":
        super().__isub__(other)
        self._provenance.mark_changed() if self._provenance else None
        return self


# This is no longer used, but kept for reference and possible future use. 
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