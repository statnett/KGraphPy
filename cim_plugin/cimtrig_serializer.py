"""CIMTrigSerializer

Inherits from rflib.plugins.serializers.trig.TrigSerializer.
orderSubjects() is overridden to do the following:
- Triples with subject blank nodes that matches the object blank node of a different triple should be serialized together.
    - Non-matching blank nodes can be serialized at the bottom as is the trig serializer's default behavior.
- Triples in the metadata header should be serialized together at the beginning, but not in its own named graph. It should be inside the named graph the header belongs too.
    - This only works if the graph.metadata_header is not None and only if the graph is a CIMGraph. Otherwise, the default behavior of TrigSerializer is used.
    - This may also require special handling of the namespaces as the metadata header carries its own namespace_manager. These may have to be merged with the graph's namespace_manager.
        Only namespaces used (in the header or the graph) should be included in the output.

Otherwise, the default behavior of TrigSerializer is used.
"""

from __future__ import annotations

from rdflib import BNode, Graph, Node
from rdflib.graph import _TripleType
from rdflib.plugins.serializers.trig import TrigSerializer
from rdflib.plugins.serializers.turtle import SUBJECT

from cim_plugin.graph import CIMGraph


class CIMTrigSerializer(TrigSerializer):
    """Trig serializer with CIM-specific ordering behavior."""

    def reset(self) -> None:
        super().reset()
        # Track how many times each bnode appears as an object
        self._object_refs = {}

    def preprocess(self) -> None:
        """Preprocess contexts and merge header namespaces when available.

        The header may carry namespace bindings not present in the graph.
        Copying those bindings into the graph namespace manager allows stable,
        meaningful prefixes in output rather than generated fallback prefixes.
        """
        for context in self.contexts:
            if isinstance(context, CIMGraph):
                header = getattr(context, "metadata_header", None)
                if header is not None:
                    for prefix, ns_uri in header.graph.namespace_manager.namespaces():
                        existing = context.namespace_manager.store.namespace(prefix)
                        if existing is None:
                            context.bind(prefix, ns_uri, override=False)

                    context += header.graph # Adds in all the header triples if they are not alreary present.

        super().preprocess()

    
    def preprocessTriple(self, triple: tuple[Node, Node, Node]) -> None:
        s, p, o = triple

        # Count object references for bnodes
        if isinstance(o, BNode):
            self._object_refs[o] = self._object_refs.get(o, 0) + 1

        # Let the parent class do its normal subject counting
        super().preprocessTriple(triple)
    

    def orderSubjects(self):  # type: ignore[override]
        """Order subjects with header and linked blank-node prioritization.

        Rules:
        - If the active store is not a CIMGraph, or if it lacks a metadata
          header, the default TrigSerializer ordering is used.
        - Header subjects are emitted first (with the main header subject first).
        - Unlinked blank-node subjects are emitted before other subjects.
        - Remaining subjects keep default ordering semantics from TrigSerializer.
        """
        if not isinstance(self.store, CIMGraph):
            return super().orderSubjects()

        header = getattr(self.store, "metadata_header", None)
        if header is None:
            return super().orderSubjects()

        default_order = super().orderSubjects()

        header_subjects = set(header.graph.subjects())
        if not header_subjects:
            return default_order

        header_priority = [header.subject] + sorted(
            [
                subject
                for subject in header_subjects
                if subject != header.subject and subject in self._subjects
            ]
        )

        ordered: list[Node] = []
        seen: set[Node] = set()

        for subject in header_priority:
            if subject in self._subjects and subject not in seen:
                ordered.append(subject)
                seen.add(subject)

        for subject in default_order:
            if subject in seen:
                continue
            # Unlinked blank nodes are invalid in cim standard. Written first, after header, to be easy to find and fix.
            if isinstance(subject, BNode):
                ordered.append(subject)
                seen.add(subject)

        for subject in default_order:
            if subject in seen:
                continue
            if not isinstance(subject, BNode):
                ordered.append(subject)
                seen.add(subject)

        for subject in default_order:
            if subject not in seen:
                ordered.append(subject)
                seen.add(subject)

        return ordered

    def p_squared(self, node: Node, position: int, newline: bool = False) -> bool:
        if not isinstance(node, BNode):
            return False
        
        if node in self._serialized:
            return False
        
        obj_refs = self._object_refs.get(node, 0)
        if obj_refs != 1:
            return False
        
        if position == SUBJECT:
            return False
        
        if not newline:
            self.write(" ")

        if self.isValidList(node):
            self.write("(")
            self.depth += 1
            self.doList(node)
            self.depth -= 1
            self.write(")")
        else:
            self.subjectDone(node)
            self.depth += 2
            self.write("[")
            self.depth -= 1
            self.predicateList(node, newline=False)
            self.write("]")
            self.depth -= 1

        return True
    