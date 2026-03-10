from linkml_runtime.utils.schemaview import SchemaView
from cim_plugin.graph import CIMGraph
from cim_plugin.header import create_header_attribute, CIMMetadataHeader
from rdflib import URIRef
from rdflib.namespace import NamespaceManager
import logging
from typing import Optional

logger = logging.getLogger('cimxml_logger')

class CIMProcessor:
    def __init__(self, graph: CIMGraph):
        self.graph: CIMGraph = graph
        self.schema: Optional[SchemaView] = None

    def set_schema(self, filepath: Optional[str]) -> None:
        if filepath:
            self.schema = SchemaView(filepath)


    def replace_header(self, header: CIMMetadataHeader | None = None) -> None:
        """Replace the header of the graph.
        
        The namespaces of the header are added to the namespaces of the graph. 
        Namespace collisions are resolved by keeping the graph's namespace.

        WARNING:
        The header namespace manager keeps its' own namespace for the prefix when there are collisions with the graph.
        This may cause problems for later serialisations. An error message warns when this happends, and .update_namespace
        can be used to fix the issue.

        Parameters:
            header (CIMMetadataHeader | None): The new header. If header=None the header will be cleared.
            
        Raises:
            TypeError: If the header is neither a CIMMetadataHeader or None.
        """
        if header is None:
            self.graph.metadata_header = None
            return
        
        if not isinstance(header, CIMMetadataHeader):
            raise TypeError("The new header must be of type CIMMetadataHeader.")

        # Check for namespace prefix collisions
        main_nm = self.graph.namespace_manager
        header_nm = header.graph.namespace_manager

        merge_namespace_managers(main_nm, header_nm)

        # Replace header
        self.graph.metadata_header = header

    def extract_header(self) -> None:
        """Move header triples from graph to the metadata_header attribute."""
        if self.graph.metadata_header:
            logger.error("Metadata header already exist. Use .replace_header instead.")
            return
        
        header = create_header_attribute(self.graph)
        self.graph.metadata_header = header
        self.graph.remove((header.subject, None, None))

        # Remove blank nodes that belong to the header
        for subject_node in header.reachable_nodes:
            self.graph.remove((subject_node, None, None))

    def merge_header(self) -> None:
        """Merge header back into graph.
        
        The namespaces of the header are added to the namespaces of the graph. 
        Namespace collisions are resolved by keeping the graph's namespace.

        WARNING:
        The header namespace manager keeps its' own namespace for the prefix when there are collisions with the graph.
        This may cause problems for later serialisations. An error message warns when this happends. Use .update_namespace
        before .merge_headers to fix the issue.
        """
        if self.graph.metadata_header:
            header = self.graph.metadata_header
            self.graph += header.graph
            merge_namespace_managers(self.graph.namespace_manager, header.graph.namespace_manager)


    # Keeping this commented out for now, in case sorting header triples first becomes necessary
    # def merge_header(self):
    #     """Merge header back into graph with header triples first."""
    #     header = self.graph.metadata_header
    #     if not header:
    #         return

    #     # Extract existing triples
    #     original_triples = list(self.graph.triples((None, None, None)))

    #     # Clear graph
    #     self.graph.remove((None, None, None))

    #     # Insert header triples first
    #     for t in header.triples:
    #         self.graph.add(t)

    #     # Insert the rest
    #     for t in original_triples:
    #         self.graph.add(t)


            

    def are_namespaces_identical_with_model(self) -> list|None:
        """Checking if all namespaces in graph are identical with namespaces in model.
        Model is the ground truth. Check only prefixes that are the same for both.

        Returns:
            list: If any not identical, else None.
        """

    def update_namespace(self) -> None:
        """Update namespace in graph and header."""

    def enrich_datatypes(self):
        """Use self.schema to enrich self.graph with datatypes."""

    def process(self, *, enrich_datatypes=False):
        """Run the full CIM processing pipeline."""
        self.extract_header()
        if enrich_datatypes:
            if not self.schema:
                logger.error("Set schema before datatype enriching.")
            else:
                self.enrich_datatypes()
        # other CIM-specific transformations can be added here

    def prepare_for_serialization(self, *, enrich_datatypes=False):
        """Prepare the graph for output formats."""
        if enrich_datatypes:
            if not self.schema:
                logger.error("Set schema before datatype enriching.")
            else:
                self.enrich_datatypes()
        self.merge_header()


def _check_for_namespace_collisions(namespaces1: NamespaceManager, namespaces2: NamespaceManager) -> bool:
    collision = False
    for prefix, ns1 in namespaces2.namespaces():
        if prefix in dict(namespaces1.namespaces()):
            main_ns = dict(namespaces1.namespaces())[prefix]
            if main_ns != ns1:
                logger.error(f"Namespace for '{prefix}' differs between graphs ({main_ns} vs {ns1}).")
                collision = True
    return collision


def merge_namespace_managers(main_nm: NamespaceManager, other_nm: NamespaceManager) -> None:
    """Merge one set of namespaces managers into another.
    
    The namespaces already in the managers are not changed. The triples are not changed.
    
    Parameters:
        main_nm (NamespaceManager): The namespace manager that will get new namespaces from other.
        other_nm (NamespaceManager): The namespace manager that new namespaces are coming from. Will not be changed.
    """
    main_dict = dict(main_nm.namespaces())
    other_temp = list(other_nm.namespaces())

    for prefix, ns in other_temp:
        if prefix not in main_dict:
            main_nm.bind(prefix, ns, override=False, replace=False) # Override and replace = False ensures that namespaces already in main_nm remains unchanged.
            main_dict[prefix] = URIRef(ns)
        else:
            if main_dict[prefix] != ns:
                logger.error(f"Namespace for '{prefix}' differs between graphs ({main_dict[prefix]} vs {ns}). {main_dict[prefix]} is kept.")
                
        # The below was removed because it is dangerous to change the namespace in the manager without changing the triples.
        # If this is needed in the future, consider how to change the triples too.
                # if not keep_other:
                #     logger.warning(f"{main_dict[prefix]} overwrites {ns} for {prefix}.")
                #     other_nm.bind(prefix, main_dict[prefix], override=True, replace=True)
                # else:
                #     logger.warning(f"{ns} overwrites {main_dict[prefix]} for {prefix}.")
                #     main_nm.bind(prefix, ns, override=True, replace=True)
                #     main_dict[prefix] = URIRef(ns)


if __name__ == "__main__":
    print("CIMProcessor for processing cim graphs.")