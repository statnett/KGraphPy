from linkml_runtime.utils.schemaview import SchemaView
from cim_plugin.graph import CIMGraph
from cim_plugin.header import create_header_attribute, CIMMetadataHeader
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

    # def replace_header(self, header: CIMMetadataHeader|None = None) -> None:
    #     self.graph.metadata_header = header

    def replace_header(self, header: CIMMetadataHeader | None = None) -> None:
        if header is None:
            self.graph.metadata_header = None
            return

        # Check for namespace prefix collisions
        main_nm = self.graph.namespace_manager
        header_nm = header.graph.namespace_manager

        for prefix, header_ns in header_nm.namespaces():
            if prefix in dict(main_nm.namespaces()):
                main_ns = dict(main_nm.namespaces())[prefix]
                if main_ns != header_ns:
                    logger.warning(f"Namespace for '{prefix}' differs between graph ({main_ns}) and new header ({header_ns}).")

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

    def merge_header(self):
        """Merge header back into graph.
        Find a way to deal with namespaces.
        """
        if self.graph.metadata_header:
            self.graph += self.graph.metadata_header.triples
            

    def are_namespaces_identical(self) -> list|None:
        """Checking if all namespaces in graph are identical with namespaces in model.
        Model is the ground truth. Check only prefixes that are the same for both.

        Returns:
            list: If any not identical, else None.
        """

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


if __name__ == "__main__":
    print("CIMProcessor for processing cim graphs.")