"""The CIMProcessor class, which allows for handling and manipulating CIM graphs in memory."""

from datetime import datetime, timezone
from linkml_runtime.utils.schemaview import SchemaView, SchemaDefinition
from cim_plugin.graph import CIMGraph
from cim_plugin.header import create_header_attribute, CIMMetadataHeader
from cim_plugin.header_validation import validate_header
from cim_plugin.namespaces import update_namespace_in_triples, MD, DCAT_EXT, validate_and_fix_namespaces_by_cimtype
from cim_plugin.enriching import _build_slot_index, resolve_datatype_from_slot, create_typed_literal
from cim_plugin.exceptions import LiteralCastingError
from cim_plugin.to_file_strategies import _select_strategy
from cim_plugin.header_conversion import convert_triple
from cim_plugin.rdf_id_selection import PROFILES
from cim_plugin.provenance import Provenance, log_provenance
from rdflib import URIRef, Literal, Graph, IdentifiedNode
from rdflib.namespace import NamespaceManager, RDF
from pathlib import Path
from copy import deepcopy
import logging
from typing import Optional, Any


logger = logging.getLogger('cimxml_logger')

class CIMProcessor:
    def __init__(self, graph: CIMGraph, provenance_description: str|None = None):
        self.graph: CIMGraph = graph
        self.schema: Optional[SchemaView] = None
        self.slot_index: Optional[dict] = None

        if provenance_description is None:
            self._provenance = None
        else:
            if not isinstance(provenance_description, str) or not provenance_description.strip():
                raise TypeError("Provenance description must be a non-empty string.")
            self._provenance = Provenance(provenance_description)
            
        self.graph._provenance = self._provenance


    @property
    def provenance(self) -> Optional[tuple[dict[str, Any], ...]]:
        """Show the provenance entries for the graph."""
        if not self._provenance:
            logger.error("No provenance available.")
            return None
        
        return self._provenance.entries
    
    def mark_graph_changed(self) -> None:
        """Mark the graph as changed, which will trigger provenance logging."""
        if self._provenance:
            self._provenance.mark_changed()

    @property
    def identifier(self) -> IdentifiedNode:
        return self.graph.identifier
    
    def set_schema(self, filepath: str|Path) -> None:
        self.schema = SchemaView(filepath)
        self.slot_index = _build_slot_index(self.schema)
        
    def convert_subject_uudi_format(self):
        """Convert all subject uuids to format 'urn:uuid:'
        Not implemented yet.
        """

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

    
    def convert_header(self) -> None:
        """Header conversion between dcat:Dataset and 552 md:FullModel.

        No other format conversions are possible.
        Triples that cannot be converted will be dropped and logged. 
        This is especially notable for dcat -> md conversion because the dcat:Dataset format typically contains more information.
        """
        if not self.graph.metadata_header or not self.graph.metadata_header.triples:
            logger.error("No metadata header found for conversion.")
            return
        
        target_format, temp_graph = _make_header_graph_for_conversion(self.graph.metadata_header)
            
        unconverted_triples = set()
        for triple in self.graph.metadata_header.triples:
            if triple[2] == self.graph.metadata_header.header_type:
                continue

            converted = convert_triple(triple, target_format=target_format)
            if converted:
                temp_graph.add(converted)
            else:
                unconverted_triples.add(triple)

        logger.info(f"Converted {len(temp_graph)-1} header triples to {target_format}.")
        self.graph.metadata_header = CIMMetadataHeader.from_graph(temp_graph)

        if unconverted_triples:
            logger.info(f"{len(unconverted_triples)} triples could not be converted and was not included in the new header:\n" +
                        "\n".join(str(triple) for triple in unconverted_triples))

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


    def namespaces_different_from_model(self) -> set[tuple[str, str, str]]|None:
        """Checking if all namespaces in graph and header are identical with namespaces in model.
        
        Model is the ground truth. Checks only prefixes that are the same for both.

        Raises:
            AttributeError: If no linkML schema has been imported.

        Returns:
            set[tuple[str, URIRef]]: The graph/header namespaces that are different if any, else None.
        """
        if not self.schema or not self.schema.schema:
            raise AttributeError("No schema detected. Import a linkML schema using .set_schema()")

        not_identical: set[tuple[str, str, str]] = set()

        graph_ns = set(self.graph.namespace_manager.store.namespaces())
        if self.graph.metadata_header:
            header = self.graph.metadata_header.graph
            if header:
                graph_ns |= set(header.namespace_manager.store.namespaces())

        schema_def: SchemaDefinition = self.schema.schema

        schema_prefixes = getattr(schema_def, "prefixes", None)
        if isinstance(schema_prefixes, dict):
            prefixes_dict = schema_prefixes
        elif isinstance(schema_prefixes, list):
            prefixes_dict = {p.prefix_prefix: p for p in schema_prefixes}
        else:
            raise TypeError(f"Unexpected prefix structure: {type(schema_prefixes).__name__}")
        
        schema_ns = {p.prefix_prefix: p.prefix_reference for p in prefixes_dict.values()}
        
        for prx, ns in graph_ns:
            schema_namespace = schema_ns.get(prx, None)
            if schema_namespace:
                if schema_namespace != ns:
                    not_identical.add((prx, str(ns), str(schema_namespace)))       

        if not_identical:
            return not_identical


    @log_provenance("update_namespace", lambda self, prefix, namespace: f"Updated namespace for '{prefix}' to '{namespace.strip()}'.")
    def update_namespace(self, prefix: str, namespace: str) -> None:
        """Update namespace in graph and header.
        
        The namespace manager and all the triples in both the main graph and the header are given the new namespace.
        If the prefix do not exist in the namespace managers it will not be added. 
        Use the standard Graph.bind for adding new prefix - namespace pairs.

        Parameters:
            prefix (str): The prefix which will be given a new namespace.
            namespace (str): The new namespace.

        Raises:
            ValueError: If namespace is None or empty. 
        """

        stripped_namespace = str(namespace).strip()
        if not namespace or not stripped_namespace:
            raise ValueError("Namespace cannot be empty.")
        
        if self.graph.metadata_header:
            header = self.graph.metadata_header.graph
            header_old_namespace = header.namespace_manager.store.namespace(prefix)
            if header_old_namespace and str(header_old_namespace) != stripped_namespace:
                header.bind(prefix, stripped_namespace, override=True, replace=True)
                update_namespace_in_triples(header, header_old_namespace, stripped_namespace)
                # self.mark_graph_changed() # Header changes are not logged in provenance. May reconsider this later.
                
        old_namespace = self.graph.namespace_manager.store.namespace(prefix)
        if old_namespace and str(old_namespace) != stripped_namespace:
            self.graph.bind(prefix, stripped_namespace, override=True, replace=True)
            update_namespace_in_triples(self.graph, old_namespace, stripped_namespace)
            self.mark_graph_changed()
        

    @log_provenance("enrich_literal_datatypes", "Enriched literal objects with datatypes.")
    def enrich_literal_datatypes(self, allow_different_namespaces: bool = False) -> None:
        """Enrich the Literals of with datatypes collected from linkML SchemaView.
        
        - Cast value to correct format and log error when that is not possible.
        - Tag with the full URI of the primitive datatype.
        - Allows the namespaces to differ between graph and schema, if the prefix is the same.

        Parameters:
            allow_different_namespaces (bool): Allows differing namespaces if True.
        """
        if not self.schema or not self.slot_index:
            logger.error("Missing schemaview or slot_index. Enriching not possible.")
            return

        unfound_predicates = set()  # For clarity. May be removed later.
        casting_errors = []
        updated_count = 0   # For clarity. May be removed later.
        
        diffs = self.namespaces_different_from_model() if allow_different_namespaces else None
        
        for s, p, o in list(self.graph):
            if not isinstance(o, Literal) or o.datatype is not None or o.language is not None:
                continue

            p_str = replace_namespace(str(p), self.graph, diffs) if diffs else str(p)

            slot = self.slot_index.get(p_str)
            if not slot:
                unfound_predicates.add(p_str)   # For clarity. May be removed later.
                continue

            datatype_uri = resolve_datatype_from_slot(self.schema, slot)
            if not datatype_uri:
                logger.info(f"No datatype found for range: {slot.range}, for {slot.name}")
                continue

            try:
                new_literal = create_typed_literal(o.value, datatype_uri, self.schema)
            except LiteralCastingError as e:
                casting_errors.append(f"Error casting {o} for {s}, {p}: {e}")
                continue

            self.graph.remove((s, p, o))
            self.graph.add((s, p, new_literal))
            updated_count += 1  # For clarity. May be removed later.

            self.mark_graph_changed()

        if casting_errors:
            logger.error("\n".join(casting_errors))

        if unfound_predicates:  # For clarity. May be removed later.
            logger.info(f"Did not find these predicates in model: {unfound_predicates}")

        logger.info(f"Enriching done. Added datatypes to {updated_count} triples.") # For clarity. May be removed later.


    def _build_copy_for_serialization(self) -> "CIMProcessor":
        """Build a copy of the processor that can be mutated and used for serialization."""
        cim = CIMGraph(identifier=self.identifier)

        for prefix, uri in self.graph.namespace_manager.namespaces():
            cim.namespace_manager.bind(prefix, uri)

        cim += self.graph
        cim.metadata_header = self.graph.metadata_header

        proc = CIMProcessor(cim)
        proc.schema = self.schema
        proc.slot_index = deepcopy(self.slot_index)

        return proc
    
    def validate_header(self, format: str = "cimxml") -> None:
        """Validate the header of the graph according to the format specification.
        
        The validation is only performed for dcat:Dataset headers. md:FullModel and custom headers are ignored.
        
        Parameters:
            format (str): The format of the file. Accepted values are 'cimxml' (default), 'trig' and 'jsonld'.
        """
        if self.graph.metadata_header:
            validate_header(self.graph.metadata_header, format=format)
        else:
            logger.error("No metadata header found. Validation not possible.")


    def validate_namespaces(self, cimxml_format: bool = False) -> None:
        """Check that namespaces conforms to CIM standards and fix if not.
        
        Only CIM standard namespaces are checked. The list can be found in the namespaces module.
        If the cimxml_format is True and the profile is a CGMES type, the CGMES outlier namespaces are checked for 'cim', 'eu' and 'md'.
        In all other cases, 'cim', 'eu' and 'eumd' follows the standard for Network Code Related Canonical Extensions 2.4.0.

        Parameters:
            cimxml_format (bool): Whether the format is CIMXML. Default is False.
        """
        cgmes_outlier = False

        if cimxml_format:
            try:
                profile = self.graph.metadata_header.profile if self.graph.metadata_header else None
            except ValueError as e:
                logger.error(f"Unable to retrieve profile. Standard namespaces will be used. Cause: {e}.")
                profile = None

            if profile in PROFILES:
                cgmes_outlier = True 
    
        validate_and_fix_namespaces_by_cimtype(self.graph, cgmes=cgmes_outlier)


    # Not tested. Should it be?
    def to_file(self, file_path: str|Path, format: str = "cimxml", **kwargs) -> None:
        """Send graph to file of given format.
        
        Parameters:
            file_path (str|Path): Name and path of the new file.
            format (str): The format of the file. Accepted values are 'cimxml' (default), 'trig' and 'jsonld'.
            **kwargs: Options for output.
                CIMXML:
                    - 'qualifier' (str): The qualifier to use for uuids.
                Trig: 
                    - 'enrich_datatypes' (bool, default False) for allowing enriching of datatypes.
                    - 'schema_path' (str) for link to linkML file with datatype information.  
        """
        output_copy = self._build_copy_for_serialization()
        strategy = _select_strategy(format, file_path, kwargs)
        strategy.serialize(output_copy)


# Not in use, but might become usefull later
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


def replace_namespace(predicate: str, graph: CIMGraph, replacements: set[tuple[str, str, str]]) -> str:
    """Replace namespace for a uri from a set of replacements.
    
    If the prefix and namespace of the predicate is in the set of replacements, the namespace will be replaced.
    Otherwise, the predicate is returned unchanged.

    Parameters:
        predicate (str): The predicate or uri where the namespace is to be replaced.
        graph (CIMGraph): The graph with the namespace manager that contains the namespace for the predicate.
        replacements (set[tuple[str, str, str]]): A set of (prefix, old namespace, new namespace).
    
    Returns:
        str: The predicate with or without changes.
    """
    try:
        prefix, ns, local = graph.compute_qname(predicate)
    except ValueError as e:
        logger.error(f"Error in compute_qname for {predicate}: {e}")
        return predicate

    for old_prefix, old_ns, new_ns in replacements:
        if prefix == old_prefix and str(ns) == old_ns:
            return new_ns + local

    return predicate


def _make_header_graph_for_conversion(header: CIMMetadataHeader) -> tuple[str, Graph]:
    """Make a graph with the correct type for conversion.
    
    The correct rdf:type triple is added to the graph. 
    
    Parameters:
        header (CIMMetadataHeader): The header to be converted.

    Raises:
        ValueError: If the header is an unknown type.
    
    Returns:
        tuple[str, Graph]: The target format and the graph with the correct type triple.
    """    
    graph = Graph()

    if header.header_type == DCAT_EXT.Dataset:
        graph.bind("md", MD)    # Must be bound explicitly because it is not a default namespace in rdflib. 
        graph.add((header.subject, RDF.type, MD.FullModel))
        target_format = "md_fullmodel"
    elif header.header_type == MD.FullModel:
        graph.add((header.subject, RDF.type, DCAT_EXT.Dataset))
        target_format = "dcat_dataset"
    else:
        raise ValueError(f"Unknown header type: {header.header_type}. Conversion not possible.")

    return target_format, graph

if __name__ == "__main__":
    print("CIMProcessor for processing cim graphs.")