"""Strategies for sending CIM graph to file."""

import logging
from pathlib import Path
from cim_plugin.jsonld_utilities import reorder_jsonld, extract_datatype_map, enrich_graph_datatypes, load_json_from_url, DEFAULT_CONTEXT_LINK
from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from cim_plugin.processor import CIMProcessor

logger = logging.getLogger('cimxml_logger')


class SerializationStrategy:
    """Base class for serialization strategies."""

    def serialize(self, processor: "CIMProcessor"):
        """Base method for serializing cim graph."""
        raise NotImplementedError


class TrigStrategy(SerializationStrategy):
    """Strategy class for sending a cim graph to trig file."""

    def __init__(self, file_path: str|Path, schema_path: Optional[str|Path] = None, enrich_datatypes: bool = False) -> None:
        self.file_path = file_path
        self.schema_path = schema_path
        self.enrich_datatypes = enrich_datatypes

    def serialize(self, processor: "CIMProcessor"):
        """Serialize cim graph to trig file.
        
        Parameters:
            processor (CIMProcessor): The CIMProcessor containing the graph to serialize.
        """
        if self.schema_path:
            processor.set_schema(self.schema_path)

        if processor.header:
            processor.merge_header()

        if self.enrich_datatypes:
            if processor.schema:
                processor.enrich_literal_datatypes()
            else:
                logger.error("Cannot enrich datatypes without schema with datatypes.")
        
        processor.graph.serialize(self.file_path, format="cimtrig")


class CIMXMLStrategy(SerializationStrategy):
    """Strategy class for sending a cim graph to CIMXML file."""
    
    def __init__(self, file_path: str|Path, qualifier: Optional[str] = None) -> None:
        self.file_path = file_path
        self.qualifier = qualifier

    def serialize(self, processor: "CIMProcessor"):
        """Serialize cim graph to CIMXML file.
        
        Parameters:
            processor (CIMProcessor): The CIMProcessor containing the graph to serialize.
        """
        if not processor.header:
            logger.error("Serializing without an extracted header may create a corrupt CIMXML file.")

        processor.graph.serialize(self.file_path, format="cimxml", qualifier=self.qualifier)

class JSONLDStrategy(SerializationStrategy):
    """Strategy class for sending a cim graph to JSON-LD file."""

    def __init__(self, file_path: str|Path, context: Optional[dict|str] = None) -> None:
        self.file_path = file_path
        self.context = context or DEFAULT_CONTEXT_LINK

    def serialize(self, processor: "CIMProcessor") -> None:
        """Serialize cim graph to JSON-LD file.
        
        Parameters:
            processor (CIMProcessor): The CIMProcessor containing the graph to serialize.
        """
        if processor.header:
            processor.merge_header()
            header_subject = processor.header.subject
        else:
            header_subject = None

        self.enrich_datatypes(processor)
        
        raw_jsonld = processor.graph.serialize(format="json-ld", context=self.context, auto_compact=True)
        reordered_jsonld = reorder_jsonld(raw_jsonld, priority_subject=header_subject)

        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(reordered_jsonld)


    def enrich_datatypes(self, processor: "CIMProcessor") -> None:
        """Enrich datatypes in the graph based on the context.
        
        Parameters:
            processor (CIMProcessor): The CIMProcessor containing the graph to enrich.

        Raises:
            TypeError: If the context is not a string or a dictionary.
        """
        if isinstance(self.context, str):
            context_data = load_json_from_url(self.context)
        elif isinstance(self.context, dict):
            context_data = self.context
        else:
            raise TypeError("Context must be a string or a dictionary.")

        datatype_map = extract_datatype_map(context_data)
        enrich_graph_datatypes(processor.graph, datatype_map)

def _select_strategy(format: str, file_path: str|Path, options: dict[str, Any]) -> SerializationStrategy:
    """Select serialization strategy based on given file format.
    
    Parameters:
        format (str): The format of the file.
        file_path (str|Path): The file name and path of the new file.
        options (dict[str, Any]): Options appropriate to the file type.

    Raises:
        ValueError: If the format input is unrecognised.

    Returns:
        SerializationStrategy: The appropriate strategy with options.
    """
    format = format.lower()

    if format == "trig":
        allowed = {"schema_path", "enrich_datatypes"}
        _validate_options(options, allowed)
        return TrigStrategy(file_path, **options)

    elif format == "cimxml":
        allowed = {"qualifier"}
        _validate_options(options, allowed)
        return CIMXMLStrategy(file_path, **options)

    elif format == "jsonld":
        allowed = {"context"}  # future options
        _validate_options(options, allowed)
        return JSONLDStrategy(file_path, **options)

    else:
        raise ValueError(f"Unknown format: {format}")
    

def _validate_options(options: dict[str, Any], allowed: set[str]) -> None:
    """Validate if options are among the allowed options.
    
    Parameters:
        options (dict[str, Any]): The options to be validated.
        allowed (set[str]): The allowed options.

    Raises:
        ValueError: If any of the options are not in the allowed set.
    """
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(
            f"Options {unknown} are not valid for this format. "
            f"Allowed options: {allowed}"
        )


if __name__ == "__main__":
    print("Strategies for serializing to file.")