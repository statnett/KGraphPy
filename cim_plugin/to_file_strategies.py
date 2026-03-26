"""Strategies for sending CIM graph to file."""

import logging
from pathlib import Path
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
        """Serialize cim graph to trig file."""
        if self.schema_path:
            processor.set_schema(self.schema_path)

        if processor.graph.metadata_header:
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
        """Serialize cim graph to CIMXML file."""
        if not processor.graph.metadata_header:
            logger.error("Serializing without an extracted header may create a corrupt CIMXML file.")

        processor.graph.serialize(self.file_path, format="cimxml", qualifier=self.qualifier)

class JSONLDStrategy(SerializationStrategy):
    """Strategy class for sending a cim graph to JSON-LD file.
    Not implemented.
    """

    def __init__(self, file_path: str|Path) -> None:
        self.file_path = file_path

    def serialize(self, processor: "CIMProcessor"):
        """Serialize cim graph to JSON-LD file. Not implemented."""
        raise NotImplementedError("JSON-LD format output is not implemented.")


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
        # allowed = {"context"}  # future options
        # self._validate_options(options, allowed)
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