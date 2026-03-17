import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from cim_plugin.processor import CIMProcessor

logger = logging.getLogger('cimxml_logger')


class SerializationStrategy:
    def serialize(self, processor: "CIMProcessor"):
        raise NotImplementedError


class TriGStrategy(SerializationStrategy):
    def __init__(self, file_path: str|Path, schema_path: Optional[str|Path] = None, enrich_datatypes: bool = False) -> None:
        self.file_path = file_path
        self.schema_path = schema_path
        self.enrich_datatypes = enrich_datatypes

    def serialize(self, processor):
        if self.schema_path:
            processor.set_schema(self.schema_path)

        if processor.graph.metadata_header:
            processor.merge_header()

        if self.enrich_datatypes:
            if processor.schema:
                processor.enrich_literal_datatypes()
            else:
                logger.error("Cannot enrich datatypes without schema.")

        processor.graph.serialize(self.file_path, format="trig")


class CIMXMLStrategy(SerializationStrategy):
    def __init__(self, file_path: str|Path, qualifier: Optional[str] = None) -> None:
        self.file_path = file_path
        self.qualifier = qualifier

    def serialize(self, processor):
        if not processor.graph.metadata_header:
            logger.error("Serializing without an extracted header may create a corrupt CIMXML file.")
        
        processor.graph.serialize(self.file_path, format="cimxml", qualifier=self.qualifier)


def _select_strategy(format: str, file_path: str|Path, options: dict[str, Any]) -> SerializationStrategy:
    format = format.lower()

    if format == "trig":
        allowed = {"schema_path", "enrich_datatypes"}
        _validate_options(options, allowed)
        return TriGStrategy(file_path, **options)

    elif format == "cimxml":
        allowed = {"qualifier"}
        _validate_options(options, allowed)
        return CIMXMLStrategy(file_path, **options)

    elif format == "jsonld":
        raise ValueError("JSON-LD format has not been implemented yet.")
        # allowed = {"context"}  # future options
        # self._validate_options(options, allowed)
        # return JSONLDStrategy(file_path, **options)

    else:
        raise ValueError(f"Unknown format: {format}")
    

def _validate_options(options: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(
            f"Options {unknown} are not valid for this format. "
            f"Allowed options: {allowed}"
        )


if __name__ == "__main__":
    print("Strategies for serialising to file.")