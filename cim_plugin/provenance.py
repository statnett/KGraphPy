"""Provenance class for handling information about the source and history of graphs in the CIM plugin."""

from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Callable, Optional, TypeVar, ParamSpec


@dataclass(frozen=True)
class ProvenanceEntry:
    step_name: str
    timestamp: str
    description: str

class Provenance:
    def __init__(self, first_description: str):
        """Initialize the Provenance instance with a first entry.
        
        Parameters:
            first_description (str): A string describing the first step or operation being logged.
        """
        self._entries: list[ProvenanceEntry] = []
        self._changed: bool = False
        self._add_entry(
            step_name="load_graph",
            description=first_description,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    @property
    def entries(self) -> tuple[dict[str, str], ...]:
        """Return a tuple of provenance entries as dictionaries. The returned entries cannot be modified."""
        return tuple([asdict(entry) for entry in self._entries])
        
    def _add_entry(self, step_name: str, description: str, timestamp: str) -> None:
        """Add a provenance entry.
        
        Parameters:
            step_name (str): A string representing the name of the step or operation being logged.
            description (str): A string describing the step or operation being logged.
            timestamp (str): A string in ISO format representing the time the step was executed.
        """
        entry = ProvenanceEntry(
            step_name=step_name,
            timestamp=timestamp,
            description=description,
        )
        self._entries.append(entry)

    def export(self, file_path: str|Path, format: str = "json") -> None:
        """Export the provenance information to a file.

        Parameters:
            file_path (str|Path): The path to the file where the provenance information will be exported.
            format (str): The format in which to export the provenance information. Currently, only "json" is supported.
        """
        if format == "json":
            with open(file_path, "w") as f:
                json.dump([entry.__dict__ for entry in self._entries], f, indent=4)
        # elif format == "": # Other formats can be implemented here
        else:
            raise ValueError(f"Unsupported format for provenance export: {format}")

    def mark_changed(self) -> None:
        """Mark that the graph has been changed and that a new provenance entry should be logged."""
        self._changed = True

    def consume_change_flag(self) -> bool:
        """Check if provenance should be written (if changes have been made.)

        Returns:
            bool: True if changes have been made since the last check, False otherwise.
        """
        changed = self._changed
        self._changed = False
        return changed


P = ParamSpec("P")
T = TypeVar("T")

def log_provenance(step_name: str, custom_description: Optional[str|Callable[..., str]] = None) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to log provenance information for a graph.

    When applied to a function or method, this decorator will automatically log a provenance entry after the function is executed, if the returned object has a _provenance attribute. 
    The _provenance is tied to each separate graph, so if the function modifies a graph, the provenance will be updated for that graph.

    Parameters:
        step_name (str): A string representing the name of the step or operation being logged.
        custom_description (Optional[str|Callable[..., str]]): An optional string or callable that generates a description of the step.
                          If a callable is provided, it will be called with the same arguments as the decorated function to generate a dynamic description. 
                          If not provided, a default description will be generated based on the step name and function arguments.

    Returns:
        Callable[[Callable[P, T]], Callable[P, T]]: A decorator function that can be applied to log what it does to the graph in its provenance record.    
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            result = func(*args, **kwargs)
            
            prov = getattr(result, "_provenance", None)
            if prov is None and args:   # Fallback for methods and functions that returns None, but modifies the graph in-place.
                prov = getattr(args[0], "_provenance", None)

            if prov and prov.consume_change_flag():  # Only log if the function/method indicates that a change has been made.
                if isinstance(custom_description, str):
                    description = custom_description
                elif callable(custom_description):
                    description = custom_description(*args, **kwargs)
                else:
                    description = f"{step_name} executed with args: {args}, kwargs: {kwargs}"

   
                prov._add_entry(
                    step_name=step_name,
                    description=description,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            return result
        return wrapper
    return decorator

if __name__ == "__main__":
    print("Provenance class")