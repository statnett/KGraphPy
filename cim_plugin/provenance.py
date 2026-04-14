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
        self._entries: list[ProvenanceEntry] = []
        self._add_entry(
            step_name="load_graph",
            description=first_description,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    @property
    def entries(self) -> tuple[dict[str, str], ...]:
        return tuple([asdict(entry) for entry in self._entries])
        
    def _add_entry(self, step_name: str, description: str, timestamp: str) -> None:
        """Add a provenance entry."""
        entry = ProvenanceEntry(
            step_name=step_name,
            timestamp=timestamp,
            description=description,
        )
        self._entries.append(entry)

    def export(self, file_path: str|Path, format: str = "json") -> None:
        """Export the provenance information to a file."""
        if format == "json":
            with open(file_path, "w") as f:
                json.dump([entry.__dict__ for entry in self._entries], f, indent=4)
        # elif format == "": # Other formats can be implemented here
        else:
            raise ValueError(f"Unsupported format for provenance export: {format}")
        
        
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
            if prov:
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