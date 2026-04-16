"""Provenance class for handling information about the source and history of graphs in the CIM plugin."""

from datetime import datetime, timezone
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable, Optional, TypeVar, ParamSpec, List, Dict, Any


@dataclass(frozen=True)
class ProvenanceEntry:
    step_name: str
    timestamp: str
    description: str
    sub_steps: List["ProvenanceEntry"] = field(default_factory=list)


    def to_dict(self) -> Dict[str, Any]:
        output: Dict[str, Any] = {
            "step_name": self.step_name,
            "timestamp": self.timestamp,
            "description": self.description,
        }
        if self.sub_steps:
            output["sub_steps"] = [sub.to_dict() for sub in self.sub_steps]
        return output


class Provenance:
    def __init__(self, first_description: str):
        """Initialize the Provenance instance with a first entry.
        
        Parameters:
            first_description (str): A string describing the first step or operation being logged.
        """
        self._entries: List[ProvenanceEntry] = []
        self._stack: List[Dict[str, Any]] = []  # For collecting subentries during function execution
        self._entries.append(ProvenanceEntry(
            step_name="load_graph",
            description=first_description,
            timestamp=datetime.now(timezone.utc).isoformat()
        ))

    @property
    def entries(self) -> tuple[dict[str, Any], ...]:
        """Return a tuple of provenance entries as dictionaries."""
        return tuple(entry.to_dict() for entry in self._entries)
    
        
    def _add_entry(self, step_name: str, description: str, timestamp: str, sub_steps: List[ProvenanceEntry]|None = None) -> None:
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
            sub_steps=sub_steps or []
        )
        if self._stack:
            self._stack[-1]["entries"].append(entry)
        else:
            self._entries.append(entry)

    def export(self, file_path: str|Path, format: str = "json") -> None:
        """Export the provenance information to a file.

        Parameters:
            file_path (str|Path): The path to the file where the provenance information will be exported.
            format (str): The format in which to export the provenance information. Currently, only "json" is supported.
        """
        if format == "json":
            with open(file_path, "w") as f:
                json.dump([entry.to_dict() for entry in self._entries], f, indent=4)
        # elif format == "": # Other formats can be implemented here
        else:
            raise ValueError(f"Unsupported format for provenance export: {format}")

    def mark_changed(self) -> None:
        """Mark that the graph has been changed and that a new provenance entry should be logged."""
        if self._stack:
            self._stack[-1]["changed"] = True
        else:
            self._entries.append(
                ProvenanceEntry(
                    step_name="illegal_entry", 
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    description="Change marked outside of a provenance logged context."
                ))


P = ParamSpec("P")
T = TypeVar("T")

def log_provenance(step_name: str, custom_description: Optional[str|Callable[..., str]] = None) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to log provenance information for a graph.

    When applied to a function or method, this decorator will automatically log a provenance entry after the function is executed, if the first argument has a _provenance attribute. 
    Will also record provenance for any called functions/methods if it also has the decorator applied, and nest those entries as sub-steps under the main entry for the outer function.

    Parameters:
        step_name (str): A string representing the name of the step or operation being logged.
        custom_description (Optional[str|Callable[..., str]]): An optional string or callable that generates a description of the step.
                          If a callable is provided, it will be called with the same arguments as the decorated function to generate a dynamic description. 
                          If not provided, a default description will be generated based on the step name and function arguments.

    Returns:
        Callable[[Callable[P, T]], Callable[P, T]]: A decorator that can be applied to a function/method to log what it does to the graph in its provenance record.    
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not args:
                return func(*args, **kwargs)  # If there are no arguments, we cannot check for provenance, so just execute the function.
            
            self_obj = args[0]  # Assuming the first argument is 'self' for methods, or the graph for functions.
            prov = getattr(self_obj, "_provenance", None)  # Try to get provenance from the first argument

            if prov is None:
                return func(*args, **kwargs)  # If no provenance is found, just execute the function without logging
            
            prov._stack.append({"entries": [], "changed": False})  # Start a new stack level to collect subentries

            try:
                result = func(*args, **kwargs)
            finally:
                frame = prov._stack.pop()  # Get the current stack level's subentries

            if not frame["changed"]:
                return result  # If no changes were made, do not log an entry
                    
            
            if isinstance(custom_description, str):
                description = custom_description
            elif callable(custom_description):
                description = custom_description(*args, **kwargs)
            else:
                description = f"{step_name} executed with args: {args}, kwargs: {kwargs}"

            prov._add_entry(
                step_name=step_name,
                description=description,
                timestamp=datetime.now(timezone.utc).isoformat(),
                sub_steps=frame["entries"]
            )

            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    print("Provenance class")