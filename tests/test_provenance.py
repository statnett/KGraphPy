import pytest
import json
from datetime import datetime, timezone
from typing import Callable, Generator
from pathlib import Path
from cim_plugin.provenance import Provenance, log_provenance
from tests.fixtures import provenance_instance, ProvenanceTestClass

# Unit tests Provenance
def test_provenance_initialization() -> None:
    prov = Provenance("Initial load")
    assert len(prov.entries) == 1
    assert prov.entries[0]["step_name"] == "load_graph"
    assert prov.entries[0]["description"] == "Initial load"
    timestamp = prov.entries[0]["timestamp"]
    assert isinstance(timestamp, str)
    assert datetime.fromisoformat(timestamp).tzinfo is not None # If it parses it is a valid ISO format.


def test_provenance_entries() -> None:
    prov = Provenance("Initial load")

    entries = prov.entries
    assert isinstance(entries, tuple)
    assert isinstance(entries[0], dict)
    entries[0]["step_name"] = "modified"
    assert prov.entries[0]["step_name"] == "load_graph" # Modifying the returned entries does not modify the actual Provenance instance.


def test_provenance_entriesnotadded() -> None:
    prov = Provenance("Initial load")
    entries = prov.entries
    with pytest.raises(IndexError):
        entries[1]["step_name"] = "second_step" # Trying to add to the entries raises an error

def test_provenance_addentry() -> None:
    prov = Provenance("Initial load")
    prov._add_entry("second_step", "Second step description", datetime.now(timezone.utc).isoformat())
    assert len(prov.entries) == 2
    assert prov.entries[1]["step_name"] == "second_step"
    assert prov.entries[1]["description"] == "Second step description"
    timestamp = prov.entries[1]["timestamp"]
    assert isinstance(timestamp, str)
    assert datetime.fromisoformat(timestamp).tzinfo is not None


def test_provenance_exportjson(tmp_path: Path) -> None:
    prov = Provenance("Initial load")
    prov._add_entry("second_step", "Second step description", datetime.now(timezone.utc).isoformat())
    file_path = tmp_path / "provenance.json"
    prov.export(file_path, format="json")

    with open(file_path, "r") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["step_name"] == "load_graph"
    assert data[0]["description"] == "Initial load"
    assert data[1]["step_name"] == "second_step"
    assert data[1]["description"] == "Second step description"

def test_provenance_exportunsupportedformat() -> None:
    prov = Provenance("Initial load")
    with pytest.raises(ValueError) as exc:
        prov.export("provenance.txt", format="txt")
    assert "Unsupported format for provenance export: txt" in str(exc.value)

# Unit tests log_provenance
def test_log_provenance_addsentry(provenance_instance: Callable[..., ProvenanceTestClass]) -> None:  
    prov = provenance_instance(5)

    # *args and **kwards should be given in the decorator if not all inputs are used
    @log_provenance("test_step", lambda x, *args, **kwargs: f"Test function executed with argument: {x}")
    def test_function(x, prov):
        prov.data *= x
        return prov
    
    result = test_function(5, prov)
    
    assert result.data == 25
    assert len(result._provenance.entries) == 2
    assert result._provenance.entries[1]["step_name"] == "test_step"
    assert "Test function executed with argument: 5" in result._provenance.entries[1]["description"]
    # Timestamp
    timestamp = result._provenance.entries[1]["timestamp"]
    assert isinstance(timestamp, str)
    assert datetime.fromisoformat(timestamp) is not None
    assert datetime.fromisoformat(timestamp).tzinfo is not None

def test_log_provenance_notallargs(provenance_instance: Callable[..., ProvenanceTestClass]) -> None:
    prov = provenance_instance(5)

    # If the function arguments is not added to the decorator callable, a TypeError is raised.
    @log_provenance("test_step", lambda: "static description")
    def test_function(x):
        prov.data *= x
        return prov

    with pytest.raises(TypeError) as exc:
        test_function(5)

    assert "takes 0 positional arguments but 1 was given" in str(exc.value)

def test_log_provenance_simpledescription(provenance_instance: Callable[..., ProvenanceTestClass]) -> None:
    prov = provenance_instance(5)

    # The description can also be a simple string, in which case it will be used as is without calling it.
    @log_provenance("test_step", "static description")
    def test_function(x):
        prov.data *= x
        return prov

    result = test_function(5)

    entry = result._provenance.entries[-1]
    assert entry["description"] == "static description"

def test_log_provenance_descriptionkwargs(provenance_instance: Callable[..., ProvenanceTestClass]) -> None:
    prov = provenance_instance(5)

    def desc_fn(x, *, scale):
        return f"x={x}, scale={scale}"

    @log_provenance("step", desc_fn)
    def test_function(x, *, scale):
        prov.data *= scale
        return prov

    result = test_function(5, scale=3)

    entry = result._provenance.entries[-1]
    assert entry["description"] == "x=5, scale=3"


def test_log_provenance_nodescription(provenance_instance: Callable[..., ProvenanceTestClass]) -> None:  
    prov = provenance_instance(5)

    @log_provenance("test_step")
    def test_function(x):
        prov.data *= 2
        return prov
    
    result = test_function(5)
    
    assert result.data == 10
    assert len(result._provenance.entries) == 2
    assert result._provenance.entries[1]["step_name"] == "test_step"
    assert "test_step executed with args: (5,), kwargs: {}" in result._provenance.entries[1]["description"]


def test_log_provenance_noprovenanceattribute() -> None:
    @log_provenance("step")
    def myfunc():
        return 123

    result = myfunc()
    assert not hasattr(result, "_provenance")
    assert result == 123


def test_log_provenance_callsfunctiononce(provenance_instance: Callable[..., ProvenanceTestClass]) -> None:
    # Making sure that the decorator does not cause the function to be called multiple times.
    prov = provenance_instance(2)
    call_count = {"n": 0}

    @log_provenance("step")
    def test_function():
        call_count["n"] += 1
        return prov

    test_function()
    assert call_count["n"] == 1

def test_log_provenance_multiplecalls(provenance_instance: Callable[..., ProvenanceTestClass]) -> None:
    prov = provenance_instance(1)

    @log_provenance("step", lambda x: f"Called test_function: {x}")
    def test_function(x):
        prov.data += x
        return prov

    test_function(1)
    test_function(2)

    # The provenance should have entries for both calls in the correct order.
    assert len(prov._provenance.entries) == 3
    assert prov._provenance.entries[1]["description"] == "Called test_function: 1"
    assert prov._provenance.entries[2]["description"] == "Called test_function: 2"


def test_log_provenance_classmethod(provenance_instance: Callable[..., ProvenanceTestClass]) -> None:
    class Dummy:
        def __init__(self):
            self.prov = provenance_instance(10)

        @log_provenance("step")
        def run(self, amount):
            self.prov.data += amount
            return self.prov

    d = Dummy()
    result = d.run(5)

    assert result.data == 15
    assert result._provenance.entries[-1]["step_name"] == "step"

if __name__ == "__main__":
    pytest.main()