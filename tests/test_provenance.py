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
    assert prov._stack == []    # The stack is empty because the decorator has not been used.


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
    assert prov._entries[1].sub_steps == [] # No sub-steps were added in this test.
    assert prov._stack == []    # The stack is still empty because the decorator has not been used.


def test_provenance_addentrywithsubsteps() -> None:
    prov = Provenance("Initial load")
    prov._stack.append({"entries": [], "changed": False})
    prov._add_entry("sub_step", "Sub step description", datetime.now(timezone.utc).isoformat())
    frame = prov._stack.pop()
    prov._add_entry("second_step", "Second step description", datetime.now(timezone.utc).isoformat(), sub_steps=frame["entries"])
    assert len(prov.entries) == 2
    assert prov.entries[1]["step_name"] == "second_step"
    assert prov.entries[1]["description"] == "Second step description"
    assert prov._entries[1].sub_steps == frame["entries"]
    assert prov.entries[1]["sub_steps"] == [sub.to_dict() for sub in frame["entries"]]
    assert prov._stack == []    # The stack is empty because of .pop().


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


def test_provenance_exportjsonnested(tmp_path: Path) -> None:
    prov = Provenance("Initial load")
    prov._stack.append({"entries": [], "changed": False}) # Simulating being inside a decorator context by adding an entry to the stack.
    prov._add_entry("sub_step", "Sub step description", datetime.now(timezone.utc).isoformat())
    frame = prov._stack.pop()
    prov._add_entry("second_step", "Second step description", datetime.now(timezone.utc).isoformat(), sub_steps=frame["entries"])
    file_path = tmp_path / "provenance.json"
    prov.export(file_path, format="json")

    with open(file_path, "r") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 2
    assert data[1]["step_name"] == "second_step"
    assert data[1]["description"] == "Second step description"
    assert isinstance(data[1]["sub_steps"], list)
    assert len(data[1]["sub_steps"]) == 1
    assert data[1]["sub_steps"][0]["step_name"] == "sub_step"
    assert data[1]["sub_steps"][0]["description"] == "Sub step description"
    assert prov._stack == []    # The stack is empty again after popping the frame.


def test_provenance_exportunsupportedformat() -> None:
    prov = Provenance("Initial load")
    with pytest.raises(ValueError) as exc:
        prov.export("provenance.txt", format="txt")
    assert "Unsupported format for provenance export: txt" in str(exc.value)

def test_mark_changed_outsidedecorator() -> None:
    prov = Provenance("Initial load")
    prov.mark_changed()
    assert prov.entries[-1]["step_name"] == "illegal_entry" # mark_changed being used outside of the decorator context gives an illegal entry.


def test_mark_changed_markingtrue() -> None:
    prov = Provenance("Initial load")
    prov._stack.append({"entries": [], "changed": False}) # Simulating being inside a decorator context by adding an entry to the stack.
    prov.mark_changed()
    assert prov._stack[-1]["changed"] == True


# Unit tests log_provenance
def test_log_provenance_addsentry(provenance_instance: Callable[..., ProvenanceTestClass]) -> None:  
    prov = provenance_instance(5)

    # *args and **kwargs should be given in the decorator if not all inputs are used
    @log_provenance("test_step", lambda prov, x: f"Test function executed with argument: {x}")
    def test_function(prov, x):
        prov.data *= x
        prov._provenance.mark_changed()
        return prov
    
    result = test_function(prov, 5)
    
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
    def test_function(prov, x):
        prov.data *= x
        prov._provenance.mark_changed()
        return prov

    with pytest.raises(TypeError) as exc:
        test_function(prov, 5)

    assert "takes 0 positional arguments but 2 were given" in str(exc.value)


def test_log_provenance_notfirstarg(provenance_instance: Callable[..., ProvenanceTestClass]) -> None:
    prov = provenance_instance(5)

    # If the class containing the _provenance attribute is not the first argument, an AttributeError is raised.
    @log_provenance("test_step", "static description")
    def test_function(x, prov):
        prov.data *= x
        prov._provenance.mark_changed()
        return prov

    with pytest.raises(AttributeError) as exc:
        test_function(prov, 5)

    assert "'int' object has no attribute 'data'" in str(exc.value)


def test_log_provenance_nestedfunctions(provenance_instance: Callable[..., ProvenanceTestClass]) -> None:  
    prov = provenance_instance(5)

    @log_provenance("inner_step", lambda prov, x: f"Inner function executed with argument: {x}")
    def inner_function(prov, x):
        prov.data *= x
        prov._provenance.mark_changed()
        return prov
    
    @log_provenance("outer_step", lambda prov, x: f"Outer function executed with argument: {x}")
    def outer_function(prov, x):
        inner_function(prov, x)
        prov._provenance.mark_changed() # Marking changed in the outer function to log an entry for it as well.

    outer_function(prov, 5)
    
    assert prov.data == 25
    assert len(prov._provenance.entries) == 2
    assert prov._provenance.entries[1]["step_name"] == "outer_step"
    assert "Outer function executed with argument: 5" in prov._provenance.entries[1]["description"]
    substeps = prov._provenance.entries[1]["sub_steps"]
    assert isinstance(substeps, list)
    assert len(substeps) == 1
    assert substeps[0]["step_name"] == "inner_step"
    assert "Inner function executed with argument: 5" in substeps[0]["description"]

def test_log_provenance_simpledescription(provenance_instance: Callable[..., ProvenanceTestClass]) -> None:
    prov = provenance_instance(5)

    # The description can also be a simple string, in which case it will be used as is without calling it.
    @log_provenance("test_step", "static description")
    def test_function(prov, x):
        prov.data *= x
        prov._provenance.mark_changed()
        return prov

    result = test_function(prov, 5)

    entry = result._provenance.entries[-1]
    assert entry["description"] == "static description"

@pytest.mark.parametrize("mark", [True, False])
def test_log_provenance_markchangedcalls(provenance_instance: Callable[..., ProvenanceTestClass], mark: bool) -> None:
    prov = provenance_instance(5)

    @log_provenance("test_step", "static description")
    def test_function(prov, x, mark_change: bool):
        prov.data *= x
        if mark_change:
            prov._provenance.mark_changed()
        return prov

    result = test_function(prov, 5, mark)

    assert result._provenance   # Provenance has been initialized.
    if mark:
        assert len(result._provenance.entries) == 2
        assert result._provenance.entries[1]["step_name"] == "test_step"
        assert result._provenance.entries[1]["description"] == "static description"
    else:
        assert len(result._provenance.entries) == 1 # No new entry is added because mark_changed was not called.
    

def test_log_provenance_descriptionkwargs(provenance_instance: Callable[..., ProvenanceTestClass]) -> None:
    # All *args and **kwargs are passed to the decorator callable, so the description can be customized based on the function inputs.
    prov = provenance_instance(5)

    def desc_fn(prov, x, *, scale):
        return f"x={x}, scale={scale}"

    @log_provenance("step", desc_fn)
    def test_function(prov, x, *, scale):
        prov.data *= scale
        prov._provenance.mark_changed()
        return prov

    result = test_function(prov, 5, scale=3)

    entry = result._provenance.entries[-1]
    assert entry["description"] == "x=5, scale=3"


def test_log_provenance_nodescription(provenance_instance: Callable[..., ProvenanceTestClass]) -> None:  
    prov = provenance_instance(5)

    @log_provenance("test_step")
    def test_function(prov, x):
        prov.data *= 2
        prov._provenance.mark_changed()
        return prov
    
    result = test_function(prov, 5)
    
    assert result.data == 10
    assert len(result._provenance.entries) == 2
    assert result._provenance.entries[1]["step_name"] == "test_step"
    # Standard description is generated if no custom description is given, which includes the function name and the arguments.
    assert f"test_step executed with args: ({prov}, 5), kwargs: " in result._provenance.entries[1]["description"]


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

    @log_provenance("step", lambda prov, x: f"Called test_function: {x}")
    def test_function(prov, x):
        prov.data += x
        prov._provenance.mark_changed()
        return prov

    test_function(prov, 1)
    test_function(prov, 2)

    # The provenance should have entries for both calls in the correct order.
    assert len(prov._provenance.entries) == 3
    assert prov._provenance.entries[1]["description"] == "Called test_function: 1"
    assert prov._provenance.entries[2]["description"] == "Called test_function: 2"


def test_log_provenance_classmethod(provenance_instance: Callable[..., ProvenanceTestClass]) -> None:
    class Dummy:
        def __init__(self):
            self.prov = provenance_instance(10)
            self._provenance = self.prov._provenance    # Works for methods as long as it has a _provenance attribute

        @log_provenance("step")
        def run(self, amount):
            self.prov.data += amount
            self.prov._provenance.mark_changed()
            return self.prov

    d = Dummy()
    result = d.run(5)

    assert result.data == 15
    assert result._provenance.entries[-1]["step_name"] == "step"

if __name__ == "__main__":
    pytest.main()