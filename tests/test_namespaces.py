import pytest
from unittest.mock import patch, MagicMock, call
from rdflib import URIRef, Namespace, Graph, Literal, BNode
from rdflib.namespace import RDF
from cim_plugin.exceptions import NamespaceEmptyError
from cim_plugin.namespaces import (
    collect_specific_namespaces, 
    update_namespace_in_triples, 
    validate_and_fix_namespaces, 
    validate_and_fix_namespaces_by_cimtype, 
    STANDARD_NAMESPACES, 
    CGMES_NAMESPACES, 
    PERSISTENT_NAMESPACES
)

# Unit tests collect_specific_namespaces
@pytest.mark.parametrize(
        "triples, expected",
        [
            pytest.param([], {}, id="Empty triples"),
            pytest.param([(URIRef("http://example.com/s1"), URIRef("p1"), URIRef("o1"))], {"ex": URIRef("http://example.com/")}, id="Subject collected"),
            pytest.param([(URIRef("s1"), URIRef("http://example.com/p1"), URIRef("o1"))], {"ex": URIRef("http://example.com/")}, id="Predicate collected"),
            pytest.param([(URIRef("s1"), URIRef("p1"), URIRef("http://example.com/o1"))], {"ex": URIRef("http://example.com/")}, id="Object collected"),
            pytest.param([(URIRef("http://example.com/s1"), URIRef("http://foo.org/ns#/p1"), URIRef("http://bar.org/o1"))], 
                         {"ex": URIRef("http://example.com/"), "foo": URIRef("http://foo.org/ns#"), "bar": URIRef("http://bar.org/")}, id="Namespaces in all"),
            pytest.param([(URIRef("http://notpresent.com/s1"), URIRef("p1"), URIRef("o1"))], {}, id="Namespace not in namespace_manager"),
            pytest.param([(URIRef("http://example.com/s1"), URIRef("p1"), URIRef("o1")), (URIRef("http://example.com/s2"), URIRef("http://bar.org/p1"), URIRef("o1"))], 
                         {"ex": URIRef("http://example.com/"), "bar": URIRef("http://bar.org/")}, id="Multiple triples"),
            pytest.param([(URIRef("http://bar.org/foo/s1"), URIRef("p1"), URIRef("http://bar.org/o1"))], 
                         {"bf": URIRef("http://bar.org/foo/"), "bar": URIRef("http://bar.org/")}, id="Overlapping namespaces"),
            pytest.param([(URIRef("http://bar.org/foo/s1"), URIRef("p1"), URIRef("o1"))], {"bf": URIRef("http://bar.org/foo/")}, id="Overlapping namespaces, only longest present"),
            pytest.param([(URIRef("s1"), URIRef("p1"), Literal("http://example.com/o1"))], {}, id="Literal not collected"),
            pytest.param([(URIRef("www.noslash.coms1"), URIRef("p1"), URIRef("o1"))], {"no": URIRef("www.noslash.com")}, id="Odd namespace"),
            pytest.param([(URIRef("http://example.com/%s1"), URIRef("p1"), URIRef("o1"))], {"ex": URIRef("http://example.com/")}, id="Symbols in name"),
            pytest.param([(URIRef("http://example.com/"), URIRef("p1"), URIRef("o1"))], {"ex": URIRef("http://example.com/")}, id="Namespace is the entire uri"),
        ]
)
def test_collect_specific_namespaces_basic(triples: list[tuple], expected: dict[str, URIRef]) ->None:
    g = Graph()
    g.bind("ex", Namespace("http://example.com/"))
    g.bind("foo", Namespace("http://foo.org/ns#"))
    g.bind("bar", Namespace("http://bar.org/"))
    g.bind("bf", Namespace("http://bar.org/foo/"))
    g.bind("no", Namespace("www.noslash.com"))
    nm = g.namespace_manager

    result = collect_specific_namespaces(triples, nm)
    assert result == expected

def test_collect_specific_namespaces_emptymanager() -> None:
    g = Graph()
    # No namespaces bound to graph
    g.add((URIRef("s1"), RDF.type, Literal("o")))

    result = collect_specific_namespaces(list(g.triples((None, None, None))), g.namespace_manager)
    assert "rdf" in result.keys() # rdf is a default namespace in rdflib
    assert len(result) == 1

def test_collect_specific_namespaces_emptygraf() -> None:
    g = Graph()

    result = collect_specific_namespaces(list(g.triples((None, None, None))), g.namespace_manager)
    assert len(result) == 0


# Unit tests update_namespace_in_triples
@pytest.mark.parametrize(
    "triple, old_ns, new_ns, expected",
    [
        pytest.param(
            (URIRef("http://old.com/a"), URIRef("x"), URIRef("y")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("http://new.com/a"), URIRef("x"), URIRef("y")),
            id="Subject update"
        ),
        pytest.param(
            (URIRef("s"), URIRef("http://old.com/p"), URIRef("o")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("s"), URIRef("http://new.com/p"), URIRef("o")),
            id="Predicate update"
        ),
        pytest.param(
            (URIRef("s"), URIRef("p"), URIRef("http://old.com/o")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("s"), URIRef("p"), URIRef("http://new.com/o")),
            id="Object update"
        ),
        pytest.param(
            (URIRef("s"), URIRef("p"), URIRef("o")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("s"), URIRef("p"), URIRef("o")),
            id="Simple no update"
        ),
        pytest.param(
            (URIRef("http://unrecognized.com/s"), URIRef("http://unrecognized.com/p"), URIRef("http://unrecognized.com/o")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("http://unrecognized.com/s"), URIRef("http://unrecognized.com/p"), URIRef("http://unrecognized.com/o")),
            id="No update, all uris."
        ),
        pytest.param(
            (BNode("s"), URIRef("http://old.com/p"), URIRef("o")),
            "http://old.com/",
            "http://new.com/",
            (BNode("s"), URIRef("http://new.com/p"), URIRef("o")),
            id="Subject is BNode"
        ),
        pytest.param(
            (URIRef("s"), URIRef("http://old.com/p"), BNode("o")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("s"), URIRef("http://new.com/p"), BNode("o")),
            id="Object is BNode"
        ),
        pytest.param(
            (URIRef("s"), URIRef("http://old.com/p"), Literal("o")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("s"), URIRef("http://new.com/p"), Literal("o")),
            id="Object is Literal"
        ),
        pytest.param(
            (URIRef("http://old.comX/a"), URIRef("x"), URIRef("y")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("http://old.comX/a"), URIRef("x"), URIRef("y")),
            id="Partial match -> No update"
        ),
        # This may cause problems!!
        pytest.param(
            (URIRef("http://old.com/a/http://old.com/b"), URIRef("x"), URIRef("y")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("http://new.com/a/http://new.com/b"), URIRef("x"), URIRef("y")),
            id="Multiple occurence of same uri"
        ),
        pytest.param(
            (URIRef("http://old.complex/foo/a"), URIRef("x"), URIRef("y")),
            "http://old.com/",
            "http://new.com/",
            (URIRef("http://old.complex/foo/a"), URIRef("x"), URIRef("y")),
            id="Uri is part of another uri -> No update"
        ),
        pytest.param(
            (URIRef("http://old.com/a"), URIRef("x"), URIRef("y")),
            "http://old.com/",
            "",
            (URIRef("a"), URIRef("x"), URIRef("y")),
            id="New namespace empty"
        ),
    ]
)
def test_update_namespace_in_triples_singletriple(triple: tuple, old_ns: str, new_ns: str, expected: tuple) -> None:
    g = Graph()
    g.add(triple)

    update_namespace_in_triples(g, old_ns, new_ns)
    for s, p, o in g:
        print(s, p, o)
    assert len(g) == 1
    assert expected in g


def test_update_namespace_in_triples_mixedtriples() -> None:
    old_ns = "http://old.com/"
    new_ns = "http://new.com/"

    g = Graph()
    t1 = (URIRef("http://old.com/a"), URIRef("p"), URIRef("o"))
    t2 = (URIRef("s"), URIRef("http://old.com/p"), URIRef("o"))
    t3 = (URIRef("s"), URIRef("p"), URIRef("http://old.com/o"))
    t4 = (URIRef("s"), URIRef("p"), URIRef("o"))  # unchanged

    for t in (t1, t2, t3, t4):
        g.add(t)

    update_namespace_in_triples(g, old_ns, new_ns)

    expected = {
        (URIRef("http://new.com/a"), URIRef("p"), URIRef("o")),
        (URIRef("s"), URIRef("http://new.com/p"), URIRef("o")),
        (URIRef("s"), URIRef("p"), URIRef("http://new.com/o")),
        t4,
    }

    assert len(g) == 4
    for triple in expected:
        assert triple in g


def test_update_namespace_in_triples_multiplereplacements() -> None:
    g = Graph()
    old_ns = "http://old.com/"
    new_ns = "http://new.com/"

    triple = (URIRef("http://old.com/s"), URIRef("http://old.com/p"), URIRef("http://old.com/o"),)
    g.add(triple)

    update_namespace_in_triples(g, old_ns, new_ns)

    expected = (URIRef("http://new.com/s"), URIRef("http://new.com/p"), URIRef("http://new.com/o"),)

    assert len(g) == 1
    assert expected in g
    assert triple not in g


def test_update_namespace_in_triples_duplicatesafterreplacement() -> None:
    # This test shows what happends if replacements leads to two uris being made identical
    g = Graph()
    old_ns = "http://old.com/"
    new_ns = "http://new.com/"

    t1 = (URIRef("http://old.com/a"), URIRef("p"), URIRef("o"))
    t2 = (URIRef("http://new.com/a"), URIRef("p"), URIRef("o"))
    g.add(t1)
    g.add(t2)

    assert len(g) == 2  # Number of triples before replacement

    update_namespace_in_triples(g, old_ns, new_ns)

    expected = (URIRef("http://new.com/a"), URIRef("p"), URIRef("o"))

    assert len(g) == 1  # Number of triples after replacement
    assert expected in g


def test_update_namespace_in_triples_emptygraph() -> None:
    g = Graph()

    update_namespace_in_triples(g, "http://old.com/", "http://new.com/")

    assert len(g) == 0


def test_update_namespace_in_triples() -> None:
    g = Graph()
    g.add((URIRef("a"), URIRef("x"), URIRef("y")))
    assert len(g) == 1

    with pytest.raises(NamespaceEmptyError, match="old_namespace cannot be an empty string"):
        update_namespace_in_triples(g, "", "http://new.com/")


# Unit tests validate_and_fix_namespaces
@patch("cim_plugin.namespaces.update_namespace_in_triples")
def test_validate_and_fix_namespaces_emptygraph(mock_update: MagicMock) -> None:
    g = Graph()

    validate_and_fix_namespaces(g, {})

    mock_update.assert_not_called()
    assert len(g) == 0


@pytest.mark.parametrize(
        "graph_ns, fix_ns",
        [
            pytest.param([], {}, id="Empty graph namespaces and no fix namespaces"),
            pytest.param([("foo", "http://foo.com/")], {}, id="Graph has namespace, fix namespaces empty"),
            pytest.param([], {"foo": "http://foo.com/"}, id="Graph has no namespaces, fix namespaces has one"),
            pytest.param([("foo", "http://foo.com/")], {"foo": "http://foo.com/"}, id="Correct namespace, no update"),
            pytest.param([("foo", "http://foo.com/")], {"bar": "http://bar.com/"}, id="No match, no update"),
        ]
)
@patch("cim_plugin.namespaces.update_namespace_in_triples")
def test_validate_and_fix_namespaces_nofixes(mock_update: MagicMock, graph_ns: tuple, fix_ns: dict) -> None:
    g = Graph()
    for prefix, ns in graph_ns:
        g.bind(prefix, ns)
    g.add((URIRef("http://foo.com/a"), URIRef("http://foo.com/p"), URIRef("http://foo.com/o")))
    old_ns = list(g.namespace_manager.store.namespaces())

    validate_and_fix_namespaces(g, fix_ns)

    mock_update.assert_not_called()
    assert len(g) == 1
    assert (URIRef("http://foo.com/a"), URIRef("http://foo.com/p"), URIRef("http://foo.com/o")) in g
    assert list(g.namespace_manager.store.namespaces()) == old_ns
    

@pytest.mark.parametrize(
        "graph_ns, fix_ns, update_calls",
        [
            pytest.param([("foo", "http://foo.com/")], {"foo": "http://newfoo.com/"}, [("http://foo.com/", "http://newfoo.com/")], id="Correct prefix, updated namespace"),
            pytest.param([("foo", "http://foo.com/"), ("bar", "www.bar.com/")], 
                         {"bar": "http://www.newbar.com/", "foo": "http://newfoo.com/"}, 
                         [("http://foo.com/", "http://newfoo.com/"), ("www.bar.com/", "http://www.newbar.com/")], 
                         id="Multiple updates of namespaces"),
            pytest.param([("foo", "http://foo.com/")], {"foo2": "http://foo.com/"}, [], id="One prefix updated"),
            pytest.param([("foo", "http://foo.com/"), ("bar", "www.bar.com/")], 
                         {"bar2": "www.bar.com/", "foo2": "http://foo.com/"}, 
                         [], 
                         id="Multiple updates of prefix"),
        ]
)
@patch("cim_plugin.namespaces.update_namespace_in_triples")
def test_validate_and_fix_namespaces_fixes(mock_update: MagicMock, graph_ns: tuple, fix_ns: dict, update_calls: list[tuple[str, str]], caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    g.bind("notfoo", "http://notfoo.com/")
    for prefix, ns in graph_ns:
        g.bind(prefix, ns)
    g.add((URIRef("http://notfoo.com/a"), URIRef("http://notfoo.com/p"), URIRef("http://notfoo.com/o")))
    old_ns = list(g.namespace_manager.store.namespaces())

    validate_and_fix_namespaces(g, fix_ns)

    assert len(g) == 1
    assert len(list(g.namespace_manager.store.namespaces())) == len(old_ns)  # No new namespaces should be added, only updated

    # Other namespaces and triples should be unchanged
    assert (URIRef("http://notfoo.com/a"), URIRef("http://notfoo.com/p"), URIRef("http://notfoo.com/o")) in g
    assert g.namespace_manager.store.namespace("notfoo") == URIRef("http://notfoo.com/")

    # Changes
    for prefix, ns in fix_ns.items():
        assert g.namespace_manager.store.namespace(prefix) == URIRef(ns)

    if update_calls:
        calls = []
        for old_ns, new_ns in update_calls:
            calls.append(call(g, old_ns, new_ns))
        mock_update.assert_has_calls(calls, any_order=True)
        assert "Wrong namespace detected for" in caplog.text
    else:
        mock_update.assert_not_called()
        assert "Wrong prefix detected for" in caplog.text


def test_validate_and_fix_namespaces_integration(caplog: pytest.LogCaptureFixture) -> None:
    g = Graph()
    g.bind("foo", "http://foo.com/")
    g.bind("notfoo", "http://notfoo.com/")
    g.bind("bar", "http://bar.com/")
    g.add((URIRef("http://foo.com/a1"), URIRef("http://foo.com/p1"), URIRef("http://foo.com/o1")))
    g.add((URIRef("http://notfoo.com/a2"), URIRef("http://notfoo.com/p2"), URIRef("http://notfoo.com/o2")))
    g.add((URIRef("http://bar.com/a3"), URIRef("http://bar.com/p3"), URIRef("http://bar.com/o3")))

    validate_and_fix_namespaces(g, {"foo": "http://newfoo.com/", "newbar": "http://bar.com/"})

    assert len(g) == 3
    assert (URIRef("http://newfoo.com/a1"), URIRef("http://newfoo.com/p1"), URIRef("http://newfoo.com/o1")) in g
    assert (URIRef("http://notfoo.com/a2"), URIRef("http://notfoo.com/p2"), URIRef("http://notfoo.com/o2")) in g
    assert (URIRef("http://bar.com/a3"), URIRef("http://bar.com/p3"), URIRef("http://bar.com/o3")) in g
    assert (URIRef("http://foo.com/a1"), URIRef("http://foo.com/p1"), URIRef("http://foo.com/o1")) not in g
    assert g.namespace_manager.store.namespace("foo") == URIRef("http://newfoo.com/")
    assert g.namespace_manager.store.namespace("newbar") == URIRef("http://bar.com/")
    assert g.namespace_manager.store.namespace("notfoo") == URIRef("http://notfoo.com/")
    assert g.namespace_manager.store.namespace("bar") is None  # Old prefix should be removed
    assert "Wrong namespace detected for 'foo': 'http://foo.com/'. Namespace corrected to 'http://newfoo.com/'." in caplog.text
    assert "Wrong prefix detected for 'http://bar.com/': 'bar'. Prefix corrected to 'newbar'." in caplog.text


def test_validate_and_fix_namespaces_namespaceempty(caplog: pytest.LogCaptureFixture) -> None:
    # If the old namespace is empty, no changes occur.
    # This is to prevent all triples without namespaces from getting the new namespace, because this gives ambiguous data.
    g = Graph()
    g.bind("foo", "")
    g.add((URIRef("a"), URIRef("p"), URIRef("o")))

    validate_and_fix_namespaces(g, {"foo": "http://foo.com/"})

    assert (URIRef("a"), URIRef("p"), URIRef("o")) in g
    assert g.namespace_manager.store.namespace("foo") == URIRef("")
    assert "Failed to update namespace for prefix 'foo' due to empty old namespace. Namespace correction skipped." in caplog.text


def test_validate_and_fix_namespaces_namespacenone() -> None:
    # If the old namespace is None, rdflib converts it to 'None'. 
    # No triples have this namespace, so no triples are changed, but the binding is updated to the new namespace.
    # This is an edge case that is very rare, but will cause silent errors when it occurs.
    g = Graph()
    g.bind("foo", None)
    g.add((URIRef("a"), URIRef("p"), URIRef("o")))
    print(g.namespace_manager.store.namespace("foo"))
    validate_and_fix_namespaces(g, {"foo": "http://foo.com/"})

    assert (URIRef("a"), URIRef("p"), URIRef("o")) in g
    assert (URIRef("http://foo.com/a"), URIRef("http://foo.com/p"), URIRef("http://foo.com/o")) not in g
    assert g.namespace_manager.store.namespace("foo") == URIRef("http://foo.com/")

def test_validate_and_fix_namespaces_newnamespaceempty() -> None:
    # A namespace can be changed to empty.
    g = Graph()
    g.bind("foo", "http://foo.com/")
    g.add((URIRef("http://foo.com/a"), URIRef("http://foo.com/p"), URIRef("http://foo.com/o")))

    validate_and_fix_namespaces(g, {"foo": ""})

    assert (URIRef("a"), URIRef("p"), URIRef("o")) in g
    assert g.namespace_manager.store.namespace("foo") == URIRef("")


def test_validate_and_fix_namespaces_defaultnamespace() -> None:
    # Default namespace is changed to a namespace with prefix.
    g = Graph()
    g.bind("", "http://foo.com/")
    g.add((URIRef("http://foo.com/a"), URIRef("http://foo.com/p"), URIRef("http://foo.com/o")))

    validate_and_fix_namespaces(g, {"foo": "http://foo.com/"})

    assert (URIRef("http://foo.com/a"), URIRef("http://foo.com/p"), URIRef("http://foo.com/o")) in g
    assert g.namespace_manager.store.namespace("foo") == URIRef("http://foo.com/")
    assert g.namespace_manager.store.namespace("") is None  # Default namespace should be removed

@patch("cim_plugin.namespaces.update_namespace_in_triples", side_effect=[None, ValueError("some other error")])
def test_validate_and_fix_namespaces_errorsfromcalledfunction(mock_update: MagicMock) -> None:
    g = Graph()
    g.bind("foo", "http://foo.com/")
    g.bind("bar", "http://bar.com/")
    g.add((URIRef("http://foo.com/a"), URIRef("http://foo.com/p"), URIRef("http://foo.com/o")))
    g.add((URIRef("http://bar.com/a"), URIRef("http://bar.com/p"), URIRef("http://bar.com/o")))

    with pytest.raises(ValueError, match="some other error"):
        validate_and_fix_namespaces(g, {"bar": "http://newbar.com/", "foo": "http://newfoo.com/"})
    
    mock_update.assert_has_calls([call(g, "http://foo.com/", "http://newfoo.com/"), call(g, "http://bar.com/", "http://newbar.com/")], any_order=True)
    # Changes made before error should be present
    assert g.namespace_manager.store.namespace("bar") == URIRef("http://newbar.com/")
    assert (URIRef("http://bar.com/a"), URIRef("http://bar.com/p"), URIRef("http://bar.com/o")) in g # Triples are not changed because update_namespace_in_triples is mocked
    # No changes after valueerror
    assert (URIRef("http://foo.com/a"), URIRef("http://foo.com/p"), URIRef("http://foo.com/o")) in g
    assert g.namespace_manager.store.namespace("foo") == URIRef("http://foo.com/")


# Unit tests validate_and_fix_namespaces_by_cimtype
@patch("cim_plugin.namespaces.validate_and_fix_namespaces")
def test_validate_and_fix_namespaces_by_cimtype_default(mock_validate: MagicMock) -> None:
    g = Graph()
    default_namespaces = STANDARD_NAMESPACES|PERSISTENT_NAMESPACES
    validate_and_fix_namespaces_by_cimtype(g, cgmes=False)

    mock_validate.assert_called_once_with(g, default_namespaces)

    assert default_namespaces["cim"] == Namespace("https://cim.ucaiug.io/ns#")  # Check one sample namespace


@patch("cim_plugin.namespaces.validate_and_fix_namespaces")
def test_validate_and_fix_namespaces_by_cimtype_cgmes(mock_validate: MagicMock) -> None:
    g = Graph()
    cgmes_exceptions = CGMES_NAMESPACES|STANDARD_NAMESPACES
    validate_and_fix_namespaces_by_cimtype(g, cgmes=True)

    mock_validate.assert_called_once_with(g, cgmes_exceptions)

    assert cgmes_exceptions["cim"] == Namespace("http://iec.ch/TC57/CIM100#")  # Check one sample namespace


def test_validate_and_fix_namespaces_by_cimtype_dictsnotmodified() -> None:
    before_standard = STANDARD_NAMESPACES.copy()
    before_persistent = PERSISTENT_NAMESPACES.copy()

    validate_and_fix_namespaces_by_cimtype(Graph(), cgmes=False)

    assert STANDARD_NAMESPACES == before_standard
    assert PERSISTENT_NAMESPACES == before_persistent


if __name__ == "__main__":
    pytest.main()