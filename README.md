# KGraphPy – Knowledge Graph DataOps for Python

**KGraphPy** is a Python library for working with knowledge graphs on top of
[RDFLib](https://github.com/RDFLib/rdflib), with a particular focus on
**CIM / CGMES**-style model exchange, change management and dataset packaging.

It extends RDFLib with:

- **CIMXML** parsing and serialisation (`format="cimxml"`)
- **CIM TriG** serialisation (`format="cimtrig"`)
- Metadata header handling (`md:FullModel` and DCAT-3 `dcat:Dataset` headers)
- JSON-LD output and datatype enrichment from LinkML schemas

## Installation

```bash
pip install kgraphpy
```

## Usage

Importing `kgraphpy` registers the parsers and serialisers with RDFLib:

```python
import kgraphpy
from kgraphpy.graph import CIMGraph

g = CIMGraph()  # a plain rdflib.Graph works too
g.parse("model.xml", format="cimxml")
g.serialize("model.trig", format="cimtrig")
```

See the [full documentation](https://github.com/statnett/KGraphPy/blob/main/ReadMe.adoc)
and the `quick_demo.ipynb` notebook in the repository for more.

## License

Apache License 2.0. See [LICENSE](https://github.com/statnett/KGraphPy/blob/main/LICENSE).
