# Incremental Graph Versioner (IGV)
# ===================================
# Production-grade library for incremental GraphRAG index management.
# Built on top of LightRAG (github.com/HKUDS/LightRAG, MIT license)
#
# Usage (fits in a Jupyter notebook):
#   from igv import IGVIndex
#   index = IGVIndex(working_dir="./my_graph")
#   await index.initialize()
#   await index.insert(new_documents)  # auto-dedup + relink + re-partition
#   results = await index.query("What are the main themes?")
from .incremental_index import IGVIndex  # noqa: F401

__version__ = "1.0.0"
