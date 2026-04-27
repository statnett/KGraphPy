"""
Functions for finding whether rdf:ID or rdf:about should be used for each uuid.
This file must be completely rewritten when that information is moved into linkML files.
Therefore these functions have only the most basic tests.
"""

import yaml
from rdflib import URIRef
from pathlib import Path
import logging

logger = logging.getLogger('cimxml_logger')

file_path = Path.cwd() /"cim_plugin" / "id_profiles.yaml"

with open(file_path) as f:
    config = yaml.safe_load(f)

PREFIXES = config["prefixes"]
PROFILES = config["profiles"]

def expand(uri: str) -> str:
    """Expand prefix:local into a full URI."""
    if ":" not in uri:
        return uri
    prefix, local = uri.split(":", 1)
    return PREFIXES.get(prefix, prefix + ":") + local


def find_rdf_id_or_about(profiles: list[str]|None, obj_type: str|URIRef) -> str:
    if not profiles:
        logger.error("No profile found. Defaults to 'about'.")
        return "about"
    
    if isinstance(obj_type, URIRef):
        obj_type = str(obj_type)

    expanded_object = expand(obj_type)
    
    keywords: set[str] = set()
    for profile in profiles:
        if profile not in PROFILES:
            keywords.add("about")
            continue
        
        expanded_exceptions = [expand(x) for x in PROFILES[profile]]
    
        if expanded_object in expanded_exceptions:
            keywords.add("about")
            continue

        keywords.add("ID")

    if "ID" in keywords:
        return "ID"
    return "about"


if __name__ == "__main__":
    print("Finding the rdf:ID special cases.")
