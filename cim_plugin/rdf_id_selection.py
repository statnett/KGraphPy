import yaml
from rdflib import URIRef
from pathlib import Path
import logging

logger = logging.getLogger('cimxml_logger')

file_path = Path.cwd() /"cim_plugin" / "id_profiles.yaml"

with open(file_path) as f:
    config = yaml.safe_load(f)

prefixes = config["prefixes"]
profiles = config["profiles"]

def expand(uri: str) -> str:
    """Expand prefix:local into a full URI."""
    if ":" not in uri:
        return uri
    prefix, local = uri.split(":", 1)
    return prefixes.get(prefix, prefix + ":") + local


def find_rdf_id_or_about(profile: str|None, obj_type: str|URIRef) -> str:
    if not profile:
        logger.error("No profile found. Defaults to 'about'.")
        return "about"
    
    if profile not in profiles:
        return "about"

    if isinstance(obj_type, URIRef):
        obj_type = str(obj_type)

    expanded_object = expand(obj_type)
    expanded_exceptions = [expand(x) for x in profiles[profile]]
    
    if expanded_object in expanded_exceptions:
        return "about"

    return "ID"


if __name__ == "__main__":
    print("Finding the rdf:ID special cases.")
