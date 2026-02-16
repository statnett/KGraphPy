from abc import ABC, abstractmethod
from rdflib import URIRef
from cim_plugin.namespaces import MODEL

uuid_namespace = MODEL

class CIMQualifierStrategy(ABC):
    """Strategy for switching between different types of uuid qualifiers in a cim graph."""

    @abstractmethod
    def matches(self, uri: str) -> bool:
        pass

    @abstractmethod
    def extract_uuid(self, uri: str) -> str:
        """Return the UUID portion."""
        pass

    @abstractmethod
    def build_special(self, uuid: str) -> str:
        pass

    @abstractmethod
    def build_default(self, uuid: str) -> str:
        pass


class UnderscoreQualifier(CIMQualifierStrategy):

    def matches(self, uri: str) -> bool:
        return uri.startswith("_") or uri.startswith("#_")

    def extract_uuid(self, uri: str) -> str:
        return uri.lstrip("#_")

    def build_special(self, uuid: str) -> str:
        return f"_{uuid}"

    def build_default(self, uuid: str) -> str:
        return f"#_{uuid}"


class URNQualifier(CIMQualifierStrategy):

    def matches(self, uri: str) -> bool:
        return uri.startswith("urn:uuid:")

    def extract_uuid(self, uri: str) -> str:
        return uri.split("urn:uuid:")[1]

    def build_special(self, uuid: str) -> str:
        return f"urn:uuid:{uuid}"

    def build_default(self, uuid: str) -> str:
        return f"urn:uuid:{uuid}"


class NamespaceQualifier(CIMQualifierStrategy):

    def matches(self, uri: str) -> bool:
        return uri.startswith(f"{uuid_namespace}:")

    def extract_uuid(self, uri: str) -> str:
        prefix = f"{uuid_namespace}:"
        return uri[len(prefix):]

    def build_special(self, uuid: str) -> str:
        return f"{uuid_namespace}:{uuid}"

    def build_default(self, uuid: str) -> str:
        return f"{uuid_namespace}:{uuid}"


class CIMQualifierResolver:
    """Convert between different types of uuid qualifiers in cim graphs."""

    strategies = [
        UnderscoreQualifier(),
        URNQualifier(),
        NamespaceQualifier(),
    ]

    def __init__(self, output_strategy: CIMQualifierStrategy):
        self.output = output_strategy

    def convert_to_special_qualifier(self, uri: URIRef) -> str:
        uri_str = str(uri)
        for s in self.strategies:
            if s.matches(uri_str):
                uuid = s.extract_uuid(uri_str)
                return self.output.build_special(uuid)
        # fallback: treat as literal UUID
        return self.output.build_special(uri_str)

    def convert_to_default_qualifier(self, uri: URIRef) -> str:
        uri_str = str(uri)
        for s in self.strategies:
            if s.matches(uri_str):
                uuid = s.extract_uuid(uri_str)
                return self.output.build_default(uuid)
        return self.output.build_default(uri_str)


if __name__ == "__main__":
    print("qualifiers for cimxml serializer")