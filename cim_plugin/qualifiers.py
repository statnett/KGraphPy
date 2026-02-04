from abc import ABC, abstractmethod
from rdflib import URIRef

class CIMQualifierStrategy(ABC):

    @abstractmethod
    def matches(self, uri: str) -> bool:
        pass

    @abstractmethod
    def extract_uuid(self, uri: str) -> str:
        """Return the UUID portion."""
        pass

    @abstractmethod
    def build_about(self, uuid: str) -> str:
        pass

    @abstractmethod
    def build_resource(self, uuid: str) -> str:
        pass


class UnderscoreQualifier(CIMQualifierStrategy):

    def matches(self, uri: str) -> bool:
        return uri.startswith("_") or uri.startswith("#_")

    def extract_uuid(self, uri: str) -> str:
        return uri.lstrip("#_")

    def build_about(self, uuid: str) -> str:
        return f"_{uuid}"

    def build_resource(self, uuid: str) -> str:
        return f"#_{uuid}"


class URNQualifier(CIMQualifierStrategy):

    def matches(self, uri: str) -> bool:
        return uri.startswith("urn:uuid:")

    def extract_uuid(self, uri: str) -> str:
        return uri.split("urn:uuid:")[1]

    def build_about(self, uuid: str) -> str:
        return f"urn:uuid:{uuid}"

    def build_resource(self, uuid: str) -> str:
        return f"urn:uuid:{uuid}"


class NamespaceQualifier(CIMQualifierStrategy):

    def matches(self, uri: str) -> bool:
        return uri.startswith(":")

    def extract_uuid(self, uri: str) -> str:
        return uri.lstrip(":")

    def build_about(self, uuid: str) -> str:
        return f":{uuid}"

    def build_resource(self, uuid: str) -> str:
        return f":{uuid}"


class CIMQualifierResolver:

    strategies = [
        UnderscoreQualifier(),
        URNQualifier(),
        NamespaceQualifier(),
    ]

    def __init__(self, output_strategy: CIMQualifierStrategy):
        self.output = output_strategy

    def convert_about(self, uri: URIRef) -> str:
        uri_str = str(uri)
        for s in self.strategies:
            if s.matches(uri_str):
                uuid = s.extract_uuid(uri_str)
                return self.output.build_about(uuid)
        # fallback: treat as literal UUID
        return self.output.build_about(uri_str)

    def convert_resource(self, uri: URIRef) -> str:
        uri_str = str(uri)
        for s in self.strategies:
            if s.matches(uri_str):
                uuid = s.extract_uuid(uri_str)
                return self.output.build_resource(uuid)
        return self.output.build_resource(uri_str)


if __name__ == "__main__":
    print("qualifiers for cimxml serializer")