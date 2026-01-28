

class CIMXMLParseError(Exception):
    def __init__(self, file_path, original_exception):
        """Error when parsing CIMXML file."""
        super().__init__(f"Failed to parse {file_path}: {original_exception}")
        self.file_path = file_path
        self.original_exception = original_exception


class LiteralCastingError(Exception):
    """Error when casting datatype of a Literal."""
    pass

if __name__ == "__main__":
    print("exceptions for cim_plugin")