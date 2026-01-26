import pytest
import uuid
from cim_plugin.utilities import _extract_uuid_from_urn
import logging

logger = logging.getLogger('cimxml_logger')

# Unit tests _extract_uuid_from_urn
@pytest.mark.parametrize(
    "input, expected, error_match", [
        pytest.param("urn:uuid:6ba7b810-9dad-11d1-80b4-00c04fd430c8", 
                     "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
                     None, 
                     id="Version 1 (time-based)"),
        pytest.param("urn:uuid:f81d4fae-7dec-3f00-a7a5-21f12f3f3e21",
                     "f81d4fae-7dec-3f00-a7a5-21f12f3f3e21",
                     None,
                     id="Version 3 (namespace + MD5)"),
        pytest.param("urn:uuid:550e8400-e29b-41d4-a716-446655440000",
                     "550e8400-e29b-41d4-a716-446655440000",
                     None,
                     id="Version 4 (random)"),
        pytest.param("urn:uuid:6ba7b811-9dad-11d1-80b4-00c04fd430c8",
                     "6ba7b811-9dad-11d1-80b4-00c04fd430c8",
                     None,
                     id="Version 5 (namespace + SHA-1)"),
        pytest.param("urn:uuid:550E8400-E29B-41D4-A716-446655440000", "550e8400-e29b-41d4-a716-446655440000", None, id="Uppercase UUID" ),
        pytest.param("urn:uuid:550e8400-e29b-41d4-a716-446655440000 ", "550e8400-e29b-41d4-a716-446655440000", "badly formed hexadecimal UUID string", id="Trailing whitespace" ),
        pytest.param(" urn:uuid:550e8400-e29b-41d4-a716-446655440000", None, "Invalid model URI:", id="Leading whitespace breaks prefix" ),
        pytest.param("urn1234", None, "Invalid model URI:", id="Invalid prefix"),
        pytest.param("urn:UUID:550e8400-e29b-41d4-a716-446655440000", None, "Invalid model URI:", id="Wrong case in prefix" ),
        pytest.param("urn:uuid:", None, "badly formed hexadecimal UUID string", id="Missing UUID after prefix" ),
        pytest.param("urn:uuid:not-a-uuid", None, "badly formed hexadecimal UUID string", id="Malformed UUID - non-hex" ),
        pytest.param("urn:uuid:1234", None, "badly formed hexadecimal UUID string", id="Malformed UUID - too short" ),
        pytest.param("urn:uuid:550e8400-e29b-41d4-a716-446655440000-extra", None, "badly formed hexadecimal UUID string", id="Extra characters after UUID" ),
    ]
)
def test_extract_uuid_from_urn(input: str, expected: str|None, error_match: str|None) -> None:
    if error_match:
        with pytest.raises(ValueError, match=error_match):
            _extract_uuid_from_urn(input)
    else:
        result = _extract_uuid_from_urn(input)
        assert result == uuid.UUID(expected)

if __name__ == "__main__":
    pytest.main()