import os
import json
import pytest
import jsonschema

SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), 
    "../neural_scalpel/route/scalpel_route.schema.json"
)

EXAMPLE_PATH = os.path.join(
    os.path.dirname(__file__), 
    "../examples/routes/example.scalpel_route.json"
)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_schema_validates_example_route():
    """Ensure the example route fully complies with the defined schema."""
    schema = load_json(SCHEMA_PATH)
    example = load_json(EXAMPLE_PATH)
    
    # This will raise jsonschema.exceptions.ValidationError if invalid
    jsonschema.validate(instance=example, schema=schema)

def test_schema_rejects_missing_fields():
    schema = load_json(SCHEMA_PATH)
    example = load_json(EXAMPLE_PATH)
    
    # Remove a required field
    del example["diagnostics"]
    
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=example, schema=schema)

def test_schema_rejects_invalid_verdict():
    schema = load_json(SCHEMA_PATH)
    example = load_json(EXAMPLE_PATH)
    
    # Set an invalid verdict
    example["diagnostics"]["verdict"] = "UNKNOWN_STATUS"
    
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=example, schema=schema)
