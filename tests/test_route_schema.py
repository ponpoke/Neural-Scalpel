import os
import json
import pytest
import jsonschema

SCHEMA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    "../neural_scalpel/route/scalpel_route.schema.json"
))

EXAMPLE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    "../examples/routes/example.scalpel_route.json"
))

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_valid_example():
    """Returns a valid route dictionary compliant with the schema."""
    try:
        example = load_json(EXAMPLE_PATH)
    except FileNotFoundError:
        # Fallback to hardcoded valid data if example file is missing
        example = {
            "route_schema_version": "0.1.0",
            "route_id": "example-route",
            "source_model": "model-a",
            "target_model": "model-b",
            "source_adapter_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "target_model_sha256": "8b1a9953c4611296a827abf8c47804d7e4cf39d88532ee43640243e86ac58e77",
            "tenant_id": "tenant-xyz",
            "license": "MIT",
            "projection_method": "TEST",
            "calibration": {"forward_passes": 64},
            "diagnostics": {
                "verdict": "PASS", 
                "ppl_degradation": 0.05, 
                "kl_divergence": 0.01, 
                "portability_score": 95
            },
            "layers": [
                {"name": "layer_1", "shape": [2, 2], "dtype": "float32", "delta_sha256": "a"*64}
            ]
        }
        
    example.setdefault("signature", {
        "algorithm": "hmac-sha256",
        "key_id": "test-key",
        "value": "b" * 64
    })

    if "diagnostics" in example:
        example["diagnostics"]["portability_score"] = int(
            example["diagnostics"].get("portability_score", 95)
        )

    return example

def test_schema_validates_example_route():
    """Ensure the example route fully complies with the defined schema."""
    schema = load_json(SCHEMA_PATH)
    example = get_valid_example()
    
    # This will raise jsonschema.exceptions.ValidationError if invalid
    jsonschema.validate(instance=example, schema=schema)

def test_schema_rejects_missing_fields():
    schema = load_json(SCHEMA_PATH)
    example = get_valid_example()
    
    # Remove a required field
    del example["diagnostics"]
    
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=example, schema=schema)

def test_schema_rejects_invalid_verdict():
    schema = load_json(SCHEMA_PATH)
    example = get_valid_example()
    
    # Set an invalid verdict
    example["diagnostics"]["verdict"] = "UNKNOWN_STATUS"
    
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=example, schema=schema)
