# Security Policy

## Supported Versions

Neural-Scalpel is currently in alpha (`v1.0.0-alpha`). There are no guarantees of security for production systems at this stage.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Neural-Scalpel (such as an issue with the Hot-Swap Runtime tenant isolation, route manifest validation bypass, or payload tampering), please report it responsibly.

Since this is an experimental research project, please open an Issue on GitHub with the tag `[SECURITY]`. Do not use Neural-Scalpel to serve untrusted external traffic in its current state.

### Known Architectural Risks
- **Pickle / PyTorch Checkpoints**: Neural-Scalpel only officially supports `.safetensors`. Avoid using standard `.pt` or `.bin` files as they can contain malicious executable code.
- **Tenant Isolation**: The FastAPI proxy (Step 4A) uses basic JWT validation. It is not hardened against advanced attacks (e.g., token replay, timing attacks).
- **KV Cache Leakage**: The internal vLLM integration is currently a mocked architecture. Deploying it without fully patched KV Cache handling in vLLM could result in cross-tenant data leakage.
