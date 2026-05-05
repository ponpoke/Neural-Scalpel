# Neural-Scalpel Security Hardening

## Authentication & Authorization

### JWT Token Validation

All inference requests require a valid JWT token with the following claims:

```json
{
  "tenant_id": "tenant_alpha",
  "iss": "neural-scalpel-auth",
  "aud": "neural-scalpel-api",
  "exp": 1717000000
}
```

**Validated fields:**
- `tenant_id`: Required, non-empty string
- `exp`: Must not be expired
- `iss`: Must match configured issuer (when set)
- `aud`: Must match configured audience (when set)
- Algorithm: HS256 (production should use RS256)

### Admin API Key

Admin endpoints (`/admin/metrics`, `/admin/audit`) require the `X-Admin-Key` header.
The key is configured via the `ADMIN_API_KEY` environment variable.

**Recommendation:** Use a high-entropy key (≥32 random bytes, base64 encoded).

### Tenant Isolation

Each route manifest specifies a `tenant_id`. The runtime enforces:
- Tenant A cannot access routes belonging to Tenant B
- Routes without `tenant_id` are treated as global (shared)

## Input Validation

### Payload Path Traversal Prevention

The payload URI in route manifests is validated to prevent path traversal attacks:
- Relative paths are resolved against a configured base directory
- `..` sequences are rejected
- Symbolic links are not followed
- Only `.safetensors` file extensions are accepted

### Payload Size Limits

Maximum payload file size is enforced before loading:
- Default: 2 GB
- Configurable via `MAX_PAYLOAD_BYTES` environment variable

### Request Size Limits

HTTP request body size is limited by middleware:
- Default: 512 KB
- Prevents memory exhaustion from oversized prompts

## Rate Limiting

Per-tenant token bucket rate limiter:
- Capacity: 200 requests
- Refill rate: 100 requests/second
- Exceeding the limit returns HTTP 429

## Cryptographic Integrity

### Route Manifest Signing

All route manifests are HMAC-SHA256 signed. The signature covers:
- All manifest fields except the `signature` block itself
- Canonical JSON serialization (sorted keys, minimal separators)

Signature verification is mandatory at registration time.
Unsigned or tampered manifests are rejected.

### Payload Integrity

Safetensors payload files are verified against SHA-256 hashes in the manifest:
- File-level hash: Verified before loading
- Per-tensor hash: Optional additional verification

## Network Security

### TLS Configuration

Neural-Scalpel does not implement TLS directly. Deploy behind a TLS-terminating
reverse proxy:

```nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/ssl/neural-scalpel.crt;
    ssl_certificate_key /etc/ssl/neural-scalpel.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
    }
}
```

### Admin Endpoint Access

Admin endpoints should be restricted by IP:
- Configure firewall rules to limit `/admin/*` access
- Or use nginx `allow/deny` directives

## Audit Trail

All security-relevant events are logged to the JSON-L audit file:
- Authentication successes and failures
- Route registration and revocation
- Quarantine events
- Tenant access denials

The audit log should be:
- Written to append-only storage
- Rotated but never deleted
- Forwarded to a centralized log aggregator

## Secret Management

| Secret | Storage | Rotation |
|--------|---------|----------|
| `JWT_SECRET` | Environment variable | Rotate monthly |
| `ADMIN_API_KEY` | Environment variable | Rotate on personnel change |
| Route signing keys | `RouteSigner` config | Rotate quarterly |

**Recommendation:** In production, use a secrets manager (Vault, AWS Secrets Manager)
instead of environment variables.
