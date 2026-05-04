import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys

class AuditLogger:
    """
    SRE-compliant Audit Logger for Neural-Scalpel Hot-Swap Runtime.
    Logs 100% of runtime events to a structured JSON-L file.
    """
    def __init__(self, log_file_path: str):
        self.logger = logging.getLogger(f"NeuralScalpelAudit_{log_file_path}")
        self.logger.setLevel(logging.INFO)
        
        # Prevent adding multiple handlers if instantiated multiple times
        if not self.logger.handlers:
            # Ensure directory exists
            Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
            
            handler = logging.FileHandler(log_file_path)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
            
            # Optional: duplicate to stdout for immediate debugging
            # stdout_handler = logging.StreamHandler(sys.stdout)
            # stdout_handler.setFormatter(logging.Formatter('AUDIT: %(message)s'))
            # self.logger.addHandler(stdout_handler)

    def log_event(self, request_id: str, tenant_id: str, route_id: str, event: str, status: str, latency_ms: float = 0.0, **kwargs):
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "tenant_id": tenant_id,
            "route_id": route_id,
            "event": event,
            "status": status,
            "latency_ms": round(latency_ms, 2)
        }
        payload.update(kwargs)
        self.logger.info(json.dumps(payload))
