from enum import Enum
from abc import ABC, abstractmethod
from typing import Any


class ServingMode(str, Enum):
    INTERNAL = "internal"
    EXTERNAL_PROXY = "external_proxy"
    NATIVE_LORA = "native_lora"
    FAIL_CLOSED = "fail_closed"
    AUTO = "auto"


class ServingEngine(ABC):
    @abstractmethod
    async def infer(self, req: Any, tenant_ctx: Any, audit_ref: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_health(self) -> dict:
        raise NotImplementedError
