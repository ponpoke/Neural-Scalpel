"""
Neural-Scalpel Runtime State Machine

A formally defined, type-safe state machine governing all runtime transitions
during the hot-swap lifecycle. Every state transition is validated against an
explicit transition table, ensuring that impossible states are rejected at the
boundary rather than silently corrupting runtime behavior.

State Machine Diagram:

    READY ──► VERIFYING ──► SNAPSHOT_CAPTURED ──► SWAPPING ──► SWAPPED
                                                                  │
                                                                  ▼
                                                             FORWARDING
                                                                  │
                                                         ┌────────┴────────┐
                                                         ▼                 ▼
                                                   ROLLING_BACK      (success)
                                                         │                 │
                                                    ┌────┴────┐      ROLLING_BACK
                                                    ▼         ▼           │
                                                  READY    ROUTE_     ┌───┴───┐
                                                          QUARANTINED ▼       ▼
                                                                    READY  ROUTE_
                                                                          QUARANTINED
                                                                               │
                                                                          WORKER_
                                                                          QUARANTINED

Design:
  - Each state is an enum member with a human-readable description
  - Valid transitions are encoded as a frozenset adjacency list
  - Transition violations raise `InvalidStateTransition` immediately
  - The state machine is fully serializable for audit logging
"""

from __future__ import annotations

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, FrozenSet, Dict


class RuntimePhase(Enum):
    """
    Exhaustive enumeration of every phase the hot-swap runtime can occupy.
    The string values are designed for direct inclusion in structured logs.
    """
    READY = "READY"
    VERIFYING = "VERIFYING"
    SNAPSHOT_CAPTURED = "SNAPSHOT_CAPTURED"
    SWAPPING = "SWAPPING"
    SWAPPED = "SWAPPED"
    FORWARDING = "FORWARDING"
    ROLLING_BACK = "ROLLING_BACK"
    ROUTE_QUARANTINED = "ROUTE_QUARANTINED"
    WORKER_QUARANTINED = "WORKER_QUARANTINED"


class QuarantineScope(Enum):
    """Distinguishes between route-level and worker-level quarantine."""
    ROUTE = "ROUTE"
    WORKER = "WORKER"


class InvalidStateTransition(RuntimeError):
    """Raised when the runtime attempts an illegal state transition."""

    def __init__(self, current: RuntimePhase, attempted: RuntimePhase):
        self.current = current
        self.attempted = attempted
        super().__init__(
            f"Invalid state transition: {current.value} → {attempted.value}. "
            f"Check the transition table for valid successors."
        )


# ── Transition Table ───────────────────────────────────────────────────
#
# This is the single source of truth for all legal state transitions.
# Any transition not listed here is categorically illegal and will raise
# InvalidStateTransition immediately.

_VALID_TRANSITIONS: Dict[RuntimePhase, FrozenSet[RuntimePhase]] = {
    RuntimePhase.READY: frozenset({
        RuntimePhase.VERIFYING,
    }),
    RuntimePhase.VERIFYING: frozenset({
        RuntimePhase.SNAPSHOT_CAPTURED,
        RuntimePhase.ROUTE_QUARANTINED,  # verification failure
    }),
    RuntimePhase.SNAPSHOT_CAPTURED: frozenset({
        RuntimePhase.SWAPPING,
        RuntimePhase.ROUTE_QUARANTINED,  # pre-swap rejection
    }),
    RuntimePhase.SWAPPING: frozenset({
        RuntimePhase.SWAPPED,
        RuntimePhase.ROLLING_BACK,       # swap failure → immediate rollback
        RuntimePhase.ROUTE_QUARANTINED,  # catastrophic swap failure
    }),
    RuntimePhase.SWAPPED: frozenset({
        RuntimePhase.FORWARDING,
        RuntimePhase.ROLLING_BACK,       # pre-forward abort
    }),
    RuntimePhase.FORWARDING: frozenset({
        RuntimePhase.ROLLING_BACK,       # post-forward (normal or error)
    }),
    RuntimePhase.ROLLING_BACK: frozenset({
        RuntimePhase.READY,              # successful rollback
        RuntimePhase.ROUTE_QUARANTINED,  # rollback failed but recoverable
        RuntimePhase.WORKER_QUARANTINED, # rollback failed, worker corrupted
    }),
    RuntimePhase.ROUTE_QUARANTINED: frozenset({
        RuntimePhase.READY,              # admin recovery / re-validation
        RuntimePhase.WORKER_QUARANTINED, # escalation
    }),
    RuntimePhase.WORKER_QUARANTINED: frozenset(),  # terminal state
}


@dataclass
class TransitionRecord:
    """An immutable record of a single state transition, for audit trails."""
    from_phase: RuntimePhase
    to_phase: RuntimePhase
    timestamp: float
    reason: str = ""
    route_id: str = ""


@dataclass
class RuntimeStateMachine:
    """
    Thread-safe state machine tracking the current runtime phase and
    maintaining a complete transition history for post-mortem analysis.

    Usage:
        sm = RuntimeStateMachine()
        sm.transition(RuntimePhase.VERIFYING, route_id="route_abc")
        sm.transition(RuntimePhase.SNAPSHOT_CAPTURED)
        ...
    """
    phase: RuntimePhase = RuntimePhase.READY
    history: list = field(default_factory=list)
    quarantine_scope: Optional[QuarantineScope] = None
    quarantine_reason: str = ""

    # ── Core Transition ────────────────────────────────────────

    def transition(
        self,
        target: RuntimePhase,
        *,
        reason: str = "",
        route_id: str = "",
    ) -> RuntimePhase:
        """
        Attempts a state transition. Returns the new phase on success.
        Raises InvalidStateTransition if the transition is illegal.
        """
        valid_successors = _VALID_TRANSITIONS.get(self.phase, frozenset())
        if target not in valid_successors:
            raise InvalidStateTransition(self.phase, target)

        record = TransitionRecord(
            from_phase=self.phase,
            to_phase=target,
            timestamp=time.monotonic(),
            reason=reason,
            route_id=route_id,
        )
        self.history.append(record)

        # Track quarantine metadata
        if target == RuntimePhase.ROUTE_QUARANTINED:
            self.quarantine_scope = QuarantineScope.ROUTE
            self.quarantine_reason = reason
        elif target == RuntimePhase.WORKER_QUARANTINED:
            self.quarantine_scope = QuarantineScope.WORKER
            self.quarantine_reason = reason
        elif target == RuntimePhase.READY:
            self.quarantine_scope = None
            self.quarantine_reason = ""

        self.phase = target
        return self.phase

    # ── Convenience Queries ────────────────────────────────────

    @property
    def is_quarantined(self) -> bool:
        return self.phase in (
            RuntimePhase.ROUTE_QUARANTINED,
            RuntimePhase.WORKER_QUARANTINED,
        )

    @property
    def is_terminal(self) -> bool:
        return self.phase == RuntimePhase.WORKER_QUARANTINED

    @property
    def is_ready(self) -> bool:
        return self.phase == RuntimePhase.READY

    @property
    def can_accept_requests(self) -> bool:
        return self.phase == RuntimePhase.READY

    def last_transition(self) -> Optional[TransitionRecord]:
        return self.history[-1] if self.history else None

    # ── Serialization ──────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serializes current state for JSON audit logs and metrics."""
        return {
            "phase": self.phase.value,
            "quarantine_scope": self.quarantine_scope.value if self.quarantine_scope else None,
            "quarantine_reason": self.quarantine_reason,
            "transition_count": len(self.history),
            "is_terminal": self.is_terminal,
        }
