"""Typed errors used across the codebase."""

from __future__ import annotations


class ContractViolation(RuntimeError):
    """Raised when an external or internal contract is violated."""


__all__ = ["ContractViolation"]
