#!/usr/bin/env python3
"""
Circuit collection wrapper for usage.py compatibility
"""

from typing import Any


class CircuitCollection:
    """
    Wrapper for circuit collections that provides both single circuit and list interfaces
    """

    def __init__(self, circuits: list[Any]):
        """
        Initialize circuit collection

        Args:
            circuits: List of quantum circuits
        """
        self.circuits = circuits if circuits else []

    def __getitem__(self, index: int) -> Any:
        """
        Allow indexing like a list

        Args:
            index: Circuit index

        Returns:
            Circuit at index
        """
        return self.circuits[index]

    def __len__(self) -> int:
        """Return number of circuits"""
        return len(self.circuits)

    def __iter__(self):
        """Allow iteration over circuits"""
        return iter(self.circuits)

    def draw(self, **kwargs) -> Any:
        """
        Draw the first circuit (for single circuit usage)

        Args:
            **kwargs: Arguments passed to circuit.draw()

        Returns:
            Drawing of first circuit
        """
        if self.circuits:
            return self.circuits[0].draw(**kwargs)
        return None

    def append(self, circuit: Any):
        """Add a circuit to the collection"""
        self.circuits.append(circuit)

    def extend(self, circuits: list[Any]):
        """Add multiple circuits to the collection"""
        self.circuits.extend(circuits)

    @property
    def first(self) -> Any:
        """Get the first circuit"""
        return self.circuits[0] if self.circuits else None

    def to_list(self) -> list[Any]:
        """Get the underlying list of circuits"""
        return self.circuits
