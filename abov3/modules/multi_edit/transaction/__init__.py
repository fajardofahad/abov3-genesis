"""
Transaction Management System

Enterprise-grade transaction management with rollback capabilities and ACID properties.
"""

from .manager import TransactionManager

__all__ = ["TransactionManager"]