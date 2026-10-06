"""Shared external network foundation."""

from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.network.config import NetworkConfig, NetworkMode
from erguoyuan_football.network.health import run_network_preflight
from erguoyuan_football.network.source_registry import ExternalSourceRegistry

__all__ = ["CoreNetworkClient", "ExternalSourceRegistry", "NetworkConfig", "NetworkMode", "run_network_preflight"]
