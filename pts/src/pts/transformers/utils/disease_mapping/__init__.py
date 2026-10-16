"""Polars port of OnToma's entity mapping, against lookup tables OnToma built on spark."""

from pts.transformers.utils.disease_mapping.mapping import map_entities, read_entity_lut

__all__ = ['map_entities', 'read_entity_lut']
