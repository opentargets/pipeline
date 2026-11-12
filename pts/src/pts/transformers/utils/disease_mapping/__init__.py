"""Entity mapping, started as a port of OnToma's, against the lookup tables built by the `ontoma_lut` steps."""

from pts.transformers.utils.disease_mapping.mapping import map_entities, read_entity_lut

__all__ = ['map_entities', 'read_entity_lut']
