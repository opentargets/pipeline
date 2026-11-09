"""Selection constants for the literature mechanism extraction."""

from __future__ import annotations

PER_DRUG = 25
PER_FILE_CAP = 300
MIN_CHARS = 30
MAX_CHARS = 1200
SECTION_RANK = {'title': 0, 'abstract': 1, 'intro': 2, 'concl': 3, 'discuss': 4, 'results': 5}
CUE = (
    r'(?i)\b(inhibitor|inhibits|inhibition|agonist|antagonist|blocker|blocks|modulator|activator|targeting|targets'
    r'|binds|binding|degrader|antibody|siRNA|antisense|selective)\b'
)
