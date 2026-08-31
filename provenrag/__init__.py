"""A minimal prototype for provenance-aware evidence retrieval."""

from .models import Evidence, EvidenceGroup, Selection
from .provenance import ProvenanceContractor
from .selectors import IndependentEvidenceSelector, NaiveDenseSelector

__all__ = [
    "Evidence",
    "EvidenceGroup",
    "Selection",
    "ProvenanceContractor",
    "IndependentEvidenceSelector",
    "NaiveDenseSelector",
]

