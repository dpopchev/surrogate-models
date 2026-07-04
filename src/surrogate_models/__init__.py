"""surrogate_models -- public package API.

Re-exports the notebook-friendly facade so callers reach it as
``from surrogate_models import load_neutron_stars`` without knowing the internal
bounded-context layout.
"""

from surrogate_models.datasets.__main__ import load_neutron_stars

__all__ = ["load_neutron_stars"]
