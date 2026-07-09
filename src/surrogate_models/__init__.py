"""surrogate_models -- public package API.

Re-exports the notebook-friendly facades so callers reach them as
``from surrogate_models import load_neutron_stars, train_run`` without knowing the
internal bounded-context layout. ``TrainRun`` is re-exported alongside ``train_run``
so a caller can build the training command the facade consumes.
"""

from surrogate_models.datasets.__main__ import load_neutron_stars
from surrogate_models.mlmodels.__main__ import train_run
from surrogate_models.mlmodels.application import TrainRun

__all__ = ["TrainRun", "load_neutron_stars", "train_run"]
