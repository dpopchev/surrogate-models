"""Mlmodels context root -- composition + the settings-driven train_run facade.

The context's composition seat: it reads settings via get_settings() and binds the
src infrastructure adapter (save_trained_run, curried on the configured
checkpoint_dir) to this context's OWN application handler (DIP), never touching the
domain ring directly -- the run id and hyperparameters arrive as edge primitives on
the TrainRun command and the domain smart constructors certify them inside the
handler. Mirrors the datasets load_neutron_stars() seat, which binds
settings.datasets.path the same way.

train_run(cmd) is the notebook-friendly entry point re-exported from the top-level
package. The src save_trained_run adapter now persists a run's UNTRAINED model as
{run_id}.ckpt (the thin training slice), so the facade returns the written checkpoint
path; when the EXPAND slice trains for real, this seat and every ring above stay
unchanged.
"""

from __future__ import annotations

import logging
from functools import partial

from surrogate_models.config import get_settings
from surrogate_models.mlmodels.application import TrainRun, handle_train_run
from surrogate_models.mlmodels.infrastructure import save_trained_run

logger = logging.getLogger(__name__)


def train_run(cmd: TrainRun) -> str:
    """Configure and train a run under the configured checkpoint dir; return its path.

    The context's composition seat: it reads ``mlmodels.checkpoint_dir`` from
    ``get_settings()``, curries the src ``save_trained_run`` adapter on it to satisfy
    the application's ``SaveTrainedRunFn``, hands ``cmd`` to ``handle_train_run``, and
    projects the terminal run's checkpoint location (a bare ``str`` -- the seat never
    names a domain type). ``unwrap`` raises at this outer edge if configuration or
    training fails, carrying the ``ErrorInfo`` cause. The thin slice makes this happy
    path reachable: the src adapter writes the run's untrained ``{run_id}.ckpt`` under
    the configured dir and this returns its path; the EXPAND slice trains for real
    without changing this seat.
    """
    settings = get_settings()
    logger.info("composing train run %s", cmd.run_id)
    save = partial(save_trained_run, settings.mlmodels.checkpoint_dir)
    return handle_train_run(save_trained_run=save, cmd=cmd).unwrap().checkpoint.location
