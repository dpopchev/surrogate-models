"""Mlmodels infrastructure -- the imperative shell of the mlmodels context.

Provides the src implementation of the application's write port: save_trained_run
trains a configured run and persists its checkpoint. It knows the domain only enough
to read a ConfiguredRun and hand back a Checkpoint -- it never exposes a domain type
outward. The REAL training (the 28-feature surrogate regressor over real data) is a
later slice, so this adapter is a typed, torch-free placeholder that fails on the
training rail with a stable code until then; the context's end-to-end proof lives in
tests, wiring the SAME application port to a test-owned adapter that trains a stub
model. Curried on ``checkpoint_dir`` at the composition root, it satisfies the
application's SaveTrainedRunFn.
"""

from __future__ import annotations

import logging
from pathlib import Path

from surrogate_models.mlmodels.domain import Checkpoint, ConfiguredRun
from surrogate_models.railway_adts import Err, ErrorInfo, Result

logger = logging.getLogger(__name__)


def save_trained_run(
    checkpoint_dir: Path, run: ConfiguredRun
) -> Result[Checkpoint, ErrorInfo]:
    """Train ``run`` and persist ``{run_id}.ckpt`` -- the src SaveTrainedRunFn adapter.

    ``checkpoint_dir`` leads so the composition root binds it via ``partial`` to yield
    the application's ``SaveTrainedRunFn``. NOT YET IMPLEMENTED in src: the real
    surrogate regressor and its data are a later slice, so this returns
    ``Err(ErrorInfo(code="TRAINING_NOT_IMPLEMENTED"))`` -- the honest current contract,
    torch-free until the model exists. When the real model lands, its Lightning
    training replaces this body and every ring above stays unchanged. The mlmodels e2e
    proves the application port today against a test-owned stub adapter.
    """
    logger.info(
        "save_trained_run placeholder: run %s would train under %s (not implemented)",
        run.run_id,
        checkpoint_dir,
    )
    return Err(
        ErrorInfo(
            code="TRAINING_NOT_IMPLEMENTED",
            message=(
                "src save_trained_run is a placeholder; the real training adapter is "
                "a later slice (see tests/mlmodels for the reference stub adapter)"
            ),
        )
    )
