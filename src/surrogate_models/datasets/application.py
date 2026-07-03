"""Datasets application -- CQRS handlers over the datasets domain.

Orchestrates the pure core via injected callables (DIP). Id generation is an
injected FactoryDatasetIdFn supplied by infrastructure, so the application mints a
DatasetID and hands it to the domain's make_dataset without the core ever touching
uuid. Command/query handlers are added under TDD, kept as ordered CQRS sections.
"""

from __future__ import annotations

from collections.abc import Callable

from surrogate_models.datasets.domain import DatasetID

# Injected port: the shell's id source (infrastructure.factory_datasetid).
FactoryDatasetIdFn = Callable[[], DatasetID]

# --- (1) Commands ---
# --- (2) Queries ---
# --- (3) Projections ---
