"""App-wide settings shell -- the single seat where the process reads its config.

Reading the environment and a config file is I/O, so it lives in this imperative
shell, never in a domain or application module. ``get_settings()`` is the ONLY way
to build ``Settings`` -- construct it nowhere else. Resolution order, highest
priority first: OS environment (and ``.env``), then ``surrogate_models.toml``,
then the field defaults.

``Settings`` is frozen and the accessor is cached, so the process reads its
configuration once. Tests reset the cache with ``get_settings.cache_clear()``
between cases (never reload the module).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

logger = logging.getLogger(__name__)

# Tag stamped on the one root handler this module installs, so a later
# _configure_logging call recognises and removes ITS OWN handler (never a caller's)
# before attaching a fresh one -- the idempotency hinge under a cleared settings cache.
_LOG_HANDLER_MARKER = "_surrogate_models_managed"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def _require_absolute(value: Path) -> Path:
    """Reject a relative path -- shape only, never a filesystem stat.

    A relative value resolves against the process working directory, the exact bug
    the absolute defaults exist to prevent; catching it here turns a silent
    wrong-directory miss into an immediate, clear config error. Shared by every path
    field's validator (datasets paths and the log file). Existence is deliberately
    NOT checked (that is I/O and belongs to the point-of-use railway, not config
    construction).
    """
    if not value.is_absolute():
        raise ValueError(
            f"must be an absolute path (resolves against the working "
            f"directory otherwise): {value!s}"
        )
    return value


class DatasetsSettings(BaseModel):
    """Configuration for the datasets bounded context.

    Both paths default to ABSOLUTE, filesystem-rooted locations so they resolve
    identically regardless of the process working directory -- a relative default
    breaks when the entry point runs from a subdirectory (e.g.
    ``load_neutron_stars()`` called from a notebook under ``notebooks/``). Override
    either per environment via the TOML file or the matching
    ``SURROGATE_MODELS__DATASETS__*`` variable.

    ``path`` is where persisted datasets live; it defaults to
    ``/var/data/surrogate_models/datasets`` and is created on first save
    (``mkdir(parents=True)``), so it need not preexist -- but the ``/var`` location
    must be a provisioned, writable directory.

    ``neutron_stars_source`` points at the raw concatenated neutron-stars ``.dat``
    the ingest reader digests; it defaults to
    ``/data/neutron-stars/neutron-stars.dat``. Having a default is safe: it is NOT a
    boot dependency -- nothing reads the file at boot, only the on-demand ingest
    reader touches it, and a missing file folds onto the ``NEUTRON_STARS_READ_FAILED``
    rail rather than crashing. This is also why existence is NOT validated here:
    config construction stays free of filesystem I/O and the railway already reports
    a missing path at the point of use. The one shape rule enforced below is that a
    resolved path is absolute -- pure, no I/O -- so the working-directory bug that
    motivated the absolute defaults cannot silently return via a relative override.
    """

    path: Path = Path("/var/data/surrogate_models/datasets")
    neutron_stars_source: Path = Path("/data/neutron-stars/neutron-stars.dat")

    @field_validator("path", "neutron_stars_source", mode="after")
    @classmethod
    def _must_be_absolute(cls, value: Path) -> Path:
        """Reject a relative dataset path -- delegates to :func:`_require_absolute`."""
        return _require_absolute(value)


class LoggingSettings(BaseModel):
    """Configuration for application-wide logging, applied on ``get_settings()``.

    ``level`` is the root logger threshold; it defaults to ``INFO`` and accepts any
    standard level name (case-insensitive, normalised to upper case). ``file`` is the
    single destination the root logger writes to; it defaults to the absolute,
    filesystem-rooted ``/var/data/surrogate_models/log/surrogate_models.log`` so it
    resolves identically regardless of the process working directory -- a relative
    default would break under a notebook subdirectory, exactly the bug the datasets
    paths already guard against. Override either per environment via the TOML file or
    the matching ``SURROGATE_MODELS__LOGGING__*`` variable.

    Only the SHAPE is validated here (a known level name, an absolute path) -- pure,
    no filesystem I/O. Actually opening the file and attaching the handler is a side
    effect, so it lives in :func:`_configure_logging`, driven from ``get_settings``.
    """

    level: str = "INFO"
    file: Path = Path("/var/data/surrogate_models/log/surrogate_models.log")

    @field_validator("level", mode="after")
    @classmethod
    def _known_level(cls, value: str) -> str:
        """Normalise to an upper-case level name, rejecting an unknown one.

        Pure shape check -- no logger is touched. ``getLevelNamesMapping`` is the
        authoritative set of names the stdlib accepts (``INFO``, ``DEBUG``,
        ``WARNING``, ...); anything else is a configuration typo caught at
        construction rather than a silent fall-through to the default level.
        """
        name = value.strip().upper()
        if name not in logging.getLevelNamesMapping():
            raise ValueError(f"unknown log level: {value!r}")
        return name

    @field_validator("file", mode="after")
    @classmethod
    def _file_must_be_absolute(cls, value: Path) -> Path:
        """Reject a relative log-file path -- delegates to :func:`_require_absolute`."""
        return _require_absolute(value)


class MLModelsSettings(BaseModel):
    """Configuration for the mlmodels bounded context.

    ``checkpoint_dir`` is where a training run writes its Lightning checkpoint
    (``.ckpt``). It defaults to the absolute, filesystem-rooted
    ``/var/data/surrogate_models/checkpoints`` so it resolves identically regardless
    of the process working directory -- the same working-directory guard the datasets
    paths use. It is created on first train (``mkdir(parents=True)``), so it need not
    preexist. Override per environment via the TOML file or the matching
    ``SURROGATE_MODELS__MLMODELS__CHECKPOINT_DIR`` variable. Only the SHAPE is
    validated here (an absolute path) -- pure, no filesystem I/O.
    """

    checkpoint_dir: Path = Path("/var/data/surrogate_models/checkpoints")

    @field_validator("checkpoint_dir", mode="after")
    @classmethod
    def _must_be_absolute(cls, value: Path) -> Path:
        """Reject a relative checkpoint path -- see :func:`_require_absolute`."""
        return _require_absolute(value)


class Settings(BaseSettings):
    """The app-wide configuration, composed of one section per bounded context.

    Env vars are prefixed ``SURROGATE_MODELS__`` and nest into sections with the
    same ``__`` delimiter, so ``SURROGATE_MODELS__DATASETS__PATH`` sets
    ``datasets.path`` and the TOML counterpart is ``[datasets] path = ...``.
    """

    model_config = SettingsConfigDict(
        env_prefix="SURROGATE_MODELS__",
        env_nested_delimiter="__",
        env_file=".env",
        toml_file="surrogate_models.toml",
        frozen=True,
        extra="ignore",
    )

    datasets: DatasetsSettings = DatasetsSettings()
    logging: LoggingSettings = LoggingSettings()
    mlmodels: MLModelsSettings = MLModelsSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order config sources by priority (first wins).

        OS environment and ``.env`` override the TOML file, which overrides the
        field defaults. The TOML source is appended here (pydantic-settings does
        not enable it by default); secrets stay last.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


def _configure_logging(config: LoggingSettings) -> None:
    """Apply ``config`` to the root logger: set the level, attach ONE file handler.

    Idempotent: any handler a previous call installed (tagged with
    ``_LOG_HANDLER_MARKER``) is removed before a fresh one is attached, so repeated
    ``get_settings()`` calls -- e.g. a test suite clearing the cache -- never stack
    duplicate handlers. FILE-ONLY by design: no console handler is added. If the log
    file cannot be created (unprovisioned directory, no permission) the ``OSError``
    is swallowed with a warning and logging degrades to the stdlib handler of last
    resort rather than crashing at ``get_settings`` time -- symmetric with the config
    philosophy of folding missing-path I/O onto a rail, never a boot crash.
    """
    root = logging.getLogger()
    root.setLevel(config.level)
    for handler in list(root.handlers):
        if getattr(handler, _LOG_HANDLER_MARKER, False):
            root.removeHandler(handler)
            handler.close()
    try:
        config.file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(config.file)
    except OSError as cause:
        logger.warning(
            "log file %s is unavailable (%s); logging degrades to the handler "
            "of last resort",
            config.file,
            cause,
        )
        return
    setattr(handler, _LOG_HANDLER_MARKER, True)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)
    logger.info("logging configured: level=%s file=%s", config.level, config.file)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide ``Settings``, built once and cached.

    The sole constructor of ``Settings``: reading env + files is done here so the
    rest of the program receives an immutable, already-resolved configuration.
    Configuring logging is a process-wide side effect, so it too rides this
    once-per-process seat -- ``_configure_logging`` applies the resolved
    ``logging`` section to the root logger right after the settings are built. Clear
    the cache with ``get_settings.cache_clear()`` in tests.
    """
    settings = Settings()
    _configure_logging(settings.logging)
    return settings
