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

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


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
        """Reject a relative dataset path -- shape only, never a filesystem stat.

        A relative value resolves against the process working directory, the exact
        bug the absolute defaults exist to prevent; catching it here turns a silent
        wrong-directory miss into an immediate, clear config error. Existence is
        deliberately NOT checked (that is I/O and belongs to the point-of-use railway,
        not config construction).
        """
        if not value.is_absolute():
            raise ValueError(
                f"must be an absolute path (resolves against the working "
                f"directory otherwise): {value!s}"
            )
        return value


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


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide ``Settings``, built once and cached.

    The sole constructor of ``Settings``: reading env + files is done here so the
    rest of the program receives an immutable, already-resolved configuration.
    Clear the cache with ``get_settings.cache_clear()`` in tests.
    """
    return Settings()
