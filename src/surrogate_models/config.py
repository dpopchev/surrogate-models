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

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class DatasetsSettings(BaseModel):
    """Configuration for the datasets bounded context.

    ``path`` is where persisted datasets live. It defaults under the gitignored
    ``var/`` state root so a fresh checkout has a working, ephemeral location with
    no setup; override it per environment via the TOML file or the
    ``SURROGATE_MODELS__DATASETS__PATH`` variable.

    ``neutron_stars_source`` points at the raw concatenated neutron-stars ``.dat``
    file the ingest reader digests. It is ``None`` by default: the app boots without
    it (ingest is an on-demand action, not a boot dependency, per the configuration
    gate). Set it via the TOML file or the
    ``SURROGATE_MODELS__DATASETS__NEUTRON_STARS_SOURCE`` variable when ingesting.
    """

    path: Path = Path("var/data/surrogate_models/datasets")
    neutron_stars_source: Path | None = None


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
