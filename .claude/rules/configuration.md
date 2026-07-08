---
paths:
  - "src/**/config.py"
  - ".env"
  - ".env.*"
  - "*.env"
---

# Configuration -- Hard Gates

Hard constraints for configuration access. The composition-root seat for
`get_settings()` (pydantic-settings, read env in the imperative shell) is single-
sourced in `.claude/rules/architecture.md` (Composition Root).

- NEVER call `os.environ` or `os.getenv` outside `infrastructure/config.py` --
  reading the environment is I/O, so it lives in the imperative shell. The frozen
  `Settings` TYPE lives in `shared_kernel`; `get_settings()` (which reads env) lives
  in `infrastructure/config.py`. `domain/` and `application/` never read env.
- NEVER construct `Settings()` directly -- always use `get_settings()`.
- NEVER hard-code DSNs, URLs, or credentials anywhere in source.
- NEVER make a config field required unless the app cannot boot without it.
- NEVER hardcode lookup tables in `src/` -- externalise to `config/`.
- ALWAYS clear cache in tests via `get_settings.cache_clear()` -- never
  `importlib.reload`.
