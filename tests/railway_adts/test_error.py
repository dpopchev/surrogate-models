"""Tests for the error primitives -- one assert per test.

``fmap_error`` wraps a lower-layer error or caught exception into an ``ErrorInfo``
cause and lifts it to a layer-specific error, so a higher layer reports its own
failure without naming the lower layer's exception type.
"""

from surrogate_models.railway_adts.error import ErrorInfo, fmap_error


def test_fmap_error_wraps_a_cause_into_errorinfo() -> None:
    wrap = fmap_error(lambda cause: cause, code="store.write_failed")
    assert wrap(ValueError("disk full")) == ErrorInfo("store.write_failed", "disk full")


def test_fmap_error_prefixes_context_and_lifts_to_a_layer_error() -> None:
    wrap = fmap_error(
        lambda cause: ("not_saved", cause), code="store.write_failed", where="/m.json"
    )
    assert wrap(OSError("denied")) == (
        "not_saved",
        ErrorInfo("store.write_failed", "/m.json: denied"),
    )


def test_errorinfo_str_is_its_message() -> None:
    assert str(ErrorInfo("store.write_failed", "disk full")) == "disk full"
