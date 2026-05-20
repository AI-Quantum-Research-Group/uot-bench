import jax
import pytest


def assert_is_jax_array(x, name: str = "value") -> None:
    """Assert that *x* is a JAX array (not numpy)."""
    assert isinstance(x, jax.Array), (
        f"Expected {name} to be a jax.Array, got {type(x).__name__!r}"
    )
