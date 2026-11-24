"""Server-side token estimates: IAedu returns no usage, so count locally.

Uses tiktoken's o200k_base (GPT-4o / GPT-5 family); falls back to chars / 4 when
the encoding cannot be loaded (e.g. offline first run).
"""

from functools import lru_cache

from .upstream import _text


@lru_cache(maxsize=1)
def _encoding():
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return None


def count_text(text: str) -> int:
    text = text.encode("utf-8", "replace").decode("utf-8")
    enc = _encoding()
    if enc is None:
        return (len(text) + 3) // 4
    return len(enc.encode(text, disallowed_special=()))


def count_messages(messages: list[dict]) -> int:
    """Chat format: ~3 tokens of framing per message plus 3 for the reply priming."""
    return sum(3 + count_text(m.get("role", "")) + count_text(_text(m)) for m in messages) + 3


def usage(messages: list[dict], completion: str) -> dict:
    p, c = count_messages(messages), count_text(completion)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
