from __future__ import annotations

import unicodedata
from typing import Any


class InvalidUnicodeInput(ValueError):
    """Raised when terminal input contains bytes that cannot be recovered safely."""


def normalize_unicode_text(value: str) -> str:
    """Return valid NFC Unicode, repairing terminal surrogate artifacts when possible."""
    if not _contains_surrogate(value):
        return unicodedata.normalize("NFC", value)

    repaired: list[str] = []
    index = 0
    while index < len(value):
        codepoint = ord(value[index])
        if 0xD800 <= codepoint <= 0xDBFF:
            if index + 1 < len(value):
                low = ord(value[index + 1])
                if 0xDC00 <= low <= 0xDFFF:
                    repaired.append(chr(0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)))
                    index += 2
                    continue
            raise InvalidUnicodeInput(_invalid_input_message(index))

        if 0xDC80 <= codepoint <= 0xDCFF:
            end = index + 1
            while end < len(value) and 0xDC80 <= ord(value[end]) <= 0xDCFF:
                end += 1
            raw = bytes(ord(char) - 0xDC00 for char in value[index:end])
            decoded = _decode_terminal_bytes(raw)
            if decoded is None:
                raise InvalidUnicodeInput(_invalid_input_message(index))
            repaired.append(decoded)
            index = end
            continue

        if 0xD800 <= codepoint <= 0xDFFF:
            raise InvalidUnicodeInput(_invalid_input_message(index))

        repaired.append(value[index])
        index += 1

    return unicodedata.normalize("NFC", "".join(repaired))


def normalize_unicode_data(value: Any) -> Any:
    """Normalize strings recursively before JSON serialization or terminal output."""
    if isinstance(value, str):
        return normalize_unicode_text(value)
    if isinstance(value, dict):
        return {normalize_unicode_text(str(key)): normalize_unicode_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_unicode_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_unicode_data(item) for item in value)
    return value


def _contains_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(char) <= 0xDFFF for char in value)


def _decode_terminal_bytes(raw: bytes) -> str | None:
    for encoding in ("utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _invalid_input_message(position: int) -> str:
    return (
        f"输入在第 {position + 1} 个字符附近包含无法解码的终端字节。"
        "请重新输入该字符；Windows Terminal/WSL 应使用 UTF-8 编码。"
    )
