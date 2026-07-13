from __future__ import annotations

import pytest

from agent.unicode_text import InvalidUnicodeInput, normalize_unicode_data, normalize_unicode_text


def test_normal_chinese_text_is_preserved() -> None:
    prompt = "分析当前目录下的所有文件，这是你自己的介绍以及出生的过程，给出后续的修改建议"
    assert normalize_unicode_text(prompt) == prompt


def test_utf16_surrogate_pair_is_repaired() -> None:
    assert normalize_unicode_text("状态：\ud83d\ude80") == "状态：🚀"


def test_gb18030_terminal_bytes_are_recovered() -> None:
    broken = "中文输入".encode("gb18030").decode("utf-8", errors="surrogateescape")
    assert normalize_unicode_text(broken) == "中文输入"


def test_unrecoverable_surrogate_has_readable_error() -> None:
    with pytest.raises(InvalidUnicodeInput, match="无法解码"):
        normalize_unicode_text("broken \ud800 input")


def test_nested_json_data_is_normalized() -> None:
    assert normalize_unicode_data({"message": "状态：\ud83d\ude80"}) == {"message": "状态：🚀"}
