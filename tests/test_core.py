from spectrida.core.ollama import _build_prompt, extract_name
from spectrida.tui.widgets.disasm import fmt_size, highlight, is_sub


def test_highlight_keeps_text():
    t = highlight("0x1000", "mov eax, 0x5")
    assert "mov" in t.plain and "eax" in t.plain and "0x5" in t.plain


def test_is_sub():
    assert is_sub("sub_140001000")
    assert is_sub("j_strcpy")
    assert not is_sub("Player$$Update")


def test_fmt_size():
    assert fmt_size(0) == ""
    assert fmt_size(512) == "512b"
    assert fmt_size(2048) == "2kb"


def test_extract_name():
    assert extract_name("NAME: do_thing\nREASON: stuff") == "do_thing"
    assert extract_name("no name here") is None


def test_prompt_uses_text_field():
    # disasm rows are {"address","text"}; the prompt must read 'text'
    p = _build_prompt([{"address": "0x1", "text": "mov eax, 1"}], ["callee"], ["caller"])
    assert "mov eax, 1" in p and "callee" in p and "caller" in p
