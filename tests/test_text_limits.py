import pytest

from pocket_tts.models.tts_model import prepare_text_prompt


def test_prepare_text_prompt_rejects_long_text(monkeypatch):
    monkeypatch.setenv("POCKET_TTS_MAX_TEXT_LENGTH", "10")

    with pytest.raises(ValueError, match="exceeds the maximum"):
        prepare_text_prompt("This prompt is definitely too long.")


def test_prepare_text_prompt_accepts_short_text(monkeypatch):
    monkeypatch.setenv("POCKET_TTS_MAX_TEXT_LENGTH", "100")

    text, frames_after_eos = prepare_text_prompt("Hello there")

    assert isinstance(text, str)
    assert frames_after_eos >= 1
