import sentencepiece
import torch

from pocket_tts.conditioners import text as text_module


class _FakeSentencePieceProcessor:
    def __init__(self, path: str):
        self.calls = 0

    def vocab_size(self) -> int:
        return 100

    def encode(self, text: str, out_type=int) -> list[int]:
        self.calls += 1
        return [1, 2, 3]


def test_sentencepiece_tokenizer_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("POCKET_TTS_TOKENIZER_CACHE_SIZE", "2")
    monkeypatch.setattr(text_module, "download_if_necessary", lambda _: tmp_path / "tok.model")
    monkeypatch.setattr(sentencepiece, "SentencePieceProcessor", _FakeSentencePieceProcessor)

    tokenizer = text_module.SentencePieceTokenizer(100, "dummy")
    first = tokenizer("hello")
    second = tokenizer("world")
    third = tokenizer("hello")

    assert torch.equal(first.tokens, third.tokens)
    assert tokenizer.sp.calls == 2
    assert second.tokens.shape == torch.Size([1, 3])
