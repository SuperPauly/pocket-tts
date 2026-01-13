import torch

from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.models.tts_model import TTSModel


def test_tokenized_text_pool_reuses_instance():
    TokenizedText._pool.clear()
    first_tokens = torch.tensor([1, 2, 3])
    instance = TokenizedText.acquire(first_tokens)
    TokenizedText.release(instance)

    second_tokens = torch.tensor([4, 5])
    reused = TokenizedText.acquire(second_tokens)

    assert instance is reused
    assert torch.equal(reused.tokens, second_tokens)


def test_should_skip_latent_empty():
    assert TTSModel._should_skip_latent(torch.empty(0))
    assert not TTSModel._should_skip_latent(torch.zeros(1, 1))
