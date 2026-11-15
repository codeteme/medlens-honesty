from types import SimpleNamespace

import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from medlens.interp.sae_eval import evaluate_on_ids
from medlens.models.anthropic_model import AnthropicModel
from medlens.models.base import CostTracker, Generation, Usage
from medlens.models.cache import ResponseCache
from medlens.models.hf_local import HFLocalModel
from medlens.models.pricing import cost_usd


# ---- tiny random Gemma 3 (no download, no license needed) -------------------
@pytest.fixture(scope="module")
def tiny() -> HFLocalModel:
    torch.manual_seed(0)
    cfg = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        sliding_window=8,
    )
    return HFLocalModel("tiny-gemma3", device="cpu", model=Gemma3ForCausalLM(cfg).float())


def ids(n: int = 10) -> torch.Tensor:
    return torch.randint(1, 64, (1, n))


def test_resid_post_capture_matches_hidden_states(tiny):
    x = ids()
    with tiny.hooks(capture=[(1, "resid_post")]) as store:
        out = tiny.model(input_ids=x, output_hidden_states=True)
    assert torch.allclose(store[(1, "resid_post")], out.hidden_states[2].float())


def test_identity_replacement_changes_nothing_and_zeroing_does(tiny):
    x = ids()
    clean, _ = tiny.forward_logits(x)
    same, _ = tiny.forward_logits(x, replace={(1, "mlp_out"): lambda a: a})
    zeroed, _ = tiny.forward_logits(x, replace={(1, "mlp_out"): torch.zeros_like})
    assert torch.allclose(clean, same)
    assert not torch.allclose(clean, zeroed)


def test_hooks_are_removed(tiny):
    x = ids()
    with tiny.hooks(replace={(0, "resid_post"): torch.zeros_like}):
        pass
    clean, _ = tiny.forward_logits(x)
    again, _ = tiny.forward_logits(x)
    assert torch.allclose(clean, again)


class IdentitySAE(torch.nn.Module):
    """Stand-in SAE that reconstructs perfectly: splice metrics must be exactly 'no harm'."""

    def __init__(
        self, d: int, transcoder: bool = False, model: HFLocalModel | None = None, layer: int = 0
    ):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.eye(d))
        self.is_transcoder = transcoder
        self._mlp = None
        if transcoder:
            mlp_layer = model.layers[layer]
            # .forward() bypasses the splice hook that is registered on this module.
            self._mlp = lambda x: mlp_layer.post_feedforward_layernorm.forward(mlp_layer.mlp(x))

    def encode(self, x):
        return torch.relu(x)

    def forward(self, x):
        return self._mlp(x) if self.is_transcoder else x


@pytest.mark.parametrize("transcoder", [False, True])
def test_perfect_reconstruction_gives_zero_error(tiny, transcoder):
    sae = IdentitySAE(32, transcoder=transcoder, model=tiny, layer=1)
    res = evaluate_on_ids(tiny, sae, layer=1, input_ids=ids(12))
    assert res["fvu"] == pytest.approx(0.0, abs=1e-5)
    assert res["delta_ce"] == pytest.approx(0.0, abs=1e-5)
    assert res["kl"] == pytest.approx(0.0, abs=1e-5)
    assert res["n_tokens"] == 11


# ---- API wrapper, cache, costs (fake client, no network) ---------------------
class FakeMessages:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        assert "temperature" not in kwargs
        assert kwargs["system"] == "be brief"
        assert all(m["role"] != "system" for m in kwargs["messages"])
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text="Answer: B"),
            ],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=500),
            stop_reason="end_turn",
            stop_details=None,
            _request_id="req_1",
        )


def test_anthropic_wrapper_cost_and_cache(tmp_path):
    fake = SimpleNamespace(messages=FakeMessages())
    model = AnthropicModel(
        "claude-sonnet-5-5", cache=ResponseCache(tmp_path / "c.sqlite"), client=fake
    )
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "Q?"}]
    first, second = model.generate(msgs, 100), model.generate(msgs, 100)
    assert first.text == "Answer: B" and not first.cached
    assert first.usage.cost_usd == pytest.approx((1000 * 2.0 + 500 * 10.0) / 1e6)
    assert second.cached and second.text == first.text and fake.messages.calls == 1

    tracker = CostTracker()
    tracker.add(first)
    tracker.add(second)
    assert tracker.calls == 2 and tracker.cached_calls == 1
    assert tracker.cost_usd == pytest.approx(first.usage.cost_usd)


def test_unknown_price_is_none_not_guessed():
    assert cost_usd("anthropic", "made-up-model", 10, 10) is None
    tracker = CostTracker()
    tracker.add(Generation(text="", model="x", usage=Usage(1, 1, None)))
    assert tracker.unpriced_calls == 1 and tracker.cost_usd == 0.0


def test_shared_passes_match_individual_evaluation(tiny):
    from medlens.interp.sae_eval import SAESpec, evaluate_text

    x = ids(9)
    res = IdentitySAE(32)
    noisy = NoisySAE(32)
    together = evaluate_text(tiny, [SAESpec("res", 1, res), SAESpec("noisy", 1, noisy)], x)
    alone = evaluate_on_ids(tiny, noisy, layer=1, input_ids=x)
    assert together[1]["delta_ce"] == pytest.approx(alone["delta_ce"], abs=1e-6)
    assert together[1]["fvu"] > 0 and alone["ce_recovered"] is not None


class NoisySAE(IdentitySAE):
    def forward(self, x):
        g = torch.Generator().manual_seed(0)
        return x + 0.5 * torch.randn(x.shape, generator=g)


def test_char_span_to_tokens():
    from medlens.interp.features import char_span_to_tokens

    offsets = [(0, 3), (3, 4), (4, 9), (9, 10), (0, 0)]
    assert char_span_to_tokens(offsets, 4, 10) == [2, 3]
    assert char_span_to_tokens(offsets, 3, 4) == [1]


class TinyLinearSAE(torch.nn.Module):
    """Exact SAE for testing: features are the coordinates themselves (ReLU)."""

    def __init__(self, d: int):
        super().__init__()
        self.W_dec = torch.nn.Parameter(torch.eye(d))

    def encode(self, x):
        return torch.relu(x)


def test_remove_and_scale_features_touch_only_chosen_coordinates():
    from medlens.interp.interventions import remove_features, scale_features

    sae = TinyLinearSAE(4)
    x = torch.tensor([[[1.0, 2.0, -1.0, 3.0], [0.5, 0.0, 2.0, 1.0]]])
    out = remove_features(sae, [1, 3])(x)
    assert torch.allclose(out, torch.tensor([[[1.0, 0.0, -1.0, 0.0], [0.5, 0.0, 2.0, 0.0]]]))
    doubled = scale_features(sae, [0], 2.0, start_pos=1)(x)
    assert torch.allclose(doubled[0, 0], x[0, 0]) and doubled[0, 1, 0] == 1.0


def test_attribution_splice_preserves_forward_and_gives_nonzero_attributions(tiny):
    from medlens.interp.attribution import _splice_hooks

    class LinearSAE(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            g = torch.Generator().manual_seed(1)
            self.W_enc = torch.nn.Parameter(torch.randn(d, 2 * d, generator=g) * 0.3)
            self.W_dec = torch.nn.Parameter(torch.randn(2 * d, d, generator=g) * 0.3)

        def encode(self, x):
            return torch.relu(x @ self.W_enc)

        def decode(self, a):
            return a @ self.W_dec

    saes = {0: LinearSAE(32), 2: LinearSAE(32)}
    x = ids(8)
    clean, _ = tiny.forward_logits(x)
    store = {}
    with torch.enable_grad(), tiny.hooks(replace=_splice_hooks(saes, store)):
        out = tiny.model(input_ids=x).logits
        metric = out[0, -1, 5] - out[0, -1, 7]
        grads = torch.autograd.grad(metric, [store[0], store[2]])
    assert torch.allclose(out.float(), clean, atol=1e-4)  # splice does not change the forward
    assert all(g.abs().sum() > 0 for g in grads)


def test_scale_features_per_row_only_on_prompt_pass():
    from medlens.interp.interventions import scale_features_per_row

    sae = TinyLinearSAE(4)
    x = torch.ones(2, 5, 4)
    hook = scale_features_per_row(sae, [0], 0.0, [[1, 2], [4]])
    out = hook(x)
    assert out[0, 1, 0] == 0 and out[0, 2, 0] == 0 and out[0, 3, 0] == 1
    assert out[1, 4, 0] == 0 and out[1, 1, 0] == 1
    step = torch.ones(2, 1, 4)
    assert torch.equal(hook(step), step)  # decoding step untouched
