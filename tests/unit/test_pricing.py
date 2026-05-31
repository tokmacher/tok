"""Tests for tok.pricing — model price lookups."""

from tok.utils.pricing import get_pricing, has_known_pricing


class TestGetPricing:
    def test_known_model_exact(self) -> None:
        rates = get_pricing("claude-sonnet-4-20250101")
        assert rates == (3.00, 15.00, 0.30, 3.75)

    def test_opus_model(self) -> None:
        rates = get_pricing("claude-opus-4-20250101")
        assert rates[0] == 15.00  # input rate

    def test_haiku_model(self) -> None:
        rates = get_pricing("claude-haiku-4-20250101")
        assert rates[0] == 0.80

    def test_unknown_model_returns_conservative_zero_rates(self) -> None:
        # An unknown model must not be silently priced at a fabricated (Sonnet)
        # rate, which would produce misleading dollar cost/savings. Conservative
        # zero rates suppress the dollar claim while leaving token savings (which
        # are rate-independent) intact.
        rates = get_pricing("totally-unknown/model-v9")
        assert rates == (0.0, 0.0, 0.0, 0.0)

    def test_has_known_pricing_distinguishes_known_from_unknown(self) -> None:
        assert has_known_pricing("claude-sonnet-4-20250101")
        assert has_known_pricing("openai/gpt-5.4-pro")
        assert not has_known_pricing("totally-unknown/model-v9")

    def test_legacy_model(self) -> None:
        rates = get_pricing("claude-3-opus-20240229")
        assert rates[0] == 15.00

    def test_openai_gpt_5_4_pro_model(self) -> None:
        rates = get_pricing("openai/gpt-5.4-pro")
        assert rates == (30.00, 180.00, 0.00, 0.00)

    def test_moonshot_model_prefix_match(self) -> None:
        rates = get_pricing("moonshotai/kimi-k2.5:free")
        assert rates == (0.38, 1.72, 0.00, 0.00)
