from decimal import Decimal
from datetime import datetime
import pytest
from ai_dev_console.models.model import AIModel, ModelCosts, SupportedModels
from ai_dev_console.models.vendor import Vendor

def test_vendor_enum_exists():
    assert Vendor.ANTHROPIC.name == "ANTHROPIC"
    assert Vendor.BEDROCK.name == "BEDROCK"

def test_model_costs_calculation():
    costs = ModelCosts(
        input_cost_per_million_tokens=Decimal("0.25"),
        output_cost_per_million_tokens=Decimal("1.25")
    )

    # Test with 1000 tokens each
    cost = costs.calculate_cost(input_tokens=1000, output_tokens=1000)
    expected = Decimal("0.00150")  # (0.25 * 1000 + 1.25 * 1000) / 1_000_000
    assert cost == expected

def test_claude_3_haiku_model_creation():
    model = AIModel.claude_3_haiku()

    assert model.name == "claude-3-haiku-20240307"
    assert model.vendor == Vendor.ANTHROPIC
    assert model.costs.input_cost_per_million_tokens == Decimal("0.25")
    assert model.costs.output_cost_per_million_tokens == Decimal("1.25")
    assert model.context_window == 200000
    assert model.max_output_tokens == 4096
    assert model.supports_vision is True
    assert model.supports_message_batches is True
    assert isinstance(model.training_cutoff, datetime)
    assert model.training_cutoff.year == 2023
    assert model.training_cutoff.month == 8
    assert model.comparative_latency == "Fastest"

def test_supported_models_initialization():
    supported_models = SupportedModels()

    # Test that all required models are available
    assert "claude-3-5-sonnet-20241022" in supported_models.available_models
    assert "claude-3-5-haiku-20241022" in supported_models.available_models
    assert "claude-3-haiku-20240307" in supported_models.available_models

    # Test model count
    assert len(supported_models.available_models) == 3

def test_supported_models_get_by_name():
    supported_models = SupportedModels()

    model = supported_models.get_model("claude-3-5-sonnet-20241022")
    assert model.name == "claude-3-5-sonnet-20241022"
    assert model.costs.input_cost_per_million_tokens == Decimal("3.0")
    assert model.costs.output_cost_per_million_tokens == Decimal("15.0")

def test_supported_models_invalid_model():
    supported_models = SupportedModels()

    with pytest.raises(ValueError, match="Model 'invalid-model' not found"):
        supported_models.get_model("invalid-model")

def test_model_cost_calculation_precision():
    model = AIModel.claude_3_haiku()
    cost = model.costs.calculate_cost(input_tokens=1_000_000, output_tokens=500_000)

    # For 1M input tokens and 500K output tokens with Claude 3 Haiku
    # Input: 1M * 0.25 / 1M = 0.25
    # Output: 500K * 1.25 / 1M = 0.625
    # Total: 0.875
    expected = Decimal("0.875")
    assert cost == expected