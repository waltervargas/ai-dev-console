from decimal import Decimal
from datetime import datetime
import pytest
from ai_dev_console.models.model import AIModel, ModelCosts, SupportedModels
from ai_dev_console.models.vendor import Vendor


class TestAIDeveloperWorkflow:
    """
    Test suite that follows a typical AI developer's journey using the console.
    """

    @pytest.fixture
    def supported_models(self):
        """Provides access to available AI models."""
        return SupportedModels()

    def test_developer_explores_available_vendors(self):
        """
        Story: A developer wants to know which AI vendors are supported
        Given the system supports multiple vendors
        When the developer checks the available vendors
        Then they should see both Anthropic and Bedrock as options
        """
        assert Vendor.ANTHROPIC.name == "ANTHROPIC"
        assert Vendor.AWS.name == "AWS"

    def test_developer_checks_available_models(self, supported_models):
        """
        Story: A developer wants to see what models are available
        Given the system is initialized
        When the developer lists available models
        Then they should see all supported Claude 3 variants
        """
        expected_models = {
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-haiku-20240307",
        }
        assert set(supported_models.available_models) == expected_models

    def test_developer_selects_cost_efficient_model(self):
        """
        Story: A developer wants to choose a cost-efficient model for their project
        Given they need a fast and affordable model
        When they select Claude 3 Haiku
        Then they should see its specifications and pricing
        """
        model = AIModel.claude_3_haiku()

        # Verify model specifications
        assert model.name == "claude-3-haiku-20240307"
        assert model.comparative_latency == "Fastest"
        assert model.context_window == 200000

        # Verify pricing structure
        assert model.costs.input_cost_per_million_tokens == Decimal("0.25")
        assert model.costs.output_cost_per_million_tokens == Decimal("1.25")


class TestCostEstimationScenarios:
    """
    Test suite for various cost estimation scenarios a developer might encounter.
    """

    @pytest.fixture
    def project_costs(self):
        """Standard cost configuration for a typical project."""
        return ModelCosts(
            input_cost_per_million_tokens=Decimal("0.25"),
            output_cost_per_million_tokens=Decimal("1.25"),
        )

    def test_developer_estimates_small_project_cost(self, project_costs):
        """
        Story: A developer wants to estimate costs for a small project
        Given they plan to process 1000 tokens each way
        When they calculate the cost
        Then they should get an accurate estimate in USD
        """
        cost = project_costs.calculate_cost(input_tokens=1000, output_tokens=1000)
        assert cost == Decimal("0.00150")

    def test_developer_estimates_large_project_cost(self, project_costs):
        """
        Story: A developer needs to budget for a large-scale project
        Given they plan to process a million tokens
        When they calculate the total cost
        Then they should get a precise estimate with proper decimal places
        """
        cost = project_costs.calculate_cost(
            input_tokens=1_000_000, output_tokens=1_000_000
        )
        assert cost == Decimal("1.50000")
        assert len(str(cost).split(".")[1]) == 5  # Ensures 5 decimal precision

    def test_developer_attempts_invalid_cost_calculation(self, project_costs):
        """
        Story: A developer accidentally inputs negative token counts
        Given they make a mistake in their input
        When they try to calculate costs with negative values
        Then they should receive a clear error message
        """
        with pytest.raises(ValueError, match="Token counts cannot be negative"):
            project_costs.calculate_cost(-1, 100)


class TestModelSelectionScenarios:
    """
    Test suite for various model selection scenarios.
    """

    @pytest.fixture
    def supported_models(self):
        return SupportedModels()

    def test_developer_selects_high_performance_model(self, supported_models):
        """
        Story: A developer needs the most capable model for complex tasks
        Given they need the highest performance model
        When they select Claude 3 Sonnet
        Then they should see its premium specifications
        """
        model = supported_models.get_model("claude-3-5-sonnet-20241022")
        assert model.costs.input_cost_per_million_tokens == Decimal("3.0")
        assert model.costs.output_cost_per_million_tokens == Decimal("15.0")

    def test_developer_tries_nonexistent_model(self, supported_models):
        """
        Story: A developer tries to use an unsupported model
        Given they request a non-existent model
        When they try to select it
        Then they should receive a helpful error message
        """
        with pytest.raises(ValueError, match="Model 'invalid-model' not found"):
            supported_models.get_model("invalid-model")

    def test_model_vendor_id_resolution(self):
        """
        Story: A developer wants to use models across different vendors
        Given a model identifier
        When resolving it for a specific vendor
        Then the correct vendor-specific ID is returned
        """
        models = SupportedModels()

        # Test canonical name to AWS ID
        aws_id = models.resolve_model_id("claude-3-haiku-20240307", Vendor.AWS)
        assert aws_id == "anthropic.claude-3-haiku-20240307-v1:0"

        # Test canonical name to Anthropic ID
        anthropic_id = models.resolve_model_id(
            "claude-3-haiku-20240307", Vendor.ANTHROPIC
        )
        assert anthropic_id == "claude-3-haiku-20240307"

        # Test direct vendor ID passthrough
        direct_id = models.resolve_model_id(
            "anthropic.claude-3-haiku-20240307-v1:0", Vendor.AWS
        )
        assert direct_id == "anthropic.claude-3-haiku-20240307-v1:0"

    def test_unknown_model_resolution(self):
        """
        Story: A developer uses an unknown model identifier
        Given an unknown model ID
        When resolving it
        Then the original ID is returned unchanged
        """
        models = SupportedModels()

        unknown_id = "unknown-model"
        resolved_id = models.resolve_model_id(unknown_id, Vendor.AWS)
        assert resolved_id == unknown_id
