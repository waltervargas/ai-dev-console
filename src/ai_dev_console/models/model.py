from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from .vendor import Vendor

@dataclass(frozen=True)
class ModelCosts:
    """
    Represents the cost structure for a model in USD.

    Attributes:
        input_cost_per_million_tokens (Decimal): Cost in USD per million input tokens
        output_cost_per_million_tokens (Decimal): Cost in USD per million output tokens
    """
    input_cost_per_million_tokens: Decimal
    output_cost_per_million_tokens: Decimal

    def __post_init__(self):
        """Validate costs are non-negative."""
        if self.input_cost_per_million_tokens < 0:
            raise ValueError("Input cost cannot be negative")
        if self.output_cost_per_million_tokens < 0:
            raise ValueError("Output cost cannot be negative")

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        """
        Calculate the cost in USD for a given number of input and output tokens.

        Args:
            input_tokens (int): Number of input tokens
            output_tokens (int): Number of output tokens

        Returns:
            Decimal: Total cost in USD

        Raises:
            ValueError: If token counts are negative
        """
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts cannot be negative")

        return (
            Decimal(input_tokens) * self.input_cost_per_million_tokens / Decimal(1_000_000) +
            Decimal(output_tokens) * self.output_cost_per_million_tokens / Decimal(1_000_000)
        ).quantize(Decimal('0.00001'))  # Round to 5 decimal places for USD

@dataclass(frozen=True)
class AIModel:
    """Represents an AI model with its capabilities and costs."""
    name: str
    vendor: Vendor
    costs: ModelCosts
    context_window: int
    max_output_tokens: int
    supports_vision: bool
    supports_message_batches: bool
    training_cutoff: datetime
    description: str
    comparative_latency: str
    vendor_model_id: Optional[str] = None  # Added to support vendor-specific IDs

    @classmethod
    def claude_3_haiku(cls) -> "AIModel":
        """Factory method for Claude 3 Haiku model."""
        return cls(
            name="claude-3-haiku-20240307",
            vendor=Vendor.ANTHROPIC,
            costs=ModelCosts(
                input_cost_per_million_tokens=Decimal("0.25"),
                output_cost_per_million_tokens=Decimal("1.25")
            ),
            context_window=200000,
            max_output_tokens=4096,
            supports_vision=True,
            supports_message_batches=True,
            training_cutoff=datetime(2023, 8, 1),
            description="Fastest and most compact model for near-instant responsiveness",
            comparative_latency="Fastest"
        )

@dataclass
class ModelMapping:
    """Maps between canonical model names and vendor-specific identifiers."""
    canonical_name: str
    vendor_ids: Dict[Vendor, str]

class SupportedModels:
    """Registry of all supported AI models."""
    def __init__(self):
        # Model mappings between vendors
        self._model_mappings = {
            "claude-3-haiku-20240307": ModelMapping(
                canonical_name="claude-3-haiku-20240307",
                vendor_ids={
                    Vendor.ANTHROPIC: "claude-3-haiku-20240307",
                    Vendor.AWS: "anthropic.claude-3-haiku-20240307-v1:0"
                }
            ),
            "claude-3-5-sonnet-20241022": ModelMapping(
                canonical_name="claude-3-5-sonnet-20241022",
                vendor_ids={
                    Vendor.ANTHROPIC: "claude-3-5-sonnet-20241022",
                    Vendor.AWS: "anthropic.claude-3-5-sonnet-20240620-v1:0"
                }
            ),
            # Add other mappings as needed
        }

        # Initialize available models
        self.available_models: Dict[str, AIModel] = {
            "claude-3-5-sonnet-20241022": AIModel(
                name="claude-3-5-sonnet-20241022",
                vendor=Vendor.ANTHROPIC,
                costs=ModelCosts(
                    input_cost_per_million_tokens=Decimal("3.0"),
                    output_cost_per_million_tokens=Decimal("15.0")
                ),
                context_window=200000,
                max_output_tokens=8192,
                supports_vision=True,
                supports_message_batches=True,
                training_cutoff=datetime(2024, 4, 1),
                description="Our most intelligent model",
                comparative_latency="Fast"
            ),
            "claude-3-5-haiku-20241022": AIModel(
                name="claude-3-5-haiku-20241022",
                vendor=Vendor.ANTHROPIC,
                costs=ModelCosts(
                    input_cost_per_million_tokens=Decimal("1.0"),
                    output_cost_per_million_tokens=Decimal("5.0")
                ),
                context_window=200000,
                max_output_tokens=8192,
                supports_vision=False,
                supports_message_batches=True,
                training_cutoff=datetime(2024, 7, 1),
                description="Our fastest model",
                comparative_latency="Fastest"
            ),
            "claude-3-haiku-20240307": AIModel.claude_3_haiku()
        }

    def get_model(self, model_name: str) -> AIModel:
        """Get a model by name."""
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not found")
        return self.available_models[model_name]

    def get_vendor_model_id(self, model_name: str, vendor: Vendor) -> str:
        """
        Get the vendor-specific model identifier.

        Args:
            model_name (str): The canonical model name
            vendor (Vendor): The target vendor

        Returns:
            str: The vendor-specific model identifier

        Raises:
            ValueError: If the model or vendor mapping is not found
        """
        # Check if we have a mapping for this model
        mapping = self._model_mappings.get(model_name)
        if mapping and vendor in mapping.vendor_ids:
            return mapping.vendor_ids[vendor]

        # If no mapping found, return the original name
        # This allows for direct use of vendor-specific IDs
        return model_name

    def resolve_model_id(self, model_id: str, vendor: Vendor) -> str:
        """
        Resolve a model identifier to its vendor-specific form.

        This method handles both canonical names and vendor-specific IDs.

        Args:
            model_id (str): The model identifier to resolve
            vendor (Vendor): The target vendor

        Returns:
            str: The vendor-specific model identifier
        """
        # First try to get a vendor-specific ID
        vendor_id = self.get_vendor_model_id(model_id, vendor)

        # If we got back the same ID and it's not in our mappings,
        # it's probably already a vendor-specific ID
        if vendor_id == model_id and model_id not in self._model_mappings:
            return model_id

        return vendor_id