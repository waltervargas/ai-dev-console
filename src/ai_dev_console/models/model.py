from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional, Tuple

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

    def __post_init__(self) -> None:
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
            Decimal(input_tokens)
            * self.input_cost_per_million_tokens
            / Decimal(1_000_000)
            + Decimal(output_tokens)
            * self.output_cost_per_million_tokens
            / Decimal(1_000_000)
        ).quantize(Decimal("0.00001"))


@dataclass(frozen=True)
class AIModel:
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
    vendor_model_id: Optional[str] = None

    @classmethod
    def claude_3_haiku(cls) -> "AIModel":
        """Factory method for Claude 3 Haiku model."""
        return cls(
            name="claude-3-haiku-20240307",
            vendor=Vendor.ANTHROPIC,
            costs=ModelCosts(
                input_cost_per_million_tokens=Decimal("0.25"),
                output_cost_per_million_tokens=Decimal("1.25"),
            ),
            context_window=200000,
            max_output_tokens=8192,
            supports_vision=True,
            supports_message_batches=True,
            training_cutoff=datetime(2023, 8, 1),
            description="Fastest and most compact model for near-instant responsiveness",
            comparative_latency="Fastest",
        )


@dataclass
class ModelMapping:
    """Maps between canonical model names and vendor-specific identifiers."""

    canonical_name: str
    vendor_ids: Dict[Vendor, str]


class SupportedModels:
    def __init__(self) -> None:
        self._model_mappings = {
            "claude-3-7-sonnet-20250219": ModelMapping(
                canonical_name="claude-3-7-sonnet-20250219",
                vendor_ids={
                    Vendor.ANTHROPIC: "claude-3-7-sonnet-20250219",
                    Vendor.AWS: "anthropic.claude-3-7-sonnet-20250219-v1:0",
                },
            ),
            "claude-3-haiku-20240307": ModelMapping(
                canonical_name="claude-3-haiku-20240307",
                vendor_ids={
                    Vendor.ANTHROPIC: "claude-3-haiku-20240307",
                    Vendor.AWS: "anthropic.claude-3-haiku-20240307-v1:0",
                },
            ),
            "claude-3-5-sonnet-20241022": ModelMapping(
                canonical_name="claude-3-5-sonnet-20241022",
                vendor_ids={
                    Vendor.ANTHROPIC: "claude-3-5-sonnet-20241022",
                    Vendor.AWS: "anthropic.claude-3-5-sonnet-20240620-v1:0",
                },
            ),
        }

        self._models_requiring_inference_profiles = {
            "claude-3-7-sonnet-20250219",
        }

        # Mapping of canonical model names to vendor specific versions.  Previous
        # versions of this code attempted to initialise ``available_models`` with
        # duplicate keys which resulted in the first definitions silently being
        # overwritten.  The structure below keeps each canonical model only once
        # and stores the vendor specific variants in a nested dictionary.
        self._models_by_vendor: Dict[str, Dict[Vendor, AIModel]] = {
            "claude-3-7-sonnet-20250219": {
                Vendor.AWS: AIModel(
                    name="claude-3-7-sonnet-20250219",
                    vendor=Vendor.AWS,
                    costs=ModelCosts(
                        input_cost_per_million_tokens=Decimal("3.0"),
                        output_cost_per_million_tokens=Decimal("15.0"),
                    ),
                    context_window=200000,
                    max_output_tokens=64000,
                    supports_vision=True,
                    supports_message_batches=True,
                    training_cutoff=datetime(2025, 2, 19),
                    description="Our most expressive model",
                    comparative_latency="Fastest",
                ),
                Vendor.ANTHROPIC: AIModel(
                    name="claude-3-7-sonnet-20250219",
                    vendor=Vendor.ANTHROPIC,
                    costs=ModelCosts(
                        input_cost_per_million_tokens=Decimal("4.0"),
                        output_cost_per_million_tokens=Decimal("20.0"),
                    ),
                    context_window=200000,
                    max_output_tokens=8192,
                    supports_vision=True,
                    supports_message_batches=True,
                    training_cutoff=datetime(2025, 2, 19),
                    description="Our most expressive model",
                    comparative_latency="Fast",
                ),
            },
            "claude-3-5-sonnet-20241022": {
                Vendor.AWS: AIModel(
                    name="claude-3-5-sonnet-20241022",
                    vendor=Vendor.AWS,
                    costs=ModelCosts(
                        input_cost_per_million_tokens=Decimal("3.0"),
                        output_cost_per_million_tokens=Decimal("15.0"),
                    ),
                    context_window=200000,
                    max_output_tokens=8192,
                    supports_vision=True,
                    supports_message_batches=True,
                    training_cutoff=datetime(2024, 4, 1),
                    description="Our most intelligent model",
                    comparative_latency="Fast",
                ),
                Vendor.ANTHROPIC: AIModel(
                    name="claude-3-5-sonnet-20241022",
                    vendor=Vendor.ANTHROPIC,
                    costs=ModelCosts(
                        input_cost_per_million_tokens=Decimal("3.0"),
                        output_cost_per_million_tokens=Decimal("15.0"),
                    ),
                    context_window=200000,
                    max_output_tokens=8192,
                    supports_vision=True,
                    supports_message_batches=True,
                    training_cutoff=datetime(2024, 4, 1),
                    description="Our most intelligent model",
                    comparative_latency="Fast",
                ),
            },
            "claude-3-haiku-20240307": {
                Vendor.ANTHROPIC: AIModel(
                    name="claude-3-haiku-20240307",
                    vendor=Vendor.ANTHROPIC,
                    costs=ModelCosts(
                        input_cost_per_million_tokens=Decimal("1.0"),
                        output_cost_per_million_tokens=Decimal("5.0"),
                    ),
                    context_window=200000,
                    max_output_tokens=8192,
                    supports_vision=False,
                    supports_message_batches=True,
                    training_cutoff=datetime(2024, 7, 1),
                    description="Our fastest model",
                    comparative_latency="Fastest",
                ),
                Vendor.AWS: AIModel(
                    name="claude-3-haiku-20240307",
                    vendor=Vendor.AWS,
                    costs=ModelCosts(
                        input_cost_per_million_tokens=Decimal("1.0"),
                        output_cost_per_million_tokens=Decimal("5.0"),
                    ),
                    context_window=200000,
                    max_output_tokens=8192,
                    supports_vision=False,
                    supports_message_batches=True,
                    training_cutoff=datetime(2024, 7, 1),
                    description="Our fastest model",
                    comparative_latency="Fastest",
                ),
            },
        }

        # Maintain backwards compatibility: expose a flat dictionary of default
        # models so existing code (and tests) that iterate over ``available_models``
        # continues to work.  Anthropic versions are preferred when available.
        self.available_models: Dict[str, AIModel] = {
            name: models.get(Vendor.ANTHROPIC, next(iter(models.values())))
            for name, models in self._models_by_vendor.items()
        }

    # TODO: This should be part of the client adapter, not part of the model.
    def requires_inference_profile(self, model_name: str) -> bool:
        """
        Check if a model requires an inference profile.

        Args:
            model_name (str): The canonical model name

        Returns:
            bool: True if the model requires an inference profile, False otherwise
        """
        return model_name in self._models_requiring_inference_profiles

    def get_inference_profile_arn(
        self, model_name: str, region: str, account_id: str
    ) -> str:
        """
        Get the ARN for the inference profile.

        Args:
            model_name (str): The canonical model name
            region (str): The AWS region
            account_id (str): The AWS account ID
        Returns:
            str: The ARN for the inference profile
        """
        if not self.requires_inference_profile(model_name):
            raise ValueError(
                f"Model '{model_name}' does not require an inference profile"
            )

        region_prefix = region[:2]
        model_id = self.get_vendor_model_id(model_name, Vendor.AWS)
        return f"arn:aws:bedrock:{region}:{account_id}:inference-profile/{region_prefix}.{model_id}"

    def get_model(self, model_name: str, vendor: Optional[Vendor] = None) -> AIModel:
        """Get a model by name and optional vendor."""
        if model_name not in self._models_by_vendor:
            raise ValueError(f"Model '{model_name}' not found")

        models = self._models_by_vendor[model_name]

        if vendor is None:
            # Default to the Anthropic version when available
            return models.get(Vendor.ANTHROPIC, next(iter(models.values())))

        if vendor not in models:
            raise ValueError(
                f"Model '{model_name}' not supported for vendor {vendor.value}"
            )

        return models[vendor]

    def get_vendor_model_id(self, model_name: str, vendor: Vendor) -> str:
        """
        Get the vendor-specific model identifier.

        Args:
            model_name (str): The canonical model name
            vendor (Vendor): The target vendor

        Returns:
            str: The vendor-specific model identifier

        Raises:
            - ValueError: Model {model_name} not supported for vendor {vendor}
            - ValueError: No mapping found for model {model_name} and vendor {vendor}
        """
        # Always check the mapping first
        mapping = self._model_mappings.get(model_name)
        if mapping:
            if vendor in mapping.vendor_ids:
                return mapping.vendor_ids[vendor]
            raise ValueError(
                f"Model '{model_name}' not supported for vendor {vendor.value}"
            )

        # If no mapping exists, check if it's a valid model for the requested vendor
        model = self.available_models.get(model_name)
        if model and model.vendor == vendor:
            return model_name

        raise ValueError(
            f"No mapping found for model '{model_name}' and vendor {vendor.value}"
        )

    def resolve_model_id(self, model_id: str, vendor: Vendor) -> str:
        """Resolve ``model_id`` to the vendor-specific identifier for ``vendor``.

        ``model_id`` may already be a vendor specific ID or a canonical name.

        Args:
            model_id: Identifier to resolve. Can be canonical or vendor specific.
            vendor: The vendor for which the ID should be resolved.

        Returns:
            The vendor-specific identifier.
        """

        # If ``model_id`` already matches a known vendor specific ID for the
        # requested vendor, return it unchanged.
        for mapping in self._model_mappings.values():
            if mapping.vendor_ids.get(vendor) == model_id:
                return model_id

        # Otherwise treat ``model_id`` as a canonical name and resolve normally.
        return self.get_vendor_model_id(model_id, vendor)

    def resolve_model_name_and_vendor(
        self, model_id: str
    ) -> Tuple[str, Optional[Vendor]]:
        """
        Resolve a model identifier to its canonical name and vendor.

        This method takes a model ID (which could be a canonical name or vendor-specific ID)
        and returns both the canonical model name and the vendor it belongs to.

        Args:
            model_id (str): The model identifier to resolve

        Returns:
            Tuple[str, Optional[Vendor]]: The canonical model name and vendor.
            If the vendor cannot be determined uniquely, vendor will be None.

        Raises:
            ValueError: If the model_id cannot be resolved to any known model
        """
        # Check if the model_id is a canonical name first
        if model_id in self._model_mappings:
            return model_id, None  # Return canonical name, but can't determine vendor

        # Check if the model_id is a vendor-specific ID
        for canonical_name, mapping in self._model_mappings.items():
            for vendor, vendor_id in mapping.vendor_ids.items():
                if vendor_id == model_id:
                    return canonical_name, vendor

        # If no match found, check the available models directly
        for name, model in self.available_models.items():
            if name == model_id:
                return name, model.vendor

        # If we get here, we couldn't resolve the model_id
        raise ValueError(f"Unable to resolve model ID: {model_id}")
