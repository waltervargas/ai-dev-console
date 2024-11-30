from abc import ABC, abstractmethod
from typing import Dict, Any, Sequence, Collection, Union, Optional
from .types import ConverseRequest, ConverseResponse, Message, ContentBlock, Role
from ..vendor import Vendor


class VendorAdapter(ABC):
    """Abstract base class for vendor-specific adapters."""

    @abstractmethod
    def adapt_request(self, request: ConverseRequest) -> Dict[str, Any]:
        """Convert our request format to vendor-specific format."""
        pass

    @abstractmethod
    def adapt_response(self, response: Dict[str, Any]) -> ConverseResponse:
        """Convert vendor-specific response to our format."""
        pass

    @staticmethod
    def create(vendor: Vendor) -> "VendorAdapter":
        """Factory method to create appropriate adapter."""
        if vendor == Vendor.ANTHROPIC:
            return AnthropicAdapter()
        elif vendor == Vendor.AWS:
            return AWSAdapter()
        raise ValueError(f"Unsupported vendor: {vendor}")


class AnthropicAdapter(VendorAdapter):
    """Adapter for Anthropic's API."""

    def adapt_request(self, request: ConverseRequest) -> Dict[str, Any]:
        """Convert to Anthropic's format."""
        # Prepare the base request with required parameters
        adapted: Dict[str, Any] = {
            "model": request.model_id,
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content[0].text,  # Anthropic expects string content
                }
                for msg in request.messages
            ],
            # max_tokens is a required parameter
            "max_tokens": (
                request.inference_config.max_tokens if request.inference_config else 500
            ),
        }

        # Optional parameters - only add if explicitly set
        if request.inference_config:
            # Add temperature if set
            if request.inference_config.temperature is not None:
                adapted["temperature"] = request.inference_config.temperature
            # Add top_p if set
            if request.inference_config.top_p is not None:
                adapted["top_p"] = request.inference_config.top_p
            # Add stop_sequences if set
            if request.inference_config.stop_sequences is not None:
                adapted["stop_sequences"] = request.inference_config.stop_sequences

        return adapted

    def adapt_response(self, response: Dict[str, Any]) -> ConverseResponse:
        """Convert vendor-specific response to our format."""
        messages = [
            Message(
                role=Role(msg["role"]),
                content=[ContentBlock(**block) for block in msg["content"]],
            )
            for msg in response["messages"]
        ]
        return ConverseResponse(
            messages=messages,
            stop_reason=response.get("stop_reason"),
            usage=response.get("usage"),
            metrics=response.get("metrics"),
        )


class AWSAdapter(VendorAdapter):
    """Adapter for AWS's API."""

    def adapt_request(self, request: ConverseRequest) -> Dict[str, Any]:
        """Convert to AWS Bedrock's format."""
        adapted = {
            "modelId": request.model_id,
            "messages": [
                {
                    "role": msg.role.value,
                    "content": [{"text": content.text} for content in msg.content]
                }
                for msg in request.messages
            ]
        }

        if request.inference_config:
            inference_config = {}
            if request.inference_config.temperature is not None:
                inference_config["temperature"] = request.inference_config.temperature
            if request.inference_config.max_tokens is not None:
                inference_config["maxTokens"] = request.inference_config.max_tokens
            if request.inference_config.top_p is not None:
                inference_config["topP"] = request.inference_config.top_p
            if request.inference_config.stop_sequences:
                inference_config["stopSequences"] = request.inference_config.stop_sequences

            if inference_config:  # Only add if there are actual values
                adapted["inferenceConfig"] = inference_config

        if request.system:
            adapted["system"] = [{"text": request.system}]

        return adapted

    def adapt_response(self, response: Dict[str, Any]) -> ConverseResponse:
        """Convert vendor-specific response to our format."""
        return ConverseResponse(
            messages=[
                Message(
                    role=Role(msg["role"]),
                    content=[ContentBlock(text=block["text"]) for block in msg["content"]],
                )
                for msg in response["messages"]
            ],
        )

def get_vendor_adapter(vendor: Vendor) -> VendorAdapter:
    """Factory method to get the appropriate vendor adapter."""
    if vendor == Vendor.ANTHROPIC:
        return AnthropicAdapter()
    elif vendor == Vendor.AWS:
        return AWSAdapter()
    raise ValueError(f"Unsupported vendor: {vendor}")
