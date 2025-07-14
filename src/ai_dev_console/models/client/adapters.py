from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Union, cast

from ..model import SupportedModels
from ..vendor import Vendor
from .types import (
    AnthropicContentBlock,
    AnthropicImageContent,
    AnthropicMessage,
    AnthropicRequestDict,
    AnthropicTextContent,
    AWSMessage,
    AWSRequestDict,
    ContentBlock,
    ConverseRequest,
    ConverseResponse,
    InferenceConfigDict,
    Message,
    MessageContent,
    Role,
    VendorRequestDict,
)


class VendorAdapter(ABC):
    """Abstract base class for vendor-specific adapters."""

    @abstractmethod
    def adapt_request(self, request: ConverseRequest) -> VendorRequestDict:
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

    def adapt_request(self, request: ConverseRequest) -> AnthropicRequestDict:
        """Convert to Anthropic's format."""
        messages: List[AnthropicMessage] = []

        # Use the original model_id - no ARN transformation needed for Anthropic
        model_id = request.model_id

        for msg in request.messages:
            if (
                len(msg.content) == 1
                and msg.content[0].text
                and not msg.content[0].image
            ):
                content: Union[str, List[AnthropicContentBlock]] = msg.content[0].text
            else:
                content_blocks: List[AnthropicContentBlock] = []
                for block in msg.content:
                    if block.text:
                        text_content: AnthropicTextContent = {
                            "type": "text",
                            "text": block.text,
                        }
                        content_blocks.append(text_content)
                    if block.image:
                        image_content: AnthropicImageContent = {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": cast(
                                    Literal["image/jpeg", "image/png"],
                                    block.image["media_type"],
                                ),
                                "data": block.image["data"],
                            },
                        }
                        content_blocks.append(image_content)
                content = content_blocks

            message: AnthropicMessage = {
                "role": cast(Literal[Role.USER, Role.ASSISTANT], msg.role.value),
                "content": content,
            }
            messages.append(message)

        # Get max_tokens with null safety
        max_tokens = (
            request.inference_config.max_tokens
            if request.inference_config
            and request.inference_config.max_tokens is not None
            else 500  # Default value
        )

        adapted: AnthropicRequestDict = {
            "model": model_id or request.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        # Optional parameters
        if request.inference_config:
            if request.inference_config.temperature is not None:
                adapted["temperature"] = request.inference_config.temperature
            if request.inference_config.top_p is not None:
                adapted["top_p"] = request.inference_config.top_p
            if request.inference_config.stop_sequences is not None:
                adapted["stop_sequences"] = list(
                    request.inference_config.stop_sequences
                )

        if request.system:
            adapted["system"] = request.system

        # Add thinking/extended reasoning for Claude 3.7 models when enabled
        if request.thinking_enabled:
            # Always add thinking support if it's enabled (even if model doesn't support it)
            # The API will simply ignore it for unsupported models

            # Debug the model ID for troubleshooting
            model_id_lower = request.model_id.lower()
            model_supports_3_7 = (
                "claude-3-7" in model_id_lower
                or "claude-3.7" in model_id_lower
                or "claude3.7" in model_id_lower
                or "claude-3-sonnet" in model_id_lower
            )

            print(
                f"Anthropic adapter: Enabling thinking for model: {request.model_id} (supports_3_7: {model_supports_3_7})"
            )

            # For direct Anthropic API, the thinking field is at the top level
            adapted["thinking"] = {
                "type": "enabled",
                "budget_tokens": request.thinking_budget,
            }

        return adapted

    def adapt_response(self, response: Dict[str, Any]) -> ConverseResponse:
        """Convert vendor-specific response to our format."""
        messages = [
            Message(
                role=Role(msg["role"]),
                content=(
                    [ContentBlock(text=msg["content"])]
                    if isinstance(msg["content"], str)
                    else [ContentBlock(**block) for block in msg["content"]]
                ),
            )
            for msg in response["messages"]
        ]

        # Extract thinking content if available in the response
        thinking_content = None
        if "thinking" in response:
            thinking_content = response["thinking"]

        return ConverseResponse(
            messages=messages,
            stop_reason=response.get("stop_reason"),
            usage=response.get("usage"),
            metrics=response.get("metrics"),
            thinking=thinking_content,
        )


class AWSAdapter(VendorAdapter):
    """Adapter for AWS's API."""

    def adapt_request(self, request: ConverseRequest) -> AWSRequestDict:
        """Convert to AWS Bedrock's format."""
        messages: List[AWSMessage] = [
            {
                "role": cast(Literal[Role.USER, Role.ASSISTANT], msg.role.value),
                "content": [
                    {"text": content.text or ""}
                    for content in msg.content
                    if content.text is not None
                ],
            }
            for msg in request.messages
        ]

        adapted: AWSRequestDict = {
            "modelId": request.model_id,
            "messages": messages,
        }

        # Add a system message if provided
        if request.system:
            adapted["system"] = [{"text": request.system}]

        # Always add inferenceConfig if provided
        if request.inference_config:
            inference: InferenceConfigDict = {}

            if request.inference_config.temperature is not None:
                inference["temperature"] = request.inference_config.temperature
            if request.inference_config.max_tokens is not None:
                inference["maxTokens"] = request.inference_config.max_tokens
            if request.inference_config.top_p is not None:
                inference["topP"] = request.inference_config.top_p
            if request.inference_config.stop_sequences is not None:
                inference["stopSequences"] = list(
                    request.inference_config.stop_sequences
                )

            if inference:  # Only add if there are actual values
                adapted["inferenceConfig"] = inference

        # Add reasoning config (thinking) for Claude 3.7 models when enabled
        if request.thinking_enabled and "claude-3-7" in request.model_id:
            # For AWS Bedrock, the parameter is named "reasoning_config"
            adapted["additionalModelRequestFields"] = {
                "reasoning_config": {
                    "type": "enabled",
                    "budget_tokens": request.thinking_budget,
                }
            }

        return adapted

    def adapt_response(self, response: Dict[str, Any]) -> ConverseResponse:
        """Convert vendor-specific response to our format."""
        messages = []
        msg = response["output"]["message"]

        for content in msg["content"]:
            content_blocks = []
            # Handle text blocks
            if "text" in content:
                content_blocks.append(ContentBlock(text=content["text"]))
            # Handle reasoning/thinking content
            elif "reasoningContent" in content:
                if "reasoningText" in content["reasoningContent"]:
                    content_blocks.append(
                        ContentBlock(
                            thinking={
                                "text": content["reasoningContent"]["reasoningText"][
                                    "text"
                                ]
                            }
                        )
                    )

            if content_blocks:
                messages.append(Message(role=Role(msg["role"]), content=content_blocks))

        return ConverseResponse(
            messages=messages,
            stop_reason=response.get("stopReason"),
            usage=response.get("usage"),
            metrics=response.get("metrics"),
        )


# The factory function is now redundant since we have VendorAdapter.create
# Consider removing this function
def get_vendor_adapter(vendor: Vendor) -> VendorAdapter:
    """Factory method to get the appropriate vendor adapter."""
    return VendorAdapter.create(vendor)
