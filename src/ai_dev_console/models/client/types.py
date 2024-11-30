from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from datetime import datetime


class ContentType(Enum):
    """Types of content that can be sent or received."""

    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"


class Role(Enum):
    """Roles in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ContentBlock:
    """Represents a block of content in a message."""

    text: Optional[str] = None
    image: Optional[Dict[str, Any]] = None
    document: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the content block to a dictionary format."""
        result = {}
        if self.text is not None:
            result["text"] = self.text
        if self.image is not None:
            result["image"] = self.image
        if self.document is not None:
            result["document"] = self.document
        return result


@dataclass
class Message:
    """Represents a message in a conversation."""

    role: Role
    content: List[ContentBlock]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary format."""
        return {
            "role": self.role.value,
            "content": [block.to_dict() for block in self.content],
        }


@dataclass
class InferenceConfiguration:
    """Configuration for model inference with sensible, cost-effective defaults."""

    temperature: Optional[float] = None  # No default, only set if explicitly needed
    top_p: Optional[float] = None  # No default, only set if explicitly needed
    max_tokens: Optional[int] = 500  # Reasonable default for most queries
    stop_sequences: Optional[List[str]] = None

    def validate(self):
        """
        Validate inference configuration parameters.

        Ensures values are within acceptable ranges and follows cost-effective practices.
        """
        if self.temperature is not None:
            if not 0 <= self.temperature <= 1:
                raise ValueError("Temperature must be between 0 and 1")

        if self.top_p is not None:
            if not 0 <= self.top_p <= 1:
                raise ValueError("Top P must be between 0 and 1")

        if self.max_tokens is not None:
            if self.max_tokens <= 0:
                raise ValueError("Max tokens must be positive")
            if self.max_tokens > 8192:  # Most models have a max of 4096
                raise ValueError("Max tokens cannot exceed 8192")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary,
        only including explicitly set parameters.
        """
        result = {}
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            result["top_p"] = self.top_p
        if self.stop_sequences:
            result["stop_sequences"] = self.stop_sequences
        return result


@dataclass
class ConverseRequest:
    """
    Request for model conversation with intelligent defaults.

    Provides a flexible interface for sending prompts to AI models
    with cost-effective and sensible default configurations.
    """

    model_id: str
    messages: List[Message]
    system: Optional[str] = None
    inference_config: Optional[InferenceConfiguration] = None

    def __post_init__(self):
        """
        Automatically apply default inference configuration
        if not explicitly provided.
        """
        if self.inference_config is None:
            self.inference_config = InferenceConfiguration()

        # Validate the configuration
        self.inference_config.validate()

    def estimate_tokens(self) -> int:
        """
        Estimate the number of tokens in the request.

        Rule of thumb: Approximately 1 token per 4-6 characters.
        This is a rough estimation and can vary by model and language.

        Returns:
            Estimated number of tokens in the request
        """

        def count_tokens(text: str) -> int:
            """Count tokens in a given text."""
            return max(1, len(text) // 6)  # Conservative token estimation

        # Count tokens in messages
        message_tokens = sum(
            count_tokens(" ".join(block.text or "" for block in message.content))
            for message in self.messages
        )

        # Add system prompt tokens if present
        system_tokens = count_tokens(self.system) if self.system else 0

        return message_tokens + system_tokens

    def validate(self):
        """
        Validate the entire request, including messages and configuration.

        Raises:
            ValueError: If the request is invalid
        """
        if not self.model_id:
            raise ValueError("Model ID is required")

        if not self.messages:
            raise ValueError("At least one message is required")

        # Validate each message
        for message in self.messages:
            if not message.content:
                raise ValueError("Each message must have content")

        # Validate inference configuration
        if self.inference_config:
            self.inference_config.validate()


@dataclass
class ConverseResponse:
    """Response from model conversation."""

    messages: List[Message]
    stop_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metrics: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConverseResponse":
        """Create a response from a dictionary format."""
        messages = [
            Message(
                role=Role(msg["role"]),
                content=[ContentBlock(**block) for block in msg["content"]],
            )
            for msg in data["messages"]
        ]
        return cls(
            messages=messages,
            stop_reason=data.get("stop_reason"),
            usage=data.get("usage"),
            metrics=data.get("metrics"),
        )
