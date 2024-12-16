from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Literal, Optional, TypedDict, Union
from datetime import datetime


# Common Enums and Base Types
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


# Content Block Types
class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class ImageSource(TypedDict):
    type: Literal["base64"]
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    data: str


class ImageContent(TypedDict):
    type: Literal["image"]
    source: ImageSource


class DocumentContent(TypedDict):
    type: Literal["document"]
    source: Dict[str, Any]


ContentBlockType = Union[TextContent, ImageContent, DocumentContent]


# Base Message Types
class MessageContent(TypedDict):
    text: str


class BaseMessage(TypedDict):
    role: Literal[Role.USER, Role.ASSISTANT]
    content: Union[str, List[MessageContent]]


# Inference Configuration
class InferenceConfigDict(TypedDict, total=False):
    """Type definition for inference configuration dictionary."""

    temperature: float
    maxTokens: int
    topP: float
    stopSequences: List[str]


# AWS-specific Types
AWSMessage = BaseMessage


class AWSRequestDict(TypedDict, total=False):
    """AWS Bedrock request format."""

    modelId: str
    messages: List[AWSMessage]
    inferenceConfig: InferenceConfigDict
    system: List[MessageContent]


# Anthropic-specific Types
class AnthropicToolSchema(TypedDict):
    """Schema for Anthropic tool definitions."""

    type: str
    properties: Dict[str, Any]
    required: List[str]


class AnthropicTool(TypedDict):
    """Anthropic tool definition."""

    name: str
    description: Optional[str]
    input_schema: AnthropicToolSchema


# Content block types for Anthropic
class AnthropicTextContent(TypedDict):
    type: Literal["text"]
    text: str


AnthropicImageSource = ImageSource


class AnthropicImageContent(TypedDict):
    type: Literal["image"]
    source: AnthropicImageSource


AnthropicContentBlock = Union[AnthropicTextContent, AnthropicImageContent]


class AnthropicMessage(TypedDict):
    """Anthropic message format."""

    role: Literal[Role.USER, Role.ASSISTANT]
    content: Union[str, List[AnthropicContentBlock]]


class AnthropicRequestDict(TypedDict, total=False):
    """Anthropic API request format.

    Reference: https://docs.anthropic.com/claude/reference/messages_post
    """

    # Required fields
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int

    # Optional fields
    metadata: Optional[Dict[str, Any]]
    stop_sequences: Optional[List[str]]
    stream: Optional[bool]
    system: Optional[str]
    temperature: Optional[float]
    tool_choice: Optional[Dict[str, Any]]
    tools: Optional[List[AnthropicTool]]
    top_k: Optional[int]
    top_p: Optional[float]


# Vendor-agnostic Types
VendorRequestDict = Union[AnthropicRequestDict, AWSRequestDict]


@dataclass
class ContentBlock:
    """Represents a block of content in a message."""

    text: Optional[str] = None
    image: Optional[Dict[str, Any]] = None
    document: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the content block to a dictionary format."""
        result: Dict[str, Any] = {}
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

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = 500
    stop_sequences: Optional[List[str]] = None

    def validate(self) -> None:
        """Validate inference configuration parameters."""
        if self.temperature is not None:
            if not 0 <= self.temperature <= 1:
                raise ValueError("Temperature must be between 0 and 1")

        if self.top_p is not None:
            if not 0 <= self.top_p <= 1:
                raise ValueError("Top P must be between 0 and 1")

        if self.max_tokens is not None:
            if self.max_tokens <= 0:
                raise ValueError("Max tokens must be positive")
            if self.max_tokens > 8192:
                raise ValueError("Max tokens cannot exceed 8192")

        if self.stop_sequences is not None:
            if not isinstance(self.stop_sequences, list):
                raise ValueError("Stop sequences must be a list of strings")
            if not all(isinstance(seq, str) for seq in self.stop_sequences):
                raise ValueError("All stop sequences must be strings")


@dataclass
class ConverseRequest:
    """Request for model conversation with intelligent defaults."""

    model_id: str
    messages: List[Message]
    system: Optional[str] = None
    inference_config: Optional[InferenceConfiguration] = None

    def __post_init__(self) -> None:
        if self.inference_config is None:
            self.inference_config = InferenceConfiguration()
        self.inference_config.validate()

    def estimate_tokens(self) -> int:
        """Estimate the number of tokens in the request."""

        def count_tokens(text: str) -> int:
            return max(1, len(text) // 6)

        message_tokens = sum(
            count_tokens(" ".join(block.text or "" for block in message.content))
            for message in self.messages
        )
        system_tokens = count_tokens(self.system) if self.system else 0
        return message_tokens + system_tokens

    def validate(self) -> None:
        """Validate the entire request."""
        if not self.model_id:
            raise ValueError("Model ID is required")
        if not self.messages:
            raise ValueError("At least one message is required")
        for message in self.messages:
            if not message.content:
                raise ValueError("Each message must have content")
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
