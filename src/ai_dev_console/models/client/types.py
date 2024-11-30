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
            "content": [block.to_dict() for block in self.content]
        }

@dataclass
class InferenceConfiguration:
    """Configuration for model inference."""
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None

    def validate(self):
        """Validate the configuration values."""
        if self.temperature is not None and not 0 <= self.temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        if self.top_p is not None and not 0 <= self.top_p <= 1:
            raise ValueError("Top P must be between 0 and 1")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary format."""
        result = {}
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.top_p is not None:
            result["top_p"] = self.top_p
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.stop_sequences is not None:
            result["stop_sequences"] = self.stop_sequences
        return result

@dataclass
class ConverseRequest:
    """Request for model conversation."""
    model_id: str
    messages: List[Message]
    system: Optional[str] = None
    inference_config: Optional[InferenceConfiguration] = None

    def validate(self):
        """Validate the request."""
        if not self.model_id:
            raise ValueError("Model ID is required")
        if not self.messages:
            raise ValueError("At least one message is required")
        if self.inference_config:
            self.inference_config.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the request to a dictionary format."""
        result = {
            "model_id": self.model_id,
            "messages": [msg.to_dict() for msg in self.messages]
        }
        if self.system:
            result["system"] = self.system
        if self.inference_config:
            result["inference_config"] = self.inference_config.to_dict()
        return result

@dataclass
class ConverseResponse:
    """Response from model conversation."""
    messages: List[Message]
    stop_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metrics: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConverseResponse':
        """Create a response from a dictionary format."""
        messages = [
            Message(
                role=Role(msg["role"]),
                content=[ContentBlock(**block) for block in msg["content"]]
            )
            for msg in data["messages"]
        ]
        return cls(
            messages=messages,
            stop_reason=data.get("stop_reason"),
            usage=data.get("usage"),
            metrics=data.get("metrics")
        )