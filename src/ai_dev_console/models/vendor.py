from enum import Enum, auto


class Vendor(Enum):
    """Supported AI model vendors."""

    ANTHROPIC = "anthropic"
    AWS = "aws"
    OPENAI = "openai"
