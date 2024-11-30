from enum import Enum, auto

class Vendor(Enum):
    """Supported AI model vendors."""
    ANTHROPIC = auto()
    AWS = auto()
    OPENAI = auto()