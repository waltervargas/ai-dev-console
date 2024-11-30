from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import anthropic
import boto3
from botocore.config import Config
from .types import ConverseRequest, ConverseResponse
from .adapters import VendorAdapter
from ..vendor import Vendor
from ..exceptions import ModelClientError

class ModelClient(ABC):
    """Abstract base class for model clients."""

    def __init__(self, vendor: Vendor, adapter: VendorAdapter):
        """Initialize the model client."""
        self.vendor = vendor
        self.adapter = adapter

    @abstractmethod
    async def converse_async(self, request: ConverseRequest) -> ConverseResponse:
        """Asynchronously send a conversation request to the model."""
        pass

    @abstractmethod
    def converse(self, request: ConverseRequest) -> ConverseResponse:
        """Synchronously send a conversation request to the model."""
        pass

class AnthropicClient(ModelClient):
    """Client implementation for Anthropic's API."""

    def __init__(self, client: anthropic.Anthropic):
        """Initialize the Anthropic client."""
        super().__init__(Vendor.ANTHROPIC, VendorAdapter.create(Vendor.ANTHROPIC))
        self.client = client

    def converse(self, request: ConverseRequest) -> ConverseResponse:
        """
        Send a synchronous conversation request to Anthropic's API.

        Args:
            request: The conversation request

        Returns:
            ConverseResponse containing the model's response

        Raises:
            ModelClientError: If the request fails
        """
        try:
            request.validate()
            adapted_request = self.adapter.adapt_request(request)
            response = self.client.messages.create(**adapted_request)
            return self.adapter.adapt_response(response.model_dump())
        except Exception as e:
            raise ModelClientError(f"Failed to process request: {str(e)}") from e

    async def converse_async(self, request: ConverseRequest) -> ConverseResponse:
        """
        Send an asynchronous conversation request to Anthropic's API.

        Args:
            request: The conversation request

        Returns:
            ConverseResponse containing the model's response

        Raises:
            ModelClientError: If the request fails
        """
        try:
            request.validate()
            adapted_request = self.adapter.adapt_request(request)
            async with anthropic.AsyncAnthropic() as client:
                response = await client.messages.create(**adapted_request)
                return self.adapter.adapt_response(response.model_dump())
        except Exception as e:
            raise ModelClientError(f"Failed to process async request: {str(e)}") from e

class AWSClient(ModelClient):
    """Client implementation for AWS Bedrock's API."""

    def __init__(self, client: 'boto3.client'):
        """Initialize the Bedrock client."""
        super().__init__(Vendor.AWS, VendorAdapter.create(Vendor.AWS))
        self.client = client

    def converse(self, request: ConverseRequest) -> ConverseResponse:
        """
        Send a synchronous conversation request to Bedrock's API.

        Args:
            request: The conversation request

        Returns:
            ConverseResponse containing the model's response

        Raises:
            ModelClientError: If the request fails
        """
        try:
            request.validate()
            adapted_request = self.adapter.adapt_request(request)
            response = self.client.converse(**adapted_request)
            return self.adapter.adapt_response(response)
        except Exception as e:
            raise ModelClientError(f"Failed to process request: {str(e)}") from e

    async def converse_async(self, request: ConverseRequest) -> ConverseResponse:
        """
        Send an asynchronous conversation request to Bedrock's API.

        Args:
            request: The conversation request

        Returns:
            ConverseResponse containing the model's response

        Raises:
            ModelClientError: If the request fails
        """
        raise NotImplementedError("Async operations not supported for Bedrock")

class ModelClientFactory:
    """Factory for creating model clients."""

    def create_client(
        self,
        vendor: Vendor,
        client: Optional[Any] = None
    ) -> ModelClient:
        """
        Create a model client for the specified vendor.

        Args:
            vendor: The vendor to create a client for
            client: Optional pre-configured client (useful for testing)

        Returns:
            ModelClient instance

        Raises:
            ValueError: If the vendor is not supported
        """
        if vendor == Vendor.ANTHROPIC:
            if client is not None:
                return AnthropicClient(client)
            return AnthropicClient(anthropic.Anthropic())

        elif vendor == Vendor.AWS:
            if client is not None:
                return AWSClient(client)

            # AWS SDK will use default credential chain
            bedrock_client = boto3.client('bedrock-runtime')
            return AWSClient(bedrock_client)

        elif vendor == Vendor.OPENAI:
            # Future implementation
            raise NotImplementedError("OpenAI client not yet implemented")

        raise ValueError(f"Unsupported vendor: {vendor}")