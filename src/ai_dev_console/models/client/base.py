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
        api_key: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None,
        client: Optional[Any] = None
    ) -> ModelClient:
        """
        Create a model client for the specified vendor.

        Args:
            vendor: The vendor to create a client for
            api_key: API key for Anthropic
            aws_access_key_id: AWS access key ID for Bedrock
            aws_secret_access_key: AWS secret access key for Bedrock
            aws_region: AWS region for Bedrock
            client: Optional pre-configured client (useful for testing)

        Returns:
            ModelClient instance

        Raises:
            ValueError: If the vendor is not supported or required credentials are missing
        """
        if vendor == Vendor.ANTHROPIC:
            if client is not None:
                return AnthropicClient(client)
            if not api_key:
                raise ValueError("API key required for Anthropic client")
            return AnthropicClient(anthropic.Anthropic(api_key=api_key))

        elif vendor == Vendor.AWS:
            if client is not None:
                return AWSClient(client)

            if not all([aws_access_key_id, aws_secret_access_key, aws_region]):
                raise ValueError("AWS credentials and region required for Bedrock client")

            config = Config(
                region_name=aws_region,
                retries={'max_attempts': 3, 'mode': 'standard'}
            )

            bedrock_client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                config=config
            )

            return AWSClient(bedrock_client)

        raise ValueError(f"Unsupported vendor: {vendor}")