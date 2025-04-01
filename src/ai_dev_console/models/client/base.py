from abc import ABC, abstractmethod
from contextlib import _GeneratorContextManager, contextmanager
from typing import Any, Dict, Generator, Iterator, Optional

import anthropic
import boto3
from botocore.config import Config

from ..exceptions import ModelClientError
from ..model import SupportedModels
from ..vendor import Vendor
from .adapters import VendorAdapter
from .types import ConverseRequest, ConverseResponse


class ModelClient(ABC):
    """Abstract base class for model clients."""

    def __init__(self, vendor: Vendor, adapter: VendorAdapter):
        """Initialize the model client."""
        self.vendor = vendor
        self.adapter = adapter

    @abstractmethod
    def converse(self, request: ConverseRequest) -> ConverseResponse:
        """Synchronously send a conversation request to the model."""
        pass

    @abstractmethod
    async def converse_async(self, request: ConverseRequest) -> ConverseResponse:
        """Asynchronously send a conversation request to the model."""
        pass

    @contextmanager
    def converse_stream(self, request: ConverseRequest) -> Iterator[str]:
        """
        Stream model responses.

        Args:
            request: The conversation request

        Yields:
            Iterator[str]: Stream of response chunks
        """
        raise NotImplementedError("Streaming not supported")


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

    @contextmanager
    def converse_stream(
        self, request: ConverseRequest
    ) -> _GeneratorContextManager[str]:
        """Stream response from Anthropic's API."""
        try:
            request.validate()
            adapted_request = self.adapter.adapt_request(request)

            with self.client.messages.stream(**adapted_request) as stream:
                # Store the stream object so its response can be accessed later
                self._stream = stream

                def generate() -> str:
                    self._generator = generate  # Store generator for access to response
                    for chunk in stream.text_stream:
                        if chunk:
                            yield chunk
                    # After streaming completes, the full response is available
                    self.response = stream.response

                yield generate()

        except Exception as e:
            raise ModelClientError(f"Streaming failed: {str(e)}")


class AWSClient(ModelClient):
    """Client implementation for AWS Bedrock's API."""

    def __init__(self, client: "boto3.client"):
        """Initialize the Bedrock client."""
        super().__init__(Vendor.AWS, VendorAdapter.create(Vendor.AWS))
        self.client = client
        self.supported_models = SupportedModels()

        # AWS Account_ID and region
        self.region = self.client.meta.region_name
        self.account_id = get_aws_account_id(client)

    def _resolve_model_id(self, model_id: str) -> str:
        """Resolve the model ID to its canonical form."""
        canonical_name = None
        for name, mapping in self.supported_models._model_mappings.items():
            if mapping.vendor_ids.get(Vendor.AWS) == model_id or name == model_id:
                canonical_name = name
                break

        if canonical_name and self.supported_models.requires_inference_profile(
            canonical_name
        ):
            return self.supported_models.get_inference_profile_arn(
                canonical_name, self.region, self.account_id
            )

        return model_id

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

    @contextmanager
    def converse_stream(self, request: ConverseRequest) -> Iterator[Iterator[str]]:
        """
        Stream response from AWS Bedrock API.

        Args:
            request: The conversation request

        Yields:
            Iterator[str]: Stream of response chunks

        Raises:
            ModelClientError: If streaming fails
        """
        try:
            request.validate()
            adapted_request = self.adapter.adapt_request(request)

            # Get the stream response
            response = self.client.converse_stream(**adapted_request)

            # Store the raw response for later access
            self._raw_response = response

            # Storage for completed message content
            full_response_text = ""
            complete_response = {}

            def generate() -> Iterator[str]:
                nonlocal full_response_text
                current_role = None
                self._generator = generate  # Store reference to generator

                for event in response["stream"]:
                    # Handle message start
                    if "messageStart" in event:
                        current_role = event["messageStart"]["role"]
                        continue

                    # Only process assistant responses
                    if current_role != "assistant":
                        continue

                    # Handle content deltas (actual text chunks)
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"]["delta"]
                        if "text" in delta and delta["text"]:
                            full_response_text += delta["text"]
                            yield delta["text"]

                    # Handle message complete - extract any thinking
                    if "messageComplete" in event:
                        # Check for content in the message that contains reasoning
                        if "message" in event["messageComplete"]:
                            message = event["messageComplete"]["message"]
                            if "content" in message:
                                for content_block in message["content"]:
                                    # Check for reasoningContent which contains the thinking in AWS Bedrock
                                    if "reasoningContent" in content_block:
                                        if (
                                            "reasoningText"
                                            in content_block["reasoningContent"]
                                        ):
                                            complete_response["thinking"] = {
                                                "text": content_block[
                                                    "reasoningContent"
                                                ]["reasoningText"]["text"]
                                            }

                    # Handle errors
                    for error_type in [
                        "internalServerException",
                        "modelStreamErrorException",
                        "validationException",
                        "throttlingException",
                        "serviceUnavailableException",
                    ]:
                        if error_type in event:
                            raise ModelClientError(
                                f"AWS Bedrock error: {event[error_type]['message']}"
                            )

                # After streaming is complete, store the final response
                complete_response["text"] = full_response_text
                self.response = complete_response

            yield generate()
        except Exception as e:
            raise ModelClientError(f"Streaming failed: {str(e)}") from e


class ModelClientFactory:
    """Factory for creating model clients."""

    def create_client(
        self, vendor: Vendor, client: Optional[Any] = None
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
            bedrock_client = boto3.client("bedrock-runtime")
            return AWSClient(bedrock_client)

        elif vendor == Vendor.OPENAI:
            # Future implementation
            raise NotImplementedError("OpenAI client not yet implemented")

        raise ValueError(f"Unsupported vendor: {vendor}")


def get_aws_account_id(client: "boto3.client") -> str:
    """Get the AWS account ID from the STS client."""
    try:
        # Try to get from client session (Mock)
        return client._session.get_credentials().get_frozen_credentials().account_id
    except (AttributeError, ValueError):
        # Fall back to STS GetCallerIdentity
        sts_client = boto3.client("sts")
        return sts_client.get_caller_identity()["Account"]
