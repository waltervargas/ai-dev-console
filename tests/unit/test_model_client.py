import pytest
from unittest.mock import Mock, patch, ANY
from typing import List, Dict, Any
from ai_dev_console.models import (
    ModelClient,
    Message,
    ContentBlock,
    InferenceConfiguration,
    ConverseRequest,
    ModelClientFactory,
    Role,
    Vendor,
)
from ai_dev_console.models.client.adapters import VendorAdapter


class TestModelClientFactory:
    """
    Test suite for model client creation scenarios.
    """

    def test_creates_anthropic_client(self):
        """
        Story: A developer wants to use Anthropic's models
        Given valid Anthropic credentials
        When creating a client for Anthropic
        Then they should get a properly configured client
        """
        with patch("anthropic.Anthropic") as mock_anthropic:
            factory = ModelClientFactory()
            client = factory.create_client(Vendor.ANTHROPIC)

            assert isinstance(client, ModelClient)
            mock_anthropic.assert_called_once_with()

    def test_creates_aws_client(self):
        """
        Story: A developer wants to use AWS models
        Given valid AWS credentials
        When creating a client for AWS
        Then they should get a properly configured client
        """
        with patch("boto3.client") as mock_boto3:
            factory = ModelClientFactory()
            client = factory.create_client(
                Vendor.AWS,
            )

            assert isinstance(client, ModelClient)
            mock_boto3.assert_called_once_with(
                "bedrock-runtime",
            )


class TestVendorAdapter:
    """
    Test suite for vendor-specific request/response adaptations.
    """

    @pytest.fixture
    def anthropic_adapter(self):
        """Provides an Anthropic adapter instance."""
        return VendorAdapter.create(Vendor.ANTHROPIC)

    @pytest.fixture
    def aws_adapter(self):
        """Provides an AWS adapter instance."""
        return VendorAdapter.create(Vendor.AWS)

    def test_anthropic_request_adaptation(self, anthropic_adapter):
        """
        Story: The system needs to adapt requests for Anthropic's API
        Given a standard request format
        When converting for Anthropic
        Then it should match Anthropic's expected format
        """
        request = ConverseRequest(
            model_id="claude-3-haiku-20240307",
            messages=[Message(role=Role.USER, content=[ContentBlock(text="Hello")])],
            inference_config=InferenceConfiguration(temperature=0.7, max_tokens=2000),
        )

        adapted = anthropic_adapter.adapt_request(request)

        assert adapted == {
            "model": "claude-3-haiku-20240307",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 2000,
            "temperature": 0.7,
        }

    def test_aws_request_adaptation(self, aws_adapter):
        """
        Story: The system needs to adapt requests for AWS Bedrock API
        Given a standard request format
        When converting for AWS
        Then it should match Bedrock's expected format
        """
        request = ConverseRequest(
            model_id="anthropic.claude-3-haiku-20240307",
            messages=[Message(role=Role.USER, content=[ContentBlock(text="Hello")])],
            inference_config=InferenceConfiguration(
                temperature=0.7,
                max_tokens=1000,
                top_p=None,  # Explicitly set to None
                stop_sequences=None,  # Explicitly set to None
            ),
        )

        adapted = aws_adapter.adapt_request(request)

        assert adapted == {
            "modelId": "anthropic.claude-3-haiku-20240307",
            "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
            "inferenceConfig": {"temperature": 0.7, "maxTokens": 1000},
        }
