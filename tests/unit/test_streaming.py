import pytest
from unittest.mock import Mock, patch, ANY
from contextlib import contextmanager
from typing import Iterator, List

from ai_dev_console.models import (
    ModelClient,
    Message,
    ContentBlock,
    InferenceConfiguration,
    ConverseRequest,
    Role,
    Vendor,
    ModelClientError,
)


class TestModelClientStreaming:
    """
    Test suite for model client streaming capabilities.
    """

    @pytest.fixture
    def mock_anthropic_stream(self):
        """Provides a mock Anthropic stream."""

        class MockStream:
            def __init__(self):
                self.text_stream = ["Hello", " World", "!"]

            def get_final_message(self):
                return {"role": "assistant", "content": "Hello World!"}

        return MockStream()

    @pytest.fixture
    def mock_anthropic_messages(self, mock_anthropic_stream):
        """Provides a mock Anthropic messages interface."""

        class MockMessages:
            @contextmanager
            def stream(self, **kwargs):
                yield mock_anthropic_stream

        return MockMessages()

    @pytest.fixture
    def mock_anthropic_client(self, mock_anthropic_messages):
        """Provides a mock Anthropic client."""
        mock = Mock()
        mock.messages = mock_anthropic_messages
        return mock

    @pytest.fixture
    def test_request(self):
        """Provides a test request."""
        return ConverseRequest(
            model_id="claude-3-haiku-20240307",
            messages=[Message(role=Role.USER, content=[ContentBlock(text="Hello")])],
            inference_config=InferenceConfiguration(temperature=0.7, max_tokens=1000),
        )

    def test_anthropic_streaming_success(self, mock_anthropic_client, test_request):
        """
        Story: A developer wants to stream responses from Anthropic's models
        Given a configured Anthropic client
        When streaming a response
        Then they should receive chunks of text in order
        """
        from ai_dev_console.models.client.base import AnthropicClient

        client = AnthropicClient(mock_anthropic_client)

        with client.stream_response(test_request) as stream:
            chunks = list(stream)
            assert chunks == ["Hello", " World", "!"]

    def test_anthropic_streaming_validation(self, mock_anthropic_client):
        """
        Story: The system should validate requests before streaming
        Given an invalid request
        When attempting to stream
        Then it should raise an appropriate error
        """
        from ai_dev_console.models.client.base import AnthropicClient

        client = AnthropicClient(mock_anthropic_client)
        invalid_request = ConverseRequest(
            model_id="", messages=[]  # Invalid empty model ID  # Invalid empty messages
        )

        with pytest.raises(ModelClientError) as exc_info:
            with client.stream_response(invalid_request):
                pass

        assert "Model ID is required" in str(exc_info.value)

    def test_anthropic_empty_chunks_filtered(self, mock_anthropic_client, test_request):
        """
        Story: The system should filter out empty chunks
        Given a stream with empty chunks
        When streaming a response
        Then it should only yield non-empty chunks
        """
        from ai_dev_console.models.client.base import AnthropicClient

        # Modify mock to include empty chunks
        class MockStreamWithEmpty:
            def __init__(self):
                self.text_stream = ["Hello", "", " World", "", "!"]

            def get_final_message(self):
                return {"role": "assistant", "content": "Hello World!"}

        mock_anthropic_client.messages.stream = Mock()

        @contextmanager
        def mock_stream(**kwargs):
            yield MockStreamWithEmpty()

        mock_anthropic_client.messages.stream.side_effect = mock_stream

        client = AnthropicClient(mock_anthropic_client)

        with client.stream_response(test_request) as stream:
            chunks = list(stream)
            assert chunks == ["Hello", " World", "!"]

    def test_base_client_streaming_not_implemented(self):
        """
        Story: The base client should not implement streaming
        Given a base ModelClient
        When attempting to stream
        Then it should raise NotImplementedError
        """

        class TestClient(ModelClient):
            def converse(self, request):
                pass

            async def converse_async(self, request):
                pass

        client = TestClient(Vendor.ANTHROPIC, Mock())

        with pytest.raises(NotImplementedError):
            with client.stream_response(Mock()):
                pass

    def test_aws_streaming_not_implemented(self, test_request):
        """
        Story: AWS streaming should not be implemented yet
        Given an AWS client
        When attempting to stream
        Then it should raise NotImplementedError
        """
        from ai_dev_console.models.client.base import AWSClient

        client = AWSClient(Mock())

        with pytest.raises(NotImplementedError):
            with client.stream_response(test_request):
                pass

    def test_anthropic_streaming_error_handling(
        self, mock_anthropic_client, test_request
    ):
        """
        Story: The system should handle streaming errors gracefully
        Given a failing stream
        When streaming a response
        Then it should raise an appropriate error
        """
        from ai_dev_console.models.client.base import AnthropicClient

        # Create a messages mock that raises an error
        class MockMessagesError:
            @contextmanager
            def stream(self, **kwargs):
                raise Exception("Stream failed")

        # Replace the messages attribute with our error-raising mock
        mock_anthropic_client.messages = MockMessagesError()

        client = AnthropicClient(mock_anthropic_client)

        with pytest.raises(ModelClientError) as exc_info:
            with client.stream_response(test_request):
                pass

        assert "Streaming failed" in str(exc_info.value)
