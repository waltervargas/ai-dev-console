import argparse
import os
import sys
from typing import List, Optional

from ai_dev_console.models import (
    ContentBlock,
    ConverseRequest,
    InferenceConfiguration,
    Message,
    ModelClientFactory,
    Role,
    Vendor,
)


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the AI prompt tool."""
    parser = argparse.ArgumentParser(
        description="Send a prompt to an AI model and get the response",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Vendors:
    - anthropic
    - aws
    - openai (future)

Environment Variables:
    ANTHROPIC_API_KEY - API key for Anthropic models
    AWS_ACCESS_KEY_ID - AWS access key for Bedrock models
    AWS_SECRET_ACCESS_KEY - AWS secret key for Bedrock models
    AWS_DEFAULT_REGION - AWS region for Bedrock models
    OPENAI_API_KEY API - key for OpenAI models (future)

Examples:
    # Basic usage
    echo "What is Python?" | ai-prompt --vendor anthropic --model claude-3-haiku-20240307

    # With custom parameters
    echo "Explain ML" | ai-prompt --vendor aws --model anthropic.claude-3-haiku-20240307 \\
        --temperature 0.7 --max-tokens 1000
        """,
    )

    parser.add_argument(
        "--vendor",
        type=str,
        choices=[v.value for v in Vendor],
        required=True,
        help="AI vendor to use for the model",
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Specific model identifier to use"
    )

    # Optional inference configuration parameters
    parser.add_argument(
        "--temperature", type=float, help="Controls randomness in response (0.0 - 1.0)"
    )

    parser.add_argument(
        "--max-tokens", type=int, help="Maximum number of tokens in the response"
    )

    return parser.parse_args(args)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the AI prompt CLI.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Parse arguments
        args = parse_arguments(argv)

        # Read prompt from stdin
        prompt = os.getenv("DEBUG_INPUT") or sys.stdin.read().strip()
        if not prompt:
            print("Error: No input provided", file=sys.stderr)
            return 1

        # Prepare inference configuration
        inference_config = None
        if args.temperature is not None or args.max_tokens is not None:
            inference_config = InferenceConfiguration(
                temperature=args.temperature, max_tokens=args.max_tokens
            )

        # Process and output response
        vendor = Vendor(args.vendor)
        factory = ModelClientFactory()
        client = factory.create_client(vendor)

        request = ConverseRequest(
            model_id=args.model,
            messages=[Message(role=Role.USER, content=[ContentBlock(text=prompt)])],
            inference_config=inference_config,
        )

        response = client.converse(request)
        print(response.messages[-1].content[0].text)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# Allow direct script execution and entry point invocation
if __name__ == "__main__":
    sys.exit(main())
