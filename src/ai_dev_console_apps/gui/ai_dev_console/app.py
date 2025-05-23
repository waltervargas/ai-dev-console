from typing import Any, Dict, List

import streamlit as st
from typing import Dict, Any, List, Optional, Iterator

from ai_dev_console.models import (
    Message,
    ContentBlock,
    ConverseRequest,
    Role,
    Vendor,
    ModelClientFactory,
    InferenceConfiguration,
    ModelClientError,
)
from ai_dev_console.models.model import SupportedModels

# # Local imports from the same package
# from .aws import saml_auth_component


def init_session_state() -> None:
    """Initialize session state variables, including loading from a saved file."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "client" not in st.session_state:
        st.session_state.client = None
    if "supported_models" not in st.session_state:
        st.session_state.supported_models = SupportedModels()

    # Initialize default values
    if "session_name" not in st.session_state:
        st.session_state.session_name = "New Chat"
    if "last_saved_state" not in st.session_state:
        st.session_state.last_saved_state = {}


# def init_session_state():
#     """Initialize session state variables."""
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "client" not in st.session_state:
#         st.session_state.client = None
#     if "supported_models" not in st.session_state:
#         st.session_state.supported_models = SupportedModels()
#     # Initialize default vendor and model if not set
#     if "vendor" not in st.session_state:
#         st.session_state.vendor = Vendor.ANTHROPIC.value
#     if "model" not in st.session_state:
#         default_models = get_available_models(Vendor.ANTHROPIC)
#         st.session_state.model = default_models[0] if default_models else None


def get_available_models(vendor: Vendor) -> List[str]:
    """Get available models for vendor."""
    models = []
    for (
        model_name,
        model_mapping,
    ) in st.session_state.supported_models._model_mappings.items():
        if vendor in model_mapping.vendor_ids:
            models.append(model_name)
    return models


def get_sidebar_config() -> Dict[str, Any]:
    """Get configuration from sidebar."""
    with st.sidebar:
        st.title("AI Dev Console")

        if st.button("New Chat"):
            st.session_state.messages = []
            st.session_state.session_name = "New Chat"
            st.session_state.last_saved_state = {}

        # Session management removed

        vendor = st.selectbox(
            "Vendor",
            options=[v.value for v in Vendor],
            key="vendor",
            on_change=on_vendor_change,
        )
        current_vendor = Vendor(vendor)

        available_models = get_available_models(current_vendor)

        if not available_models:
            st.error(f"No models available for vendor {vendor}")
            return {}

        if (
            "model" not in st.session_state
            or st.session_state.model not in available_models
        ):
            st.session_state.model = available_models[0]

        model = st.selectbox("Model", options=available_models, key="model")

        model_info = st.session_state.supported_models.get_model(model)
        resolved_model_id = st.session_state.supported_models.resolve_model_id(
            model, current_vendor
        )

        st.caption(f"Description: {model_info.description}")
        st.caption(f"Latency: {model_info.comparative_latency}")
        st.caption(f"Vision support: {'Yes' if model_info.supports_vision else 'No'}")
        st.caption(f"Model ID: {resolved_model_id}")

        system_prompt = st.text_area(
            "System Prompt", value=st.session_state.get("system_prompt", ""), height=100
        )

        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.4)
        max_tokens = st.number_input(
            "Max Tokens", min_value=100, max_value=120000, value=8092
        )
        top_k = st.number_input("Top K", min_value=0, max_value=100, value=5, step=1)

        # Add thinking controls for claude-3-7 models
        thinking_enabled = False
        thinking_budget = 16000
        if model and "claude-3-7" in model:
            st.markdown("---")
            st.subheader("Extended Reasoning")

            thinking_enabled = st.checkbox(
                "Enable thinking/extended reasoning", value=False
            )

            if thinking_enabled:
                thinking_budget = st.slider(
                    "Thinking budget (tokens)",
                    min_value=1000,
                    max_value=32000,
                    value=16000,
                    step=1000,
                )
                st.caption("Higher budget allows for more thorough reasoning")

        # You should either alter temperature or top_p, but not both.
        # https://docs.anthropic.com/en/api/complete
        # top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.8, step=0.05)

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        # Enable thinking for Claude 3.7 models on both Anthropic and AWS
        thinking_is_supported = "claude-3-7" in model

        return {
            "vendor": current_vendor,
            "model": model,
            "model_id": resolved_model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "system_prompt": system_prompt,
            "thinking_enabled": thinking_enabled if thinking_is_supported else False,
            "thinking_budget": thinking_budget,
        }


def on_vendor_change() -> None:
    """Handle vendor change by updating model selection."""
    if "vendor" in st.session_state:
        current_vendor = Vendor(st.session_state.vendor)
        available_models = get_available_models(current_vendor)
        if available_models:
            st.session_state.model = available_models[0]


def display_chat_messages() -> None:
    """Display chat message history."""
    for msg in st.session_state.messages:
        with st.chat_message(msg.role.value):
            st.write(msg.content[0].text)


def process_chat_stream(
    client: Any, request: ConverseRequest, placeholder: Any
) -> Optional[str]:
    """Handle streaming chat response."""
    try:
        # Print request details for debugging
        # Create request info with basic details
        debug_info = {
            "client_type": type(client).__name__,
            "client_vendor": client.vendor.value,
            "request_model_id": request.model_id,
        }

        # Add full request details in JSON format
        request_dict = {
            "model_id": request.model_id,
            "system": request.system,
            "thinking_enabled": request.thinking_enabled,
            "thinking_budget": (
                request.thinking_budget if request.thinking_enabled else None
            ),
            "model_id_for_thinking": request.model_id,
            "model_contains_claude_3_7": (
                "claude-3-7" in request.model_id
                or "claude-3.7" in request.model_id
                or "claude3.7" in request.model_id
            ),
            "inference_config": {
                "temperature": request.inference_config.temperature,
                "max_tokens": request.inference_config.max_tokens,
                "top_p": request.inference_config.top_p,
                "stop_sequences": request.inference_config.stop_sequences,
            },
            "messages": [
                {
                    "role": msg.role.value,
                    "content": [
                        {
                            "text": block.text,
                            "image": block.image,
                            "document": block.document,
                        }
                        for block in msg.content
                    ],
                }
                for msg in request.messages
            ],
        }

        # Add adapted request that will be sent to the API
        if client.vendor == Vendor.ANTHROPIC:
            adapted_request = client.adapter.adapt_request(request)
            debug_info["adapted_anthropic_request"] = adapted_request
            # Highlight thinking configuration if enabled
            if request.thinking_enabled and "thinking" in adapted_request:
                debug_info["extended_thinking"] = {
                    "enabled": True,
                    "budget_tokens": adapted_request["thinking"].get(
                        "budget_tokens", 16000
                    ),
                }
        elif client.vendor == Vendor.AWS:
            adapted_request = client.adapter.adapt_request(request)
            debug_info["adapted_aws_request"] = adapted_request
            # Highlight thinking configuration if enabled
            if request.thinking_enabled:
                # Look for reasoning_config in additionalModelRequestFields
                if (
                    "additionalModelRequestFields" in adapted_request
                    and "reasoning_config"
                    in adapted_request["additionalModelRequestFields"]
                ):
                    reasoning_config = adapted_request["additionalModelRequestFields"][
                        "reasoning_config"
                    ]
                    debug_info["extended_thinking"] = {
                        "enabled": True,
                        "budget_tokens": reasoning_config.get("budget_tokens", 16000),
                        "source": "additionalModelRequestFields.reasoning_config",
                    }

        debug_info["request"] = request_dict
        st.session_state["debug_info"] = debug_info

        # For AWS models that require inference profiles, ensure ARN is properly resolved
        if client.vendor == Vendor.AWS and hasattr(client, "_resolve_model_id"):
            # Get the resolved model_id directly from the AWS client
            resolved_model_id = client._resolve_model_id(request.model_id)
            st.session_state["debug_info"]["resolved_model_id"] = resolved_model_id
            # Explicitly update the request's model_id with the resolved one
            # This ensures the ARN is properly set for claude-sonnet-3-7
            request.model_id = resolved_model_id

        with client.converse_stream(request) as response_stream:
            response_text = ""

            for chunk in response_stream:
                response_text += chunk
                placeholder.markdown(response_text + "▌")

            placeholder.markdown(response_text)

            # Extract thinking content from the response
            thinking_content = None

            # Check client object for response with thinking content
            if hasattr(client, "response") and client.response:
                if isinstance(client.response, dict) and "thinking" in client.response:
                    thinking_content = client.response["thinking"]
                elif hasattr(client.response, "thinking"):
                    thinking_content = client.response.thinking

            # For Anthropic: try accessing through the stream property
            if (
                not thinking_content
                and hasattr(client, "_stream")
                and hasattr(client._stream, "response")
            ):
                if hasattr(client._stream.response, "thinking"):
                    thinking_content = client._stream.response.thinking

            # Store thinking content in session state if available
            if thinking_content:
                st.session_state["thinking_content"] = thinking_content
                # Also add to debug info for convenience
                if "debug_info" in st.session_state:
                    st.session_state["debug_info"]["claude_thinking"] = {
                        "available": True,
                        "content_type": type(thinking_content).__name__,
                    }

            return response_text

    except Exception as e:
        error_msg = f"Error processing chat stream: {str(e)}"
        # Add debug info to the error message
        if "debug_info" in st.session_state:
            error_msg += f"\nDebug info: {st.session_state['debug_info']}"
        raise ModelClientError(error_msg)


def prepare_messages_for_request(messages: List[Message]) -> List[Message]:
    prepared_messages = []
    for msg in messages:
        if msg.role in [Role.USER, Role.ASSISTANT]:
            prepared_messages.append(msg)

    # Ensure the last message is from the user
    if prepared_messages and prepared_messages[-1].role == Role.ASSISTANT:
        prepared_messages.pop()

    return prepared_messages


def main() -> None:
    st.set_page_config(page_title="AI Dev Console", layout="wide")
    init_session_state()

    # Get configuration from sidebar
    config = get_sidebar_config()

    # Initialize/update client if needed
    if (
        st.session_state.client is None
        or getattr(st.session_state.client, "vendor", None) != config["vendor"]
    ):
        factory = ModelClientFactory()
        st.session_state.client = factory.create_client(config["vendor"])

    # Display chat interface
    st.write(config)

    # Debug information for the AWS client, if using AWS
    if config["vendor"] == Vendor.AWS and st.session_state.client is not None:
        st.caption("AWS Client Debug:")
        # Add these fields if you're running AWS client
        aws_client = st.session_state.client
        if hasattr(aws_client, "region"):
            st.caption(f"AWS Region: {aws_client.region}")
        if hasattr(aws_client, "account_id"):
            st.caption(f"AWS Account ID: {aws_client.account_id}")

        # Display if the model requires inference profile
        models = SupportedModels()
        model_name = config["model"]
        if models.requires_inference_profile(model_name):
            st.caption(f"Model requires inference profile: Yes")
            if hasattr(aws_client, "region") and hasattr(aws_client, "account_id"):
                profile_arn = models.get_inference_profile_arn(
                    model_name, aws_client.region, aws_client.account_id
                )
                st.caption(f"Generated Profile ARN: {profile_arn}")
        else:
            st.caption(f"Model requires inference profile: No")

    display_chat_messages()

    # Chat input
    if prompt := st.chat_input("Message..."):
        # Add and display user message immediately
        user_message = Message(role=Role.USER, content=[ContentBlock(text=prompt)])
        st.session_state.messages.append(user_message)

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Prepare messages for the API request
        prepared_messages = prepare_messages_for_request(st.session_state.messages)
        # st.write(prepared_messages)

        # Create the request
        request = ConverseRequest(
            model_id=config["model_id"],
            messages=prepared_messages,
            system=config.get("system_prompt"),
            inference_config=InferenceConfiguration(
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            ),
            thinking_enabled=config.get("thinking_enabled", False),
            thinking_budget=config.get("thinking_budget", 16000),
        )

        # For debugging
        st.session_state["last_request"] = request

        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                if response_text := process_chat_stream(
                    st.session_state.client, request, placeholder
                ):
                    st.session_state.messages.append(
                        Message(
                            role=Role.ASSISTANT,
                            content=[ContentBlock(text=response_text)],
                        )
                    )

                    # Display debug info and thinking content if available
                    col1, col2 = st.columns(2)

                    with col1:
                        if "debug_info" in st.session_state:
                            with st.expander("Debug Info (Click to expand)"):
                                st.json(st.session_state["debug_info"])

                    with col2:
                        # Show thinking expander either if thinking content exists or if thinking was enabled
                        if "thinking_content" in st.session_state or (
                            "debug_info" in st.session_state
                            and "extended_thinking" in st.session_state["debug_info"]
                        ):
                            with st.expander(
                                "Claude's Thinking Process (Click to expand)"
                            ):
                                thinking = st.session_state.get("thinking_content")

                                if thinking is None or (
                                    isinstance(thinking, dict) and not thinking
                                ):
                                    st.warning(
                                        "Extended reasoning was enabled but no thinking content was returned by the model."
                                    )
                                    if "debug_info" in st.session_state:
                                        if (
                                            "extended_thinking"
                                            in st.session_state["debug_info"]
                                        ):
                                            st.markdown("#### Debug Information")
                                            st.json(
                                                st.session_state["debug_info"][
                                                    "extended_thinking"
                                                ]
                                            )
                                else:
                                    # Anthropic format
                                    if (
                                        isinstance(thinking, dict)
                                        and "value" in thinking
                                    ):
                                        st.markdown("### Extended Reasoning")
                                        st.markdown(f"```\n{thinking['value']}\n```")

                                    # AWS format
                                    elif (
                                        isinstance(thinking, dict)
                                        and "text" in thinking
                                    ):
                                        st.markdown("### Extended Reasoning")
                                        st.markdown(f"```\n{thinking['text']}\n```")

                                    # Simple string format
                                    elif isinstance(thinking, str):
                                        st.markdown("### Extended Reasoning")
                                        st.markdown(f"```\n{thinking}\n```")

                                    # Unknown format - display as JSON
                                    else:
                                        st.markdown(
                                            "### Extended Reasoning (Raw Format)"
                                        )
                                        st.json(thinking)

                                # Display token usage if available
                                if (
                                    "usage" in st.session_state.get("debug_info", {})
                                    and "thinking_tokens"
                                    in st.session_state["debug_info"]["usage"]
                                ):
                                    st.caption(
                                        f"Thinking tokens used: {st.session_state['debug_info']['usage']['thinking_tokens']}"
                                    )

            except ModelClientError as e:
                st.session_state.messages.pop()
                st.error(e)

                # On error, also display the debug info to help troubleshoot
                if "debug_info" in st.session_state:
                    with st.expander("Debug Info (Click to expand)"):
                        st.json(st.session_state["debug_info"])


if __name__ == "__main__":
    main()
