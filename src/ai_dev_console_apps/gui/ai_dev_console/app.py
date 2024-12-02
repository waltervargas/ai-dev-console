import streamlit as st
from typing import Dict, Any, List

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


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "client" not in st.session_state:
        st.session_state.client = None
    if "supported_models" not in st.session_state:
        st.session_state.supported_models = SupportedModels()
    # Initialize default vendor and model if not set
    if "vendor" not in st.session_state:
        st.session_state.vendor = Vendor.ANTHROPIC.value
    if "model" not in st.session_state:
        default_models = get_available_models(Vendor.ANTHROPIC)
        st.session_state.model = default_models[0] if default_models else None


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

        # Model Configuration
        vendor = st.selectbox(
            "Vendor",
            options=[v.value for v in Vendor],
            key="vendor",
            on_change=on_vendor_change,
        )
        current_vendor = Vendor(vendor)

        # Get available models for this vendor
        available_models = get_available_models(current_vendor)

        # Ensure we have a valid model for the current vendor
        if not available_models:
            st.error(f"No models available for vendor {vendor}")
            return {}

        if (
            "model" not in st.session_state
            or st.session_state.model not in available_models
        ):
            st.session_state.model = available_models[0]

        model = st.selectbox("Model", options=available_models, key="model")

        # Get model info and resolved ID
        model_info = st.session_state.supported_models.get_model(model)
        resolved_model_id = st.session_state.supported_models.resolve_model_id(
            model, current_vendor
        )

        # Show model details
        st.caption(f"Description: {model_info.description}")
        st.caption(f"Latency: {model_info.comparative_latency}")
        st.caption(f"Vision support: {'Yes' if model_info.supports_vision else 'No'}")
        st.caption(f"Model ID: {resolved_model_id}")

        # Parameters
        st.subheader("Parameters")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        max_tokens = st.number_input("Max Tokens", 100, 4096, 1000)

        # Clear Chat
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        return {
            "vendor": current_vendor,
            "model": model,
            "model_id": resolved_model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }


def on_vendor_change():
    """Handle vendor change by updating model selection."""
    if "vendor" in st.session_state:
        current_vendor = Vendor(st.session_state.vendor)
        available_models = get_available_models(current_vendor)
        if available_models:
            st.session_state.model = available_models[0]


def display_chat_messages():
    """Display chat message history."""
    for msg in st.session_state.messages:
        with st.chat_message(msg.role.value):
            st.write(msg.content[0].text)


def process_chat_stream(client, request: ConverseRequest, placeholder: st.empty):
    """Handle streaming chat response."""
    try:
        with client.stream_response(request) as response_stream:
            response_text = ""

            for chunk in response_stream:
                response_text += chunk
                placeholder.markdown(response_text + "▌")

            placeholder.markdown(response_text)
            return response_text

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def main():
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
    display_chat_messages()

    # Chat input
    if prompt := st.chat_input("Message..."):
        # Add and display user message immediately
        user_message = Message(role=Role.USER, content=[ContentBlock(text=prompt)])
        st.session_state.messages.append(user_message)

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        request = ConverseRequest(
            model_id=config["model_id"],
            messages=[user_message],
            inference_config=InferenceConfiguration(
                temperature=config["temperature"], max_tokens=config["max_tokens"]
            ),
        )

        # Process response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            if response_text := process_chat_stream(
                st.session_state.client, request, placeholder
            ):
                st.session_state.messages.append(
                    Message(
                        role=Role.ASSISTANT, content=[ContentBlock(text=response_text)]
                    )
                )


if __name__ == "__main__":
    main()
