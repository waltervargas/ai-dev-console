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


def get_available_models(vendor: Vendor) -> List[str]:
    """Get available models for vendor."""
    models = []
    for model_name, model in st.session_state.supported_models.available_models.items():
        if model.vendor == vendor:
            models.append(model_name)
    return models


def get_sidebar_config() -> Dict[str, Any]:
    """Get configuration from sidebar."""
    with st.sidebar:
        st.title("AI Dev Console")

        # Model Configuration
        vendor = st.selectbox("Vendor", options=[v.value for v in Vendor], key="vendor")

        current_vendor = Vendor(vendor)
        available_models = get_available_models(current_vendor)

        model = st.selectbox("Model", options=available_models, key="model")

        # Show model details (optional)
        if model:
            model_info = st.session_state.supported_models.get_model(model)
            st.caption(f"Description: {model_info.description}")
            st.caption(f"Latency: {model_info.comparative_latency}")
            st.caption(
                f"Vision support: {'Yes' if model_info.supports_vision else 'No'}"
            )

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
            "temperature": temperature,
            "max_tokens": max_tokens,
        }


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
    st.title("Chat")
    display_chat_messages()

    # Chat input
    if prompt := st.chat_input("Message..."):
        # Add and display user message immediately
        user_message = Message(role=Role.USER, content=[ContentBlock(text=prompt)])
        st.session_state.messages.append(user_message)

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Create request
        request = ConverseRequest(
            model_id=config["model"],
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
