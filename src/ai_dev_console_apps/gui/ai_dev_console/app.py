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

from session_storage import FileSessionStorage

from aws import saml_auth_component


def init_session_state():
    """Initialize session state variables, including loading from a saved file."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "client" not in st.session_state:
        st.session_state.client = None
    if "supported_models" not in st.session_state:
        st.session_state.supported_models = SupportedModels()
    if "session_storage" not in st.session_state:
        st.session_state.session_storage = FileSessionStorage()

    # Load the last saved session, if available
    last_session = st.session_state.session_storage.list_sessions()
    if last_session:
        st.session_state.session_name = last_session[0]
        st.session_state.last_saved_state = (
            st.session_state.session_storage.load_session(st.session_state.session_name)
        )
    else:
        st.session_state.session_name = "New Chat"
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

        with st.expander("AWS SAML Authentication"):
            saml_auth_component()

        if st.button("New Chat"):
            st.session_state.messages = []
            st.session_state.session_name = "New Chat"
            st.session_state.last_saved_state = {}

        available_sessions = st.session_state.session_storage.list_sessions()
        if available_sessions:
            selected_session = st.selectbox(
                "Load Chat", options=available_sessions, index=0
            )
            if st.button("Load Chat"):
                st.session_state.last_saved_state = (
                    st.session_state.session_storage.load_session(selected_session)
                )
                st.session_state.session_name = selected_session
                st.session_state.messages = st.session_state.last_saved_state.get(
                    "messages", []
                )

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
            "Max Tokens", min_value=100, max_value=4096, value=4096
        )
        top_k = st.number_input("Top K", min_value=0, max_value=100, value=5, step=1)

        # You should either alter temperature or top_p, but not both.
        # https://docs.anthropic.com/en/api/complete
        # top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.8, step=0.05)

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        return {
            "vendor": current_vendor,
            "model": model,
            "model_id": resolved_model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "system_prompt": system_prompt,
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
        # Print request details for debugging
        st.session_state["debug_info"] = {
            "client_type": type(client).__name__,
            "client_vendor": client.vendor.value,
            "request_model_id": request.model_id,
        }

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
            return response_text

    except Exception as e:
        error_msg = f"Error processing chat stream: {str(e)}"
        # Add debug info to the error message
        if "debug_info" in st.session_state:
            error_msg += f"\nDebug info: {st.session_state['debug_info']}"
        raise ModelClientError(error_msg)


def prepare_messages_for_request(messages):
    prepared_messages = []
    for msg in messages:
        if msg.role in [Role.USER, Role.ASSISTANT]:
            prepared_messages.append(msg)

    # Ensure the last message is from the user
    if prepared_messages and prepared_messages[-1].role == Role.ASSISTANT:
        prepared_messages.pop()

    return prepared_messages


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

                    # If debug_info is available, display it in a collapsed expander
                    if "debug_info" in st.session_state:
                        with st.expander("Debug Info (Click to expand)"):
                            st.json(st.session_state["debug_info"])

            except ModelClientError as e:
                st.session_state.messages.pop()
                st.error(e)

                # On error, also display the debug info to help troubleshoot
                if "debug_info" in st.session_state:
                    with st.expander("Debug Info (Click to expand)"):
                        st.json(st.session_state["debug_info"])


if __name__ == "__main__":
    main()
