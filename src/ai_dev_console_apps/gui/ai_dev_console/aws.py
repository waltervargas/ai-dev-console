import base64
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional

import boto3
import pyperclip
import streamlit as st

from ai_dev_console.models.client.base import ModelClientFactory
from ai_dev_console.models.vendor import Vendor


# Decode SAML assertion and parse XML
def decode_saml_assertion(saml_assertion: str) -> ET.Element:
    try:
        decoded_assertion = base64.b64decode(saml_assertion)
        return ET.fromstring(decoded_assertion)
    except Exception as e:
        raise ValueError(f"Failed to decode SAML assertion: {e}")


# Extract AWS account IDs and roles
def extract_roles_from_saml(saml_xml: ET.Element) -> list[str]:
    roles = []
    # Search for Attribute elements containing roles
    for attr in saml_xml.findall(".//{urn:oasis:names:tc:SAML:2.0:assertion}Attribute"):
        if attr.attrib.get("Name") == "https://aws.amazon.com/SAML/Attributes/Role":
            for value in attr.findall(
                ".//{urn:oasis:names:tc:SAML:2.0:assertion}AttributeValue"
            ):
                roles.append(value.text)
    return roles


# Parse role ARNs and organize by account ID
def parse_roles(roles: list[str]) -> dict[str, list[dict[str, str]]]:
    parsed_roles = {}
    for role in roles:
        role_arn, saml_provider_arn = role.split(",")
        account_id = role_arn.split(":")[4]
        if account_id not in parsed_roles:
            parsed_roles[account_id] = []
        parsed_roles[account_id].append(
            {
                "role_arn": role_arn,
                "saml_provider_arn": saml_provider_arn,
                "role_name": role_arn.split("role")[-1].lstrip("/"),
            }
        )
    return parsed_roles


# Assume a role using AssumeRoleWithSAML API
def assume_role_with_saml(
    saml_assertion: str, role_arn: str, saml_provider_arn: str
) -> dict[str, Any]:
    try:
        session = boto3.session.Session()
        sts_client = session.client("sts")

        response = sts_client.assume_role_with_saml(
            RoleArn=role_arn,
            PrincipalArn=saml_provider_arn,
            SAMLAssertion=saml_assertion,
            DurationSeconds=3600,
        )

        # Set credentials in the environment
        credentials = response["Credentials"]
        os.environ["AWS_ACCESS_KEY_ID"] = credentials["AccessKeyId"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = credentials["SecretAccessKey"]
        os.environ["AWS_SESSION_TOKEN"] = credentials["SessionToken"]

        boto3.setup_default_session(
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
        )
        return credentials
    except Exception as e:
        raise ValueError(f"Failed to assume role: {e}")


def get_role_options(
    selected_account: str, parsed_roles: dict[str, list[dict[str, str]]]
) -> list[str]:
    role_options = []
    for account_id, role_list in parsed_roles.items():
        if account_id == selected_account:
            for role in role_list:
                role_options.append(f"{role['role_name']}")
    return role_options


def get_saml_provider_arn(
    selected_account: str, parsed_roles: dict[str, list[dict[str, str]]]
) -> str:
    for account_id, role_list in parsed_roles.items():
        if account_id == selected_account:
            for role in role_list:
                return role["saml_provider_arn"]


def get_role_arn(
    selected_account: str,
    selected_role: str,
    parsed_roles: dict[str, list[dict[str, str]]],
) -> str:
    for account_id, role_list in parsed_roles.items():
        if account_id == selected_account:
            for role in role_list:
                if role["role_name"] == selected_role:
                    return role["role_arn"]


def saml_auth_component() -> None:
    if "saml_assertion" not in st.session_state:
        st.session_state.saml_assertion = ""
    if "selected_account" not in st.session_state:
        st.session_state.selected_account = ""
    if "selected_role" not in st.session_state:
        st.session_state.selected_role = ""
    if "saml_provider_arn" not in st.session_state:
        st.session_state.saml_provider_arn = ""
    if "role_arn" not in st.session_state:
        st.session_state.role_arn = ""

    st.session_state.saml_assertion = st.text_area(
        "Paste your SAML Assertion here:",
        value=st.session_state.saml_assertion,
        height=200,
    )

    if st.session_state.saml_assertion:
        try:
            saml_xml = decode_saml_assertion(st.session_state.saml_assertion)
            roles = extract_roles_from_saml(saml_xml)
            parsed_roles = parse_roles(roles)
            st.session_state.selected_account = st.selectbox(
                "Select Account ID:", list(parsed_roles.keys())
            )
            st.session_state.saml_provider_arn = get_saml_provider_arn(
                st.session_state.selected_account, parsed_roles
            )
            role_options = get_role_options(
                st.session_state.selected_account, parsed_roles
            )
            st.session_state.selected_role = st.selectbox(
                "Select Role (search by role name):", role_options
            )
            st.session_state.role_arn = get_role_arn(
                st.session_state.selected_account,
                st.session_state.selected_role,
                parsed_roles,
            )

            if st.session_state.selected_role:
                if st.button("Assume Role"):

                    try:
                        credentials = assume_role_with_saml(
                            st.session_state.saml_assertion,
                            st.session_state.role_arn,
                            st.session_state.saml_provider_arn,
                        )

                        os.environ["AWS_ACCESS_KEY_ID"] = credentials["AccessKeyId"]
                        os.environ["AWS_SECRET_ACCESS_KEY"] = credentials[
                            "SecretAccessKey"
                        ]
                        os.environ["AWS_SESSION_TOKEN"] = credentials["SessionToken"]

                        boto3.setup_default_session(
                            aws_access_key_id=credentials["AccessKeyId"],
                            aws_secret_access_key=credentials["SecretAccessKey"],
                            aws_session_token=credentials["SessionToken"],
                        )

                        st.session_state.client = ModelClientFactory().create_client(
                            Vendor.AWS
                        )

                        st.success(
                            "Role assumed successfully. AWS credentials are set."
                        )
                        try:
                            sts_client = boto3.client("sts")
                            caller_identity = sts_client.get_caller_identity()
                            st.info(caller_identity.get("Arn"))
                        except Exception as e:
                            st.error(f"Failed to retrieve caller identity: {e}")
                    except Exception as e:
                        st.error(f"Failed to assume role: {e}")

        except Exception as e:
            st.error(f"Invalid SAML assertion: {e}")


def main() -> None:
    # Retrieve the SAML assertion from the clipboard
    print("Fetching SAML assertion from clipboard...")
    saml_assertion_base64 = pyperclip.paste().strip()

    if not saml_assertion_base64:
        print("Error: Clipboard is empty or does not contain a SAML assertion.")
        return

    try:
        # Step 1: Decode and parse the SAML assertion
        print("Decoding SAML assertion...")
        saml_xml = decode_saml_assertion(saml_assertion_base64)
        print("SAML assertion successfully decoded.")

        # Step 2: Extract roles
        print("Extracting roles from SAML assertion...")
        roles = extract_roles_from_saml(saml_xml)
        if not roles:
            print("No roles found in the SAML assertion.")
            return
        print("Roles found in the SAML assertion.")

        # Step 3: Request Account ID and Role Name from user
        account_id = input("Enter the AWS Account ID: ").strip()
        role_name = input("Enter the AWS Role Name: ").strip()

        # Step 4: Find matching Role and SAML Provider ARNs
        parsed_roles = parse_roles(roles)
        if account_id not in parsed_roles:
            print(f"Error: No roles found for account ID {account_id}.")
            return

        matching_role = None
        for role in parsed_roles[account_id]:
            if role["role_arn"].endswith(f"role/{role_name}"):
                matching_role = role
                break

        if not matching_role:
            print(f"Error: Role '{role_name}' not found for account ID {account_id}.")
            return

        role_arn = matching_role["role_arn"]
        saml_provider_arn = matching_role["saml_provider_arn"]

        # Step 5: Assume the selected role
        print(f"\nAssuming role {role_arn}...")
        credentials = assume_role_with_saml(
            saml_assertion_base64, role_arn, saml_provider_arn
        )
        print("Role assumed successfully.")
        print("Temporary AWS credentials:")
        print(f"AWS_ACCESS_KEY_ID: {credentials['AccessKeyId']}")
        print(f"AWS_SECRET_ACCESS_KEY: {credentials['SecretAccessKey']}")
        print(f"AWS_SESSION_TOKEN: {credentials['SessionToken']}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
