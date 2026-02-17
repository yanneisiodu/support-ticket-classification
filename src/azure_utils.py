import logging
import os
from typing import Dict, Optional

import yaml
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)

_REQUIRED_ENV_VARS = [
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_STORAGE_ACCOUNT_NAME",
    "AZURE_STORAGE_ACCOUNT_KEY",
]


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"Required environment variable {name} is not set. "
            f"See .env.example for the full list of required variables."
        )
    return value


def create_container(
    container_name: str,
    blob_service_client: Optional[BlobServiceClient] = None,
    account_name: Optional[str] = None,
    account_key: Optional[str] = None,
    default_endpoints_protocol: str = "https",
    endpoint_suffix: str = "core.windows.net",
    exist_ok: bool = True,
) -> BlobServiceClient:
    if blob_service_client is None:
        account_name = account_name or _require_env("AZURE_STORAGE_ACCOUNT_NAME")
        account_key = account_key or _require_env("AZURE_STORAGE_ACCOUNT_KEY")
        blob_service_client = BlobServiceClient.from_connection_string(
            blob_connection_string(
                account_name,
                account_key,
                default_endpoints_protocol=default_endpoints_protocol,
                endpoint_suffix=endpoint_suffix,
            )
        )
    try:
        blob_service_client.create_container(container_name)
        logger.info("Created blob container %s", container_name)
    except ResourceExistsError:
        if exist_ok:
            logger.info("Blob container %s already exists", container_name)
        else:
            raise
    return blob_service_client


def blob_connection_string(
    account_name: str,
    account_key: str,
    default_endpoints_protocol: str = "https",
    endpoint_suffix: str = "core.windows.net",
) -> str:
    return (
        f"DefaultEndpointsProtocol={default_endpoints_protocol};"
        f"AccountName={account_name};"
        f"AccountKey={account_key};"
        f"EndpointSuffix={endpoint_suffix}"
    )


def load_azure_conf(conf_path: Optional[str] = None) -> Dict[str, str]:
    conf_path = conf_path or os.path.join("src", "azure_conf.yml")
    with open(conf_path, "r") as f:
        azure_conf = yaml.load(f, Loader=yaml.FullLoader)
    azure_conf["SUBSCRIPTION_ID"] = _require_env("AZURE_SUBSCRIPTION_ID")
    azure_conf.setdefault("STORAGE", {})
    azure_conf["STORAGE"]["AccountName"] = _require_env("AZURE_STORAGE_ACCOUNT_NAME")
    azure_conf["STORAGE"]["AccountKey"] = _require_env("AZURE_STORAGE_ACCOUNT_KEY")
    return azure_conf
