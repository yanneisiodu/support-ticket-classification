import logging
import os

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient

from src.azure_utils import create_container, load_azure_conf

logger = logging.getLogger(__name__)


def write_csv_azure(
    csv: str,
    csv_name: str,
    blob_service_client: BlobServiceClient,
    container_name: str,
    exist_ok: bool = True,
) -> None:
    try:
        blob_service_client.get_blob_client(container_name, csv_name).upload_blob(csv)
        logger.info("Uploaded %s to container %s", csv_name, container_name)
    except ResourceExistsError:
        if exist_ok:
            logger.info("Blob %s already exists in %s, skipping", csv_name, container_name)
        else:
            raise


def load_csv_as_str(csv_path: str) -> str:
    with open(csv_path) as f:
        csv = "".join(f.readlines())
    return csv


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    azure_conf = load_azure_conf()

    blob_service_client = create_container(
        azure_conf["CONTAINER_NAME"],
        account_name=azure_conf["STORAGE"]["AccountName"],
        account_key=azure_conf["STORAGE"]["AccountKey"],
        exist_ok=True,
    )
    write_csv_azure(
        csv=load_csv_as_str(azure_conf["LOCAL_DATASET_PATH"]),
        csv_name=azure_conf["LOCAL_DATASET_PATH"].split(os.sep)[-1],
        blob_service_client=blob_service_client,
        container_name=azure_conf["CONTAINER_NAME"],
    )
    logger.info("Data uploaded to Azure!")
