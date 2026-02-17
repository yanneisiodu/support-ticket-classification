"""
Endpoint to launch an experiment on AzureML.

NOTE: This uses Azure ML SDK v1 (azureml-core / Estimator). This is kept as a
legacy optional path. A future migration to azure.ai.ml v2 is planned.
"""

import logging
import os
from os.path import dirname
from typing import Optional

from azureml.core import Datastore, Experiment, Run, Workspace
from azureml.train.estimator import Estimator

from src.azure_utils import load_azure_conf
from src.utils import pip_packages

logger = logging.getLogger(__name__)


def run_azure_experiment_with_storage(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    datastore_name: str,
    container_name: str,
    storage_account_name: str,
    storage_account_key: str,
    compute_name: str,
    experiment_name: Optional[str] = None,
    source_directory: Optional[str] = None,
    image_name: Optional[str] = None,
    use_gpu: bool = True,
) -> Run:
    workspace = Workspace(subscription_id, resource_group, workspace_name)
    data_store = Datastore.register_azure_blob_container(
        workspace=workspace,
        datastore_name=datastore_name,
        container_name=container_name,
        account_name=storage_account_name,
        account_key=storage_account_key,
    )
    source_directory = source_directory or dirname(__file__)
    if compute_name not in workspace.compute_targets:
        raise ValueError(
            f"Compute '{compute_name}' is not created in '{workspace_name}' workspace"
        )
    estimator = Estimator(
        source_directory=source_directory,
        script_params={"--data-folder": data_store.as_mount()},
        compute_target=workspace.compute_targets[compute_name],
        pip_packages=pip_packages(),
        entry_script=os.path.join(source_directory, "azure_train.py"),
        use_gpu=use_gpu,
        custom_docker_image=image_name,
    )
    experiment_name = experiment_name or __file__.split(os.sep)[-1].split(".py")[0]
    experiment = Experiment(workspace=workspace, name=experiment_name)
    run = experiment.submit(estimator)
    logger.info("Submitted experiment %s, run id: %s", experiment_name, run.id)
    return run


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    azure_conf = load_azure_conf()
    run = run_azure_experiment_with_storage(
        subscription_id=azure_conf["SUBSCRIPTION_ID"],
        resource_group=azure_conf["RESOURCE_GROUP"],
        workspace_name=azure_conf["WORKSPACE_NAME"],
        datastore_name=azure_conf["DATASTORE_NAME"],
        container_name=azure_conf["CONTAINER_NAME"],
        storage_account_name=azure_conf["STORAGE"]["AccountName"],
        storage_account_key=azure_conf["STORAGE"]["AccountKey"],
        compute_name=azure_conf["COMPUTE_NAME"],
        experiment_name=__file__.split(os.sep)[-1].split(".py")[0],
        source_directory=os.path.dirname(__file__),
        image_name=azure_conf["IMAGE_NAME"],
        use_gpu=True,
    )
