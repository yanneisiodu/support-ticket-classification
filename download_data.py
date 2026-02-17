import logging
import sys

import pandas as pd

logger = logging.getLogger(__name__)

SUBSET_DATA_URL = "https://raw.githubusercontent.com/jitender18/IT_Support_Ticket_Classification_with_AWS_Integration/master/latest_ticket_data.csv"
ALL_DATA_URL = "https://privdatastorage.blob.core.windows.net/github/support-tickets-classification/datasets/all_tickets.csv"


def download_data() -> None:
    logger.info("Downloading subset data from %s", SUBSET_DATA_URL)
    df_subset_data = pd.read_csv(SUBSET_DATA_URL)
    df_subset_data.columns = [c.lower() for c in df_subset_data.columns]
    df_subset_data = df_subset_data.rename(columns={"description": "body"})
    assert "body" in df_subset_data.columns and "category" in df_subset_data.columns
    df_subset_data.to_csv("subset_tickets.csv", index=False)
    logger.info("Saved subset_tickets.csv (%d rows)", len(df_subset_data))

    logger.info("Downloading all data from %s", ALL_DATA_URL)
    df_all_data = pd.read_csv(ALL_DATA_URL)
    df_all_data.to_csv("all_tickets.csv", index=False)
    logger.info("Saved all_tickets.csv (%d rows)", len(df_all_data))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    download_data()
