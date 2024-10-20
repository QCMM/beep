import qcfractal.interface as ptl
import logging
from typing import List, Tuple, Dict
from collections import Counter
from pydantic import ValidationError
from qcfractal.interface.client import FractalClient
from qcelemental.models.molecule import Molecule
from qcfractal.interface.collections import Dataset, OptimizationDataset, ReactionDataset

from .logging_utils import *


def check_jobs_status(
    client: FractalClient,
    job_ids: List[str],
    logger: logging.Logger,
    wait_interval: int = 600
) -> None:
    """
    Continuously monitors and reports the status of computations for given job IDs.

    Parameters:
    client (FractalClient): The client object used to interact with the QCFractal server.
    job_ids (List[str]): A list of job IDs whose status is to be checked.
    wait_interval (int): Interval in seconds between status checks.

    Returns:
    None: This function prints the status updates but does not return anything.
    """
    all_complete = False

    while not all_complete:
        status_counts = {"COMPLETE": 0, "INCOMPLETE": 0, "ERROR": 0}


        job_stats = client.query_procedures(job_ids)
        for job in job_stats:
            if job:
                status = job.status.upper()
                if status in status_counts:
                    status_counts[status] += 1
                else:
                    logger.info(f"Job ID {job_id}: Unknown status - {status}")
            else:
                logger.info(f"Job ID {job_id}: Not found in the database")

        # Log the status summary
        logger.info(
            f"Job Status Summary: {status_counts['INCOMPLETE']} INCOMPLETE, "
            f"{status_counts['COMPLETE']} COMPLETE, {status_counts['ERROR']} ERROR\n"
        )

        if status_counts["ERROR"] > 0:
            logger.info("Some jobs have ERROR status. Proceed with caution.")

        if status_counts["INCOMPLETE"] == 0:
            all_complete = True
            logger.info("All jobs are COMPLETE. Continuing with the execution.")
        else:
            time.sleep(wait_interval)  # Wait before the next check


