import qcfractal.interface as ptl
import logging
from typing import List, Tuple, Dict
from collections import Counter
from pydantic import ValidationError
from qcfractal.interface.client import FractalClient
from qcelemental.models.molecule import Molecule
from qcfractal.interface.collections import Dataset, OptimizationDataset, ReactionDataset
import time

from .logging_utils import *

def check_jobs_status(
    client: FractalClient,
    job_ids: List[str],
    logger: logging.Logger,
    wait_interval: int = 600,
    print_job_ids = False
) -> None:
    """
    Continuously monitors and reports the status of computations for given job IDs, 
    processing job IDs in chunks if their number exceeds the query limit.

    Parameters:
    client (FractalClient): The client object used to interact with the QCFractal server.
    job_ids (List[str]): A list of job IDs whose status is to be checked.
    wait_interval (int): Interval in seconds between status checks.

    Returns:
    None: This function prints the status updates but does not return anything.
    """
    all_complete = False
    chunk_size = 1000  # The maximum query limit

    while not all_complete:
        status_counts = {"COMPLETE": 0, "INCOMPLETE": 0, "ERROR": 0}

        # Process job_ids in chunks of 1000 or less
        for i in range(0, len(job_ids), chunk_size):
            chunk = job_ids[i:i + chunk_size]
            if print_job_ids:
                logger.info(f"List with job_ids: {chunk}")
            job_stats = client.query_procedures(chunk)

            for job in job_stats:
                if job:
                    status = job.status.upper()
                    if status in status_counts:
                        status_counts[status] += 1
                    else:
                        logger.info(f"Job ID {job.id}: Unknown status - {status}")
                else:
                    logger.info(f"Job ID {job.id}: Not found in the database")

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

