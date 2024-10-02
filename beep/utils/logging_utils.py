import logging
from typing import Any, Dict, List, Tuple, Union, NoReturn
import pandas as pd


def setup_logging(prefix: str, molecule_name: str) -> logging.Logger:
    """
    Sets up a logging system for a specific molecule sampling process.

    This function configures logging for the beep_sampling module. It creates a log file with a name based 
    on the given molecule name. It also sets up logging to both the file and the console with INFO level. 
    The log messages are formatted to include only the message part.

    Parameters:
    molecule_name (str): The name of the molecule, which will be used to create the log file name.

    Returns:
    logging.Logger: The configured logger object for the beep_sampling module.
    """
    logger = logging.getLogger("beep")
    logger.setLevel(logging.INFO)

    # File handler setup
    log_file =  prefix +  f"_{molecule_name}.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    # Console handler setup
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger

#def log_formatted_list(logger: logging.Logger, my_list: List[str], description: str):
#    """
#    Converts a list into a formatted string and logs it using the provided logger.
#
#    Parameters:
#    logger (logging.Logger): The logger object to use for logging the information.
#    my_list (List[str]): The list to be formatted and logged.
#    """
#    # Define the format for each item (e.g., "- Item")
#    formatted_items = [f"* {item}" for item in my_list]
#
#    # Join the formatted items into a single string with line breaks
#    formatted_string = "\n".join(formatted_items)
#
#    # Log the formatted string
#    logger.info(f"{description}\n{formatted_string}")


def log_formatted_list(logger: logging.Logger, my_list: List[str], description: str, max_rows: int = 10):
    """
    Converts a list into formatted strings organized in a table with at most max_rows rows
    and logs it using the provided logger.

    Parameters:
    logger (logging.Logger): The logger object to use for logging the information.
    my_list (List[str]): The list to be formatted and logged.
    max_rows (int): Maximum number of rows in the table.
    """
    # Validate max_rows
    if max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")

    # Determine the number of columns needed
    num_columns = len(my_list) // max_rows + (1 if len(my_list) % max_rows else 0)

    # Create a list of rows, each containing up to num_columns items
    table = []
    for row_idx in range(max_rows):
        row = []
        for col_idx in range(num_columns):
            # Calculate the index in the original list
            item_idx = col_idx * max_rows + row_idx
            if item_idx < len(my_list):
                row.append(f"* {my_list[item_idx]}")
            else:
                row.append("")  # Empty string if no item for this slot
        table.append(row)

    # Calculate the width of the widest item for formatting
    max_width = max(len(item) for item in my_list) + 4  # Add 4 for the "* " prefix and some padding

    # Format each row with aligned columns
    formatted_rows = ["\t".join(item.ljust(max_width) for item in row) for row in table]

    # Join the rows into a formatted string
    formatted_string = "\n".join(formatted_rows)

    # Log the formatted string
    logger.info(f"{description}\n{formatted_string}")


def dict_to_log(logger: logging.Logger, data: dict):
    """
    Logs the contents of a dictionary where each key has a list of items.
    Each key is logged followed by a two-column table with the items of the list.

    Parameters:
    logger (logging.Logger): The logger object used for logging.
    data (dict): The dictionary to be logged.
    """
    for key, items in data.items():
        logger.info(f"\nXC Functional Type: {key}")
        logger.info(f"{'Index':<10} | {'Item':<30}")
        logger.info(f"{'-' * 10}+{'-' * 30}")
        for index, item in enumerate(items):
            logger.info(f"{index:<10} | {item:<30}")

def padded_log(logger: logging.Logger, message: str, variable: str ='', padding_char: str = '-', total_length: int = 90):
    """
    Logs a message with custom padding characters on both sides, padding to a fixed total length,
    and includes the value of a variable within the message.

    Parameters:
    logger (logging.Logger): The logger object used for logging.
    message (str): The message template to be logged, with placeholders for variables.
    variable (str): The variable's value to be inserted into the message.
    padding_char (str): The character used for padding.
    total_length (int): The total fixed length of the log message, including padding.
    """
    # Format the message with the variable value
    if '{}' in message and variable:
        formatted_message = message.format(variable)
    else:
        formatted_message = message

    # Length of the text within the padding
    text_length = len(formatted_message)

    # Calculate the number of padding characters needed for padding on each side
    padding_each_side = (total_length - text_length - 2) // 2  # -2 for the spaces after and before the padding chars

    # Adjust for odd total lengths
    padding_extra = "" if (total_length - text_length) % 2 == 0 else padding_char

    # Construct the padded message
    padded_message = f"\n{padding_char * padding_each_side} {formatted_message} {padding_char * padding_each_side}{padding_extra}\n"

    logger.info(padded_message)

def log_progress(logger: logging.Logger, current_step: int, total_steps: int, bar_length: int = 50) -> NoReturn:
    """
    Logs a progress bar to the given logger.

    This function logs a progress bar by calculating the percentage completion and represents it as a bar of '=' characters.

    Parameters:
    logger (logging.Logger): The logger to which the progress bar will be logged.
    current_step (int): The current step (or iteration) of the operation.
    total_steps (int): The total number of steps (or iterations) in the operation.
    bar_length (int): The character length of the progress bar. Defaults to 50.

    Returns:
    NoReturn
    """
    if not (0 <= current_step <= total_steps):
        raise ValueError("Current step is out of the bounds of total steps")

    proportion_complete = current_step / total_steps
    filled_length = int(proportion_complete * bar_length)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    logger.info(f"[{bar}] {current_step}/{total_steps} Complete")

def log_dataframe_averages(logger: logging.Logger, df: pd.DataFrame):
    """
    Logs the average values of each group in the dataframe, excluding the first unnecessary column,
    with the functional group as a header above each group's table. Also logs a summary table at
    the end with the functional group and level of theory corresponding to the minimum average value.

    Parameters:
    logger (logging.Logger): The logger object used for logging.
    df (pd.DataFrame): The dataframe containing the data.
    """
    df = df.iloc[:, 1:]  # Drop the first column
    grouped_columns = df.columns.to_series().groupby(lambda x: x.split('_')[1] + '_' + x.split('_')[2]).groups
    summary_data = {}

    for group, columns in grouped_columns.items():
        logger.info(f"\nFunctional Group: {group}")
        logger.info(f"{'Level of Theory':<30} | Average")
        logger.info('-' * 45)

        group_averages = df[columns].mean()
        min_avg_value = group_averages.min()
        min_avg_theory = group_averages.idxmin().split('_')[3:]  # Assuming the level of theory is from the 4th split
        min_avg_theory = '_'.join(min_avg_theory)

        for col in columns:
            level_of_theory = '_'.join(col.split('_')[3:])
            avg_value = df[col].mean()
            mark = '*' if avg_value == min_avg_value else ' '
            logger.info(f"{level_of_theory:<30} | {avg_value:.6f} {mark}")

        summary_data[group] = (min_avg_theory, min_avg_value)
    # Log the summary table
    logger.info("\nSummary of minimum average RMSD values per functional group:")
    logger.info(f"{'Functional Group':<30} | {'Level of Theory':<30} | {'Min Average RMSD':<30}")
    logger.info('-' * 95)
    for group, (theory, min_avg) in summary_data.items():
        logger.info(f"{group:<30} | {theory:<30} | {min_avg:.6f}")



def log_energy_mae(logger: logging.Logger, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Mean Absolute Error (MAE) for each method_basis pair in a DataFrame and log the results.

    This function takes a DataFrame where each row corresponds to a different structure and each column
    corresponds to a different computational method. It computes the MAE for each method_basis pair,
    and logs the 15 best-performing methods (those with the lowest MAE) for each pair. The best overall
    method is marked with an asterisk.

    Parameters:
    logger (logging.Logger): The logger object to use for logging the MAE results.
    df (pd.DataFrame): The DataFrame containing the computational methods' errors.

    Returns:
    pd.DataFrame: A new DataFrame containing the MAEs for each method_basis pair.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Step 1: Extract the method_basis pair and add it as a new column
    df_copy['method_basis'] = df_copy.index.map(lambda x: '/'.join(x.split('_')[-2:]))
    abs_columns = df_copy.columns.difference(['method_basis'])
    df_copy[abs_columns] = df_copy[abs_columns].abs()
    mae_results = df_copy.groupby('method_basis').mean()

    # Determine the maximum width for method names for formatting
    max_method_len = max(len(method) for method in mae_results.columns) + 1  # Adding space for the pipe symbol

    # Header for the log
    header = f"\n{'Level of Theory':<{max_method_len}} | MAE\n" + "-" * (max_method_len + 6)  # 6 for " | MAE" and newline

    # Log the header for MAE results
    logger.info(header)

    # New logging capability with neat formatting
    for index, row in mae_results.iterrows():
        sorted_row = row.sort_values()
        min_value = sorted_row.iloc[0]
        log_message = [f"Lowest MAEs for {index} geometry:\n"]

        # Formatting each line to match the requested structure
        for col in sorted_row.index[:15]:
            value = sorted_row[col]
            star = "*" if value == min_value else " "
            method_formatted = col.ljust(max_method_len - 2)  # Adjust for the length of " |"
            log_message.append(f"{method_formatted} | {value:.6f}{star}\n")

        # Combine the log message and log it
        log_message_str = ''.join(log_message)
        logger.info(log_message_str)
    
    return mae_results

