from optimagic.logging.read_from_database import read_last_rows
from optimagic.logging.write_to_database import append_row, update_row


def log_scheduled_steps_and_get_ids(steps, logging, database):
    """Add scheduled steps to the steps table of the database and get their ids.

    The ids are only determined once the steps are written to the database and the
    ids of all previously existing steps are known.

    Args:
        steps (list): List of dicts with entries for the steps table.
        logging (bool): Whether to actually write to the database.
        database (DataBase):

    Returns:
        list: List of integers with the step ids.

    """
    default_row = {"status": "scheduled"}
    if logging:
        for row in steps:
            data = {**default_row, **row}

            append_row(
                data=data,
                table_name="steps",
                database=database,
            )

        step_ids = read_last_rows(
            table_name="steps",
            n_rows=len(steps),
            return_type="dict_of_lists",
            database=database,
        )["rowid"]
    else:
        step_ids = list(range(len(steps)))

    return step_ids


def update_step_status(step, new_status, database):
    step = int(step)

    assert new_status in ["scheduled", "running", "complete", "skipped"]

    update_row(
        data={"status": new_status},
        rowid=step,
        table_name="steps",
        database=database,
    )
