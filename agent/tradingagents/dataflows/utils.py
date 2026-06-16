import os
import json
import pandas as pd
from datetime import date, timedelta, datetime
from typing import Annotated

SavePathType = Annotated[str, "File path to save data. If None, data is not saved."]


def save_output(data: pd.DataFrame, tag: str, save_path: SavePathType = None) -> None:
    if save_path:
        data.to_csv(save_path)
        print(f"{tag} saved to {save_path}")


def get_current_date():
    return date.today().strftime("%Y-%m-%d")


def get_next_weekday(date):
    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")
    if date.weekday() >= 5:
        days_to_add = 7 - date.weekday()
        return date + timedelta(days=days_to_add)
    return date
