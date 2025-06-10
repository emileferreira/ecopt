from datetime import datetime


# `strftime` does not support a cross-platform way of removing leading zeros
today = datetime.today()
__version__ = f"{today.year}.{today.month}.{today.day}"
