import pandas as pd


def dict_to_csv(filename: str, content: dict):
    """Write the keys of `content` as headings and the associated lists as rows
    to `filename`.csv."""
    df = pd.DataFrame(content)
    df.to_csv(f"{filename}.csv", index=False)
