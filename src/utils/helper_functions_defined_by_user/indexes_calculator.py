from typing import List
import pyspark.sql.functions as f
from pyspark.sql import Column, DataFrame


def logit(groups: List[str], n_windows: int, round_to: int = 5) -> List[Column]:
    #####INTERESTS LOGIT VARIABLE: IDF
    min_idf = f.first("min")
    total_diff = f.first("max") - min_idf

    limit = 0.02298

    return [
        (
            f.round(
                (
                    1
                    / (
                        1
                        + f.exp(
                            (
                                -(
                                    0.25
                                    + 0.5
                                    * (
                                        (f.first(f"stat_{group}") - min_idf)
                                        / total_diff
                                    )
                                    * f.sum(f.col(group))
                                    / n_windows
                                )
                                + 4 * f.exp(-(f.sum(f.col(group)) / n_windows))
                            )
                        )
                    )
                ),
                round_to,
            )
            - limit
        ).alias(group)
        for group in groups
    ]


def indexes_calculator(
    data: DataFrame,
    interests_list: List[str],
    windows_number: int,
    level_of_distinction: List[str],
) -> DataFrame:
    return data.groupBy(*level_of_distinction).agg(
        *logit(groups=interests_list, n_windows=windows_number)
    )
