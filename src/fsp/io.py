"""Load a single dataframe from whatever format it arrives as."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read(path: str | Path) -> pd.DataFrame:
    """Read CSV / parquet / Excel / SPSS-SAS-Stata into a DataFrame."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if suffix in {".sav", ".dta", ".sas7bdat"}:
        import pyreadstat  # lazy: heavy optional path

        reader = {
            ".sav": pyreadstat.read_sav,
            ".dta": pyreadstat.read_dta,
            ".sas7bdat": pyreadstat.read_sas7bdat,
        }[suffix]
        df, _ = reader(str(p))
        return df
    # CSV / TSV — try utf-8, then detect encoding on failure.
    try:
        return pd.read_csv(p)
    except UnicodeDecodeError:
        from charset_normalizer import from_path

        best = from_path(p).best()
        encoding = best.encoding if best is not None else "latin-1"
        return pd.read_csv(p, encoding=encoding)
