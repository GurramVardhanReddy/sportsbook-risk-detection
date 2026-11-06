"""Real Premier League odds -> one row per selection the simulator can price.

Source: football-data.co.uk Premier League CSVs. For every selection we keep
- the book's prices: Bet365 opening and closing odds. Between them the book's
  price moves linearly in implied probability over the PRICE_WINDOW_H hours
  before kickoff (a modelling assumption; the CSVs have only two snapshots);
- the fair closing probability: Pinnacle closing odds with the margin removed
  by normalising the implied probabilities (multiplicative method);
- whether the selection won.

Seasons stop at 2024/25 because Pinnacle closing odds are missing for 170 of
the 380 matches in 2025/26.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
SEASONS = ["2122", "2223", "2324", "2425"]
PRICE_WINDOW_H = 96.0

# market -> [(selection, book open column, book close column, Pinnacle close column)]
MARKETS: dict[str, list[tuple[str, str, str, str]]] = {
    "1X2": [
        ("H", "B365H", "B365CH", "PSCH"),
        ("D", "B365D", "B365CD", "PSCD"),
        ("A", "B365A", "B365CA", "PSCA"),
    ],
    "OU25": [
        ("O", "B365>2.5", "B365C>2.5", "PC>2.5"),
        ("U", "B365<2.5", "B365C<2.5", "PC<2.5"),
    ],
}
ODDS_COLUMNS = [c for sels in MARKETS.values() for s in sels for c in s[1:]]


def download_season(code: str, raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Fetch one season's CSV unless it is already on disk."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"E0_{code}.csv"
    if force or not path.exists():
        resp = requests.get(BASE_URL.format(code=code), timeout=30)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


def load_matches(seasons: list[str] = SEASONS, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Matches with kickoff times, results and every odds column the simulator needs."""
    frames = []
    for code in seasons:
        df = pd.read_csv(download_season(code, raw_dir), encoding="latin-1")
        df["season"] = code
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    keep = ["season", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"] + ODDS_COLUMNS
    df = df[keep].dropna().copy()
    df["kickoff"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
    df = df.sort_values("kickoff").reset_index(drop=True)
    df["match_id"] = np.arange(len(df))
    return df


def _won(df: pd.DataFrame, market: str, selection: str) -> pd.Series:
    if market == "1X2":
        return df["FTR"] == selection
    goals = df["FTHG"] + df["FTAG"]
    return goals > 2.5 if selection == "O" else goals < 2.5


def build_selections(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, market, selection) with book prices and fair close probability."""
    rows = []
    for market, sels in MARKETS.items():
        close_cols = [s[3] for s in sels]
        overround = (1.0 / matches[close_cols]).sum(axis=1)
        for selection, open_col, book_close_col, close_col in sels:
            rows.append(pd.DataFrame({
                "match_id": matches["match_id"].to_numpy(),
                "season": matches["season"].to_numpy(),
                "kickoff": matches["kickoff"].to_numpy(),
                "market": market,
                "selection": selection,
                "book_open": matches[open_col].to_numpy(dtype=float),
                "book_close": matches[book_close_col].to_numpy(dtype=float),
                "p_fair": (1.0 / matches[close_col] / overround).to_numpy(dtype=float),
                "won": _won(matches, market, selection).to_numpy(),
            }))
    sel = pd.concat(rows, ignore_index=True)
    sel = sel.sort_values(["match_id", "market", "selection"]).reset_index(drop=True)
    sel["sel_id"] = np.arange(len(sel))
    return sel


def book_price(book_open, book_close, hours_before, window_h: float = PRICE_WINDOW_H):
    """Book price at `hours_before` kickoff: linear in implied probability from open to close."""
    hours = np.clip(np.asarray(hours_before, dtype=float), 0.0, window_h)
    progress = 1.0 - hours / window_h  # 0 at the open, 1 at kickoff
    q_open = 1.0 / np.asarray(book_open, dtype=float)
    q_close = 1.0 / np.asarray(book_close, dtype=float)
    return 1.0 / (q_open + (q_close - q_open) * progress)
