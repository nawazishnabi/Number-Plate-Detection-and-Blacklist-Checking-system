"""
blacklist.py
------------
Handles all blacklist operations:
  - Creating default CSV if absent
  - Loading and normalising plate numbers
  - Exact + fuzzy matching
"""

import re
import os
import pandas as pd
from difflib import SequenceMatcher
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import BLACKLIST_PATH, FUZZY_MATCH_THRESHOLD


class BlacklistManager:
    def __init__(self):
        self._ensure_file_exists()
        self.df = self._load()
        print(f"[Blacklist] Loaded {len(self.df)} entries from {BLACKLIST_PATH}")

    # ── Public API ────────────────────────────────────────────────────

    def check(self, plate_text: str) -> tuple[bool, str]:
        """
        Returns (is_blacklisted, reason).
        Tries exact match first, then fuzzy.
        """
        if not plate_text or plate_text in ("UNREADABLE", "ERROR"):
            return False, ""

        try:
            clean = re.sub(r'[^A-Z0-9]', '', plate_text.upper())

            # 1. Exact match
            match = self.df[self.df["PlateNumber"] == clean]
            if not match.empty:
                return True, match.iloc[0]["Reason"]

            # 2. Fuzzy match
            for _, row in self.df.iterrows():
                similarity = SequenceMatcher(None, clean, row["PlateNumber"]).ratio()
                if similarity >= FUZZY_MATCH_THRESHOLD:
                    return True, row["Reason"]

            return False, ""

        except Exception as e:
            print(f"[Blacklist] check error: {e}")
            return False, ""

    def reload(self):
        """Hot-reload the CSV (useful if user edits it while app is running)."""
        self.df = self._load()

    # ── Private helpers ───────────────────────────────────────────────

    def _ensure_file_exists(self):
        if not os.path.exists(BLACKLIST_PATH):
            os.makedirs(os.path.dirname(BLACKLIST_PATH), exist_ok=True)
            sample = {
                "PlateNumber": ["JK08D4356", "JK01AB1234", "JK14SU3550", "JK08XP1434"],
                "Reason":      ["Traffic Violation", "Stolen Vehicle",
                                "Duplicate Number Plate", "Unregistered Vehicle"],
            }
            pd.DataFrame(sample).to_csv(BLACKLIST_PATH, index=False)
            print(f"[Blacklist] Created default file at {BLACKLIST_PATH}")

    def _load(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(BLACKLIST_PATH)
            df["PlateNumber"] = (
                df["PlateNumber"]
                .str.upper()
                .str.replace(r'[^A-Z0-9]', '', regex=True)
            )
            return df
        except Exception as e:
            print(f"[Blacklist] Load error: {e}")
            return pd.DataFrame(columns=["PlateNumber", "Reason"])