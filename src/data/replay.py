"""Simulate a real-time data feed by advancing a pointer through the
historical TLC panel.

Use case: the dashboard wants to feel live, but NYC TLC only publishes
trip data monthly. ``ReplayClock`` keeps a "current hour" pointer that
moves forward at configurable speed; the dashboard queries it to get
the state of every zone at that moment.

The clock is persisted in a small JSON file so it survives Streamlit
reruns and pod restarts.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


STATE_PATH = Path("data/processed/replay_state.json")


@dataclass
class ReplayState:
    """The on-disk state of the replay clock.

    Attributes
    ----------
    current_hour : ISO timestamp
        The "now" pointer - rows with ``hour <= current_hour`` are visible.
    speed_x : float
        Replay speed multiplier. ``speed_x=60`` means one real minute
        advances the clock by 60 simulated hours (great for demos).
    last_tick_unix : float
        Wall-clock time of the most recent tick. Used to compute drift.
    """

    current_hour: str
    speed_x: float = 60.0
    last_tick_unix: float = 0.0


class ReplayClock:
    """Pointer into a historical hourly panel that advances over time."""

    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = "hour",
        speed_x: float = 60.0,
        state_path: Path = STATE_PATH,
    ) -> None:
        self.df = df.sort_values(date_col).reset_index(drop=True)
        self.date_col = date_col
        self.state_path = Path(state_path)
        self.min_hour = pd.to_datetime(df[date_col].min())
        self.max_hour = pd.to_datetime(df[date_col].max())
        self.state = self._load_or_init(speed_x)

    def _load_or_init(self, speed_x: float) -> ReplayState:
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    raw = json.load(f)
                return ReplayState(**raw)
            except Exception:
                pass
        # Default: start a few days into the data so lag features are populated.
        start = self.min_hour + pd.Timedelta(days=7)
        return ReplayState(
            current_hour=start.isoformat(),
            speed_x=speed_x,
            last_tick_unix=time.time(),
        )

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(asdict(self.state), f, indent=2)

    def tick(self) -> pd.Timestamp:
        """Advance the clock by the elapsed real time * speed_x."""
        now = time.time()
        elapsed_real = max(0.0, now - self.state.last_tick_unix)
        sim_hours = elapsed_real / 60.0 * self.state.speed_x
        current = pd.to_datetime(self.state.current_hour) + pd.Timedelta(hours=sim_hours)
        if current > self.max_hour:
            current = self.min_hour + pd.Timedelta(days=7)
        self.state.current_hour = current.isoformat()
        self.state.last_tick_unix = now
        self._save()
        return current

    def reset(self) -> None:
        """Rewind the clock to one week after the start of the data."""
        start = self.min_hour + pd.Timedelta(days=7)
        self.state.current_hour = start.isoformat()
        self.state.last_tick_unix = time.time()
        self._save()

    @property
    def now(self) -> pd.Timestamp:
        return pd.to_datetime(self.state.current_hour)

    def visible_df(self) -> pd.DataFrame:
        """Return only the rows up to and including ``now``."""
        return self.df[pd.to_datetime(self.df[self.date_col]) <= self.now]

    def latest_per_zone(self, n_hours: int = 24) -> pd.DataFrame:
        """Last ``n_hours`` of pickups per zone, ending at ``now``."""
        window_start = self.now - pd.Timedelta(hours=n_hours)
        d = self.visible_df()
        return d[pd.to_datetime(d[self.date_col]) > window_start]


def get_replay_clock(
    featured_path: str | Path = "data/processed/trips_featured.parquet",
    speed_x: float = 60.0,
) -> ReplayClock:
    """Helper to load the featured panel and return a clock."""
    df = pd.read_parquet(featured_path)
    return ReplayClock(df, speed_x=speed_x)


if __name__ == "__main__":
    clock = get_replay_clock()
    print(f"Replay clock initialised at {clock.now}")
    print(f"Data range: {clock.min_hour} -> {clock.max_hour}")
    print(f"Visible rows: {len(clock.visible_df()):,}")
