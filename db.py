"""
db.py — Async SQLite layer (aiosqlite).

Tables
------
trades            — every executed order (paper or live)
signals           — every arbitrage signal detected, whether acted upon or not
portfolio_snapshots — periodic balance / P&L snapshots
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,           -- ISO-8601 UTC
    market_id       TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,           -- BTCUSDT / ETHUSDT
    timeframe       TEXT    NOT NULL,           -- 5m / 15m
    side            TEXT    NOT NULL,           -- YES / NO
    size_usdc       REAL    NOT NULL,           -- USDC spent
    entry_price     REAL    NOT NULL,           -- fractional (0-1)
    contracts       REAL    NOT NULL,           -- number of contracts bought
    cex_prob        REAL    NOT NULL,
    poly_prob       REAL    NOT NULL,
    edge_pct        REAL    NOT NULL,
    confidence      REAL    NOT NULL,
    kelly_fraction  REAL    NOT NULL,
    paper           INTEGER NOT NULL DEFAULT 1, -- 1 = paper, 0 = live
    status          TEXT    NOT NULL DEFAULT 'OPEN',  -- OPEN / WON / LOST / CANCELLED
    exit_price      REAL,
    pnl             REAL,
    closed_at       TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    market_id       TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    timeframe       TEXT    NOT NULL,
    cex_prob        REAL    NOT NULL,
    poly_prob       REAL    NOT NULL,
    lag_pct         REAL    NOT NULL,
    edge_pct        REAL    NOT NULL,
    confidence      REAL    NOT NULL,
    acted           INTEGER NOT NULL DEFAULT 0, -- 1 if trade was placed
    skip_reason     TEXT                        -- why not traded (if applicable)
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    balance         REAL    NOT NULL,           -- free USDC
    open_positions  REAL    NOT NULL,           -- cost basis of open positions
    realized_pnl    REAL    NOT NULL,
    unrealized_pnl  REAL    NOT NULL,
    equity          REAL    NOT NULL,           -- balance + open_positions + unrealized_pnl
    daily_drawdown  REAL    NOT NULL
);
"""

# ── Dataclasses returned by queries ───────────────────────────────────────────

@dataclass
class TradeRow:
    id: int
    timestamp: str
    market_id: str
    symbol: str
    timeframe: str
    side: str
    size_usdc: float
    entry_price: float
    contracts: float
    cex_prob: float
    poly_prob: float
    edge_pct: float
    confidence: float
    kelly_fraction: float
    paper: bool
    status: str
    exit_price: Optional[float]
    pnl: Optional[float]
    closed_at: Optional[str]


@dataclass
class SnapshotRow:
    id: int
    timestamp: str
    balance: float
    open_positions: float
    realized_pnl: float
    unrealized_pnl: float
    equity: float
    daily_drawdown: float


# ── Database class ─────────────────────────────────────────────────────────────

class Database:
    def __init__(self, path: str = "polyradar.db") -> None:
        self._path = path
        self._db: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Open connection and initialise schema."""
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_DDL)
        await self._db.commit()
        logger.info("Database ready: %s", self._path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ── Trades ────────────────────────────────────────────────────────────────

    async def insert_trade(
        self,
        market_id: str,
        symbol: str,
        timeframe: str,
        side: str,
        size_usdc: float,
        entry_price: float,
        contracts: float,
        cex_prob: float,
        poly_prob: float,
        edge_pct: float,
        confidence: float,
        kelly_fraction: float,
        paper: bool,
    ) -> int:
        """Insert a new OPEN trade and return its row id."""
        now = _utcnow()
        async with self._lock:
            assert self._db
            cur = await self._db.execute(
                """
                INSERT INTO trades (
                    timestamp, market_id, symbol, timeframe, side,
                    size_usdc, entry_price, contracts,
                    cex_prob, poly_prob, edge_pct, confidence, kelly_fraction, paper
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    now, market_id, symbol, timeframe, side,
                    size_usdc, entry_price, contracts,
                    cex_prob, poly_prob, edge_pct, confidence, kelly_fraction,
                    int(paper),
                ),
            )
            await self._db.commit()
            return cur.lastrowid  # type: ignore[return-value]

    async def close_trade(
        self, trade_id: int, exit_price: float, pnl: float, status: str
    ) -> None:
        """Mark a trade WON/LOST with final P&L."""
        async with self._lock:
            assert self._db
            await self._db.execute(
                """
                UPDATE trades
                SET status=?, exit_price=?, pnl=?, closed_at=?
                WHERE id=?
                """,
                (status, exit_price, pnl, _utcnow(), trade_id),
            )
            await self._db.commit()

    async def get_recent_trades(self, n: int = 10) -> List[TradeRow]:
        """Return the *n* most recently inserted trades."""
        assert self._db
        async with self._db.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (n,)
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_trade(r) for r in rows]

    async def get_open_trades(self) -> List[TradeRow]:
        assert self._db
        async with self._db.execute(
            "SELECT * FROM trades WHERE status='OPEN' ORDER BY timestamp"
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_trade(r) for r in rows]

    async def get_daily_stats(self) -> Dict[str, Any]:
        """Realized P&L, win-rate, and trade count for today (UTC)."""
        today = date.today().isoformat()
        assert self._db
        async with self._db.execute(
            """
            SELECT
                COUNT(*)                                    AS total,
                SUM(CASE WHEN status='WON'  THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN status='LOST' THEN 1 ELSE 0 END) AS losses,
                COALESCE(SUM(pnl), 0)                       AS realized_pnl
            FROM trades
            WHERE DATE(timestamp) = ? AND status != 'OPEN'
            """,
            (today,),
        ) as cur:
            row = await cur.fetchone()
        total = row["total"] or 0
        wins  = row["wins"]  or 0
        return {
            "total": total,
            "wins": wins,
            "losses": row["losses"] or 0,
            "realized_pnl": row["realized_pnl"] or 0.0,
            "win_rate": (wins / total) if total > 0 else 0.0,
        }

    async def get_all_time_win_rate(self) -> float:
        assert self._db
        async with self._db.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status='WON' THEN 1 ELSE 0 END) AS wins
            FROM trades WHERE status != 'OPEN'
            """
        ) as cur:
            row = await cur.fetchone()
        total = row["total"] or 0
        wins  = row["wins"]  or 0
        return (wins / total) if total > 0 else 0.0

    # ── Signals ───────────────────────────────────────────────────────────────

    async def insert_signal(
        self,
        market_id: str,
        symbol: str,
        timeframe: str,
        cex_prob: float,
        poly_prob: float,
        lag_pct: float,
        edge_pct: float,
        confidence: float,
        acted: bool,
        skip_reason: Optional[str] = None,
    ) -> None:
        async with self._lock:
            assert self._db
            await self._db.execute(
                """
                INSERT INTO signals (
                    timestamp, market_id, symbol, timeframe,
                    cex_prob, poly_prob, lag_pct, edge_pct, confidence,
                    acted, skip_reason
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    _utcnow(), market_id, symbol, timeframe,
                    cex_prob, poly_prob, lag_pct, edge_pct, confidence,
                    int(acted), skip_reason,
                ),
            )
            await self._db.commit()

    # ── Portfolio snapshots ───────────────────────────────────────────────────

    async def insert_snapshot(
        self,
        balance: float,
        open_positions: float,
        realized_pnl: float,
        unrealized_pnl: float,
        equity: float,
        daily_drawdown: float,
    ) -> None:
        async with self._lock:
            assert self._db
            await self._db.execute(
                """
                INSERT INTO portfolio_snapshots (
                    timestamp, balance, open_positions,
                    realized_pnl, unrealized_pnl, equity, daily_drawdown
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    _utcnow(), balance, open_positions,
                    realized_pnl, unrealized_pnl, equity, daily_drawdown,
                ),
            )
            await self._db.commit()

    async def get_latest_snapshot(self) -> Optional[SnapshotRow]:
        assert self._db
        async with self._db.execute(
            "SELECT * FROM portfolio_snapshots ORDER BY id DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return SnapshotRow(**dict(row))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_trade(row: aiosqlite.Row) -> TradeRow:
    d = dict(row)
    d["paper"] = bool(d["paper"])
    return TradeRow(**d)
