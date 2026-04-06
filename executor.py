"""
executor.py — Trade execution: paper simulation and live CLOB orders.

Paper trading
-------------
Positions are tracked in-memory.  When the candle closes the current
Polymarket price (or full payout of 1.0) is used to mark the trade WON/LOST.

Live trading
------------
Uses py-clob-client to submit limit orders at the best ask (YES/NO side).
Order status is polled until MATCHED or a 30-second timeout elapses.

Both modes share the Portfolio class which tracks cash, open positions, and
realised P&L.  The daily high-water mark and drawdown are maintained here and
exposed to the kill-switch logic in bot.py.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable, Dict, List, Optional

from arbitrage import Signal
from db import Database

logger = logging.getLogger(__name__)

# ── Position ──────────────────────────────────────────────────────────────────

@dataclass
class Position:
    trade_id: int
    market_id: str
    symbol: str
    timeframe: str
    side: str           # YES / NO
    contracts: float    # number of contracts purchased
    entry_price: float  # cost per contract
    size_usdc: float    # USDC spent
    opened_at: datetime
    clob_order_id: Optional[str] = None   # live only
    # Set when position is closed
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    closed_at: Optional[datetime] = None
    status: str = "OPEN"   # OPEN / WON / LOST / CANCELLED

    @property
    def unrealized_pnl(self) -> float:
        """Estimated P&L using current market price (not yet resolved)."""
        if self.exit_price is None:
            return 0.0
        gross = self.contracts * self.exit_price
        return gross - self.size_usdc


# ── Portfolio ─────────────────────────────────────────────────────────────────

class Portfolio:
    """
    Tracks cash balance, open positions, realised P&L, and daily drawdown.
    Thread-safe via asyncio.Lock (must be used within the event loop).
    """

    def __init__(self, initial_balance: float, paper: bool = True) -> None:
        self.paper        = paper
        self.balance      = initial_balance   # free USDC
        self.realized_pnl = 0.0
        self.positions: Dict[int, Position] = {}   # keyed by trade_id
        self._lock        = asyncio.Lock()

        # Daily drawdown tracking
        self._today       = date.today()
        self._day_start_equity = initial_balance
        self._peak_equity = initial_balance

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def open_position_cost(self) -> float:
        return sum(p.size_usdc for p in self.positions.values() if p.status == "OPEN")

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values() if p.status == "OPEN")

    @property
    def equity(self) -> float:
        return self.balance + self.open_position_cost + self.unrealized_pnl

    @property
    def daily_drawdown(self) -> float:
        """Fraction of day-start equity lost today (positive = drawdown)."""
        self._maybe_reset_daily()
        eq = self.equity
        if self._peak_equity > 0:
            self._peak_equity = max(self._peak_equity, eq)
        dd = (self._peak_equity - eq) / self._peak_equity if self._peak_equity > 0 else 0.0
        return dd

    # ── Mutation ───────────────────────────────────────────────────────────────

    async def open_position(
        self,
        trade_id: int,
        signal: Signal,
        clob_order_id: Optional[str] = None,
    ) -> Position:
        async with self._lock:
            cost = signal.size_usdc
            if cost > self.balance:
                cost = self.balance  # partial fill at best
            contracts = cost / signal.cex_prob  # approximate contracts received

            pos = Position(
                trade_id=trade_id,
                market_id=signal.market_id,
                symbol=signal.symbol,
                timeframe=signal.timeframe,
                side=signal.side,
                contracts=contracts,
                entry_price=signal.cex_prob,
                size_usdc=cost,
                opened_at=datetime.now(timezone.utc),
                clob_order_id=clob_order_id,
            )
            self.balance -= cost
            self.positions[trade_id] = pos
            return pos

    async def close_position(
        self,
        trade_id: int,
        exit_price: float,
        status: str,
    ) -> Optional[Position]:
        """Close a position with the given exit price (0 = loss, 1 = win)."""
        async with self._lock:
            pos = self.positions.get(trade_id)
            if pos is None or pos.status != "OPEN":
                return None
            gross     = pos.contracts * exit_price
            pnl       = gross - pos.size_usdc
            pos.exit_price  = exit_price
            pos.pnl         = pnl
            pos.closed_at   = datetime.now(timezone.utc)
            pos.status      = status
            self.balance   += gross
            self.realized_pnl += pnl
            return pos

    def _maybe_reset_daily(self) -> None:
        today = date.today()
        if today != self._today:
            self._today           = today
            self._day_start_equity = self.equity
            self._peak_equity     = self.equity


# ── Paper executor ────────────────────────────────────────────────────────────

class PaperExecutor:
    """
    Simulates trades without touching the CLOB API.

    Resolution: when a candle closes the binance_feed CandleState `is_closed`
    flag becomes True.  The bot loop calls `resolve_closed_candles()` which
    marks positions WON (exit_price=1.0) or LOST (exit_price=0.0) based on
    whether the final close was above the candle open.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        db: Database,
        on_close: Optional[Callable[[Position], None]] = None,
    ) -> None:
        self._portfolio = portfolio
        self._db = db
        self._on_close = on_close

    async def execute(self, signal: Signal) -> Optional[Position]:
        """
        Immediately 'fill' the paper order at the signal's entry price.
        Returns the opened Position or None if insufficient balance.
        """
        if self._portfolio.balance < 1.0:
            logger.warning("Paper balance too low to execute trade.")
            return None

        trade_id = await self._db.insert_trade(
            market_id=signal.market_id,
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            side=signal.side,
            size_usdc=signal.size_usdc,
            entry_price=signal.cex_prob,
            contracts=signal.size_usdc / max(signal.cex_prob, 0.01),
            cex_prob=signal.cex_prob,
            poly_prob=signal.poly_prob,
            edge_pct=signal.edge_pct,
            confidence=signal.confidence,
            kelly_fraction=signal.kelly_sized,
            paper=True,
        )

        pos = await self._portfolio.open_position(trade_id, signal)
        logger.info(
            "[PAPER] OPEN %s %s/%s  side=%s  size=$%.2f  entry=%.3f  edge=%.1f%%",
            signal.market_id[:8], signal.symbol, signal.timeframe,
            signal.side, signal.size_usdc, signal.cex_prob, signal.edge_pct * 100,
        )
        return pos

    async def resolve_candle(
        self,
        trade_id: int,
        candle_final_price: float,
        candle_open_price: float,
    ) -> Optional[Position]:
        """
        Resolve a paper position when its candle closes.

        Args:
            trade_id:           DB trade id
            candle_final_price: actual close price of the 5m/15m candle
            candle_open_price:  open price of the candle
        """
        pos = self._portfolio.positions.get(trade_id)
        if pos is None or pos.status != "OPEN":
            return None

        price_went_up = candle_final_price > candle_open_price
        if pos.side == "YES":
            won = price_went_up
        else:
            won = not price_went_up

        exit_price = 1.0 if won else 0.0
        status     = "WON" if won else "LOST"

        closed = await self._portfolio.close_position(trade_id, exit_price, status)
        if closed:
            await self._db.close_trade(
                trade_id, exit_price, closed.pnl or 0.0, status
            )
            logger.info(
                "[PAPER] CLOSE %s %s  pnl=$%.2f  status=%s",
                pos.symbol, pos.timeframe, closed.pnl or 0.0, status,
            )
            if self._on_close:
                self._on_close(closed)
        return closed


# ── Live executor ─────────────────────────────────────────────────────────────

class LiveExecutor:
    """
    Places real orders via the Polymarket CLOB.

    IMPORTANT: Only instantiated when all three live-trading flags are set.
    """

    _ORDER_TIMEOUT = 30.0   # seconds to wait for fill confirmation

    def __init__(
        self,
        portfolio: Portfolio,
        db: Database,
        clob_client: object,
        on_close: Optional[Callable[[Position], None]] = None,
    ) -> None:
        self._portfolio  = portfolio
        self._db         = db
        self._client     = clob_client
        self._on_close   = on_close

    async def execute(self, signal: Signal) -> Optional[Position]:
        """Submit a limit order and return the position on (partial) fill."""
        loop = asyncio.get_running_loop()
        try:
            order_response = await asyncio.wait_for(
                loop.run_in_executor(None, self._submit_order_sync, signal),
                timeout=self._ORDER_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("[LIVE] Order submission timed out for %s.", signal.market_id)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("[LIVE] Order submission failed: %s", exc)
            return None

        order_id = self._extract_order_id(order_response)
        logger.info(
            "[LIVE] OPEN %s %s/%s  side=%s  size=$%.2f  order_id=%s",
            signal.market_id[:8], signal.symbol, signal.timeframe,
            signal.side, signal.size_usdc, order_id,
        )

        trade_id = await self._db.insert_trade(
            market_id=signal.market_id,
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            side=signal.side,
            size_usdc=signal.size_usdc,
            entry_price=signal.cex_prob,
            contracts=signal.size_usdc / max(signal.cex_prob, 0.01),
            cex_prob=signal.cex_prob,
            poly_prob=signal.poly_prob,
            edge_pct=signal.edge_pct,
            confidence=signal.confidence,
            kelly_fraction=signal.kelly_sized,
            paper=False,
        )

        pos = await self._portfolio.open_position(trade_id, signal, clob_order_id=order_id)
        return pos

    async def close_position_at_resolution(
        self,
        trade_id: int,
        won: bool,
    ) -> Optional[Position]:
        """Call after on-chain resolution is confirmed."""
        exit_price = 1.0 if won else 0.0
        status     = "WON" if won else "LOST"
        closed = await self._portfolio.close_position(trade_id, exit_price, status)
        if closed:
            await self._db.close_trade(
                trade_id, exit_price, closed.pnl or 0.0, status
            )
            if self._on_close:
                self._on_close(closed)
        return closed

    # ── Internals ─────────────────────────────────────────────────────────────

    def _submit_order_sync(self, signal: Signal) -> object:
        """Blocking CLOB order submission (run in thread pool)."""
        from py_clob_client.clob_types import OrderArgs, OrderType  # type: ignore[import-untyped]

        token_id = (
            signal.token_id_yes if signal.side == "YES" else signal.token_id_no
        )
        price = signal.cex_prob  # place limit at our model's fair value

        args = OrderArgs(
            token_id=token_id,
            price=round(price, 4),
            size=round(signal.size_usdc / max(price, 0.01), 4),
            side="BUY",
        )
        return self._client.create_and_post_order(args)  # type: ignore[attr-defined]

    @staticmethod
    def _extract_order_id(response: object) -> str:
        try:
            if isinstance(response, dict):
                return (
                    response.get("orderID")
                    or response.get("order_id")
                    or response.get("id")
                    or "unknown"
                )
            return getattr(response, "order_id", "unknown")
        except Exception:  # noqa: BLE001
            return "unknown"
