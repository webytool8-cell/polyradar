"""
binance_feed.py — Real-time price feed from Binance via WebSocket.

Subscribes to combined kline streams for BTCUSDT and ETHUSDT at 5-minute
and 15-minute intervals.  Maintains a live CandleState per (symbol, timeframe)
and exposes a GBM-based probability model used by the arbitrage engine.

Reconnects automatically with exponential back-off on any disconnection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, Tuple

import websockets
from websockets.exceptions import ConnectionClosedError, WebSocketException
from scipy.stats import norm  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MINUTES_PER_YEAR = 365 * 24 * 60  # 525 600
_SYMBOLS = ("btcusdt", "ethusdt")
_TIMEFRAMES = ("5m", "15m")
_TIMEFRAME_MINUTES = {"5m": 5, "15m": 15}

# Build combined-stream path
_STREAMS = "/".join(
    f"{sym}@kline_{tf}" for sym in _SYMBOLS for tf in _TIMEFRAMES
)

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class CandleState:
    symbol: str               # BTCUSDT / ETHUSDT
    timeframe: str            # 5m / 15m
    open_price: float = 0.0
    current_price: float = 0.0
    high: float = 0.0
    low: float = 0.0
    open_time_ms: int = 0     # candle open  (epoch ms)
    close_time_ms: int = 0    # candle close (epoch ms)
    is_closed: bool = False   # True once the candle bar has finalised
    last_update_ms: int = 0

    @property
    def elapsed_minutes(self) -> float:
        """Minutes elapsed since candle open (wall-clock time)."""
        now_ms = int(time.time() * 1000)
        return max(0.0, (now_ms - self.open_time_ms) / 60_000)

    @property
    def total_minutes(self) -> float:
        return _TIMEFRAME_MINUTES.get(self.timeframe, 5)

    @property
    def remaining_minutes(self) -> float:
        return max(0.0, self.total_minutes - self.elapsed_minutes)

    @property
    def current_return(self) -> float:
        """Log-return since candle open."""
        if self.open_price <= 0:
            return 0.0
        return math.log(self.current_price / self.open_price)


# ── Probability model ─────────────────────────────────────────────────────────

def cex_probability(candle: CandleState, annual_vol: float) -> float:
    """
    Estimate P(price at candle close > candle open) given current price.

    Uses a Geometric Brownian Motion (GBM) model with zero drift (valid over
    short horizons):

        P(S_T > S_0 | S_t) = Φ( log(S_t / S_0) / σ_remaining )

    where σ_remaining = σ_annual / √(minutes_per_year) × √(remaining_minutes)

    Args:
        candle:     current CandleState snapshot
        annual_vol: annualised volatility for this asset (e.g. 0.80 for BTC)

    Returns:
        Probability in [0, 1].
    """
    remaining = candle.remaining_minutes

    # Candle hasn't started yet or bad data
    if candle.open_price <= 0 or candle.current_price <= 0:
        return 0.5

    # Candle already closed — return definitive outcome
    if remaining <= 0 or candle.is_closed:
        return 1.0 if candle.current_price >= candle.open_price else 0.0

    vol_per_minute = annual_vol / math.sqrt(MINUTES_PER_YEAR)
    sigma_remaining = vol_per_minute * math.sqrt(remaining)

    if sigma_remaining < 1e-10:
        return 1.0 if candle.current_return > 0 else 0.0

    z = candle.current_return / sigma_remaining
    return float(norm.cdf(z))


def model_confidence(p_cex: float) -> float:
    """
    Translate a raw probability into a confidence score.

    Confidence is how far the model is from 50/50:
        confidence = 2 × |p - 0.5|

    50 % → 0.0 (coin-flip, no edge)
    85 % → 0.70
    95 % → 0.90
    99 % → 0.98
    """
    return min(1.0, 2.0 * abs(p_cex - 0.5))


# ── Feed ──────────────────────────────────────────────────────────────────────

# Callback type: receives a (symbol, timeframe, CandleState) update
PriceCallback = Callable[[str, str, CandleState], None]


class BinanceFeed:
    """
    Maintains live CandleState for each (symbol × timeframe) pair and fires
    optional callbacks on every price update.

    Usage::

        feed = BinanceFeed(ws_base_url="wss://stream.binance.com:9443")
        asyncio.create_task(feed.run())
        # Later:
        candle = feed.get_candle("BTCUSDT", "5m")
    """

    def __init__(
        self,
        ws_base_url: str = "wss://stream.binance.com:9443",
        on_update: Optional[PriceCallback] = None,
    ) -> None:
        self._url = f"{ws_base_url}/stream?streams={_STREAMS}"
        self._on_update = on_update
        self._candles: Dict[Tuple[str, str], CandleState] = {}
        self._connected = asyncio.Event()
        self._stop = asyncio.Event()

        # Initialise empty states so callers never get KeyError
        for sym in ("BTCUSDT", "ETHUSDT"):
            for tf in _TIMEFRAMES:
                self._candles[(sym, tf)] = CandleState(symbol=sym, timeframe=tf)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_candle(self, symbol: str, timeframe: str) -> CandleState:
        """Return the latest CandleState for a symbol/timeframe pair."""
        return self._candles.get(
            (symbol.upper(), timeframe),
            CandleState(symbol=symbol.upper(), timeframe=timeframe),
        )

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    async def wait_until_connected(self, timeout: float = 30.0) -> bool:
        """Block until first message received or timeout."""
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def stop(self) -> None:
        self._stop.set()

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Connect, receive messages, and reconnect forever until stop()."""
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._connect_and_receive()
                backoff = 1.0  # successful session — reset backoff
            except asyncio.CancelledError:
                break
            except (ConnectionClosedError, WebSocketException, OSError) as exc:
                self._connected.clear()
                logger.warning(
                    "Binance WS disconnected (%s). Reconnecting in %.0fs…",
                    exc, backoff,
                )
            except Exception as exc:  # noqa: BLE001
                self._connected.clear()
                logger.error(
                    "Unexpected Binance WS error (%s). Reconnecting in %.0fs…",
                    exc, backoff,
                )
            if not self._stop.is_set():
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _connect_and_receive(self) -> None:
        logger.info("Connecting to Binance WebSocket: %s", self._url)
        async with websockets.connect(  # type: ignore[attr-defined]
            self._url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            logger.info("Binance WebSocket connected.")
            async for raw in ws:
                if self._stop.is_set():
                    break
                try:
                    self._handle_message(raw)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to parse Binance message: %s", exc)

    def _handle_message(self, raw: str) -> None:
        msg = json.loads(raw)
        # Combined stream wraps payload under "data"
        data = msg.get("data", msg)
        if data.get("e") != "kline":
            return

        k = data["k"]
        symbol: str = k["s"].upper()       # e.g. "BTCUSDT"
        tf: str = k["i"]                   # e.g. "5m"

        if (symbol, tf) not in self._candles:
            return  # not a pair we track

        candle = self._candles[(symbol, tf)]
        candle.open_price     = float(k["o"])
        candle.current_price  = float(k["c"])
        candle.high           = float(k["h"])
        candle.low            = float(k["l"])
        candle.open_time_ms   = int(k["t"])
        candle.close_time_ms  = int(k["T"])
        candle.is_closed      = bool(k["x"])
        candle.last_update_ms = int(time.time() * 1000)

        self._connected.set()

        if self._on_update:
            try:
                self._on_update(symbol, tf, candle)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Price callback raised: %s", exc)

        logger.debug(
            "%s %s  open=%.2f  cur=%.2f  remaining=%.1fm",
            symbol, tf, candle.open_price, candle.current_price,
            candle.remaining_minutes,
        )
