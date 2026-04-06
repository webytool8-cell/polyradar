"""
arbitrage.py — Signal detection and position-sizing engine.

Pipeline
--------
1. For every (symbol, timeframe) pair, read current CandleState from Binance
   and the latest MarketInfo from Polymarket.
2. Compute the CEX-derived probability using the GBM model in binance_feed.py.
3. Compare with Polymarket's implied probability.
4. If the lag exceeds the configured threshold, construct a Signal.
5. Validate the signal against edge, confidence, and position constraints.
6. Size the position using half-Kelly (capped at max_position_pct of equity).

Half-Kelly for binary prediction markets
-----------------------------------------
For a contract where:
    p_cex   = true probability (our estimate)
    p_poly  = market-implied probability (cost per YES contract, in [0,1])
    payout  = 1.00 per contract

Kelly fraction of bankroll:
    f* = (p_cex - p_poly) / (1 - p_poly)

Half-Kelly: f = kelly_fraction × f*  (default kelly_fraction = 0.5)
Position in USDC: size = f × equity, capped at max_position_pct × equity.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from binance_feed import BinanceFeed, CandleState, cex_probability, model_confidence
from polymarket_feed import MarketInfo, PolymarketFeed

logger = logging.getLogger(__name__)

# ── Signal dataclass ──────────────────────────────────────────────────────────

@dataclass
class Signal:
    timestamp: datetime
    symbol: str           # BTCUSDT / ETHUSDT
    timeframe: str        # 5m / 15m
    market_id: str
    token_id_yes: str
    token_id_no: str

    # Probabilities
    cex_prob: float       # GBM-derived P(up at close)
    poly_prob: float      # Polymarket YES mid-price (implied prob)
    lag_pct: float        # abs(cex_prob - poly_prob)  — in probability points

    # Trade parameters
    side: str             # 'YES' or 'NO'
    edge_pct: float       # (cex_prob - cost) / cost  — fraction
    confidence: float     # 2 × |cex_prob - 0.5|
    kelly_raw: float      # raw Kelly fraction before cap/scaling
    kelly_sized: float    # after half-Kelly and max-position cap
    size_usdc: float      # dollar amount to trade

    # Candle context
    candle_open: float    = 0.0
    current_price: float  = 0.0
    elapsed_minutes: float = 0.0
    remaining_minutes: float = 0.0

    # Validation gate result
    valid: bool = False
    skip_reason: Optional[str] = None


# ── Engine ────────────────────────────────────────────────────────────────────

class ArbitrageEngine:
    """
    Scans all monitored (symbol, timeframe) pairs and returns validated
    Signals ready for execution.
    """

    def __init__(
        self,
        config: "Config",           # noqa: F821
        binance_feed: BinanceFeed,
        poly_feed: PolymarketFeed,
        equity_fn: "Callable[[], float]",  # noqa: F821 — returns current portfolio equity
    ) -> None:
        self._cfg = config
        self._binance = binance_feed
        self._poly = poly_feed
        self._equity_fn = equity_fn
        # Track open position count per market to avoid doubling up
        self._open_position_ids: "set[str]" = set()

    # ── Public API ─────────────────────────────────────────────────────────────

    def scan(self) -> List[Signal]:
        """
        Synchronous scan — call from asyncio loop, returns list of Signal objects.
        Returns all signals (valid and invalid) for logging/dashboard purposes.
        """
        signals: List[Signal] = []
        pairs = [
            ("BTCUSDT", "5m"),
            ("BTCUSDT", "15m"),
            ("ETHUSDT", "5m"),
            ("ETHUSDT", "15m"),
        ]
        for symbol, tf in pairs:
            sig = self._evaluate(symbol, tf)
            if sig is not None:
                signals.append(sig)
        return signals

    def mark_position_open(self, market_id: str) -> None:
        self._open_position_ids.add(market_id)

    def mark_position_closed(self, market_id: str) -> None:
        self._open_position_ids.discard(market_id)

    # ── Core evaluation ───────────────────────────────────────────────────────

    def _evaluate(self, symbol: str, tf: str) -> Optional[Signal]:
        candle = self._binance.get_candle(symbol, tf)
        market = self._poly.get_market(symbol, tf)

        if market is None or not market.active:
            return None

        if candle.open_price <= 0 or candle.current_price <= 0:
            logger.debug("%s/%s — no candle data yet, skipping.", symbol, tf)
            return None

        annual_vol = self._cfg.annual_vol_for(symbol)
        p_cex = cex_probability(candle, annual_vol)
        p_poly = market.yes_price   # implied probability from orderbook mid

        # Determine which side we'd trade
        if p_cex > p_poly:
            side = "YES"
            cost = market.yes_ask   # we pay the ask to buy YES
            lag  = p_cex - p_poly
        else:
            # We think NO is more likely — buy NO (equivalent to p_no_cex > p_no_poly)
            side = "NO"
            cost = market.no_price  # approximate; use 1 - yes_bid
            p_cex_effective = 1.0 - p_cex
            p_poly_effective = 1.0 - p_poly
            lag  = p_cex_effective - p_poly_effective
            # Shadow variables for sizing
            p_cex  = p_cex_effective
            p_poly = p_poly_effective

        confidence = model_confidence(p_cex)
        edge_pct   = self._edge(p_cex, cost)
        kelly_raw, kelly_sized, size_usdc = self._size_position(p_cex, cost)

        sig = Signal(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            timeframe=tf,
            market_id=market.market_id,
            token_id_yes=market.token_id_yes,
            token_id_no=market.token_id_no,
            cex_prob=p_cex,
            poly_prob=p_poly,
            lag_pct=lag,
            side=side,
            edge_pct=edge_pct,
            confidence=confidence,
            kelly_raw=kelly_raw,
            kelly_sized=kelly_sized,
            size_usdc=size_usdc,
            candle_open=candle.open_price,
            current_price=candle.current_price,
            elapsed_minutes=candle.elapsed_minutes,
            remaining_minutes=candle.remaining_minutes,
        )

        sig.valid, sig.skip_reason = self._validate(sig, market)
        return sig

    # ── Sizing ────────────────────────────────────────────────────────────────

    def _edge(self, p_true: float, cost: float) -> float:
        """Edge as a fraction of cost: (p_true - cost) / cost."""
        if cost <= 0:
            return 0.0
        return (p_true - cost) / cost

    def _size_position(
        self, p_true: float, cost: float
    ) -> Tuple[float, float, float]:
        """
        Returns (kelly_raw, kelly_sized, size_usdc).

        Kelly fraction of equity for a binary prediction market:
            f* = (p_true - cost) / (1 - cost)

        After half-Kelly and max_position_pct cap:
            f = min(kelly_fraction * f*, max_position_pct)
        """
        equity = self._equity_fn()
        if equity <= 0 or cost <= 0 or cost >= 1:
            return 0.0, 0.0, 0.0

        payout_per_dollar = (1.0 - cost) / cost  # net odds
        numerator  = p_true * (payout_per_dollar + 1) - 1
        denominator = payout_per_dollar
        kelly_raw  = max(0.0, numerator / denominator) if denominator > 0 else 0.0

        kelly_sized = min(
            self._cfg.kelly_fraction * kelly_raw,
            self._cfg.max_position_pct,
        )
        size_usdc = kelly_sized * equity
        return kelly_raw, kelly_sized, size_usdc

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(
        self, sig: Signal, market: MarketInfo
    ) -> Tuple[bool, Optional[str]]:
        """Gate check — returns (valid, skip_reason)."""
        cfg = self._cfg

        # 1 — lag threshold (informational flag, not a block by itself)
        if sig.lag_pct < cfg.lag_threshold:
            return False, f"lag {sig.lag_pct:.1%} < threshold {cfg.lag_threshold:.1%}"

        # 2 — minimum edge
        if sig.edge_pct < cfg.min_edge:
            return False, f"edge {sig.edge_pct:.1%} < min_edge {cfg.min_edge:.1%}"

        # 3 — minimum confidence
        if sig.confidence < cfg.min_confidence:
            return False, (
                f"confidence {sig.confidence:.1%} < min_confidence {cfg.min_confidence:.1%}"
            )

        # 4 — position size must be > $1 to be worth executing
        if sig.size_usdc < 1.0:
            return False, f"size ${sig.size_usdc:.2f} too small"

        # 5 — no existing open position in this market
        if sig.market_id in self._open_position_ids:
            return False, "already have an open position in this market"

        # 6 — simulated market (no IDs to trade against)
        if market.market_id == "SIMULATED":
            return False, "market is simulated, no real trading possible"

        # 7 — don't trade near candle boundaries (< 30 s remaining)
        if sig.remaining_minutes < 0.5:
            return False, "too close to candle close"

        return True, None
