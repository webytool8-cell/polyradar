"""
polymarket_feed.py — Polymarket CLOB feed.

Polls the CLOB API every `poll_interval` seconds for each tracked market and
extracts the current YES implied-probability from the best bid/ask.

Market discovery
----------------
If condition IDs are not supplied in the config the feed will query the CLOB
`get_markets()` endpoint and filter by known title patterns for BTC/ETH 5 min
and 15 min up/down markets.  Matched IDs are written to stdout so you can
persist them in your .env.

Reconnection / error handling
------------------------------
Each poll cycle is wrapped in a try/except.  Persistent errors are logged and
the cycle is retried after an exponential back-off capped at 60 s.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Market metadata ────────────────────────────────────────────────────────────

# Patterns used to auto-discover markets if condition IDs are not configured.
# Keys are (symbol, timeframe) tuples; values are substrings to match in
# the market question/description (case-insensitive).
_DISCOVERY_PATTERNS: Dict[Tuple[str, str], List[str]] = {
    ("BTCUSDT", "5m"):  ["btc", "bitcoin", "5 min", "5min", "5-min"],
    ("BTCUSDT", "15m"): ["btc", "bitcoin", "15 min", "15min", "15-min"],
    ("ETHUSDT", "5m"):  ["eth", "ethereum", "5 min", "5min", "5-min"],
    ("ETHUSDT", "15m"): ["eth", "ethereum", "15 min", "15min", "15-min"],
}


@dataclass
class MarketInfo:
    """Snapshot of a Polymarket market's current state."""
    market_id: str          # condition ID (hex string)
    token_id_yes: str       # ERC-1155 token ID for YES outcome
    token_id_no: str        # ERC-1155 token ID for NO outcome
    symbol: str             # BTCUSDT / ETHUSDT
    timeframe: str          # 5m / 15m
    question: str           # full market question text
    yes_price: float = 0.50 # best-ask price for YES (≈ implied probability)
    no_price: float  = 0.50
    yes_bid: float   = 0.0
    yes_ask: float   = 0.50
    spread: float    = 0.0
    last_update: Optional[datetime] = field(default=None)
    active: bool = True


# ── Feed class ────────────────────────────────────────────────────────────────

class PolymarketFeed:
    """
    Polls Polymarket CLOB for YES/NO prices on configured markets.

    Usage::

        feed = PolymarketFeed(config)
        asyncio.create_task(feed.run())
        # Later:
        info = feed.get_market("BTCUSDT", "5m")
        yes_prob = info.yes_price
    """

    def __init__(self, config: "Config") -> None:  # noqa: F821
        self._cfg = config
        self._markets: Dict[Tuple[str, str], MarketInfo] = {}
        self._client: Optional[object] = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._backoff = 1.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_market(self, symbol: str, timeframe: str) -> Optional[MarketInfo]:
        return self._markets.get((symbol.upper(), timeframe))

    @property
    def markets(self) -> Dict[Tuple[str, str], MarketInfo]:
        return dict(self._markets)

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    async def wait_until_ready(self, timeout: float = 60.0) -> bool:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def stop(self) -> None:
        self._stop.set()

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Initialise client, discover markets, then poll continuously."""
        await self._init_client()
        await self._discover_or_load_markets()

        while not self._stop.is_set():
            try:
                await self._poll_prices()
                self._ready.set()
                self._backoff = 1.0
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("Polymarket poll error: %s. Retry in %.0fs…", exc, self._backoff)
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, 60.0)
                continue

            await asyncio.sleep(self._cfg.poly_poll_interval)

    # ── Initialisation ────────────────────────────────────────────────────────

    async def _init_client(self) -> None:
        """Instantiate py-clob-client ClobClient (runs in thread pool)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._create_client_sync)

    def _create_client_sync(self) -> None:
        try:
            from py_clob_client.client import ClobClient  # type: ignore[import-untyped]
            from py_clob_client.clob_types import ApiCreds  # type: ignore[import-untyped]

            if self._cfg.poly_api_key:
                creds = ApiCreds(
                    api_key=self._cfg.poly_api_key,
                    api_secret=self._cfg.poly_api_secret,
                    api_passphrase=self._cfg.poly_passphrase,
                )
                self._client = ClobClient(
                    host=self._cfg.poly_host,
                    creds=creds,
                )
            else:
                # Public read-only client (no trading)
                self._client = ClobClient(host=self._cfg.poly_host)
            logger.info("Polymarket CLOB client initialised.")
        except ImportError:
            logger.warning(
                "py-clob-client not installed — running in simulated feed mode."
            )
            self._client = None

    async def _discover_or_load_markets(self) -> None:
        """Populate self._markets from config IDs or via API discovery."""
        cfg = self._cfg
        configured = {
            ("BTCUSDT", "5m"):  cfg.btc_5m_condition_id,
            ("BTCUSDT", "15m"): cfg.btc_15m_condition_id,
            ("ETHUSDT", "5m"):  cfg.eth_5m_condition_id,
            ("ETHUSDT", "15m"): cfg.eth_15m_condition_id,
        }

        missing = [k for k, v in configured.items() if not v]

        if missing and self._client is not None:
            logger.info(
                "Auto-discovering Polymarket markets for: %s",
                [f"{s}/{tf}" for s, tf in missing],
            )
            await self._discover_markets(missing, configured)
        elif missing:
            logger.warning(
                "No condition IDs configured and client unavailable. "
                "Using simulated market data."
            )
            self._init_simulated_markets()
            return

        for (symbol, tf), condition_id in configured.items():
            if not condition_id:
                continue
            if (symbol, tf) not in self._markets:
                self._markets[(symbol, tf)] = MarketInfo(
                    market_id=condition_id,
                    token_id_yes="",
                    token_id_no="",
                    symbol=symbol,
                    timeframe=tf,
                    question=f"{symbol} {tf} up/down",
                )

    async def _discover_markets(
        self,
        missing: List[Tuple[str, str]],
        configured: Dict[Tuple[str, str], str],
    ) -> None:
        loop = asyncio.get_running_loop()
        try:
            markets_raw = await loop.run_in_executor(
                None, self._fetch_all_markets_sync
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Market discovery failed: %s", exc)
            self._init_simulated_markets()
            return

        found: Dict[Tuple[str, str], dict] = {}
        for m in markets_raw:
            question: str = (m.get("question") or m.get("description") or "").lower()
            for (sym, tf), patterns in _DISCOVERY_PATTERNS.items():
                if (sym, tf) in found:
                    continue
                if configured.get((sym, tf)):
                    continue  # already configured
                if all(p.lower() in question for p in patterns):
                    found[(sym, tf)] = m
                    logger.info(
                        "Discovered %s/%s market: %s  id=%s",
                        sym, tf, question[:60], m.get("condition_id", "?"),
                    )

        for (sym, tf), m in found.items():
            tokens = m.get("tokens", [])
            yes_token = next((t for t in tokens if t.get("outcome") == "Yes"), {})
            no_token  = next((t for t in tokens if t.get("outcome") == "No"),  {})
            self._markets[(sym, tf)] = MarketInfo(
                market_id=m.get("condition_id", ""),
                token_id_yes=yes_token.get("token_id", ""),
                token_id_no=no_token.get("token_id", ""),
                symbol=sym,
                timeframe=tf,
                question=m.get("question", ""),
            )
            # Print for user to persist in .env
            print(
                f"[DISCOVERY] {sym}/{tf} → condition_id={m.get('condition_id', '')}"
            )

        # Any still-missing markets fall back to simulation
        for pair in missing:
            if pair not in self._markets and pair not in found:
                logger.warning("Could not discover market for %s/%s.", *pair)
                sym, tf = pair
                self._markets[(sym, tf)] = MarketInfo(
                    market_id="SIMULATED",
                    token_id_yes="",
                    token_id_no="",
                    symbol=sym,
                    timeframe=tf,
                    question=f"[simulated] {sym} {tf} up/down",
                )

    def _fetch_all_markets_sync(self) -> List[dict]:
        """Synchronous CLOB call to list markets (run in executor)."""
        assert self._client is not None
        result = self._client.get_markets()  # type: ignore[attr-defined]
        if isinstance(result, dict):
            return result.get("data", [])
        return list(result) if result else []

    def _init_simulated_markets(self) -> None:
        for sym in ("BTCUSDT", "ETHUSDT"):
            for tf in ("5m", "15m"):
                self._markets[(sym, tf)] = MarketInfo(
                    market_id="SIMULATED",
                    token_id_yes="",
                    token_id_no="",
                    symbol=sym,
                    timeframe=tf,
                    question=f"[simulated] {sym} {tf} up/down",
                )

    # ── Price polling ─────────────────────────────────────────────────────────

    async def _poll_prices(self) -> None:
        loop = asyncio.get_running_loop()
        for (sym, tf), info in list(self._markets.items()):
            if info.market_id == "SIMULATED":
                # Leave default 0.50 prices — arbitrage engine will ignore these
                info.last_update = datetime.now(timezone.utc)
                continue
            try:
                prices = await loop.run_in_executor(
                    None, self._fetch_orderbook_sync, info
                )
                info.yes_bid, info.yes_ask = prices
                # Mid-price as implied probability
                info.yes_price = (info.yes_bid + info.yes_ask) / 2.0
                info.no_price  = 1.0 - info.yes_price
                info.spread    = max(0.0, info.yes_ask - info.yes_bid)
                info.last_update = datetime.now(timezone.utc)
                logger.debug(
                    "%s/%s  YES bid=%.3f ask=%.3f mid=%.3f spread=%.4f",
                    sym, tf,
                    info.yes_bid, info.yes_ask, info.yes_price, info.spread,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to poll %s/%s orderbook: %s", sym, tf, exc)

    def _fetch_orderbook_sync(self, info: MarketInfo) -> Tuple[float, float]:
        """
        Fetch best bid/ask for the YES token.

        Returns (best_bid, best_ask) as floats in [0, 1].
        Falls back to (0.49, 0.51) on any error.
        """
        if self._client is None or not info.token_id_yes:
            return 0.49, 0.51

        try:
            book = self._client.get_order_book(info.token_id_yes)  # type: ignore[attr-defined]
            bids = book.bids or []
            asks = book.asks or []
            best_bid = float(bids[0].price)  if bids else 0.0
            best_ask = float(asks[-1].price) if asks else 1.0
            # Clamp to sensible range
            best_bid = max(0.01, min(0.99, best_bid))
            best_ask = max(0.01, min(0.99, best_ask))
            return best_bid, best_ask
        except (IndexError, AttributeError, TypeError, ValueError):
            return 0.49, 0.51
