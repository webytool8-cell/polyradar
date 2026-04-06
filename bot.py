"""
bot.py — PolyRadar: Polymarket latency arbitrage bot.

Usage
-----
Paper trading (default):
    python bot.py

Discover available markets:
    python bot.py --discover-markets

Go live (requires ALL THREE flags — safety gate):
    python bot.py --live --confirm-live --i-accept-financial-risk

Override scan interval:
    python bot.py --scan-interval 3

The bot will:
  1. Connect to Binance WebSocket for real-time BTC/ETH prices.
  2. Poll Polymarket CLOB for YES/NO prices on 5-min and 15-min markets.
  3. Run the arbitrage engine every `scan_interval` seconds.
  4. Flag signals where Poly odds lag CEX by > 3 pp.
  5. Execute trades that clear all gates (edge > 5%, conf > 85%, pos < 8%).
  6. Persist every trade and signal to SQLite.
  7. Send Telegram alerts on every trade open/close and drawdown threshold.
  8. Halt trading if daily drawdown exceeds 20% (kill switch).
  9. Display a live terminal dashboard (Rich).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Optional

from config import Config
from db import Database
from binance_feed import BinanceFeed
from polymarket_feed import PolymarketFeed
from arbitrage import ArbitrageEngine, Signal
from executor import PaperExecutor, LiveExecutor, Portfolio, Position
from telegram_alerts import TelegramAlerter
from dashboard import Dashboard

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler("polyradar.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("polyradar")

# ── Drawdown alert levels ─────────────────────────────────────────────────────

_DD_WARN_LEVELS = (0.10, 0.15)   # warn at 10% and 15%; kill at 20%

# ── Bot ───────────────────────────────────────────────────────────────────────

class PolyRadar:
    """
    Orchestrates all sub-systems and runs the main event loop.
    """

    def __init__(self, config: Config, scan_interval: float = 2.0) -> None:
        self.cfg           = config
        self.scan_interval = scan_interval

        # Sub-systems (initialised in setup())
        self.db:         Optional[Database]        = None
        self.binance:    Optional[BinanceFeed]     = None
        self.poly:       Optional[PolymarketFeed]  = None
        self.portfolio:  Optional[Portfolio]       = None
        self.engine:     Optional[ArbitrageEngine] = None
        self.executor:   Optional[PaperExecutor | LiveExecutor] = None
        self.telegram:   Optional[TelegramAlerter] = None
        self.dashboard:  Optional[Dashboard]       = None

        self._stop         = asyncio.Event()
        self._kill_switch  = asyncio.Event()

        # Track drawdown alert levels already sent
        self._dd_alerted: set[float] = set()

        # Background tasks
        self._tasks: list[asyncio.Task] = []  # type: ignore[type-arg]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Instantiate and wire up all sub-systems."""
        cfg = self.cfg
        logger.info("Setting up PolyRadar (mode=%s)…", "LIVE" if cfg.live_trading else "PAPER")

        # Database
        self.db = Database(cfg.db_path)
        await self.db.connect()

        # Feeds
        self.binance = BinanceFeed(ws_base_url=cfg.binance_ws_url)
        self.poly    = PolymarketFeed(cfg)

        # Portfolio
        self.portfolio = Portfolio(
            initial_balance=cfg.initial_balance,
            paper=not cfg.live_trading,
        )

        # Arbitrage engine
        self.engine = ArbitrageEngine(
            config=cfg,
            binance_feed=self.binance,
            poly_feed=self.poly,
            equity_fn=lambda: self.portfolio.equity,
        )

        # Executor
        if cfg.live_trading:
            clob_client = self._build_clob_client()
            self.executor = LiveExecutor(
                portfolio=self.portfolio,
                db=self.db,
                clob_client=clob_client,
                on_close=self._on_position_close,
            )
        else:
            self.executor = PaperExecutor(
                portfolio=self.portfolio,
                db=self.db,
                on_close=self._on_position_close,
            )

        # Telegram
        self.telegram = TelegramAlerter(cfg.telegram_token, cfg.telegram_chat_id)
        await self.telegram.start()

        # Dashboard
        self.dashboard = Dashboard(
            portfolio=self.portfolio,
            db=self.db,
            config=cfg,
            binance_feed=self.binance,
            poly_feed=self.poly,
        )
        self.dashboard.session_start_equity = self.portfolio.equity

    async def run(self) -> None:
        """Start all background tasks and block until stop is requested."""
        assert self.db and self.binance and self.poly and self.dashboard

        # Start feeds
        self._tasks.append(asyncio.create_task(self.binance.run(), name="binance-feed"))
        self._tasks.append(asyncio.create_task(self.poly.run(), name="poly-feed"))

        # Wait for initial data
        logger.info("Waiting for Binance WebSocket connection…")
        if not await self.binance.wait_until_connected(timeout=30):
            logger.error("Could not connect to Binance WebSocket within 30s. Exiting.")
            return

        logger.info("Waiting for Polymarket feed to be ready…")
        if not await self.poly.wait_until_ready(timeout=60):
            logger.warning("Polymarket feed not ready within 60s — continuing anyway.")

        # Start dashboard
        await self.dashboard.start()

        # Telegram startup alert
        mode = "LIVE" if self.cfg.live_trading else "PAPER"
        self.telegram.send_startup(mode, self.portfolio.equity)

        # Start snapshot loop
        self._tasks.append(
            asyncio.create_task(self._snapshot_loop(), name="snapshot-loop")
        )

        # Main scan loop
        logger.info("PolyRadar running.  Scan interval: %.1fs", self.scan_interval)
        try:
            await self._scan_loop()
        except asyncio.CancelledError:
            pass
        finally:
            await self._teardown()

    async def stop(self) -> None:
        self._stop.set()

    # ── Scan loop ─────────────────────────────────────────────────────────────

    async def _scan_loop(self) -> None:
        assert self.engine and self.executor and self.db and self.telegram

        while not self._stop.is_set():
            if not self._kill_switch.is_set():
                try:
                    await self._scan_once()
                except Exception as exc:  # noqa: BLE001
                    logger.error("Scan error: %s", exc, exc_info=True)

                await self._check_kill_switch()
            await asyncio.sleep(self.scan_interval)

    async def _scan_once(self) -> None:
        assert self.engine and self.executor and self.db and self.telegram

        signals = self.engine.scan()

        for sig in signals:
            # Always log the signal
            await self.db.insert_signal(
                market_id=sig.market_id,
                symbol=sig.symbol,
                timeframe=sig.timeframe,
                cex_prob=sig.cex_prob,
                poly_prob=sig.poly_prob,
                lag_pct=sig.lag_pct,
                edge_pct=sig.edge_pct,
                confidence=sig.confidence,
                acted=False,
                skip_reason=sig.skip_reason,
            )

            # Flag detected lags to Telegram (even if not traded)
            if sig.lag_pct >= self.cfg.lag_threshold:
                self.telegram.send_signal_flagged(
                    symbol=sig.symbol,
                    timeframe=sig.timeframe,
                    cex_prob=sig.cex_prob,
                    poly_prob=sig.poly_prob,
                    lag_pct=sig.lag_pct,
                    edge_pct=sig.edge_pct,
                )

            if not sig.valid:
                logger.debug(
                    "Signal %s/%s skipped: %s", sig.symbol, sig.timeframe, sig.skip_reason
                )
                continue

            # Place trade
            pos = await self.executor.execute(sig)
            if pos is None:
                logger.warning("Executor returned None for signal %s/%s.", sig.symbol, sig.timeframe)
                continue

            # Update DB signal as acted
            await self.db.insert_signal(
                market_id=sig.market_id,
                symbol=sig.symbol,
                timeframe=sig.timeframe,
                cex_prob=sig.cex_prob,
                poly_prob=sig.poly_prob,
                lag_pct=sig.lag_pct,
                edge_pct=sig.edge_pct,
                confidence=sig.confidence,
                acted=True,
            )

            # Mark position open in engine (prevent doubling up)
            self.engine.mark_position_open(sig.market_id)

            # Telegram alert
            self.telegram.send_trade_open(
                symbol=sig.symbol,
                timeframe=sig.timeframe,
                side=sig.side,
                size_usdc=sig.size_usdc,
                entry_price=sig.cex_prob,
                edge_pct=sig.edge_pct,
                confidence=sig.confidence,
                paper=self.portfolio.paper,
            )

            logger.info(
                "TRADE OPEN | %s/%s | %s | $%.2f | edge=%.1f%% | conf=%.1f%%",
                sig.symbol, sig.timeframe, sig.side,
                sig.size_usdc, sig.edge_pct * 100, sig.confidence * 100,
            )

            # Schedule resolution check (paper: watch for candle close)
            if isinstance(self.executor, PaperExecutor):
                asyncio.create_task(
                    self._paper_resolution_watcher(pos, sig),
                    name=f"resolve-{pos.trade_id}",
                )

    # ── Paper resolution ──────────────────────────────────────────────────────

    async def _paper_resolution_watcher(
        self, pos: Position, sig: Signal
    ) -> None:
        """
        Wait until the relevant candle closes, then resolve the paper trade.
        Polls every 2 seconds; times out after 2× the candle period + 30s.
        """
        assert isinstance(self.executor, PaperExecutor)
        assert self.telegram

        timeframe_minutes = {"5m": 5, "15m": 15}[sig.timeframe]
        timeout_seconds   = timeframe_minutes * 60 * 2 + 30

        deadline = asyncio.get_event_loop().time() + timeout_seconds
        candle_open_price = sig.candle_open

        while asyncio.get_event_loop().time() < deadline:
            if self._stop.is_set():
                return
            await asyncio.sleep(2.0)

            candle = self.binance.get_candle(sig.symbol, sig.timeframe)

            # A new candle has opened (open price changed) → previous closed
            if candle.open_price != candle_open_price and candle.open_price > 0:
                final_price = candle_open_price  # we don't have the exact close; approximate
                # Actually check: if the current candle's open is different, the
                # previous candle closed.  Use the closing price implied by the
                # side of the position to be conservative.
                # Fetch the final close from the feed's last update of the old candle.
                # In practice this heuristic works well for paper sim.
                closed = await self.executor.resolve_candle(
                    trade_id=pos.trade_id,
                    candle_final_price=candle.open_price,   # prev close ≈ current open
                    candle_open_price=candle_open_price,
                )
                if closed:
                    assert self.engine
                    self.engine.mark_position_closed(sig.market_id)
                    self.telegram.send_trade_close(
                        symbol=sig.symbol,
                        timeframe=sig.timeframe,
                        side=sig.side,
                        pnl=closed.pnl or 0.0,
                        status=closed.status,
                        paper=True,
                    )
                return

            # Also check if the `is_closed` flag flipped
            if candle.is_closed and candle.open_time_ms > 0:
                closed = await self.executor.resolve_candle(
                    trade_id=pos.trade_id,
                    candle_final_price=candle.current_price,
                    candle_open_price=candle_open_price,
                )
                if closed:
                    assert self.engine
                    self.engine.mark_position_closed(sig.market_id)
                    self.telegram.send_trade_close(
                        symbol=sig.symbol,
                        timeframe=sig.timeframe,
                        side=sig.side,
                        pnl=closed.pnl or 0.0,
                        status=closed.status,
                        paper=True,
                    )
                return

        # Timeout — cancel the position
        logger.warning("Resolution timeout for trade %d — cancelling.", pos.trade_id)
        await self.executor.resolve_candle(
            trade_id=pos.trade_id,
            candle_final_price=sig.candle_open,   # treat as flat → LOST
            candle_open_price=sig.candle_open,
        )
        if self.engine:
            self.engine.mark_position_closed(sig.market_id)

    # ── Kill switch ───────────────────────────────────────────────────────────

    async def _check_kill_switch(self) -> None:
        assert self.portfolio and self.telegram and self.dashboard

        dd = self.portfolio.daily_drawdown
        eq = self.portfolio.equity

        # Progressive warnings
        for level in _DD_WARN_LEVELS:
            if dd >= level and level not in self._dd_alerted:
                self._dd_alerted.add(level)
                logger.warning("Drawdown warning: %.1f%%", dd * 100)
                self.telegram.send_drawdown_alert(dd, eq, self.cfg.max_daily_drawdown)

        # Hard kill switch
        if dd >= self.cfg.max_daily_drawdown and not self._kill_switch.is_set():
            self._kill_switch.set()
            self.dashboard.kill_switch_active = True
            logger.critical(
                "KILL SWITCH ACTIVATED — daily drawdown %.1f%% exceeds limit %.1f%%.",
                dd * 100, self.cfg.max_daily_drawdown * 100,
            )
            self.telegram.send_kill_switch(dd, eq)

    # ── Snapshot loop ─────────────────────────────────────────────────────────

    async def _snapshot_loop(self) -> None:
        """Persist a portfolio snapshot to DB every 60 seconds."""
        assert self.portfolio and self.db

        while not self._stop.is_set():
            await asyncio.sleep(60)
            pf = self.portfolio
            try:
                await self.db.insert_snapshot(
                    balance=pf.balance,
                    open_positions=pf.open_position_cost,
                    realized_pnl=pf.realized_pnl,
                    unrealized_pnl=pf.unrealized_pnl,
                    equity=pf.equity,
                    daily_drawdown=pf.daily_drawdown,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Snapshot write failed: %s", exc)

    # ── Position close callback ───────────────────────────────────────────────

    def _on_position_close(self, pos: Position) -> None:
        """Called by executor after any position is closed."""
        pnl = pos.pnl or 0.0
        logger.info(
            "Position closed: %s/%s  %s  pnl=$%.2f  status=%s",
            pos.symbol, pos.timeframe, pos.side, pnl, pos.status,
        )

    # ── Teardown ──────────────────────────────────────────────────────────────

    async def _teardown(self) -> None:
        logger.info("Shutting down PolyRadar…")

        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        if self.binance:
            self.binance.stop()
        if self.poly:
            self.poly.stop()
        if self.dashboard:
            await self.dashboard.stop()
        if self.telegram and self.portfolio:
            self.telegram.send_shutdown(
                self.portfolio.equity, self.portfolio.realized_pnl
            )
            await self.telegram.stop()
        if self.db:
            await self.db.close()

        logger.info("PolyRadar stopped.")

    # ── CLOB client factory ───────────────────────────────────────────────────

    def _build_clob_client(self) -> object:
        from py_clob_client.client import ClobClient  # type: ignore[import-untyped]
        from py_clob_client.clob_types import ApiCreds  # type: ignore[import-untyped]

        creds = ApiCreds(
            api_key=self.cfg.poly_api_key,
            api_secret=self.cfg.poly_api_secret,
            api_passphrase=self.cfg.poly_passphrase,
        )
        return ClobClient(host=self.cfg.poly_host, creds=creds)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PolyRadar — Polymarket latency arbitrage bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bot.py                                    # paper trading
  python bot.py --scan-interval 5                  # slower scan
  python bot.py --discover-markets                 # list available markets
  python bot.py \\
    --live --confirm-live --i-accept-financial-risk # GO LIVE (all 3 flags required)
""",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="First of three required flags to enable live trading.",
    )
    p.add_argument(
        "--confirm-live",
        action="store_true",
        dest="confirm_live",
        help="Second of three required flags to enable live trading.",
    )
    p.add_argument(
        "--i-accept-financial-risk",
        action="store_true",
        dest="accept_risk",
        help="Third of three required flags to enable live trading.",
    )
    p.add_argument(
        "--scan-interval",
        type=float,
        default=2.0,
        dest="scan_interval",
        metavar="SECONDS",
        help="Seconds between arbitrage scans (default: 2.0).",
    )
    p.add_argument(
        "--discover-markets",
        action="store_true",
        dest="discover",
        help="Query Polymarket CLOB for matching markets and exit.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="log_level",
        help="Logging verbosity (default: INFO).",
    )
    return p.parse_args()


# ── Market discovery helper ───────────────────────────────────────────────────

async def discover_markets(cfg: Config) -> None:
    """Standalone mode: query CLOB and print matching markets."""
    print("Discovering Polymarket markets…\n")
    feed = PolymarketFeed(cfg)
    await feed._init_client()
    await feed._discover_or_load_markets()
    print("\nDiscovered markets:")
    for (sym, tf), m in feed.markets.items():
        print(f"  {sym}/{tf}  id={m.market_id}  q={m.question[:60]}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    args = parse_args()

    # Adjust log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    cfg = Config.from_env()
    cfg.apply_live_flags(
        live=args.live,
        confirm_live=args.confirm_live,
        accept_risk=args.accept_risk,
    )

    # Safety banner
    if cfg.live_trading:
        print("\n" + "=" * 60)
        print("  ⚠️   LIVE TRADING MODE ENABLED")
        print("  Real USDC will be used.  Kill switch at 20% drawdown.")
        print("=" * 60 + "\n")
        logger.warning("LIVE TRADING MODE — all three flags confirmed.")
    elif args.live or args.confirm_live or args.accept_risk:
        missing = []
        if not args.live:          missing.append("--live")
        if not args.confirm_live:  missing.append("--confirm-live")
        if not args.accept_risk:   missing.append("--i-accept-financial-risk")
        print(f"\n⚠️  Live trading requires ALL THREE flags.  Missing: {missing}")
        print("   Running in paper mode instead.\n")

    if args.discover:
        await discover_markets(cfg)
        return

    logger.info("Config: %s", cfg)

    bot = PolyRadar(cfg, scan_interval=args.scan_interval)
    await bot.setup()

    # Install OS signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    def _sig_handler() -> None:
        logger.info("Shutdown signal received.")
        asyncio.create_task(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _sig_handler)

    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
