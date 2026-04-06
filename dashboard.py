"""
dashboard.py — Rich terminal dashboard.

Layout (refreshed every second)
────────────────────────────────
┌──────────────────────────────────────────────────────────────────────┐
│  PolyRadar  │  Mode  │  Equity  │  Daily P&L  │  Win Rate  │  Status │
├─────────────────────────┬────────────────────────────────────────────┤
│  Live Prices (CEX)      │  Polymarket Odds                           │
├─────────────────────────┴────────────────────────────────────────────┤
│  Open Positions                                                       │
├──────────────────────────────────────────────────────────────────────┤
│  Last 10 Trades                                                       │
└──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from db import Database, TradeRow
from executor import Portfolio, Position

logger = logging.getLogger(__name__)

_REFRESH_RATE = 1   # Hz

# ── Helpers ───────────────────────────────────────────────────────────────────

def _pnl_color(v: float) -> str:
    return "green" if v >= 0 else "red"

def _fmt_pnl(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"[{_pnl_color(v)}]{sign}${v:.2f}[/]"

def _fmt_pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1%}"

def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")


# ── Dashboard class ───────────────────────────────────────────────────────────

class Dashboard:
    """
    Async Rich Live dashboard.  Call `start()` to launch the background
    refresh task, and mutate the shared state attributes to update what's shown.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        db: Database,
        config: "Config",           # noqa: F821
        binance_feed: "BinanceFeed",    # noqa: F821
        poly_feed: "PolymarketFeed",    # noqa: F821
    ) -> None:
        self._portfolio  = portfolio
        self._db         = db
        self._cfg        = config
        self._binance    = binance_feed
        self._poly       = poly_feed
        self._console    = Console()
        self._live: Optional[Live] = None
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._stop       = asyncio.Event()

        # Mutable state updated by the bot loop
        self.kill_switch_active: bool = False
        self.session_start_equity: float = portfolio.balance
        self.last_signals: List[dict] = []   # recent signal dicts for display

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="dashboard")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._live:
            self._live.stop()

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        layout = self._build_layout()
        with Live(
            layout,
            console=self._console,
            refresh_per_second=_REFRESH_RATE,
            screen=True,
        ) as live:
            self._live = live
            while not self._stop.is_set():
                try:
                    recent_trades = await self._db.get_recent_trades(10)
                    open_trades   = await self._db.get_open_trades()
                    stats         = await self._db.get_daily_stats()
                    win_rate_all  = await self._db.get_all_time_win_rate()
                    self._refresh(layout, recent_trades, open_trades, stats, win_rate_all)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Dashboard refresh error: %s", exc)
                await asyncio.sleep(1.0 / _REFRESH_RATE)

    # ── Layout ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_layout() -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header",   size=3),
            Layout(name="mid",      size=10),
            Layout(name="positions", size=8),
            Layout(name="trades"),
        )
        layout["mid"].split_row(
            Layout(name="prices",  ratio=1),
            Layout(name="odds",    ratio=1),
        )
        return layout

    def _refresh(
        self,
        layout: Layout,
        recent_trades: List[TradeRow],
        open_trades: List[TradeRow],
        stats: dict,
        win_rate_all: float,
    ) -> None:
        pf = self._portfolio
        layout["header"].update(self._make_header(stats, win_rate_all))
        layout["prices"].update(self._make_prices())
        layout["odds"].update(self._make_odds())
        layout["positions"].update(self._make_positions(pf))
        layout["trades"].update(self._make_trades(recent_trades))

    # ── Panels ────────────────────────────────────────────────────────────────

    def _make_header(self, stats: dict, win_rate_all: float) -> Panel:
        pf   = self._portfolio
        mode = "[red bold]LIVE[/]" if not pf.paper else "[yellow bold]PAPER[/]"
        ks   = "[red bold] ⛔ KILL SWITCH [/]" if self.kill_switch_active else "[green]Active[/]"

        session_pnl = pf.equity - self.session_start_equity
        daily_dd    = pf.daily_drawdown

        t = Table.grid(padding=(0, 2))
        t.add_column(justify="left")
        t.add_column(justify="right")
        t.add_column(justify="right")
        t.add_column(justify="right")
        t.add_column(justify="right")
        t.add_column(justify="right")
        t.add_column(justify="right")
        t.add_row(
            f"[bold cyan]PolyRadar[/]  {mode}",
            f"Equity: [bold]${pf.equity:,.2f}[/]",
            f"Session P&L: {_fmt_pnl(session_pnl)}",
            f"Today P&L: {_fmt_pnl(stats.get('realized_pnl', 0.0))}",
            f"Win Rate: [bold]{win_rate_all:.1%}[/] ({stats.get('total', 0)} trades today)",
            f"Drawdown: [{_pnl_color(-daily_dd)}]{daily_dd:.1%}[/]",
            f"Status: {ks}   {_now_str()}",
        )
        return Panel(t, style="bold", box=box.SIMPLE)

    def _make_prices(self) -> Panel:
        table = Table(
            title="CEX Prices (Binance)",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold cyan",
            expand=True,
        )
        table.add_column("Pair / TF", style="cyan", width=14)
        table.add_column("Open", justify="right", width=12)
        table.add_column("Current", justify="right", width=12)
        table.add_column("Ret%", justify="right", width=8)
        table.add_column("Elapsed", justify="right", width=9)
        table.add_column("P(up)", justify="right", width=8)

        from binance_feed import cex_probability, model_confidence
        import math

        for sym in ("BTCUSDT", "ETHUSDT"):
            for tf in ("5m", "15m"):
                c = self._binance.get_candle(sym, tf)
                if c.open_price <= 0:
                    table.add_row(f"{sym[:3]}/{tf}", "-", "-", "-", "-", "-")
                    continue
                ret   = (c.current_price / c.open_price - 1) * 100
                ann_v = self._cfg.annual_vol_for(sym)
                p_up  = cex_probability(c, ann_v)
                color = "green" if ret >= 0 else "red"
                table.add_row(
                    f"{sym[:3]}/{tf}",
                    f"${c.open_price:,.2f}",
                    f"[{color}]${c.current_price:,.2f}[/]",
                    f"[{color}]{ret:+.2f}%[/]",
                    f"{c.elapsed_minutes:.1f}m",
                    f"[bold]{p_up:.1%}[/]",
                )
        return Panel(table, box=box.ROUNDED)

    def _make_odds(self) -> Panel:
        table = Table(
            title="Polymarket Odds",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )
        table.add_column("Market", style="magenta", width=14)
        table.add_column("YES mid", justify="right", width=10)
        table.add_column("NO mid",  justify="right", width=10)
        table.add_column("Spread",  justify="right", width=8)
        table.add_column("Lag",     justify="right", width=8)
        table.add_column("Updated", justify="right", width=10)

        from binance_feed import cex_probability
        for sym in ("BTCUSDT", "ETHUSDT"):
            for tf in ("5m", "15m"):
                m = self._poly.get_market(sym, tf)
                c = self._binance.get_candle(sym, tf)
                if m is None:
                    table.add_row(f"{sym[:3]}/{tf}", "-", "-", "-", "-", "-")
                    continue
                ann_v = self._cfg.annual_vol_for(sym)
                p_cex = cex_probability(c, ann_v) if c.open_price > 0 else 0.5
                lag   = abs(p_cex - m.yes_price)
                lag_color = "red bold" if lag >= self._cfg.lag_threshold else "white"
                updated = (
                    m.last_update.strftime("%H:%M:%S")
                    if m.last_update else "-"
                )
                table.add_row(
                    f"{sym[:3]}/{tf}",
                    f"{m.yes_price:.3f}",
                    f"{m.no_price:.3f}",
                    f"{m.spread:.4f}",
                    f"[{lag_color}]{lag:.1%}[/]",
                    updated,
                )
        return Panel(table, box=box.ROUNDED)

    def _make_positions(self, pf: Portfolio) -> Panel:
        table = Table(
            title=f"Open Positions  (equity ${pf.equity:,.2f}  |  free ${pf.balance:,.2f})",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold yellow",
            expand=True,
        )
        table.add_column("#",       width=5)
        table.add_column("Market",  width=14)
        table.add_column("Side",    width=5)
        table.add_column("Size",    justify="right", width=10)
        table.add_column("Entry",   justify="right", width=8)
        table.add_column("Contr.",  justify="right", width=8)
        table.add_column("Opened",  width=10)

        open_pos = [p for p in pf.positions.values() if p.status == "OPEN"]
        if not open_pos:
            table.add_row("—", "No open positions", "", "", "", "", "")
        for pos in open_pos:
            table.add_row(
                str(pos.trade_id),
                f"{pos.symbol[:3]}/{pos.timeframe}",
                f"[cyan]{pos.side}[/]",
                f"${pos.size_usdc:.2f}",
                f"{pos.entry_price:.3f}",
                f"{pos.contracts:.2f}",
                pos.opened_at.strftime("%H:%M:%S"),
            )
        return Panel(table, box=box.ROUNDED)

    def _make_trades(self, trades: List[TradeRow]) -> Panel:
        table = Table(
            title="Last 10 Trades",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold white",
            expand=True,
        )
        table.add_column("#",       width=5)
        table.add_column("Time",    width=10)
        table.add_column("Market",  width=14)
        table.add_column("Side",    width=5)
        table.add_column("Size",    justify="right", width=10)
        table.add_column("Edge",    justify="right", width=7)
        table.add_column("Status",  width=8)
        table.add_column("P&L",     justify="right", width=10)
        table.add_column("Mode",    width=6)

        if not trades:
            table.add_row("—", "No trades yet", "", "", "", "", "", "", "")

        for tr in trades:
            status_color = {
                "WON":       "green",
                "LOST":      "red",
                "OPEN":      "yellow",
                "CANCELLED": "dim",
            }.get(tr.status, "white")
            pnl_str = _fmt_pnl(tr.pnl) if tr.pnl is not None else "-"
            table.add_row(
                str(tr.id),
                tr.timestamp[11:19],   # HH:MM:SS
                f"{tr.symbol[:3]}/{tr.timeframe}",
                tr.side,
                f"${tr.size_usdc:.2f}",
                f"{tr.edge_pct:.1%}",
                f"[{status_color}]{tr.status}[/]",
                pnl_str,
                "paper" if tr.paper else "[red]live[/]",
            )
        return Panel(table, box=box.ROUNDED)
