"""
telegram_alerts.py — Non-blocking Telegram notifications.

All sends are fire-and-forget (queued in a background task).
If the Telegram token / chat ID are not configured, every method
becomes a no-op so the rest of the bot runs unchanged.
"""

from __future__ import annotations

import asyncio
import html
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_EMOJI = {
    "trade_open":   "📥",
    "trade_won":    "✅",
    "trade_lost":   "❌",
    "drawdown":     "⚠️",
    "kill_switch":  "🛑",
    "signal":       "🔍",
    "daily":        "📊",
    "startup":      "🚀",
    "shutdown":     "💤",
}


class TelegramAlerter:
    """
    Sends Telegram messages via the Bot HTTP API.
    Uses python-telegram-bot's async Bot directly to avoid a full Application.
    """

    def __init__(self, token: str, chat_id: str) -> None:
        self._token   = token
        self._chat_id = chat_id
        self._bot: Optional[object] = None
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._enabled = bool(token and chat_id)
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not self._enabled:
            logger.info("Telegram alerts disabled (no token/chat_id configured).")
            return
        try:
            from telegram import Bot  # type: ignore[import-untyped]
            self._bot = Bot(token=self._token)
            self._task = asyncio.create_task(self._sender_loop(), name="telegram-sender")
            logger.info("Telegram alerter started.")
        except ImportError:
            logger.warning(
                "python-telegram-bot not installed — Telegram alerts disabled."
            )
            self._enabled = False

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ── Public alert methods ──────────────────────────────────────────────────

    def send_trade_open(
        self,
        symbol: str,
        timeframe: str,
        side: str,
        size_usdc: float,
        entry_price: float,
        edge_pct: float,
        confidence: float,
        paper: bool,
    ) -> None:
        mode = "[PAPER]" if paper else "[LIVE]"
        text = (
            f"{_EMOJI['trade_open']} <b>Trade Opened {mode}</b>\n"
            f"Market : <code>{symbol} {timeframe}</code>\n"
            f"Side   : <b>{side}</b>\n"
            f"Size   : <b>${size_usdc:.2f}</b>\n"
            f"Entry  : {entry_price:.3f}\n"
            f"Edge   : {edge_pct:.1%}\n"
            f"Conf.  : {confidence:.1%}\n"
            f"Time   : {_now()}"
        )
        self._enqueue(text)

    def send_trade_close(
        self,
        symbol: str,
        timeframe: str,
        side: str,
        pnl: float,
        status: str,
        paper: bool,
    ) -> None:
        emoji  = _EMOJI["trade_won"] if status == "WON" else _EMOJI["trade_lost"]
        mode   = "[PAPER]" if paper else "[LIVE]"
        sign   = "+" if pnl >= 0 else ""
        text   = (
            f"{emoji} <b>Trade Closed {mode}</b>\n"
            f"Market : <code>{symbol} {timeframe}</code>\n"
            f"Side   : {side}  →  <b>{status}</b>\n"
            f"P&amp;L   : <b>{sign}${pnl:.2f}</b>\n"
            f"Time   : {_now()}"
        )
        self._enqueue(text)

    def send_drawdown_alert(
        self,
        daily_drawdown: float,
        equity: float,
        threshold: float,
    ) -> None:
        text = (
            f"{_EMOJI['drawdown']} <b>Drawdown Warning</b>\n"
            f"Daily drawdown : <b>{daily_drawdown:.1%}</b> "
            f"(threshold {threshold:.0%})\n"
            f"Equity         : <b>${equity:.2f}</b>\n"
            f"Time           : {_now()}"
        )
        self._enqueue(text)

    def send_kill_switch(self, daily_drawdown: float, equity: float) -> None:
        text = (
            f"{_EMOJI['kill_switch']} <b>KILL SWITCH ACTIVATED</b>\n"
            f"Daily drawdown {daily_drawdown:.1%} exceeded 20% limit.\n"
            f"All trading halted.\n"
            f"Equity : <b>${equity:.2f}</b>\n"
            f"Time   : {_now()}"
        )
        self._enqueue(text)

    def send_signal_flagged(
        self,
        symbol: str,
        timeframe: str,
        cex_prob: float,
        poly_prob: float,
        lag_pct: float,
        edge_pct: float,
    ) -> None:
        """Alert for a flagged signal (lag > threshold), even if not traded."""
        text = (
            f"{_EMOJI['signal']} <b>Lag Signal Detected</b>\n"
            f"Market : <code>{symbol} {timeframe}</code>\n"
            f"CEX prob  : {cex_prob:.1%}\n"
            f"Poly prob : {poly_prob:.1%}\n"
            f"Lag       : <b>{lag_pct:.1%}</b>\n"
            f"Edge      : {edge_pct:.1%}\n"
            f"Time      : {_now()}"
        )
        self._enqueue(text)

    def send_daily_summary(
        self,
        equity: float,
        realized_pnl: float,
        win_rate: float,
        total_trades: int,
        daily_drawdown: float,
    ) -> None:
        sign = "+" if realized_pnl >= 0 else ""
        text = (
            f"{_EMOJI['daily']} <b>Daily Summary</b>\n"
            f"Equity    : <b>${equity:.2f}</b>\n"
            f"Realized  : <b>{sign}${realized_pnl:.2f}</b>\n"
            f"Win rate  : {win_rate:.1%} ({total_trades} trades)\n"
            f"Drawdown  : {daily_drawdown:.1%}\n"
            f"Date      : {_now()}"
        )
        self._enqueue(text)

    def send_startup(self, mode: str, initial_equity: float) -> None:
        text = (
            f"{_EMOJI['startup']} <b>PolyRadar Started</b>\n"
            f"Mode   : <b>{mode}</b>\n"
            f"Equity : ${initial_equity:.2f}\n"
            f"Time   : {_now()}"
        )
        self._enqueue(text)

    def send_shutdown(self, equity: float, realized_pnl: float) -> None:
        sign = "+" if realized_pnl >= 0 else ""
        text = (
            f"{_EMOJI['shutdown']} <b>PolyRadar Stopped</b>\n"
            f"Final equity : ${equity:.2f}\n"
            f"Session P&amp;L  : {sign}${realized_pnl:.2f}\n"
            f"Time         : {_now()}"
        )
        self._enqueue(text)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _enqueue(self, text: str) -> None:
        if self._enabled:
            try:
                self._queue.put_nowait(text)
            except asyncio.QueueFull:
                logger.debug("Telegram queue full, dropping message.")

    async def _sender_loop(self) -> None:
        """Drain the queue and send messages one by one."""
        while True:
            text = await self._queue.get()
            try:
                await self._send(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Telegram send failed: %s", exc)
            finally:
                self._queue.task_done()

    async def _send(self, text: str) -> None:
        assert self._bot is not None
        await self._bot.send_message(  # type: ignore[attr-defined]
            chat_id=self._chat_id,
            text=text,
            parse_mode="HTML",
        )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
