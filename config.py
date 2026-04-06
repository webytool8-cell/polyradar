"""
config.py — Configuration dataclass loaded from .env + CLI overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ── Polymarket CLOB ──────────────────────────────────────────────────────
    poly_api_key: str = ""
    poly_api_secret: str = ""
    poly_passphrase: str = ""
    poly_host: str = "https://clob.polymarket.com"

    # Condition IDs for the four markets we monitor.
    # If empty at start-up the bot will attempt auto-discovery.
    btc_5m_condition_id: str = ""
    btc_15m_condition_id: str = ""
    eth_5m_condition_id: str = ""
    eth_15m_condition_id: str = ""

    # ── Binance ──────────────────────────────────────────────────────────────
    binance_ws_url: str = "wss://stream.binance.com:9443"

    # ── Signal thresholds ────────────────────────────────────────────────────
    lag_threshold: float = 0.03    # 3 pp — flag when Poly lags CEX by this much
    min_edge: float = 0.05         # 5 % minimum edge to trade
    max_position_pct: float = 0.08 # 8 % of portfolio max per position
    min_confidence: float = 0.85   # 85 % model confidence required
    kelly_fraction: float = 0.5    # half-Kelly

    # ── Risk management ───────────────────────────────────────────────────────
    max_daily_drawdown: float = 0.20  # 20 % — kill switch threshold

    # ── Paper-trading ─────────────────────────────────────────────────────────
    # paper_trading stays True unless all three live flags are supplied.
    paper_trading: bool = True
    initial_balance: float = 10_000.0

    # ── Live-trading gate (3 explicit flags required) ─────────────────────────
    live_flag_1: bool = False   # --live
    live_flag_2: bool = False   # --confirm-live
    live_flag_3: bool = False   # --i-accept-financial-risk

    # ── Volatility (annualised, used in GBM probability model) ───────────────
    btc_annual_vol: float = 0.80
    eth_annual_vol: float = 0.90

    # ── Misc ─────────────────────────────────────────────────────────────────
    poly_poll_interval: float = 2.0  # seconds between CLOB orderbook polls
    db_path: str = "polyradar.db"

    # ── Telegram ─────────────────────────────────────────────────────────────
    telegram_token: str = ""
    telegram_chat_id: str = ""

    # ── Internal (set after __post_init__) ───────────────────────────────────
    live_trading: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.live_trading = (
            not self.paper_trading
            and self.live_flag_1
            and self.live_flag_2
            and self.live_flag_3
        )

    # ── Factory ───────────────────────────────────────────────────────────────
    @classmethod
    def from_env(cls) -> "Config":
        """Load base config from .env / environment variables."""
        return cls(
            poly_api_key=os.getenv("POLY_API_KEY", ""),
            poly_api_secret=os.getenv("POLY_API_SECRET", ""),
            poly_passphrase=os.getenv("POLY_PASSPHRASE", ""),
            poly_host=os.getenv("POLY_HOST", "https://clob.polymarket.com"),
            btc_5m_condition_id=os.getenv("BTC_5M_CONDITION_ID", ""),
            btc_15m_condition_id=os.getenv("BTC_15M_CONDITION_ID", ""),
            eth_5m_condition_id=os.getenv("ETH_5M_CONDITION_ID", ""),
            eth_15m_condition_id=os.getenv("ETH_15M_CONDITION_ID", ""),
            binance_ws_url=os.getenv("BINANCE_WS_URL", "wss://stream.binance.com:9443"),
            lag_threshold=float(os.getenv("LAG_THRESHOLD", "0.03")),
            min_edge=float(os.getenv("MIN_EDGE", "0.05")),
            max_position_pct=float(os.getenv("MAX_POSITION_PCT", "0.08")),
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.85")),
            kelly_fraction=float(os.getenv("KELLY_FRACTION", "0.5")),
            max_daily_drawdown=float(os.getenv("MAX_DAILY_DRAWDOWN", "0.20")),
            initial_balance=float(os.getenv("INITIAL_BALANCE", "10000.0")),
            btc_annual_vol=float(os.getenv("BTC_ANNUAL_VOL", "0.80")),
            eth_annual_vol=float(os.getenv("ETH_ANNUAL_VOL", "0.90")),
            poly_poll_interval=float(os.getenv("POLY_POLL_INTERVAL", "2.0")),
            db_path=os.getenv("DB_PATH", "polyradar.db"),
            telegram_token=os.getenv("TELEGRAM_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )

    def apply_live_flags(
        self,
        live: bool = False,
        confirm_live: bool = False,
        accept_risk: bool = False,
    ) -> None:
        """Apply the three CLI live flags and recompute live_trading."""
        self.live_flag_1 = live
        self.live_flag_2 = confirm_live
        self.live_flag_3 = accept_risk
        if live and confirm_live and accept_risk:
            self.paper_trading = False
        self.__post_init__()

    def annual_vol_for(self, symbol: str) -> float:
        """Return annualised volatility for a given symbol (BTCUSDT / ETHUSDT)."""
        return self.btc_annual_vol if "BTC" in symbol.upper() else self.eth_annual_vol

    def __repr__(self) -> str:
        mode = "LIVE" if self.live_trading else "PAPER"
        return (
            f"Config(mode={mode}, lag={self.lag_threshold:.0%}, "
            f"min_edge={self.min_edge:.0%}, max_pos={self.max_position_pct:.0%}, "
            f"min_conf={self.min_confidence:.0%}, kelly={self.kelly_fraction}x)"
        )
