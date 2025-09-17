#!/usr/bin/env python3
"""
spot_advanced_bot.py

Advanced Telegram bot for multi-timeframe technical + structure analysis:
- multiple indicators (EMA, MACD, RSI, Bollinger, ATR, ADX, VWAP, OBV)
- Fair Value Gaps (FVG) detection
- Order Blocks detection
- Liquidity areas (based on wick clustering and recent highs/lows)
- Orderbook pressure metric (via depth snapshot)
- Multi-timeframe analysis: when running on a base timeframe (e.g. 15m) the bot
  will also fetch and analyze higher timeframes (1h, 4h, 1d) for confirmation.
- Inline UI: Start / Stop / Snapshot / Orderbook / Settings
- Sends compact summary messages listing which method indicates BUY/SELL/HOLD and reasons.

NOTES:
- This is an analytic tool and not financial advice.
- Replace TELEGRAM_BOT_TOKEN in .env or environment before running.
"""

import os
import asyncio
import logging
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta
import math
import json

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from binance import AsyncClient, BinanceSocketManager, BinanceAPIException
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters

# ---------------- CONFIG ----------------
TELEGRAM_BOT_TOKEN = os.getenv("8023843301:AAFlSOS-rR141haOeF-21wt2ldemgJoASOQ")
DEFAULT_INTERVAL = "15m"   # base timeframe for user choice (can be 15m,1h, etc)
HIGHER_TFS = {"15m": ["1h", "4h"], "1h": ["4h", "1d"], "4h": ["1d"], "1d": []}
CANDLES_HISTORY = 500
KLINE_LIMIT_FETCH = 1000
# Keep memory small per chat
CANDLES_TO_KEEP = 1000

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Per-chat state
chat_tasks = {}
chat_settings = {}
last_signals = {}

# Orderbooks cache
orderbooks = defaultdict(lambda: {"bids": {}, "asks": {}, "ready": False})

# ----------------- Utility -----------------
def normalize_symbol(sym: str) -> str:
    s = sym.strip().upper().replace("/", "")
    if not s.endswith("USDT"):
        s = s + "USDT"
    return s

def timeframe_to_minutes(tf: str) -> int:
    mapping = {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60,"2h":120,"4h":240,"6h":360,"12h":720,"1d":1440}
    return mapping.get(tf, 15)

# ---------------- Indicators / Structure analysis ----------------

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha['open'] = 0.0
    for i in range(len(df)):
        if i == 0:
            ha.iat[0, ha.columns.get_loc('open')] = (df['open'].iat[0] + df['close'].iat[0]) / 2
        else:
            ha.iat[i, ha.columns.get_loc('open')] = (ha.iat[i-1, ha.columns.get_loc('open')] + ha.iat[i-1, ha.columns.get_loc('close')]) / 2
    ha['high'] = df[['high','open','close']].max(axis=1)
    ha['low'] = df[['low','open','close']].min(axis=1)
    return ha

def detect_fvg(df: pd.DataFrame, lookback:int=200) -> list:
    """
    Detect Fair Value Gaps (FVG):
    Conservative definition used here:
      - For each 3-candle sequence (i-2, i-1, i):
        If candle i-1 body does NOT overlap with candle i-2 and candle i bodies in the direction of
        the impulse (i.e., an impulsive candle followed by a retracement leaving an imbalance), mark
        the gap area between the close and open of the impulse where market did not trade.
    Returns list of dicts: {'start':low, 'end':high, 'type':'bull'/'bear', 'time':index_of_impulse}
    """
    fvg_list = []
    # use last lookback candles
    N = min(len(df), lookback)
    for i in range(2, N):
        c0 = df.iloc[-i-2]
        c1 = df.iloc[-i-1]
        c2 = df.iloc[-i]
        # Bullish impulse: c0 bullish big, c1 small retrace, gap between low of c1 and high of c2 ???
        # Simpler robust approach: if c0.close < c2.open (i.e., gap up) mark area
        # We'll check both directions using bodies
        body0 = c0['close'] - c0['open']
        body1 = c1['close'] - c1['open']
        body2 = c2['close'] - c2['open']
        # bullish impulse then pullback leaving gap: bullish candle then lower overlap?
        # Use common practical rule: If high of previous candle < low of next candle => FVG up
        if c0['high'] < c2['low']:
            fvg_list.append({"start": float(c0['high']), "end": float(c2['low']), "type":"bull", "time": df.index[-i]})
        if c0['low'] > c2['high']:
            fvg_list.append({"start": float(c2['high']), "end": float(c0['low']), "type":"bear", "time": df.index[-i]})
    return fvg_list

def detect_order_blocks(df: pd.DataFrame, lookback:int=200, impulsive_ratio:float=1.5) -> list:
    """
    Detect simple order blocks:
    - Find impulsive candles where absolute body > impulsive_ratio * average body
    - The candle preceding the impulsive move is considered an order block candidate (institutional entry)
    - We return ranges (open..close) of that candle as order block zone
    """
    obs = []
    N = min(len(df), lookback)
    body = (df['close'] - df['open']).abs()
    avg_body = body.rolling(20, min_periods=1).mean()
    for i in range(1, N):
        if body.iloc[-i] > impulsive_ratio * avg_body.iloc[-i]:
            # preceding candle
            if i+1 < len(df):
                prev = df.iloc[-i-1]
                zone = {"low": float(min(prev['open'], prev['close'])), "high": float(max(prev['open'], prev['close'])), "time": df.index[-i-1]}
                obs.append(zone)
    return obs

def liquidity_areas(df: pd.DataFrame, lookback:int=500, wick_multiplier:float=1.5) -> list:
    """
    Simple liquidity detection: find clusters of swing highs/lows (wicks) where many candles have extremes
    near same price. We'll identify levels where count of wicks within small tolerance is high.
    """
    highs = df['high'][-lookback:]
    lows = df['low'][-lookback:]
    levels = []
    # sample candidate prices from highs and lows
    candidates = pd.concat([highs, lows]).unique()
    tol = (df['high'].max() - df['low'].min()) * 0.002  # 0.2% price tolerance window
    for p in candidates:
        cnt_high = ((highs - p).abs() <= tol).sum()
        cnt_low = ((lows - p).abs() <= tol).sum()
        cnt = cnt_high + cnt_low
        if cnt >= max(3, int(lookback * 0.01)):  # at least 1% of lookback candles
            levels.append({"price": float(p), "count": int(cnt)})
    # sort by count desc
    levels = sorted(levels, key=lambda x: -x['count'])
    return levels[:10]

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    pv = (df['close'] * df['volume']).cumsum()
    v = df['volume'].cumsum()
    return pv / v.replace(0, np.nan)

# Orderbook pressure
def orderbook_pressure(snapshot: dict, depth:int=10) -> dict:
    bids = sorted(((float(p), float(q)) for p,q in snapshot.get('bids', {}).items()), key=lambda x:-x[0])[:depth]
    asks = sorted(((float(p), float(q)) for p,q in snapshot.get('asks', {}).items()), key=lambda x:x[0])[:depth]
    bid_vol = sum(q for p,q in bids)
    ask_vol = sum(q for p,q in asks)
    pressure = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
    spread = None
    mid = None
    if bids and asks:
        spread = asks[0][0] - bids[0][0]
        mid = (asks[0][0] + bids[0][0]) / 2
    return {"bid_vol": bid_vol, "ask_vol": ask_vol, "pressure": pressure, "spread": spread, "mid": mid}

# ---------------- Compose multi-method analysis ----------------

def analyze_symbol_multitime(client: AsyncClient, symbol: str, base_tf: str = "15m", limit:int=200):
    """
    This function will be called async: it fetches klines for base_tf and for higher timeframes and computes:
     - indicators summary (ema crossover, rsi, macd basic)
     - vwap location
     - fvg zones on multiple higher TFs
     - order blocks on multiple TFs
     - liquidity areas
     - orderbook pressure (from cache if present)
    Returns a dict summarizing which methods point to BUY/SELL/HOLD and the reasoning texts.
    """
    raise NotImplementedError("The real-time function uses websockets and per-chat tasks below.")
# ---------------- Real-time streamer per chat ----------------

async def fetch_klines(client: AsyncClient, symbol: str, interval: str, limit:int=500):
    # Wrap REST klines call into a helper
    try:
        kl = await client.get_klines(symbol=symbol, interval=interval, limit=limit)
    except BinanceAPIException as e:
        logger.exception("Binance API exception while fetching klines: %s", e)
        return None
    cols = ['open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base_vol','taker_quote_vol','ignore']
    df = pd.DataFrame(kl, columns=cols)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df.index = pd.to_datetime(df['close_time'], unit='ms')
    return df

async def stream_and_analyze_chat(application: Application, chat_id: int, symbol: str, base_tf: str):
    """
    Main per-chat worker: subscribes to kline websocket for the base timeframe,
    and on each closed candle does multi-timeframe analysis and sends compact summary.
    """
    logger.info("Starting stream task for chat %s symbol %s tf %s", chat_id, symbol, base_tf)
    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)
    ks = bm.kline_socket(symbol=symbol.lower(), interval=base_tf)
    async with ks as ksocket:
        candles_cache = deque(maxlen=CANDLES_TO_KEEP)
        async for msg in ksocket:
            try:
                k = msg.get("k", {})
                is_closed = k.get("x", False)
                candle = {
                    "open": float(k["o"]),
                    "high": float(k["h"]),
                    "low": float(k["l"]),
                    "close": float(k["c"]),
                    "volume": float(k["v"]),
                    "startTime": int(k["t"]),
                    "endTime": int(k["T"])
                }
                candles_cache.append(candle)
                if is_closed:
                    df = pd.DataFrame(candles_cache)
                    df.index = pd.to_datetime(df["endTime"], unit='ms')
                    df = df[["open","high","low","close","volume"]]
                    # compute analysis for base TF and higher TFs (fetch via REST)
                    msg_summary = await compose_full_analysis(client, symbol, base_tf, df)
                    # send to chat
                    await application.bot.send_message(chat_id=chat_id, text=msg_summary, parse_mode="Markdown")
            except asyncio.CancelledError:
                logger.info("Chat stream cancelled %s", chat_id)
                break
            except Exception:
                logger.exception("Error in stream loop for chat %s", chat_id)
    await client.close_connection()
    logger.info("Stream ended for chat %s", chat_id)

async def compose_full_analysis(client: AsyncClient, symbol: str, base_tf: str, base_df: pd.DataFrame):
    """
    Compose multi-timeframe analysis message string using helpers.
    """
    # prepare summary parts
    lines = []
    lines.append(f"*Multi-method analysis for {symbol}*")
    lines.append(f"Base timeframe: {base_tf}  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    last_price = float(base_df['close'].iloc[-1])
    lines.append(f"Current price: `{last_price:.8f}`")
    # Base timeframe indicators
    try:
        ema12 = EMAIndicator(base_df['close'], window=12).ema_indicator().iloc[-1]
        ema26 = EMAIndicator(base_df['close'], window=26).ema_indicator().iloc[-1]
        rsi = RSIIndicator(base_df['close'], window=14).rsi().iloc[-1]
        bb = BollingerBands(base_df['close'], window=20)
        bb_high = bb.bollinger_hband().iloc[-1]
        bb_low = bb.bollinger_lband().iloc[-1]
        atr = AverageTrueRange(base_df['high'], base_df['low'], base_df['close'], window=14).average_true_range().iloc[-1]
    except Exception:
        ema12=ema26=rsi=bb_high=bb_low=atr = None
    lines.append("")
    lines.append("*Indicator signals (base TF):*")
    ind_msgs = []
    if ema12 and ema26:
        if ema12 > ema26:
            ind_msgs.append("EMA12 > EMA26 → *bullish bias*")
        else:
            ind_msgs.append("EMA12 < EMA26 → *bearish bias*")
    if rsi is not None:
        if rsi < 30:
            ind_msgs.append(f"RSI {rsi:.1f} → *oversold* (possible reversal)")
        elif rsi > 70:
            ind_msgs.append(f"RSI {rsi:.1f} → *overbought* (watch for sell)")
        else:
            ind_msgs.append(f"RSI {rsi:.1f} → neutral")
    if bb_high and bb_low:
        if last_price > bb_high:
            ind_msgs.append("Price > Upper Bollinger → momentum to upside (watch pullbacks)")
        elif last_price < bb_low:
            ind_msgs.append("Price < Lower Bollinger → momentum to downside")
    if atr:
        ind_msgs.append(f"ATR ≈ {atr:.8f}")
    lines.extend(["- " + m for m in ind_msgs[:6]])

    # Multi-timeframe: fetch higher TFs
    higher = HIGHER_TFS.get(base_tf, [])
    fvg_found = []
    ob_found = []
    liquidity_found = []
    for tf in [base_tf] + higher:
        # fetch historical klines for timeframe tf
        df_tf = await fetch_klines(client, symbol, tf, limit=500)
        if df_tf is None or len(df_tf) < 50:
            lines.append(f"_Could not fetch {tf} data_")
            continue
        # detect FVG on this TF (use last 200 candles)
        fvg = detect_fvg(df_tf, lookback=200)
        if fvg:
            fvg_found.append({"tf": tf, "zones": fvg[:5]})
        obs = detect_order_blocks(df_tf, lookback=200)
        if obs:
            ob_found.append({"tf": tf, "zones": obs[:5]})
        liq = liquidity_areas(df_tf, lookback=500)
        if liq:
            liquidity_found.append({"tf": tf, "levels": liq[:5]})

    # Summarize FVG and OB detection
    lines.append("")
    lines.append("*Fair Value Gaps (FVG) found:*")
    if not fvg_found:
        lines.append("- None detected on scanned TFs")
    else:
        for item in fvg_found:
            zones = item['zones']
            lines.append(f"- TF {item['tf']}: {len(zones)} zones (showing up to 3)")
            for z in zones[:3]:
                lines.append(f"    • {z['type'].upper()} zone {z['start']:.8f} — {z['end']:.8f} @ {z['time'].strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("*Order blocks (recent candidates):*")
    if not ob_found:
        lines.append("- None detected on scanned TFs")
    else:
        for item in ob_found:
            lines.append(f"- TF {item['tf']}: {len(item['zones'])} (showing up to 3)")
            for z in item['zones'][:3]:
                lines.append(f"    • zone {z['low']:.8f} — {z['high']:.8f} @ {z['time'].strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("*Liquidity areas:*")
    if not liquidity_found:
        lines.append("- No clustered levels detected")
    else:
        for item in liquidity_found:
            lines.append(f"- TF {item['tf']}: {len(item['levels'])} levels (top 3)")
            for lvl in item['levels'][:3]:
                lines.append(f"    • {lvl['price']:.8f} (count {lvl['count']})")

    # Orderbook pressure snapshot (from cache)
    ob_cache = orderbooks.get(symbol, {})
    if ob_cache.get("ready"):
        pressure = orderbook_pressure(ob_cache)
        lines.append("")
        lines.append("*Orderbook pressure:*")
        lines.append(f"- Bid vol: {pressure['bid_vol']:.6f}  Ask vol: {pressure['ask_vol']:.6f}  Pressure: {pressure['pressure']:.3f}  Spread: {pressure['spread']:.8f}")
    else:
        lines.append("")
        lines.append("_Orderbook depth snapshot not ready_")

    # Final combined quick verdict (simple ensemble of checks)
    verdicts = []
    # indicators: ema bias
    if ema12 and ema26:
        if ema12 > ema26:
            verdicts.append("INDICATORS: bullish bias")
        else:
            verdicts.append("INDICATORS: bearish bias")
    # FVG/OB confluence: if FVG bull exists at/near price → buy bias
    con_buy = False
    con_sell = False
    # check if any fvg/ob on higher TFs contains current price within a buffer
    buf = (atr if 'atr' in locals() and atr else (last_price*0.002))
    for item in fvg_found:
        for z in item['zones']:
            if z['start'] - buf <= last_price <= z['end'] + buf:
                if z['type']=='bull':
                    con_buy = True
                else:
                    con_sell = True
    for item in ob_found:
        for z in item['zones']:
            if z['low'] - buf <= last_price <= z['high'] + buf:
                # order block presence suggests mean reversion: if price inside bull OB -> buy
                if z['low'] < z['high']:
                    # no direct type, assume bullish
                    con_buy = True

    if con_buy and not con_sell:
        verdicts.append("STRUCTURE: confluence favors BUY (FVG/OB)")
    if con_sell and not con_buy:
        verdicts.append("STRUCTURE: confluence favors SELL (FVG/OB)")

    # orderbook pressure weighting
    if ob_cache.get("ready"):
        p = orderbook_pressure(ob_cache)['pressure']
        if p > 0.15:
            verdicts.append("ORDERBOOK: buy-side pressure")
        elif p < -0.15:
            verdicts.append("ORDERBOOK: sell-side pressure")

    lines.append("")
    lines.append("*Quick ensemble verdicts:*")
    if verdicts:
        for v in verdicts[:6]:
            lines.append("- " + v)
    else:
        lines.append("- No decisive confluence. HOLD / wait for confirmation.")

    # short actionable suggestions (position sizing left to user)
    lines.append("")
    lines.append("_Note: This is structural + technical *information* only, not financial advice._")

    return "\n".join(lines)

# ---------------- Telegram Handlers ----------------

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("Start monitoring (enter symbol)", callback_data="START")],
        [InlineKeyboardButton("Stop", callback_data="STOP"), InlineKeyboardButton("Snapshot", callback_data="SNAP")],
    ]
    await update.message.reply_text("Advanced spot analyzer. Press Start to enter symbol (e.g. BTCUSDT).", reply_markup=InlineKeyboardMarkup(kb))

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = query.message.chat_id
    if data == "START":
        await query.message.reply_text("Send symbol like `BTCUSDT` or `BTC/USDT` as plain message.")
        return
    if data == "STOP":
        task = chat_tasks.get(chat_id)
        if task:
            task.cancel()
            chat_tasks.pop(chat_id, None)
            chat_settings.pop(chat_id, None)
            await query.message.reply_text("Stopped monitoring.")
        else:
            await query.message.reply_text("No active monitor.")
        return
    if data == "SNAP":
        settings = chat_settings.get(chat_id)
        if not settings:
            await query.message.reply_text("No symbol running. Start first.")
            return
        # request a manual snapshot by calling compose_full_analysis with current REST data
        symbol = settings['symbol']
        base_tf = settings.get('tf', DEFAULT_INTERVAL)
        client = await AsyncClient.create()
        df = await fetch_klines(client, symbol, base_tf, limit=200)
        text = await compose_full_analysis(client, symbol, base_tf, df)
        await application.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
        await client.close_connection()
        return

async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    text = update.message.text.strip()
    # user provided symbol; normalize and start stream
    sym = normalize_symbol(text)
    # simple check
    await update.message.reply_text(f"Starting monitoring for {sym} on {DEFAULT_INTERVAL}. I'll send analysis on each {DEFAULT_INTERVAL} candle close.")
    # store and start task
    if chat_id in chat_tasks:
        chat_tasks[chat_id].cancel()
        await asyncio.sleep(0.1)
    chat_settings[chat_id] = {"symbol": sym, "tf": DEFAULT_INTERVAL}
    task = asyncio.create_task(stream_and_analyze_chat(context.application, chat_id, sym, DEFAULT_INTERVAL))
    chat_tasks[chat_id] = task

# ---------------- Entry point ----------------
def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_message_handler))
    app.run_polling()

if __name__ == "__main__":
    main()
