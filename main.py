
import os
import math
import traceback
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# =============================
# Technical Indicator Helpers
# =============================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def detect_macd_cross(macd_line: pd.Series, sig: pd.Series) -> str:
    if len(macd_line) < 2 or len(sig) < 2:
        return "None"
    prev = macd_line.iloc[-2] - sig.iloc[-2]
    last = macd_line.iloc[-1] - sig.iloc[-1]
    if prev < 0 and last > 0:
        return "Up"
    if prev > 0 and last < 0:
        return "Down"
    return "None"

def trend_signal(close, ma50, ma200, macd_line, macd_signal, rsi_val):
    vals = [ma50, ma200, macd_line, macd_signal, rsi_val]
    if any(pd.isna(v) for v in vals):
        return "Insufficient Data"
    if close > ma50 > ma200 and macd_line > macd_signal and rsi_val > 50:
        return "Bullish"
    if close < ma50 < ma200 and macd_line < macd_signal and rsi_val < 50:
        return "Bearish"
    return "Neutral"

def candle_pattern(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return ""
    prev = df.iloc[-2]
    last = df.iloc[-1]
    prev_high_body = max(prev['Close'], prev['Open'])
    prev_low_body = min(prev['Close'], prev['Open'])
    last_high_body = max(last['Close'], last['Open'])
    last_low_body = min(last['Close'], last['Open'])
    if (last['Close'] > last['Open'] and prev['Close'] < prev['Open']
        and last_high_body >= prev_high_body and last_low_body <= prev_low_body):
        return "Bullish Engulfing"
    if (last['Close'] < last['Open'] and prev['Close'] > prev['Open']
        and last_high_body >= prev_high_body and last_low_body <= prev_low_body):
        return "Bearish Engulfing"
    return ""

# =============================
# Data & Analysis
# =============================
def safe_download(ticker: str, period="6mo", interval="1d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            # Single out the level for this ticker
            df = df.xs(ticker, level=1, axis=1)
        df = df.dropna().reset_index()
        df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
        return df
    except Exception as e:
        print(f"[WARN] download failed {ticker}: {e}")
        return pd.DataFrame()

def compute_levels(df: pd.DataFrame) -> pd.DataFrame:
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df['RSI14'] = rsi(df['Close'], 14)
    macd_line, sig, hist = macd(df['Close'])
    df['MACD'] = macd_line
    df['MACD_Signal'] = sig
    df['MACD_Hist'] = hist
    # Support/Resistance windows
    df['Support20'] = df['Low'].rolling(20).min()
    df['Resistance20'] = df['High'].rolling(20).max()
    df['Support50'] = df['Low'].rolling(50).min()
    df['Resistance50'] = df['High'].rolling(50).max()
    return df

def build_trade_plan(row_prev: pd.Series, row: pd.Series) -> dict:
    macd_cross = None
    if not (pd.isna(row_prev.get('MACD')) or pd.isna(row_prev.get('MACD_Signal')) or
            pd.isna(row.get('MACD')) or pd.isna(row.get('MACD_Signal'))):
        prev = row_prev['MACD'] - row_prev['MACD_Signal']
        last = row['MACD'] - row['MACD_Signal']
        if prev < 0 and last > 0:
            macd_cross = "Up"
        elif prev > 0 and last < 0:
            macd_cross = "Down"
        else:
            macd_cross = "None"
    else:
        macd_cross = "None"

    plan = {
        "MACD_Cross": macd_cross,
        "Trend": ("Uptrend" if (row['MA50'] > row['MA200']) else
                  "Downtrend" if (row['MA50'] < row['MA200']) else "Sideways"),
        "Entry": None,
        "StopLoss": None,
        "TakeProfit": None,
        "Bias": None
    }

    # Long setup
    if (row['Close'] > row['MA50']) and (row['RSI14'] is not np.nan and row['RSI14'] > 55) and (row['MACD'] > row['MACD_Signal']):
        plan['Bias'] = "Long"
        plan['Entry'] = float(row['Close'])
        # SL at nearest support, TP at resistance
        sl_candidates = [row.get('Support20'), row.get('Support50')]
        sl_candidates = [c for c in sl_candidates if not pd.isna(c)]
        if sl_candidates:
            plan['StopLoss'] = float(max(sl_candidates))  # tighter SL (nearest strong support)
        tp_candidates = [row.get('Resistance20'), row.get('Resistance50')]
        tp_candidates = [c for c in tp_candidates if not pd.isna(c)]
        if tp_candidates:
            plan['TakeProfit'] = float(min(tp_candidates))  # nearer resistance
    # Short setup
    elif (row['Close'] < row['MA50']) and (row['RSI14'] is not np.nan and row['RSI14'] < 45) and (row['MACD'] < row['MACD_Signal']):
        plan['Bias'] = "Short"
        plan['Entry'] = float(row['Close'])
        sl_candidates = [row.get('Resistance20'), row.get('Resistance50')]
        sl_candidates = [c for c in sl_candidates if not pd.isna(c)]
        if sl_candidates:
            plan['StopLoss'] = float(min(sl_candidates))  # tighter SL above
        tp_candidates = [row.get('Support20'), row.get('Support50')]
        tp_candidates = [c for c in tp_candidates if not pd.isna(c)]
        if tp_candidates:
            plan['TakeProfit'] = float(max(tp_candidates))  # nearer support
    else:
        plan['Bias'] = "No-Trade"

    return plan

def analyze_ticker(ticker: str) -> dict:
    df = safe_download(ticker, period="6mo", interval="1d")
    if df.empty:
        return {"ticker": ticker, "error": "No data", "df": pd.DataFrame()}
    df = compute_levels(df)
    pattern = candle_pattern(df[['Open', 'High', 'Low', 'Close']])
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    signal = trend_signal(last['Close'], last['MA50'], last['MA200'], last['MACD'], last['MACD_Signal'], last['RSI14'])
    macd_cross = detect_macd_cross(df['MACD'], df['MACD_Signal'])
    plan = build_trade_plan(prev, last)

    info = {
        "ticker": ticker,
        "time": str(last['Date']),
        "close": float(last['Close']),
        "rsi14": float(last['RSI14']) if not math.isnan(last['RSI14']) else None,
        "ma50": float(last['MA50']) if not math.isnan(last['MA50']) else None,
        "ma200": float(last['MA200']) if not math.isnan(last['MA200']) else None,
        "macd": float(last['MACD']) if not math.isnan(last['MACD']) else None,
        "macd_signal": float(last['MACD_Signal']) if not math.isnan(last['MACD_Signal']) else None,
        "macd_cross": macd_cross,
        "support20": float(last['Support20']) if not math.isnan(last['Support20']) else None,
        "resistance20": float(last['Resistance20']) if not math.isnan(last['Resistance20']) else None,
        "support50": float(last['Support50']) if not math.isnan(last['Support50']) else None,
        "resistance50": float(last['Resistance50']) if not math.isnan(last['Resistance50']) else None,
        "pattern": pattern,
        "signal": signal,
        "trend": "Uptrend" if (not pd.isna(last['MA50']) and not pd.isna(last['MA200']) and last['MA50'] > last['MA200']) else
                 "Downtrend" if (not pd.isna(last['MA50']) and not pd.isna(last['MA200']) and last['MA50'] < last['MA200']) else "Sideways",
        "plan": plan,
        "df": df
    }
    return info

# =============================
# Reporting
# =============================
def build_excel_report(results: list, out_path: str) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Summary
        rows = []
        for r in results:
            if "error" in r:
                rows.append({"Ticker": r["ticker"], "Status": f"ERROR: {r['error']}"})
                continue
            plan = r["plan"]
            rows.append({
                "Ticker": r["ticker"],
                "Last Time": r["time"],
                "Last Close": r["close"],
                "Signal": r["signal"],
                "Trend": r["trend"],
                "RSI14": r["rsi14"],
                "MACD": r["macd"],
                "MACD_Signal": r["macd_signal"],
                "MACD_Cross": r["macd_cross"],
                "Support20": r["support20"],
                "Resistance20": r["resistance20"],
                "Support50": r["support50"],
                "Resistance50": r["resistance50"],
                "CandlePattern": r["pattern"],
                "Bias": plan["Bias"],
                "Entry": plan["Entry"],
                "StopLoss": plan["StopLoss"],
                "TakeProfit": plan["TakeProfit"],
            })
        pd.DataFrame(rows).to_excel(writer, sheet_name="Summary", index=False)

        # Full data per ticker
        for r in results:
            name = r.get("ticker", "NA")[:31]
            df = r.get("df", pd.DataFrame())
            if df.empty:
                pd.DataFrame({"Status": [r.get("error", "No data")]}).to_excel(writer, sheet_name=name, index=False)
            else:
                df.to_excel(writer, sheet_name=name, index=False)

def send_telegram_text(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=30).raise_for_status()
    except Exception as e:
        print(f"[WARN] Telegram text failed: {e}")

def send_telegram_document(token: str, chat_id: str, file_path: str, caption: str = None) -> None:
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    try:
        with open(file_path, "rb") as f:
            files = {"document": f}
            data = {"chat_id": chat_id}
            if caption:
                data["caption"] = caption
            requests.post(url, files=files, data=data, timeout=120).raise_for_status()
    except Exception as e:
        print(f"[WARN] Telegram document failed: {e}")

# =============================
# Main
# =============================
def main():
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Missing TELEGRAM_TOKEN or CHAT_ID")

    # Default extended asset list (editable via ASSETS env var)
    default_assets = "GC=F,CL=F,BTC-USD,^GSPC,EURUSD=X,TASI.SR"
    tickers = os.environ.get("ASSETS", default_assets)
    tickers = [t.strip() for t in tickers.split(",") if t.strip()]

    results = []
    for t in tickers:
        try:
            results.append(analyze_ticker(t))
        except Exception as e:
            results.append({"ticker": t, "error": str(e), "df": pd.DataFrame()})

    # Build Excel
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_name = f"market_report_pro_{ts}.xlsx"
    out_path = os.path.join(os.getcwd(), out_name)
    build_excel_report(results, out_path)

    # Telegram summary (rich per asset)
    lines = ["<b>Daily Markets Report — Pro</b>"]
    for r in results:
        if "error" in r:
            lines.append(f"{r['ticker']}: ERROR — {r['error']}")
            continue
        plan = r["plan"]
        macd_cross = r["macd_cross"]
        rsi_txt = f"RSI={r['rsi14']:.1f}" if r['rsi14'] is not None else "RSI=NA"
        sr_txt = []
        if r['support20'] is not None and r['resistance20'] is not None:
            sr_txt.append(f"S20={r['support20']:.2f}/R20={r['resistance20']:.2f}")
        if r['support50'] is not None and r['resistance50'] is not None:
            sr_txt.append(f"S50={r['support50']:.2f}/R50={r['resistance50']:.2f}")
        plan_txt = f"{plan['Bias']}"
        if plan['Entry']:
            plan_txt += f" @ {plan['Entry']:.2f}"
        if plan['StopLoss']:
            plan_txt += f" | SL {plan['StopLoss']:.2f}"
        if plan['TakeProfit']:
            plan_txt += f" | TP {plan['TakeProfit']:.2f}"

        lines.append(
            f"{r['ticker']}: {r['signal']} | {rsi_txt} | MACD Cross {macd_cross} | "
            f"{' | '.join(sr_txt)} | {plan_txt}"
        )

    send_telegram_text(token, chat_id, "\n".join(lines))
    send_telegram_document(token, chat_id, out_path, caption="Pro Excel report (Summary + full sheets)")
    print("Done:", out_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()
        token = os.environ.get("TELEGRAM_TOKEN")
        chat_id = os.environ.get("CHAT_ID")
        if token and chat_id:
            send_telegram_text(token, chat_id, f"⚠️ Bot error: {e}")
