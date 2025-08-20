import requests
import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# قراءة التوكن و chat_id من متغيرات البيئة
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    response = requests.post(url, data=payload)
    return response.json()

def fetch_market_data(symbol, period="1d", interval="5m"):
    df = yf.download(symbol, period=period, interval=interval)
    return df

def create_excel_report(data_dict):
    filename = f"Market_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    writer = pd.ExcelWriter(filename, engine="xlsxwriter")
    for name, df in data_dict.items():
        df.to_excel(writer, sheet_name=name)
    writer.save()
    return filename

def main():
    if not TELEGRAM_TOKEN or not CHAT_ID:
        raise RuntimeError("⚠️ اضبط TELEGRAM_TOKEN و CHAT_ID عبر GitHub Secrets")

    send_telegram_message("📈 بوت التداول يعمل الآن ✅")

    symbols = {
        "الذهب": "XAUUSD=X",
        "النفط": "CL=F",
        "البيتكوين": "BTC-USD",
        "S&P 500": "^GSPC",
        "تداول السعودية": "^TASI"
    }

    market_data = {}
    for name, symbol in symbols.items():
        df = fetch_market_data(symbol)
        market_data[name] = df.tail(50)

    report_path = create_excel_report(market_data)

    send_telegram_message(f"✅ تم إنشاء تقرير السوق: {report_path}")

if __name__ == "__main__":
    main()
