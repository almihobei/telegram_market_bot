
# Telegram Daily Markets Bot — Pro (9:30 Asia/Riyadh)

نسخة موسّعة بتقرير أكثر تفصيلاً: إشارات، دعم/مقاومة 20/50، MACD Cross، توصية دخول/خروج، وقف خسارة وهدف.

## ما الجديد؟
- **ملخص غني في Excel**: الأعمدة تشمل Signal, Trend, RSI, MACD, MACD_Cross, S/R20/50, CandlePattern, Bias, Entry, StopLoss, TakeProfit.
- **رسالة تيليجرام تفصيلية لكل أصل** بسطر واضح وجاهز للتنفيذ.
- **خطة تداول آلية (اختيارية)**:
  - Long إذا السعر > MA50 و RSI>55 و MACD>Signal.
  - Short إذا السعر < MA50 و RSI<45 و MACD<Signal.
  - SL عند أقرب دعم/مقاومة، TP عند أقرب مقاومة/دعم مقابلة.

## الأصول الافتراضية
`GC=F, CL=F, BTC-USD, ^GSPC, EURUSD=X, TASI.SR`
> يمكنك تعديلها عبر متغير البيئة `ASSETS` في ورشة العمل.

## الهيكل
```
your-repo/
├─ main.py
├─ requirements.txt
├─ README.md
└─ .github/workflows/
   └─ daily_bot.yml
```

## ملاحظات
- رموز السعودية عبر Yahoo قد تكون `TASI.SR` وليس `^TASI`. إذا لم تُجلب بيانات سيظهر تحذير في الملخص.
- المؤشرات تعليمية وليست نصيحة استثمارية.
- التوقيت مضبوط: 6:30 UTC = 9:30 Asia/Riyadh.
