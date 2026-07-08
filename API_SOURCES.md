# External API sources — Postman / curl reference

This document lists every **third-party HTTP API** used by the TradingAgents cron
pipeline for report appendices (FRED, Polymarket, StockTwits, Reddit) plus the
Yahoo Finance endpoints used indirectly via `yfinance`.

Use these URLs in **Postman**, **curl**, or your browser to verify connectivity
before debugging report content.

**Related code**

| Area | Path |
|------|------|
| FRED fetcher | `agent/tradingagents/dataflows/fred.py` |
| Polymarket fetcher | `agent/tradingagents/dataflows/polymarket.py` |
| StockTwits fetcher | `agent/tradingagents/dataflows/stocktwits.py` |
| Reddit fetcher | `agent/tradingagents/dataflows/reddit.py` |
| Prefetch bundle | `agent/tradingagents/agents/utils/prefetch_context.py` |
| Local smoke test | `agent/scripts/smoke_source_appendices.py` |
| Live API checks | `agent/scripts/run_api_checks.py` |
| Env template | `env.example` |

---

## Environment variables

| Variable | Required? | Purpose |
|----------|-----------|---------|
| `FRED_API_KEY` | **Yes** for real macro data | Free key from St. Louis Fed |
| `TRADINGAGENTS_VENDOR_MACRO_DATA` | Optional | Default vendor for macro (`fred`) |
| `TRADINGAGENTS_VENDOR_PREDICTION_MARKETS` | Optional | Default vendor for odds (`polymarket`) |
| `REDDIT_INTER_REQUEST_DELAY_SEC` | Optional | Pause between subreddit calls (default `2.5`) |
| `SOCIAL_FETCH_CACHE_TTL_SEC` | Optional | In-process cache TTL (default `300`) |

Cron loads env from `swing-trader/.env.local` and `swing-trader/.env` (see
`agent/write_recommendation_cache.py`).

---

## Required vs optional — full picture

### Cron / agent (must have for production runs)

| Variable | Required? | What happens if missing |
|----------|-----------|-------------------------|
| `DATABASE_URL` | **Required** | Cron cannot write recommendations |
| `RECOMMENDATION_TICKERS` | **Required** | Batch job has nothing to analyze |
| `Z_API_KEY` (or `GLM_API_KEY` / `ANTHROPIC_AUTH_TOKEN`) | **Required** | LLM agent does not run |
| `LLM_PROVIDER`, `LLM_BACKEND_URL`, `DEEP_THINK_LLM`, `QUICK_THINK_LLM` | **Required** | LLM client misconfigured |

### Report data sources (appendix APIs)

| Source | API key? | Required for cron? | If missing / fails |
|--------|----------|-------------------|---------------------|
| **Yahoo / yfinance** | No | **De facto required** (prices, news, fundamentals) | Agent run fails market-data gates |
| **FRED** | **Yes** (`FRED_API_KEY`) | **Optional** | Report shows `DATA_UNAVAILABLE`; news analyst uses Yahoo headlines only |
| **Polymarket** | No | **Optional** | Report section empty or “no markets matched” |
| **StockTwits** | No | **Optional** | Placeholder in report; **NSE `.NS` tickers not on StockTwits anyway** |
| **Reddit** | No | **Optional** | Placeholder; may 429 if rate-limited |

### Tuning (all optional)

| Variable | Default | Purpose |
|----------|---------|---------|
| `TRADINGAGENTS_VENDOR_MACRO_DATA` | `fred` | Macro vendor selection |
| `TRADINGAGENTS_VENDOR_PREDICTION_MARKETS` | `polymarket` | Prediction-market vendor |
| `REDDIT_INTER_REQUEST_DELAY_SEC` | `2.5` | Pause between subreddit RSS calls |
| `SOCIAL_FETCH_CACHE_TTL_SEC` | `300` | Cache Reddit/StockTwits in one batch run |
| `BATCH_TICKER_DELAY_SEC` | `30` | Pause between tickers in CircleCI batch |
| `DATA_VENDOR_*` | `yfinance` | Price/news/fundamentals vendor |

**Summary:** Only **FRED** needs a separate API key among the appendix sources. Polymarket, StockTwits, and Reddit are **keyless**. FRED is **optional** but recommended for real macro numbers in reports.

---

## Live API test results

**Run date:** 2026-07-08 (from this machine)  
**Script:** `agent/scripts/run_api_checks.py`  
**Env:** `FRED_API_KEY` **not set** in `swing-trader/.env.local` at test time  

| API | Auth | Result | Detail |
|-----|------|--------|--------|
| FRED CPI (`CPIAUCSL`) | API key | **SKIP** | `FRED_API_KEY` not configured locally |
| FRED unemployment (`UNRATE`) | API key | **SKIP** | same |
| FRED fed funds (`FEDFUNDS`) | API key | **SKIP** | same |
| FRED 10Y Treasury (`DGS10`) | API key | **SKIP** | same |
| Polymarket `Fed rate cut` | None | **PASS** | HTTP 200, 16 events |
| Polymarket `recession 2026` | None | **PASS** | HTTP 200, 3 events |
| Polymarket `btc` | None | **PASS** | HTTP 200, 20 events |
| StockTwits `AAPL` | None | **PASS** | HTTP 200, 30 messages |
| StockTwits `BTC.X` | None | **PASS** | HTTP 200, 30 messages |
| StockTwits `SUNPHARMA.NS` | None | **FAIL (expected)** | HTTP 404 — NSE tickers not listed on StockTwits |
| Reddit `r/IndiaInvestments` + `SUNPHARMA` | User-Agent only | **PASS** | HTTP 200, Atom/XML RSS |
| Reddit `r/stocks` + `AAPL` | User-Agent only | **FAIL** | HTTP 429 — IP rate-limited from earlier tests; retry later |
| Yahoo news `SUNPHARMA.NS` (raw curl URL) | None | **FAIL** | HTTP 500 — unofficial endpoint; **app uses `yfinance` in Python instead** |
| Yahoo news `BTC-USD` (raw curl URL) | None | **FAIL** | HTTP 500 — same; use yfinance for real news |

### Re-run these checks yourself

```bash
cd trade-circleci-cron/agent
python scripts/run_api_checks.py
```

After adding FRED key to `swing-trader/.env.local`:

```env
FRED_API_KEY=your_key_here
```

re-run the script — all four FRED rows should show **PASS** with `observations=…` in the detail column.

---

## 1. FRED — US macro (CPI, unemployment, rates, yields)

### Official docs & websites

| Resource | URL |
|----------|-----|
| FRED home | https://fred.stlouisfed.org/ |
| API overview | https://fred.stlouisfed.org/docs/api/fred/ |
| Get API key | https://fred.stlouisfed.org/docs/api/api_key.html |
| API base | https://api.stlouisfed.org/fred |
| Browse series (CPI) | https://fred.stlouisfed.org/series/CPIAUCSL |
| Browse unemployment | https://fred.stlouisfed.org/series/UNRATE |
| Browse fed funds | https://fred.stlouisfed.org/series/FEDFUNDS |
| Browse 10Y yield | https://fred.stlouisfed.org/series/DGS10 |

### Series IDs used by this project

| App alias | FRED `series_id` | Description |
|-----------|------------------|-------------|
| `cpi` | `CPIAUCSL` | Consumer Price Index |
| `unemployment` | `UNRATE` | Unemployment rate |
| `fed_funds_rate` | `FEDFUNDS` | Federal funds rate |
| `10y_treasury` | `DGS10` | 10-year Treasury yield |

### A) Series metadata

**Method:** `GET`

```
https://api.stlouisfed.org/fred/series?series_id=CPIAUCSL&api_key=YOUR_FRED_API_KEY&file_type=json
```

**curl**

```bash
curl -G "https://api.stlouisfed.org/fred/series" \
  --data-urlencode "series_id=CPIAUCSL" \
  --data-urlencode "api_key=YOUR_FRED_API_KEY" \
  --data-urlencode "file_type=json"
```

**PowerShell**

```powershell
curl.exe -G "https://api.stlouisfed.org/fred/series" `
  --data-urlencode "series_id=CPIAUCSL" `
  --data-urlencode "api_key=YOUR_FRED_API_KEY" `
  --data-urlencode "file_type=json"
```

### B) Series observations (what the agent renders in reports)

**Method:** `GET`

```
https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&observation_start=2026-01-01&observation_end=2026-07-08&sort_order=asc&api_key=YOUR_FRED_API_KEY&file_type=json
```

**curl — CPI**

```bash
curl -G "https://api.stlouisfed.org/fred/series/observations" \
  --data-urlencode "series_id=CPIAUCSL" \
  --data-urlencode "observation_start=2026-01-01" \
  --data-urlencode "observation_end=2026-07-08" \
  --data-urlencode "sort_order=asc" \
  --data-urlencode "api_key=YOUR_FRED_API_KEY" \
  --data-urlencode "file_type=json"
```

**curl — unemployment**

```bash
curl -G "https://api.stlouisfed.org/fred/series/observations" \
  --data-urlencode "series_id=UNRATE" \
  --data-urlencode "observation_start=2026-01-01" \
  --data-urlencode "observation_end=2026-07-08" \
  --data-urlencode "sort_order=asc" \
  --data-urlencode "api_key=YOUR_FRED_API_KEY" \
  --data-urlencode "file_type=json"
```

**curl — fed funds**

```bash
curl -G "https://api.stlouisfed.org/fred/series/observations" \
  --data-urlencode "series_id=FEDFUNDS" \
  --data-urlencode "observation_start=2026-01-01" \
  --data-urlencode "observation_end=2026-07-08" \
  --data-urlencode "sort_order=asc" \
  --data-urlencode "api_key=YOUR_FRED_API_KEY" \
  --data-urlencode "file_type=json"
```

**curl — 10-year Treasury**

```bash
curl -G "https://api.stlouisfed.org/fred/series/observations" \
  --data-urlencode "series_id=DGS10" \
  --data-urlencode "observation_start=2026-01-01" \
  --data-urlencode "observation_end=2026-07-08" \
  --data-urlencode "sort_order=asc" \
  --data-urlencode "api_key=YOUR_FRED_API_KEY" \
  --data-urlencode "file_type=json"
```

### Expected success

- HTTP `200`
- JSON with `observations` array; each row has `date` and `value`

### Expected failure (no key)

If `FRED_API_KEY` is unset, the app writes:

`DATA_UNAVAILABLE: optional macro_data could not be retrieved (FRED_API_KEY environment variable is not set...)`

---

## 2. Polymarket — prediction-market odds

### Official docs & websites

| Resource | URL |
|----------|-----|
| Polymarket | https://polymarket.com/ |
| Gamma API (public, no key) | https://gamma-api.polymarket.com |
| Gamma API docs | https://docs.polymarket.com/developers/gamma-markets-api/overview |

### Search markets by topic

**Method:** `GET`

```
https://gamma-api.polymarket.com/public-search?q=Fed%20rate%20cut&limit_per_type=20
```

**curl — Fed rate cut**

```bash
curl -G "https://gamma-api.polymarket.com/public-search" \
  --data-urlencode "q=Fed rate cut" \
  --data-urlencode "limit_per_type=20"
```

**curl — recession 2026**

```bash
curl -G "https://gamma-api.polymarket.com/public-search" \
  --data-urlencode "q=recession 2026" \
  --data-urlencode "limit_per_type=20"
```

**curl — BTC (crypto tickers)**

```bash
curl -G "https://gamma-api.polymarket.com/public-search" \
  --data-urlencode "q=btc" \
  --data-urlencode "limit_per_type=20"
```

### Expected success

- HTTP `200`
- JSON with `events[]` → each event has `markets[]`
- Each market has `outcomePrices`, `outcomes`, `volumeNum`, `endDate`

### Auth

None required.

---

## 3. StockTwits — retail sentiment stream

### Official docs & websites

| Resource | URL |
|----------|-----|
| StockTwits | https://stocktwits.com/ |
| API docs | https://api.stocktwits.com/developers/docs |
| Symbol streams | https://api.stocktwits.com/developers/docs/api/streams/symbol |

### Symbol message stream

**Method:** `GET`

```
https://api.stocktwits.com/api/2/streams/symbol/{SYMBOL}.json
```

### curl examples

**US stock — AAPL (should work)**

```bash
curl -s "https://api.stocktwits.com/api/2/streams/symbol/AAPL.json" \
  -H "User-Agent: tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)" \
  -H "Accept: application/json"
```

**Crypto — BTC (use `.X` suffix, not `BTC-USD`)**

```bash
curl -s "https://api.stocktwits.com/api/2/streams/symbol/BTC.X.json" \
  -H "User-Agent: tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)" \
  -H "Accept: application/json"
```

**Indian NSE — SUNPHARMA.NS (expected 404 — not supported)**

```bash
curl -s -o /dev/null -w "%{http_code}\n" \
  "https://api.stocktwits.com/api/2/streams/symbol/SUNPHARMA.NS.json" \
  -H "User-Agent: tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)" \
  -H "Accept: application/json"
```

On Windows PowerShell:

```powershell
curl.exe -s "https://api.stocktwits.com/api/2/streams/symbol/AAPL.json" `
  -H "User-Agent: tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)" `
  -H "Accept: application/json"
```

### Expected success

- HTTP `200`
- JSON root key `messages` (array of posts with `body`, `user`, `entities.sentiment`)

### Limitation

StockTwits primarily covers **US-listed** symbols and **crypto** (`BTC.X`).
**NSE/BSE tickers** (`.NS`, `.BO`) typically return **404**. The app documents
this in the report and uses Reddit + Yahoo news for Indian equities instead.

### Auth

None required.

---

## 4. Reddit — community discussion (RSS feed)

### Official docs & websites

| Resource | URL |
|----------|-----|
| Reddit | https://www.reddit.com/ |
| API rules / etiquette | https://github.com/reddit-archive/reddit/wiki/API |
| OAuth API docs | https://www.reddit.com/dev/api/ |

This project uses the **public RSS search feed** (no OAuth). The JSON search
endpoint (`/search.json`) is often blocked (`403`) for anonymous clients.

### RSS search (what the agent uses)

**Method:** `GET`

```
https://www.reddit.com/r/{subreddit}/search.rss?q={query}&restrict_sr=on&sort=new&t=week&limit=5
```

**Required header**

```
User-Agent: tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)
```

### Subreddits by market

| Market | Subreddits |
|--------|------------|
| US equities | `wallstreetbets`, `stocks`, `investing` |
| Indian NSE/BSE (`.NS`, `.BO`) | `IndiaInvestments`, `IndianStreetBets`, `dalalstreet` |

### Search term rules

| Input ticker | Reddit search `q` |
|--------------|-------------------|
| `SUNPHARMA.NS` | `SUNPHARMA` |
| `BTC-USD` | `BTC` |
| `AAPL` | `AAPL` |

### curl — Indian stock (IndiaInvestments)

```bash
curl -s "https://www.reddit.com/r/IndiaInvestments/search.rss?q=SUNPHARMA&restrict_sr=on&sort=new&t=week&limit=5" \
  -H "User-Agent: tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
```

### curl — US stock (stocks)

```bash
curl -s "https://www.reddit.com/r/stocks/search.rss?q=AAPL&restrict_sr=on&sort=new&t=week&limit=5" \
  -H "User-Agent: tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
```

### curl — crypto (wallstreetbets)

```bash
curl -s "https://www.reddit.com/r/wallstreetbets/search.rss?q=BTC&restrict_sr=on&sort=new&t=week&limit=5" \
  -H "User-Agent: tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
```

### Expected success

- HTTP `200`
- Atom/XML feed (`<feed>`, `<entry>` elements)

### Rate limits

- Too many rapid requests → HTTP **429**
- Wait **15–30 minutes** and retry
- In production, set `REDDIT_INTER_REQUEST_DELAY_SEC=2.5` (or higher for batch runs)

### Auth

None required for RSS (public feed).

---

## 5. Yahoo Finance — news & prices (via yfinance)

The Python agent uses the **`yfinance`** library, not a single fixed REST URL.
These unofficial Yahoo endpoints are useful for manual Postman/curl checks.

### Websites

| Resource | URL |
|----------|-----|
| Yahoo Finance | https://finance.yahoo.com/ |
| Example quote page | https://finance.yahoo.com/quote/SUNPHARMA.NS |
| yfinance project | https://github.com/ranaroussi/yfinance |

### Ticker news (approximate)

```bash
curl -s "https://query2.finance.yahoo.com/v2/finance/news?symbols=SUNPHARMA.NS"
```

```bash
curl -s "https://query2.finance.yahoo.com/v2/finance/news?symbols=BTC-USD"
```

### Symbol search (approximate)

```bash
curl -s "https://query1.finance.yahoo.com/v1/finance/search?q=SUNPHARMA.NS"
```

**Note:** Yahoo endpoints are undocumented and may change. Prefer FRED /
Polymarket / StockTwits / Reddit docs above for stable public APIs.

---

## 6. Quick verification matrix

| Source | Auth | Test symbol | Expect |
|--------|------|-------------|--------|
| FRED | API key | `CPIAUCSL` | `200` + observations |
| Polymarket | None | `Fed rate cut` | `200` + events |
| StockTwits | None | `AAPL` | `200` + messages |
| StockTwits | None | `BTC.X` | `200` + messages |
| StockTwits | None | `SUNPHARMA.NS` | `404` (normal) |
| Reddit RSS | User-Agent | `SUNPHARMA` in `IndiaInvestments` | `200` + XML |
| Yahoo news | None | `SUNPHARMA.NS` | Raw curl URL may 500; **use yfinance in app** |

---

## 7. Local smoke tests (no Postman)

From `trade-circleci-cron/agent`:

**A) Live API connectivity (this file’s curl equivalents)**

```bash
python scripts/run_api_checks.py
```

**B) Report appendix assembly (no LLM)**

```bash
python scripts/smoke_source_appendices.py SUNPHARMA.NS
```

Writes a sample report to:

`agent/tradingagents/reports/_smoke_SUNPHARMA.NS.md`

---

## 8. Import into Postman

1. Open Postman → **Import** → **Raw text**
2. Paste any curl block from this file
3. Postman will create the request with URL, query params, and headers
4. For FRED, replace `YOUR_FRED_API_KEY` in params or use a Postman environment variable `fred_api_key`

**Suggested Postman environment variables**

| Variable | Example |
|----------|---------|
| `fred_api_key` | your FRED key |
| `ticker` | `SUNPHARMA.NS` |
| `reddit_query` | `SUNPHARMA` |
| `trade_date` | `2026-07-08` |

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| FRED `DATA_UNAVAILABLE` | `FRED_API_KEY` not set | Add key to `swing-trader/.env.local` + CircleCI |
| StockTwits 404 for `.NS` | Platform does not list NSE tickers | Expected; use Reddit + Yahoo |
| Reddit 429 | IP rate-limited | Wait 15–30 min; increase `REDDIT_INTER_REQUEST_DELAY_SEC` |
| Polymarket empty `events` | No open markets for query | Try broader topic (`btc`, `Fed rate cut`) |
| Yahoo curl fails | Endpoint changed | Use yfinance in Python or Yahoo quote page in browser |
