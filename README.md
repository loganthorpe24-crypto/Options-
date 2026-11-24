# QQQ 0DTE options bot (backtester + paper runner)

This is a Python implementation of the QQQ 0DTE options strategy described in an Instagram reel by
**@nickafischer (Nov 20, 2026)**. It uses Alpaca for market data and for **paper** trading.

> **Safety.** This is educational code that implements a creator's described rules. It is **not financial
> advice**. 0DTE options can, and often do, go to zero. The runner trades **paper only** unless you pass
> `--live`. Nothing in `config.yaml` can turn on live trading, and API keys are read **only** from
> environment variables.

---

## The strategy, exactly as the creator described it

| Rule | Value |
|---|---|
| Universe | QQQ options, same-day expiry (0DTE) only |
| Opening range | high/low of QQQ 9:30–9:45 ET (first 15 one-minute bars) |
| Noise band | for each minute after 9:30: mean absolute move from the open at that same minute over the last 20 trading days, × 1.25. Upper = open + band, lower = open − band, recomputed for the minute being checked |
| VWAP | session VWAP from 9:30, from 1-minute bars |
| ATR | 5-day ATR from daily bars: True Range, simple average of the last 5 days |
| Entry window | 10:00–14:00 ET, checked on every 10-minute close. First signal only |
| **Buy puts** | close < OR low − 0.3×ATR **and** close < lower band **and** close < VWAP |
| **Buy calls** | close > OR high + 0.3×ATR **and** close > upper band **and** close > VWAP |
| Trades/day | 1 max. No re-entry, win or lose |
| Hard exit | 15:45 ET at market, no exceptions |
| Trailing exit | arm at +4 ATR of favorable move from entry; once armed, exit when 50% of the peak favorable move is given back |

These live in `config.yaml` under `strategy:` and are tagged `[CREATOR]`. If you change one, the bot
logs a WARNING that you are no longer running his strategy.

## What the creator did *not* specify (assumptions)

Every gap is a named config variable tagged `[ASSUMPTION]` in `config.yaml`. All of them are logged at
every start (`python -m qqq0dte assumptions` prints them) and saved with each backtest.

| Config variable | Default | Meaning |
|---|---|---|
| `strike_selection` (STRIKE_SELECTION) | `atm` | listed 0DTE strike closest to QQQ at the signal close (`itm`/`otm` + `strike_offset` also available) |
| `atm_tie_break` | `lower` | which strike wins when QQQ is exactly halfway between two |
| `contracts` | `10` | position size (no sizing rule given; his example used 10) |
| `session_open_price` | `first_bar_open` | "the open" = open of the 9:30 one-minute bar |
| `noise_move_units` | `dollars` | `|price − open|` in $; `percent` = `|price/open − 1|` |
| `vwap_price` | `hlc3` | per-bar VWAP price: `(H+L+C)/3`, `bar_vwap` (Alpaca `vw`), or `close` |
| `atr_source` | `alpaca_daily` | Alpaca 1Day bars, or `rth_from_minutes` (daily OHLC from 9:30–16:00 bars) |
| `entry_window_end_inclusive` | `true` | the 14:00 check counts (25 checks: 10:00 … 14:00) |
| `trail_check_interval_minutes` | `1` | trailing stop evaluated on every QQQ 1-minute close |
| `fill_model` | `next_bar_open` | backtest fill = open of the option's first 1-min bar at/after the decision |
| `option_fill_max_staleness_minutes` | `5` | no option trade within 5 min → the trade is a **data error**, never a made-up price |
| `slippage_per_share`, `commission_per_contract` | `0` | backtest costs |
| `early_close_policy` | `skip` | half days (1pm close) can't hold a 3:45 exit, so they are skipped |
| `data.max_bar_gap_minutes` | `5` | if the newest QQQ bar at a check or trailing-stop minute is older than this, it's a data error (no stale prices) |
| `data.strike_grid_fallback` | `1.0` | if Alpaca lists no contracts for a past expiry, assume a $1 grid (the contract must still have real trade bars) |
| `runner.late_check_tolerance_seconds` | `120` | if the runner starts late and a missed check was the first signal, no trade |

Fixed behavior that is also not specified by the creator, and is logged too:
- The price "at" 10:30 is the close of the 10:29–10:30 bar.
- The noise band and the ATR use only completed days before the trade date.
- Entry and trailing exits are market orders.
- The 3:45 exit takes priority over the trailing stop.

### The trailing stop's measurement basis (the biggest ambiguity)

The creator says "+4 ATR" but not *what* is measured in ATR units. That choice is one constant at the
top of [`qqq0dte/exits.py`](qqq0dte/exits.py):

```python
TRAIL_MEASUREMENT_BASIS = "UNDERLYING_MOVE_OVER_ATR5"   # default, as specified
```

- `UNDERLYING_MOVE_OVER_ATR5` (default): favorable QQQ move since entry ÷ 5-day ATR, on 1-minute closes.
- `OPTION_PREMIUM_RETURN` (alternative, *not* the stated rule): `(option − entry premium) / entry premium`,
  so 4.0 = +400%.

Every trade records the basis, peak (value/time/QQQ price), whether and when it armed, the exit value, and
the retracement from peak. Heads-up: QQQ's 5-day ATR is typically several dollars, so +4 ATR is a very
large intraday move. Expect the trailing stop to arm rarely and most trades to exit at 15:45. The
backtest summary reports how many trades armed.

---

## Install

Python 3.10+.

```bash
git clone <this repo> && cd Options-
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -e ".[dev]"
```

## Set your Alpaca keys (environment variables only)

Create **paper** keys at <https://app.alpaca.markets>: switch to your Paper account, open *API Keys*, then
*Generate*.

```bash
export APCA_API_KEY_ID='your-paper-key-id'
export APCA_API_SECRET_KEY='your-paper-secret'
```

If either is missing, every command stops and names the missing variable. Paper and live accounts have
different keys.

**What your Alpaca plan needs to cover**
- Historical QQQ 1-minute and daily bars with `feed=sip`. Free plans can query SIP history older than 15
  minutes.
- Historical option bars (`/v1beta1/options/bars`). Alpaca's options history starts **Feb 2024**, so
  backtests before that fail loudly.
- The option contract list (`/v2/options/contracts`) for strikes, and `/v2/calendar` for trading days and
  half days.
- For the live/paper runner: **real-time** SIP stock data needs a paid plan. On the free plan set
  `data.live_stock_feed: iex`. IEX volume is a small slice of the tape, so VWAP and the opening range
  will differ from SIP backtests. The runner checks this at startup and tells you.

If any of these is unavailable, the code raises with the exact endpoint, symbol and date. It never
substitutes prices.

**Backtesting without the trading API.** The calendar and the strike list are the only things the
backtest takes from the trading API (`paper-api.alpaca.markets`). If you can only reach market data,
set `data.calendar_source: pandas_market_calendars` (NYSE calendar incl. half days; `pip install
pandas_market_calendars`) and `data.strike_source: grid` ($1 strikes; every chosen contract must
still have real Alpaca trade bars). Set `APCA_AUTH_MODE=proxy` only if an outbound proxy you control
adds the Alpaca credentials to each request. The process then holds no keys.

## Run the backtest

```bash
python -m qqq0dte backtest --start 2025-01-02 --end 2026-11-20
python -m qqq0dte backtest --start 2025-01-02 --end 2026-11-20 --offline   # re-run from the local cache only
```

Downloads are cached in `data_cache/`. Results go to `output/backtest_<start>_<end>/`:

| File | Contents |
|---|---|
| `trades.csv` | date, signal time, side, strike, option symbol, contracts, entry time/price, exit time/price, exit reason, P&L, P&L %, plus ATR, OR, bands, VWAP, fill bars, and trailing-stop peak / arming / retracement |
| `checks.csv` | every 10-minute check with all six conditions (3 put, 3 call) true/false and the levels |
| `days.csv` | each day: `traded`, `no_signal`, `skipped` (half day) or `data_error` (with the reason) |
| `summary.txt` / `summary.json` | trades, win rate, avg win/loss, largest win/loss, total P&L, max drawdown (on cumulative trade P&L), days with no trade (split into no-signal / half-day / data-error), exit reasons, trailing-armed count |
| `assumptions.txt` | every rule and assumption used for the run |

## Validation: the creator's Nov 20, 2026 trade

```bash
python -m qqq0dte validate                 # default date 2026-11-20
```

The creator's trade was: 10:30am, buy 10× $632 0DTE puts at $11.26 ($11,260), sell at 3:45pm at $25.76
($25,760), profit $14,500 (+128.8%). His caption's "$11,260 in, $14,500 out" is read as proceeds $25,760
and profit $14,500, per the video.

The command runs the normal backtest for that day with your config and prints:
1. The setup (open, opening range, ATR, put/call triggers) and **every 10-minute check** with each
   condition true/false.
2. A field-by-field comparison with the creator (side, signal time, strike, prices, P&L).
3. `SIGNAL CHECK: PASS/FAIL`. If it fails, it shows how far each condition was from flipping at 10:30.
4. A diagnostic pricing of the creator's exact contract `QQQ261120P00632000` from Alpaca at 10:30 and 15:45
   (both fill models), with QQQ's price and the contract's intrinsic value.
5. A sensitivity table re-running the day under each alternative assumption. It is diagnostic only and
   does not change your config.

Exit codes: `0` = put signal at 10:30, `1` = different/no signal, `3` = signal but option data missing,
`2` = configuration/credential/data error. **Nothing is tuned to force a match.**

Result on Alpaca data (SIP feed, default assumptions): the rules fire a **PUT at 11:10, not 10:30**.
At 10:30 QQQ was $632.23 (so $632 was the at-the-money strike), but the close was still $4.07 above
`OR low − 0.3×ATR` (the 5-day ATR was an unusually large $19.02), and $0.16 above the lower noise band.
Alpaca's tape shows the creator's $632 put trading around $4.48 at 10:30 and $22.44 at 15:45, not
$11.26 and $25.76. The backtest's own trade that day was 10× $628 puts, $4.35 → $18.25, +$13,900.

## Paper trading

```bash
python -m qqq0dte run --dry-run   # signals + logs only, NO orders (marks at the latest option trade)
python -m qqq0dte run             # PAPER orders to paper-api.alpaca.markets
python -m qqq0dte run --loop      # stay running, one session per trading day
python -m qqq0dte run --live      # REAL MONEY. Only with this explicit flag.
```

Start it before 9:45 ET. What it does:
- Before the session it loads the 20-day noise profile and the 5-day ATR.
- At 9:45 it builds the opening range, then evaluates each 10-minute check from 10:00 to 14:00.
- On the first signal it buys `contracts` of the 0DTE strike chosen by `strike_selection` (market order,
  paper).
- It then tracks the trailing stop every minute and closes at 15:45 at market.
- Every check is logged with all three conditions for both sides: console, `logs/run_<date>.log`,
  `logs/checks_<date>.csv`.
- Fills and P&L go to `logs/trades.csv`.
- State is saved per mode in `state/<date>_<paper|live|dryrun>.json`, so paper and live never mix (a
  paper run warns loudly if a live position from today is still open).
- A restart never re-enters: an entry order caught mid-submit is looked up by its client order id, and
  a first signal missed while the runner was down means no trade that day.
- After a restart mid-trade, the minutes missed while it was down are replayed from QQQ bars for the
  trailing stop.
- Once a position is open the runner does not give up. A data/API error skips that minute's
  trailing-stop check, and the exit sell is retried until the position is flat or the market closes.
  Before each retry it checks earlier exit orders and the live position, so it never sells more than
  is held.

Scheduling with cron (server clock in ET):

```cron
CRON_TZ=America/New_York
25 9 * * 1-5  cd /path/to/Options- && . .venv/bin/activate && APCA_API_KEY_ID=... APCA_API_SECRET_KEY=... python -m qqq0dte run >> logs/cron.log 2>&1
```

(Or keep it running with `run --loop`. Holidays and half days come from Alpaca's calendar.)

## Tests

```bash
pytest
```

The tests use **synthetic** prices (`tests/synthetic.py`) to check the rule logic, lookahead safety, fills,
the trailing stop, the runner's order flow and restart behavior, the cache, and that missing data errors
out instead of being faked. The backtester and runner never import that module. They only ever use
Alpaca data or the cache of it.

## Layout

```
config.yaml            creator rules + named assumptions
qqq0dte/
  exits.py             TRAIL_MEASUREMENT_BASIS constant + trailing stop
  indicators.py        opening range, noise band, VWAP, compute_atr() (the one ATR function)
  signals.py           the three entry conditions (shared by backtest and runner)
  backtest.py          day-by-day simulation with real option trade bars
  report.py            summary stats + output files
  validate.py          Nov 20, 2026 comparison
  runner.py            paper/live runner
  alpaca.py            REST client (paper by default; live needs --live)
  data.py              Alpaca data provider + cache
  options.py           OCC symbols, strike selection
  config.py            config loading + assumption registry
tests/                 unit tests (synthetic data only)
```
