import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Priyanka@ F&O Strong Trend Scanner",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("📊 Priyanka@ F&O Strong Trend Scanner")
st.caption("2M • 5M • 15M | Strong Bullish / Strong Bearish | Sideways Filter")

# ============================================================
# HARD-CODED F&O STOCK UNIVERSE
# NSE F&O UNDERLYINGS
# ============================================================

FNO_STOCKS = [
    "AARTIIND", "ABB", "ABBOTINDIA", "ACC",
    "ADANIENT", "ADANIPORTS", "ABCAPITAL", "ABFRL",
    "ALKEM", "AMBUJACEM", "APOLLOHOSP", "APOLLOTYRE",
    "ASHOKLEY", "ASIANPAINT", "ASTRAL", "ATUL",
    "AUBANK", "AUROPHARMA", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV",
    "BALKRISIND", "BALRAMCHIN", "BANDHANBNK",
    "BANKBARODA", "BATAINDIA", "BERGEPAINT",
    "BEL", "BHARATFORG", "BHEL", "BPCL",
    "BHARTIARTL", "BIOCON", "BSOFT", "BOSCHLTD",
    "BRITANNIA", "CANFINHOME", "CANBK",
    "CHAMBLFERT", "CHOLAFIN", "CIPLA", "CUB",
    "COALINDIA", "COFORGE", "COLPAL", "CONCOR",
    "COROMANDEL", "CROMPTON", "CUMMINSIND",
    "DABUR", "DALBHARAT", "DEEPAKNTR", "DELTACORP",
    "DIVISLAB", "DIXON", "DLF", "LALPATHLAB",
    "DRREDDY", "EICHERMOT", "ESCORTS", "EXIDEIND",
    "GAIL", "GLENMARK", "GMRINFRA", "GODREJCP",
    "GODREJPROP", "GRANULES", "GRASIM", "GUJGASLTD",
    "GNFC", "HAVELLS", "HCLTECH", "HDFCAMC",
    "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HAL", "HINDCOPPER", "HINDPETRO",
    "HINDUNILVR", "ICICIBANK", "ICICIGI",
    "ICICIPRULI", "IDFCFIRSTB", "IBULHSGFIN",
    "INDIAMART", "IEX", "IOC", "IRCTC", "IGL",
    "INDUSTOWER", "INDUSINDBK", "NAUKRI", "INFY",
    "INTELLECT", "INDIGO", "IPCALAB", "ITC",
    "JINDALSTEL", "JKCEMENT", "JSWSTEEL",
    "JUBLFOOD", "KOTAKBANK", "L&TFH", "LTTS",
    "LTIM", "LT", "LAURUSLABS", "LICHSGFIN",
    "LUPIN", "MGL", "M&MFIN", "M&M",
    "MANAPPURAM", "MARICO", "MARUTI", "MFSL",
    "METROPOLIS", "MOTHERSON", "MPHASIS", "MRF",
    "MUTHOOTFIN", "NATIONALUM", "NAVINFLUOR",
    "NESTLEIND", "NMDC", "NTPC", "OBEROIRLTY",
    "ONGC", "OFSS", "PAGEIND", "PERSISTENT",
    "PETRONET", "PIIND", "PIDILITIND", "PEL",
    "POLYCAB", "PFC", "POWERGRID", "PNB",
    "PVRINOX", "RAIN", "RBLBANK", "RECLTD",
    "RELIANCE", "SBICARD", "SBILIFE", "SHREECEM",
    "SHRIRAMFIN", "SIEMENS", "SRF", "SBIN",
    "SAIL", "SUNPHARMA", "SUNTV", "SYNGENE",
    "TATACHEM", "TATACOMM", "TCS", "TATACONSUM",
    "TATAMOTORS", "TATAPOWER", "TATASTEEL",
    "TECHM", "FEDERALBNK", "INDIACEM", "INDHOTEL",
    "RAMCOCEM", "TITAN", "TORNTPHARM", "TRENT",
    "TVSMOTOR", "ULTRACEMCO", "UBL", "MCDOWELL-N",
    "UPL", "VEDL", "IDEA", "VOLTAS", "WHIRLPOOL",
    "WIPRO", "ZEEL", "ZYDUSLIFE"
]

# ============================================================
# SETTINGS
# ============================================================

st.sidebar.header("⚙️ Scanner Settings")

ATR_LEN = st.sidebar.number_input(
    "ATR Length",
    min_value=5,
    max_value=50,
    value=14
)

RSI_LEN = st.sidebar.number_input(
    "RSI Length",
    min_value=5,
    max_value=50,
    value=14
)

MIN_DISTANCE_ATR = st.sidebar.number_input(
    "Minimum EMA-MA Distance × ATR",
    min_value=0.05,
    max_value=1.00,
    value=0.10,
    step=0.05
)

BULL_RSI = st.sidebar.number_input(
    "Bullish RSI",
    min_value=50,
    max_value=70,
    value=55
)

BEAR_RSI = st.sidebar.number_input(
    "Bearish RSI",
    min_value=30,
    max_value=50,
    value=45
)

MIN_BODY_ATR = st.sidebar.number_input(
    "Minimum Candle Body × ATR",
    min_value=0.05,
    max_value=1.00,
    value=0.20,
    step=0.05
)

# Strong confirmation:
# 3/3 timeframes must agree
REQUIRE_ALL_TF = st.sidebar.checkbox(
    "Require 2M + 5M + 15M confirmation",
    value=True
)

# ============================================================
# FUNCTIONS
# ============================================================

def calculate_rsi(series, length=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / length,
        min_periods=length,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / length,
        min_periods=length,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_atr(df, length=14):

    high_low = df["High"] - df["Low"]

    high_close = (
        df["High"] - df["Close"].shift(1)
    ).abs()

    low_close = (
        df["Low"] - df["Close"].shift(1)
    ).abs()

    tr = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / length,
        min_periods=length,
        adjust=False
    ).mean()


def prepare_dataframe(df):

    if df is None or df.empty:
        return None

    df = df.copy()

    # Handle MultiIndex returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):

        df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in df.columns
        ]

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for col in required:
        if col not in df.columns:
            return None

    df = df[required].copy()

    df.dropna(
        subset=["Open", "High", "Low", "Close"],
        inplace=True
    )

    if len(df) < 50:
        return None

    # ========================================================
    # EMA / MA
    # ========================================================

    df["EMA10"] = df["Close"].ewm(
        span=10,
        adjust=False
    ).mean()

    df["MA10"] = df["Close"].rolling(
        10
    ).mean()

    # ========================================================
    # ATR
    # ========================================================

    df["ATR14"] = calculate_atr(
        df,
        ATR_LEN
    )

    # ========================================================
    # RSI
    # ========================================================

    df["RSI14"] = calculate_rsi(
        df["Close"],
        RSI_LEN
    )

    # ========================================================
    # EMA-MA DISTANCE
    # ========================================================

    df["EMA_MA_DISTANCE"] = (
        df["EMA10"] - df["MA10"]
    ).abs()

    df["DIST_ATR_RATIO"] = (
        df["EMA_MA_DISTANCE"] /
        df["ATR14"].replace(0, np.nan)
    )

    # ========================================================
    # EMA SLOPE
    # ========================================================

    df["EMA_SLOPE"] = (
        df["EMA10"] - df["EMA10"].shift(1)
    )

    # ========================================================
    # CANDLE BODY
    # ========================================================

    df["BODY"] = (
        df["Close"] - df["Open"]
    ).abs()

    df["BODY_ATR_RATIO"] = (
        df["BODY"] /
        df["ATR14"].replace(0, np.nan)
    )

    # ========================================================
    # CROSS
    # ========================================================

    df["BULL_CROSS"] = (
        (df["EMA10"] > df["MA10"]) &
        (df["EMA10"].shift(1) <= df["MA10"].shift(1))
    )

    df["BEAR_CROSS"] = (
        (df["EMA10"] < df["MA10"]) &
        (df["EMA10"].shift(1) >= df["MA10"].shift(1))
    )

    return df


# ============================================================
# TIMEFRAME ANALYSIS
# ============================================================

def analyze_timeframe(df, timeframe):

    df = prepare_dataframe(df)

    if df is None or len(df) < 30:
        return None

    row = df.iloc[-1]

    close = float(row["Close"])
    ema = float(row["EMA10"])
    ma = float(row["MA10"])
    atr = float(row["ATR14"])
    rsi = float(row["RSI14"])
    distance_ratio = float(row["DIST_ATR_RATIO"])
    body_ratio = float(row["BODY_ATR_RATIO"])
    slope = float(row["EMA_SLOPE"])

    if any(
        pd.isna(x)
        for x in [
            close,
            ema,
            ma,
            atr,
            rsi,
            distance_ratio,
            body_ratio,
            slope
        ]
    ):
        return None

    # ========================================================
    # CORE CONDITIONS
    # ========================================================

    bullish_ema = ema > ma
    bearish_ema = ema < ma

    bullish_slope = slope > 0
    bearish_slope = slope < 0

    bullish_price = close > ema
    bearish_price = close < ema

    strong_distance = (
        distance_ratio >= MIN_DISTANCE_ATR
    )

    strong_body = (
        body_ratio >= MIN_BODY_ATR
    )

    bullish_candle = (
        row["Close"] > row["Open"]
    )

    bearish_candle = (
        row["Close"] < row["Open"]
    )

    bullish_rsi = rsi > BULL_RSI
    bearish_rsi = rsi < BEAR_RSI

    # ========================================================
    # SIDEWAYS FILTER
    # ========================================================

    # EMA and MA too close
    sideways_distance = (
        distance_ratio < MIN_DISTANCE_ATR
    )

    # EMA practically flat compared with ATR
    slope_ratio = abs(slope) / atr

    sideways_slope = (
        slope_ratio < 0.02
    )

    sideways = (
        sideways_distance and sideways_slope
    )

    # ========================================================
    # BULLISH SCORE
    # ========================================================

    bull_score = 0

    bull_score += int(bullish_ema)
    bull_score += int(bullish_slope)
    bull_score += int(bullish_price)
    bull_score += int(strong_distance)
    bull_score += int(bullish_rsi)
    bull_score += int(bullish_candle)
    bull_score += int(strong_body)

    # ========================================================
    # BEARISH SCORE
    # ========================================================

    bear_score = 0

    bear_score += int(bearish_ema)
    bear_score += int(bearish_slope)
    bear_score += int(bearish_price)
    bear_score += int(strong_distance)
    bear_score += int(bearish_rsi)
    bear_score += int(bearish_candle)
    bear_score += int(strong_body)

    # ========================================================
    # FINAL TF SIGNAL
    # ========================================================

    signal = "SIDEWAYS"

    if not sideways:

        if bull_score >= 6:
            signal = "BULLISH"

        elif bear_score >= 6:
            signal = "BEARISH"

    # ========================================================
    # ENTRY / SL / TARGET
    # ========================================================

    entry = close

    if signal == "BULLISH":

        recent_low = float(
            df["Low"].tail(5).min()
        )

        sl = recent_low

        risk = entry - sl

        if risk > 0:
            t1 = entry + risk
            t2 = entry + risk * 2
            t3 = entry + risk * 3
        else:
            t1 = np.nan
            t2 = np.nan
            t3 = np.nan

    elif signal == "BEARISH":

        recent_high = float(
            df["High"].tail(5).max()
        )

        sl = recent_high

        risk = sl - entry

        if risk > 0:
            t1 = entry - risk
            t2 = entry - risk * 2
            t3 = entry - risk * 3
        else:
            t1 = np.nan
            t2 = np.nan
            t3 = np.nan

    else:

        sl = np.nan
        t1 = np.nan
        t2 = np.nan
        t3 = np.nan

    return {
        "TF": timeframe,
        "Signal": signal,
        "Close": close,
        "EMA10": ema,
        "MA10": ma,
        "RSI": rsi,
        "ATR": atr,
        "Distance_ATR": distance_ratio,
        "Slope_ATR": slope_ratio,
        "Body_ATR": body_ratio,
        "BullScore": bull_score,
        "BearScore": bear_score,
        "Entry": entry,
        "SL": sl,
        "T1": t1,
        "T2": t2,
        "T3": t3
    }


# ============================================================
# DOWNLOAD DATA
# ============================================================

@st.cache_data(ttl=120, show_spinner=False)
def download_market_data(symbols):

    result = {
        "2m": {},
        "5m": {},
        "15m": {}
    }

    ticker_list = [
        f"{symbol}.NS"
        for symbol in symbols
    ]

    # --------------------------------------------------------
    # 2 MINUTE
    # --------------------------------------------------------

    try:

        data_2m = yf.download(
            ticker_list,
            period="60d",
            interval="2m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )

        for symbol in symbols:

            ticker = f"{symbol}.NS"

            try:

                if isinstance(
                    data_2m.columns,
                    pd.MultiIndex
                ):

                    if ticker in data_2m.columns.get_level_values(0):

                        result["2m"][symbol] = (
                            data_2m[ticker]
                            .dropna(how="all")
                        )

            except Exception:
                pass

    except Exception:
        pass

    # --------------------------------------------------------
    # 5 MINUTE
    # --------------------------------------------------------

    try:

        data_5m = yf.download(
            ticker_list,
            period="60d",
            interval="5m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )

        for symbol in symbols:

            ticker = f"{symbol}.NS"

            try:

                if isinstance(
                    data_5m.columns,
                    pd.MultiIndex
                ):

                    if ticker in data_5m.columns.get_level_values(0):

                        result["5m"][symbol] = (
                            data_5m[ticker]
                            .dropna(how="all")
                        )

            except Exception:
                pass

    except Exception:
        pass

    # --------------------------------------------------------
    # 15 MINUTE
    # --------------------------------------------------------

    try:

        data_15m = yf.download(
            ticker_list,
            period="60d",
            interval="15m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )

        for symbol in symbols:

            ticker = f"{symbol}.NS"

            try:

                if isinstance(
                    data_15m.columns,
                    pd.MultiIndex
                ):

                    if ticker in data_15m.columns.get_level_values(0):

                        result["15m"][symbol] = (
                            data_15m[ticker]
                            .dropna(how="all")
                        )

            except Exception:
                pass

    except Exception:
        pass

    return result


# ============================================================
# SCAN ONE STOCK
# ============================================================

def scan_stock(symbol, market_data):

    analyses = {}

    for tf in ["2m", "5m", "15m"]:

        df = market_data[tf].get(symbol)

        if df is None:
            return None

        result = analyze_timeframe(
            df,
            tf
        )

        if result is None:
            return None

        analyses[tf] = result

    s2 = analyses["2m"]
    s5 = analyses["5m"]
    s15 = analyses["15m"]

    signals = [
        s2["Signal"],
        s5["Signal"],
        s15["Signal"]
    ]

    # ========================================================
    # ALL TIMEFRAME AGREEMENT
    # ========================================================

    bullish_all = all(
        x == "BULLISH"
        for x in signals
    )

    bearish_all = all(
        x == "BEARISH"
        for x in signals
    )

    # ========================================================
    # FINAL SIGNAL
    # ========================================================

    if bullish_all:

        final_signal = "STRONG BULLISH"

    elif bearish_all:

        final_signal = "STRONG BEARISH"

    else:

        return None

    # ========================================================
    # COMBINED STRENGTH
    # ========================================================

    if final_signal == "STRONG BULLISH":

        avg_rsi = np.mean([
            s2["RSI"],
            s5["RSI"],
            s15["RSI"]
        ])

        avg_distance = np.mean([
            s2["Distance_ATR"],
            s5["Distance_ATR"],
            s15["Distance_ATR"]
        ])

        strength_score = (
            min(avg_rsi / 70 * 50, 50)
            +
            min(avg_distance / 0.50 * 50, 50)
        )

    else:

        avg_rsi = np.mean([
            s2["RSI"],
            s5["RSI"],
            s15["RSI"]
        ])

        avg_distance = np.mean([
            s2["Distance_ATR"],
            s5["Distance_ATR"],
            s15["Distance_ATR"]
        ])

        strength_score = (
            min((100 - avg_rsi) / 70 * 50, 50)
            +
            min(avg_distance / 0.50 * 50, 50)
        )

    # ========================================================
    # ENTRY / SL / TARGET FROM 5M
    # ========================================================

    base = s5

    return {
        "Stock": symbol,
        "Signal": final_signal,

        "2M": s2["Signal"],
        "5M": s5["Signal"],
        "15M": s15["Signal"],

        "RSI 2M": s2["RSI"],
        "RSI 5M": s5["RSI"],
        "RSI 15M": s15["RSI"],

        "Distance 2M": s2["Distance_ATR"],
        "Distance 5M": s5["Distance_ATR"],
        "Distance 15M": s15["Distance_ATR"],

        "Entry": base["Entry"],
        "SL": base["SL"],
        "T1": base["T1"],
        "T2": base["T2"],
        "T3": base["T3"],

        "Strength": round(
            strength_score,
            1
        )
    }


# ============================================================
# SCANNER
# ============================================================

def run_scanner(symbols):

    market_data = download_market_data(
        symbols
    )

    results = []

    progress = st.progress(0)

    total = len(symbols)

    for i, symbol in enumerate(symbols):

        try:

            result = scan_stock(
                symbol,
                market_data
            )

            if result is not None:
                results.append(result)

        except Exception:
            pass

        progress.progress(
            int((i + 1) / total * 100)
        )

    progress.empty()

    return pd.DataFrame(results)


# ============================================================
# CONTROL BUTTON
# ============================================================

st.sidebar.markdown("---")

st.sidebar.write(
    f"**Hardcoded F&O Stocks:** {len(FNO_STOCKS)}"
)

scan_button = st.sidebar.button(
    "🔄 SCAN NOW",
    use_container_width=True
)

# ============================================================
# AUTO FIRST RUN
# ============================================================

if "scan_done" not in st.session_state:
    st.session_state.scan_done = False

if scan_button:
    st.session_state.scan_done = True

# ============================================================
# RUN
# ============================================================

if st.session_state.scan_done:

    with st.spinner(
        "2M + 5M + 15M F&O stocks scanning..."
    ):

        result_df = run_scanner(
            FNO_STOCKS
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    if result_df.empty:

        st.warning(
            "अभी कोई Strong Bullish / Strong Bearish stock नहीं मिला। "
            "यह भी sideways filter का परिणाम हो सकता है।"
        )

    else:

        bullish_df = result_df[
            result_df["Signal"] == "STRONG BULLISH"
        ].copy()

        bearish_df = result_df[
            result_df["Signal"] == "STRONG BEARISH"
        ].copy()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "F&O Scanned",
            len(FNO_STOCKS)
        )

        col2.metric(
            "🟢 Strong Bullish",
            len(bullish_df)
        )

        col3.metric(
            "🔴 Strong Bearish",
            len(bearish_df)
        )

        st.markdown("---")

        # ====================================================
        # BULLISH
        # ====================================================

        st.subheader(
            "🟢 STRONG BULLISH"
        )

        if not bullish_df.empty:

            bullish_df = bullish_df.sort_values(
                "Strength",
                ascending=False
            )

            display_bull = bullish_df[
                [
                    "Stock",
                    "Signal",
                    "2M",
                    "5M",
                    "15M",
                    "RSI 2M",
                    "RSI 5M",
                    "RSI 15M",
                    "Entry",
                    "SL",
                    "T1",
                    "T2",
                    "T3",
                    "Strength"
                ]
            ].copy()

            st.dataframe(
                display_bull,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "RSI 2M": st.column_config.NumberColumn(
                        "RSI 2M",
                        format="%.1f"
                    ),
                    "RSI 5M": st.column_config.NumberColumn(
                        "RSI 5M",
                        format="%.1f"
                    ),
                    "RSI 15M": st.column_config.NumberColumn(
                        "RSI 15M",
                        format="%.1f"
                    ),
                    "Entry": st.column_config.NumberColumn(
                        "Entry",
                        format="%.2f"
                    ),
                    "SL": st.column_config.NumberColumn(
                        "SL",
                        format="%.2f"
                    ),
                    "T1": st.column_config.NumberColumn(
                        "T1",
                        format="%.2f"
                    ),
                    "T2": st.column_config.NumberColumn(
                        "T2",
                        format="%.2f"
                    ),
                    "T3": st.column_config.NumberColumn(
                        "T3",
                        format="%.2f"
                    ),
                    "Strength": st.column_config.NumberColumn(
                        "Strength",
                        format="%.1f"
                    )
                }
            )

        else:

            st.info(
                "अभी कोई Strong Bullish setup नहीं।"
            )

        # ====================================================
        # BEARISH
        # ====================================================

        st.subheader(
            "🔴 STRONG BEARISH"
        )

        if not bearish_df.empty:

            bearish_df = bearish_df.sort_values(
                "Strength",
                ascending=False
            )

            display_bear = bearish_df[
                [
                    "Stock",
                    "Signal",
                    "2M",
                    "5M",
                    "15M",
                    "RSI 2M",
                    "RSI 5M",
                    "RSI 15M",
                    "Entry",
                    "SL",
                    "T1",
                    "T2",
                    "T3",
                    "Strength"
                ]
            ].copy()

            st.dataframe(
                display_bear,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "RSI 2M": st.column_config.NumberColumn(
                        "RSI 2M",
                        format="%.1f"
                    ),
                    "RSI 5M": st.column_config.NumberColumn(
                        "RSI 5M",
                        format="%.1f"
                    ),
                    "RSI 15M": st.column_config.NumberColumn(
                        "RSI 15M",
                        format="%.1f"
                    ),
                    "Entry": st.column_config.NumberColumn(
                        "Entry",
                        format="%.2f"
                    ),
                    "SL": st.column_config.NumberColumn(
                        "SL",
                        format="%.2f"
                    ),
                    "T1": st.column_config.NumberColumn(
                        "T1",
                        format="%.2f"
                    ),
                    "T2": st.column_config.NumberColumn(
                        "T2",
                        format="%.2f"
                    ),
                    "T3": st.column_config.NumberColumn(
                        "T3",
                        format="%.2f"
                    ),
                    "Strength": st.column_config.NumberColumn(
                        "Strength",
                        format="%.1f"
                    )
                }
            )

        else:

            st.info(
                "अभी कोई Strong Bearish setup नहीं।"
            )

        # ====================================================
        # DOWNLOAD
        # ====================================================

        csv_data = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Scanner CSV",
            data=csv_data,
            file_name="fo_strong_trend_scanner.csv",
            mime="text/csv"
        )

else:

    st.info(
        "👈 बाईं ओर **SCAN NOW** दबाकर F&O scanner शुरू करें।"
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Priyanka@ F&O Strong Trend Scanner | "
    "EMA10 + MA10 + ATR14 + RSI14 + Slope + Candle Strength"
)

st.caption(
    "यह scanner केवल technical screening के लिए है; "
    "यह buy/sell recommendation नहीं है।"
)
