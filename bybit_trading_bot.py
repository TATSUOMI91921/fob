import os
import hmac
import hashlib
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import logging
import math
from scipy.stats import linregress
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input  # Add Input import
from tensorflow.keras.optimizers import Adam
import pickle
import os.path
import json
import sys
import websockets
import asyncio

# Bybit WebSocketのURL
WS_URL = "wss://stream.bybit.com/v5/private"

# WebSocket認証データ生成
def generate_ws_auth():
    timestamp = str(int(time.time() * 1000))
    message = f"{timestamp}{API_KEY}"
    signature = hmac.new(
        API_SECRET.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    return {"api_key": API_KEY, "sign": signature, "timestamp": timestamp}

# WebSocketの処理
async def ws_handler():
    async with websockets.connect(WS_URL) as ws:
        # 認証
        auth_payload = generate_ws_auth()
        await ws.send(json.dumps({"op": "auth", **auth_payload}))
        auth_response = await ws.recv()
        logging.info(f"認証結果: {auth_response}")

        # トピックに購読
        subscriptions = {
            "op": "subscribe",
            "args": ["position", "order"]
        }
        await ws.send(json.dumps(subscriptions))
        logging.info("positionとorderトピックに購読しました")

        # メッセージ処理
        try:
            while True:
                message = await ws.recv()
                process_ws_message(json.loads(message))
        except websockets.ConnectionClosed as e:
            logging.error(f"WebSocket接続が切断されました: {e}")
            await reconnect_ws()

# メッセージの処理
def process_ws_message(message):
    if "topic" not in message:
        logging.warning("トピックのないメッセージを受信しました")
        return

    topic = message["topic"]
    data = message["data"]
    logging.info(f"受信トピック: {topic}")

    if topic == "position":
        update_positions(data)
    elif topic == "order":
        update_orders(data)

# ポジションの更新
positions = {}

def update_positions(data):
    global positions
    for position in data:
        symbol = position["symbol"]
        positions[symbol] = {
            "size": position["size"],
            "entry_price": position["entry_price"],
            "direction": "long" if position["size"] > 0 else "short"
        }
    logging.info(f"ポジション更新: {positions}")

# 注文の更新
orders = {}

def update_orders(data):
    global orders
    for order in data:
        order_id = order["order_id"]
        orders[order_id] = {
            "symbol": order["symbol"],
            "status": order["status"],
            "price": order["price"],
            "qty": order["qty"]
        }
    logging.info(f"注文更新: {orders}")

# WebSocket再接続
async def reconnect_ws():
    logging.info("WebSocket再接続中...")
    await asyncio.sleep(5)
    await ws_handler()

def restart_bot():
    """ボットを再起動する"""
    print("[INFO] Restarting the bot...")
    time.sleep(2)  # 再起動前に2秒待機
    os.execv(sys.executable, ['python'] + sys.argv)

def is_model_updated():
    """モデルが更新されたかを判定"""
    try:
        # モデルの最終更新時刻を記録したファイルを確認
        last_update_file = "model_last_update.txt"
        if not os.path.exists(last_update_file):
            return False
        
        # ファイルの内容（タイムスタンプ）を取得
        with open(last_update_file, 'r') as f:
            last_update_time = float(f.read().strip())
        
        # モデルの現在の更新時刻を取得（例: モデルファイルのタイムスタンプ）
        model_file = "trained_model.pkl"  # モデルファイルの名前
        current_update_time = os.path.getmtime(model_file)
        
        # モデルが更新されたかを判定
        return current_update_time > last_update_time
    except Exception as e:
        print(f"[ERROR] Error checking model update: {e}")
        return False

def update_model_flag():
    """モデルの更新フラグを記録"""
    try:
        model_file = "trained_model.pkl"
        
        # モデルファイルが存在しない場合、新規作成
        if not os.path.exists(model_file):
            print("[INFO] Model file not found. Creating a new model...")
            initialize_model(model_file)

        # モデルファイルの更新時刻を取得
        current_update_time = os.path.getmtime(model_file)
        
        # 更新時刻を記録
        with open("model_last_update.txt", 'w') as f:
            f.write(str(current_update_time))
        print("[INFO] Model update flag recorded.")
    except Exception as e:
        print(f"[ERROR] Error updating model flag: {e}")

def initialize_model(model_file):
    """初期モデルを作成"""
    try:
        model = {
            "trained_data": [],  # 初期データ
            "parameters": {},    # モデルパラメータ
            "metrics": {}        # モデル評価指標
        }
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"[INFO] New model created: {model_file}")
    except Exception as e:
        print(f"[ERROR] Error initializing model: {e}")



# 環境変数の読み込み
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = os.getenv("BASE_URL", "https://api.bybit.com")



# 利確・損切りロジックのトラッキング
active_trades = {}

# タイムスタンプ取得
def get_server_timestamp():
    url = f"{BASE_URL}/v5/market/time"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("time")  # タイムスタンプを返す
    else: 
        raise ValueError("Failed to fetch server timestamp")

# 署名の生成
def create_signature(secret, params, timestamp):
    """
    Bybit API v5の仕様に準拠した署名生成
    """
    try:
        param_str = ''
        if isinstance(params, dict):
            # パラメータをソートして文字列化
            sorted_params = dict(sorted(params.items()))
            param_str = '&'.join([f"{key}={sorted_params[key]}" for key in sorted_params])

        # 署名文字列の生成 (timestamp + api_key + recv_window + parameters)
        sign_str = str(timestamp) + API_KEY + "5000" + param_str
        
        # HMAC SHA256による署名生成
        signature = hmac.new(
            bytes(secret, 'utf-8'),
            sign_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
        
    except Exception as e:
        logging.error(f"Error creating signature: {e}")
        raise

# エラーログ記録
def log_error(symbol, message):
    log_file = os.path.join(os.path.expanduser("~"), "OneDrive", "デスクトップ", "error_log.txt")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} - Symbol: {symbol}, Error: {message}\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_message)
    print(f"[ERROR LOGGED] {log_message.strip()}")

# ウォレット残高取得関数
def get_wallet_balance():
    """認証付きでウォレット残高を取得する"""
    try:
        endpoint = "/v5/account/wallet-balance"
        params = {
            "accountType": "UNIFIED"
        }

        timestamp = str(int(time.time() * 1000))  # ミリ秒単位のタイムスタンプ
        
        # 署名を生成
        signature = create_signature(API_SECRET, params, timestamp)

        headers = {
            "X-BAPI-API-KEY": API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "X-BAPI-RECV-WINDOW": "5000",
            "Content-Type": "application/json"
        }

        url = f"{BASE_URL}{endpoint}"
        
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data["retCode"] == 0:
                return float(data["result"]["list"][0]["totalWalletBalance"])
            else:
                logging.error(f"API error: {data['retMsg']}")
                return None
        else:
            logging.error(f"HTTP error: {response.status_code}")
            return None

    except Exception as e:
        logging.error(f"Error fetching wallet balance: {e}")
        return None

def get_top_volume_symbols():
    """
    出来高上位2～500位のUSDT建てシンボルを取得します。
    """
    endpoint = "/v5/market/tickers"
    url = f"{BASE_URL}{endpoint}"

    params = {"category": "linear"}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data.get("retCode") == 0:
            tickers = data.get("result", {}).get("list", [])

            # USDT建てのペアのみを対象とするフィルタ
            filtered_tickers = [
                ticker for ticker in tickers if "USDT" in ticker["symbol"]
            ]

            # 出来高順にソートして上位300を取得
            sorted_tickers = sorted(
                filtered_tickers, key=lambda x: float(x.get("turnover24h", 0.0)), reverse=True
            )
            top_symbols = [
                ticker["symbol"] for ticker in sorted_tickers[2:500]
            ]
            return top_symbols
        else:
            raise Exception(f"Error fetching tickers: {data.get('retMsg')}")
    else:
        raise Exception(f"Error fetching tickers: {response.text}")


def fetch_market_price(symbol):
    """
    指定したシンボルの最新市場価格を取得します。
    """
    endpoint = "/v5/market/tickers"
    url = f"{BASE_URL}{endpoint}"
    timestamp = get_server_timestamp()
    params = {"category": "linear"}

    # 署名の生成
    signature = create_signature(API_SECRET, params, timestamp)

    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": str(timestamp),
        "X-BAPI-SIGN": signature,
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            tickers = response.json().get("result", {}).get("list", [])
            for ticker in tickers:
                if ticker["symbol"] == symbol:
                    return float(ticker["lastPrice"])
            raise ValueError(f"Symbol {symbol} not found in market tickers")
        else:
            raise Exception(f"Failed to fetch market price: {response.text}")
    except Exception as e:
        print(f"[ERROR] Error fetching market price for {symbol}: {e}")
        log_error(symbol, str(e))
        raise

def fetch_kline_data(symbol, interval="15", limit=100):  # limitを増やす
    """
    Klineデータ取得関数の改善
    """
    try:
        endpoint = "/v5/market/kline"
        url = f"{BASE_URL}{endpoint}"

        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0:
                kline_data = data.get("result", {}).get("list", [])
                
                # データ取得状況のログ
                logging.debug(f"Fetched {len(kline_data)} klines for {symbol}")
                
                if len(kline_data) < limit:
                    logging.warning(f"Received fewer klines than requested for {symbol}. Got {len(kline_data)}, Expected {limit}")
                
                # データが存在する場合は時系列順にソート
                if kline_data:
                    kline_data.sort(key=lambda x: int(x[0]))  # タイムスタンプでソート
                    return kline_data
                
                logging.warning(f"No kline data available for {symbol}")
                return None
            
            logging.error(f"API error for {symbol}: {data.get('retMsg')}")
            return None
            
        logging.error(f"HTTP error {response.status_code} for {symbol}")
        return None

    except Exception as e:
        logging.error(f"Error fetching kline data for {symbol}: {e}")
        return None


def analyze_indicators(
    rsi, adx, macd_line, signal_line, close_price,
    bollinger_upper, bollinger_lower, atr, obv_buy, obv_sell, btc_trend
):
    """
    指標を分析して、エントリー条件を満たすかを判定します。
    デバッグ用に各指標を出力します。
    """
    conditions_met_long = 0
    conditions_met_short = 0

    # デバッグ: 指標の値を出力
    print(f"[DEBUG] Indicators:")
    print(f"  RSI: {rsi}")
    print(f"  ADX: {adx}")
    print(f"  MACD Line: {macd_line}, Signal Line: {signal_line}")
    print(f"  Close Price: {close_price}")
    print(f"  Bollinger Bands: Upper={bollinger_upper}, Lower={bollinger_lower}")
    print(f"  ATR: {atr}")
    print(f"  OBV: Buy={obv_buy}, Sell={obv_sell}")
    print(f"  BTC Trend: {btc_trend}")

    # BTCトレンドの確認
    if btc_trend == "uptrend":
        conditions_met_long += 10
        conditions_met_short -= 5
    elif btc_trend == "downtrend":
        conditions_met_long -= 5
        conditions_met_short += 10

    # ADXによるトレンド強度
    if adx >= 30:
        conditions_met_long += 7
        conditions_met_short += 7
    elif adx < 15:
        conditions_met_long -= 3
        conditions_met_short -= 3

    # MACDの確認
    if macd_line > signal_line:
        conditions_met_long += 5
    elif macd_line < signal_line:
        conditions_met_short += 5

    # RSIによる過熱感の確認
    if rsi > 70:
        conditions_met_long -= 10
    elif rsi < 30:
        conditions_met_short -= 10

    # 出来高の分析
    if obv_buy > obv_sell * 1.2:
        conditions_met_long += 7
    elif obv_sell > obv_buy * 1.2:
        conditions_met_short += 7

    # デバッグ: スコアを出力
    print(f"[DEBUG] Conditions Scores:")
    print(f"  Long Score: {conditions_met_long}")
    print(f"  Short Score: {conditions_met_short}")

    # 最終判定
    if conditions_met_long >= 15:
        print(f"[DEBUG] Decision: LONG (Score: {conditions_met_long})")
        return True, "long"
    elif conditions_met_short >= 15:
        print(f"[DEBUG] Decision: SHORT (Score: {conditions_met_short})")
        return True, "short"
    
    print(f"[DEBUG] Decision: NO ENTRY")
    return False, None

def analyze_btc_trend():
    try:
        # データ取得
        btc_15m_kline = fetch_kline_data("BTCUSDT", interval="15", limit=4)
        btc_1h_kline = fetch_kline_data("BTCUSDT", interval="60", limit=4)
        btc_4h_kline = fetch_kline_data("BTCUSDT", interval="240", limit=4)

        # データが新しい順で返される場合、タイムスタンプでソート
        btc_15m_kline = sorted(btc_15m_kline, key=lambda x: x[0])
        btc_1h_kline = sorted(btc_1h_kline, key=lambda x: x[0])
        btc_4h_kline = sorted(btc_4h_kline, key=lambda x: x[0])

        # データ整形
        close_15m = [float(k[4]) for k in btc_15m_kline]
        close_1h = [float(k[4]) for k in btc_1h_kline]
        close_4h = [float(k[4]) for k in btc_4h_kline]

        # トレンド評価
        trend_15m = evaluate_trend(close_15m, 0.003, weight=1)
        trend_1h = evaluate_trend(close_1h, 0.005, weight=2)
        trend_4h = evaluate_trend(close_4h, 0.01, weight=3)

        # 統合スコアの計算
        total_score = trend_15m + trend_1h + trend_4h

        # 最終トレンドの決定
        if total_score > 0:
            final_trend = "uptrend"
        elif total_score < 0:
            final_trend = "downtrend"
        else:
            final_trend = "sideways"

        logging.debug(f"BTC Trend: 15m={trend_15m}, 1h={trend_1h}, 4h={trend_4h}, Final={final_trend}")
        return final_trend

    except Exception as e:
        logging.error(f"Error analyzing BTC trend: {e}")
        return "sideways"


def evaluate_trend(close_prices, threshold, weight):
    """
    スコア付きトレンド判定ロジック
    """
    try:
        start_price = close_prices[-4]
        end_price = close_prices[-1]
        change_rate = abs((end_price - start_price) / start_price)

        # トレンド判定
        if end_price - start_price > 0 and change_rate >= threshold:
            return weight  # 上昇トレンドの場合はスコア加算
        elif end_price - start_price < 0 and change_rate >= threshold:
            return -weight  # 下降トレンドの場合はスコア減算
        else:
            return 0  # 横ばい相場の場合はスコアなし
    except Exception as e:
        logging.error(f"Error evaluating trend: {e}")
        return 0


def find_pivot_points(prices, window=5):
    """
    価格データからピボットポイントを検出する
    """
    pivots = []
    for i in range(window, len(prices) - window):
        if all(prices[i] > prices[i-j] for j in range(1, window+1)) and \
           all(prices[i] > prices[i+j] for j in range(1, window+1)):
            pivots.append((i, prices[i], "high"))
        elif all(prices[i] < prices[i-j] for j in range(1, window+1)) and \
             all(prices[i] < prices[i+j] for j in range(1, window+1)):
            pivots.append((i, prices[i], "low"))
    return pivots

def determine_triangle_effect(high_prices, low_prices, close_prices, plot_flag=False):
    """
    三角持ち合いの影響を判定し、トレンド転換やブレイクアウトを評価します。
    """
    try:
        # detect_triangle_break の結果を取得
        triangle_up, triangle_down, bounds = detect_triangle_break(high_prices, low_prices, close_prices)

        # データ不足時の処理
        if bounds is None:
            logging.warning("Triangle detection returned insufficient bounds. Skipping analysis.")
            return "no_breakout"

        # デバッグ用ログ
        logging.debug(f"Triangle Bounds: Upper Bound={bounds[0]}, Lower Bound={bounds[1]}")
        logging.debug(f"Breakout Status: Up={triangle_up}, Down={triangle_down}")

        # プロット（オプション）
        if plot_flag:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Candlestick(
                x=list(range(len(close_prices))),
                open=close_prices,
                high=high_prices,
                low=low_prices,
                close=close_prices
            )])
            fig.add_trace(go.Scatter(
                x=list(range(len(close_prices))),
                y=[bounds[0]] * len(close_prices),
                mode='lines',
                name='Upper Bound',
                line=dict(color='green', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=list(range(len(close_prices))),
                y=[bounds[1]] * len(close_prices),
                mode='lines',
                name='Lower Bound',
                line=dict(color='red', dash='dash')
            ))
            fig.show()

        # ブレイクアウトの結果に基づく判定
        if triangle_up:
            logging.info("Triangle breakout up detected.")
            return "triangle_up"
        elif triangle_down:
            logging.info("Triangle breakout down detected.")
            return "triangle_down"
        else:
            logging.info("No triangle breakout detected.")
            return "no_breakout"

    except Exception as e:
        logging.error(f"Error in determining triangle effect: {e}")
        return "error"


def detect_triangle_break(high_prices, low_prices, close_prices, lookback=20):
    """
    三角持ち合いのブレイクを検出する関数を改善
    """
    try:
        # データ量の確認とログ出力
        logging.debug(f"Data points received - High: {len(high_prices)}, Low: {len(low_prices)}, Close: {len(close_prices)}")

        # 最小必要データ数を20に減少
        if len(high_prices) < lookback or len(low_prices) < lookback:
            logging.warning(f"Insufficient data points. Required: {lookback}, Got - High: {len(high_prices)}, Low: {len(low_prices)}")
            return False, False, None

        # 分析に使用するデータを直近の期間に限定
        high_prices = high_prices[-lookback:]
        low_prices = low_prices[-lookback:]
        close_prices = close_prices[-lookback:]

        # ピボットポイントの計算（より少ないデータでも機能するように調整）
        def pivot_point(prices, pivot_type):
            pivots = []
            for i in range(1, len(prices) - 1):  # 前後1本での判定に変更
                if pivot_type == "high":
                    if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                        pivots.append((i, prices[i]))
                elif pivot_type == "low":
                    if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                        pivots.append((i, prices[i]))
            return pivots

        high_pivots = pivot_point(high_prices, "high")
        low_pivots = pivot_point(low_prices, "low")

        # 最小ピボットポイント数を2に減少
        if len(high_pivots) < 2 or len(low_pivots) < 2:
            logging.warning(f"Insufficient pivot points. High: {len(high_pivots)}, Low: {len(low_pivots)}")
            return False, False, None

        # 高値と安値のトレンドラインを計算
        high_indices, high_values = zip(*high_pivots) if high_pivots else ([], [])
        low_indices, low_values = zip(*low_pivots) if low_pivots else ([], [])

        if len(high_indices) < 2 or len(low_indices) < 2:
            return False, False, None

        # 高値のトレンドライン
        high_slope, high_intercept, high_r_value, _, _ = linregress(high_indices, high_values)
        # 安値のトレンドライン
        low_slope, low_intercept, low_r_value, _, _ = linregress(low_indices, low_values)

        # 相関係数の基準を緩和
        if high_r_value ** 2 < 0.6 or low_r_value ** 2 < 0.6:  # 0.8から0.6に緩和
            logging.debug(f"Low correlation coefficients. High R²: {high_r_value**2:.2f}, Low R²: {low_r_value**2:.2f}")
            return False, False, None

        # 最新の価格でのブレイクアウトを判定
        latest_idx = len(close_prices) - 1
        high_line = high_slope * latest_idx + high_intercept
        low_line = low_slope * latest_idx + low_intercept
        latest_close = close_prices[-1]

        # ブレイクアウトの判定
        breakout_threshold = 0.02  # 2%のブレイクアウトしきい値
        triangle_up = latest_close > high_line * (1 + breakout_threshold)
        triangle_down = latest_close < low_line * (1 - breakout_threshold)

        bounds = (high_line, low_line)
        
        return triangle_up, triangle_down, bounds

    except Exception as e:
        logging.error(f"Error in triangle detection: {str(e)}")
        return False, False, None

def analyze_triangle_break(symbol, high_prices, low_prices, close_prices, ):
    """
    三角持ち合いのブレイクを分析し、結果を返す。エラー処理を強化。
    """
    try:
        # 最小データ数のチェック
        min_required_data = 20
        if (len(high_prices) < min_required_data or 
            len(low_prices) < min_required_data or 
            len(close_prices) < min_required_data):
            logging.warning(f"{symbol}: Insufficient data for triangle analysis. " +
                          f"Required: {min_required_data}, Got: {len(high_prices)}")
            return "no_breakout"

        # detect_triangle_breakを使用して三角持ち合いブレイクを検出
        triangle_up, triangle_down, bounds = detect_triangle_break(
            high_prices, low_prices, close_prices
        )

        if bounds is None:
            logging.info(f"{symbol}: No triangle pattern detected.")
            return "no_breakout"

        if triangle_up:
            logging.info(f"{symbol}: Triangle breakout detected (UP).")
            return "triangle_up"
        elif triangle_down:
            logging.info(f"{symbol}: Triangle breakout detected (DOWN).")
            return "triangle_down"
        else:
            logging.info(f"{symbol}: No triangle breakout detected.")
            return "no_breakout"

    except Exception as e:
        logging.error(f"{symbol}: Error in analyze_triangle_break: {e}")
        return "error"

def detect_triangle_height(high_prices, low_prices):
    """
    高値と安値の差から三角持ち合いの高さを計算。

    Args:
        high_prices (list): 高値のリスト
        low_prices (list): 安値のリスト

    Returns:
        float: 三角持ち合いの高さ
    """
    try:
        if len(high_prices) < 2 or len(low_prices) < 2:
            logging.warning("Insufficient data to calculate triangle height.")
            return 0.0

        # 高値と安値の最大差を計算
        max_high = max(high_prices)
        min_low = min(low_prices)
        height = max_high - min_low

        logging.debug(f"Triangle height calculated: {height}")
        return height
    except Exception as e:
        logging.error(f"Error calculating triangle height: {e}")
        return 0.0


def detect_channel(high_prices, low_prices, close_prices, lookback=20):
    """
    平行チャネルを検出する関数
    """
    try:
        # データ量のチェック
        if len(high_prices) < lookback or len(low_prices) < lookback:
            return False, False, None

        # 直近のデータに限定
        high_prices = high_prices[-lookback:]
        low_prices = low_prices[-lookback:]
        close_prices = close_prices[-lookback:]

        # 高値と安値の回帰直線を計算
        x = np.array(range(len(high_prices)))
        high_slope, high_intercept, high_r_value, _, _ = linregress(x, high_prices)
        low_slope, low_intercept, low_r_value, _, _ = linregress(x, low_prices)

        # チャネルの平行性をチェック（傾きの差が小さいことを確認）
        slope_diff = abs(high_slope - low_slope)
        if slope_diff > 0.01:  # 傾きの差の許容範囲
            return False, False, None

        # チャネルの方向を判定
        if abs(high_slope) < 0.001:
            channel_direction = "sideways"
        else:
            channel_direction = "up" if high_slope > 0 else "down"

        # チャネルの上限/下限ライン
        upper_line = high_slope * (len(high_prices) - 1) + high_intercept
        lower_line = low_slope * (len(low_prices) - 1) + low_intercept

        # チャネル幅を計算
        channel_width = upper_line - lower_line

        # 最小チャネル幅のチェック
        if channel_width < 0.001:  # 最小チャネル幅
            return False, False, None

        # チャネル情報を格納
        channel_info = {
            'direction': channel_direction,
            'upper_line': upper_line,
            'lower_line': lower_line,
            'channel_width': channel_width,
            'high_slope': high_slope,
            'low_slope': low_slope,
            'high_r_value': high_r_value,
            'low_r_value': low_r_value
        }

        # 価格がチャネル内に収まっているかチェック
        latest_close = close_prices[-1]
        in_channel = lower_line <= latest_close <= upper_line

        return True, in_channel, channel_info

    except Exception as e:
        logging.error(f"Error in detect_channel: {str(e)}")
        return False, False, None

def detect_channel_break(high_prices, low_prices, close_prices):
    """
    平行チャネルのブレイクを検出する関数
    """
    try:
        # チャネルの検出
        channel_up, channel_down, channel_info = detect_channel(high_prices, low_prices, close_prices)
        
        if channel_info is None:
            return False, False, None

        # 最新の価格がチャネルの上限/下限を突破しているか確認
        latest_close = close_prices[-1]
        upper_line = channel_info['upper_line']
        lower_line = channel_info['lower_line']
        
        # チャネル幅の2%をブレイクアウトの閾値として使用
        breakout_threshold = (upper_line - lower_line) * 0.02
        
        # ブレイクアウトの判定
        if latest_close > upper_line + breakout_threshold:
            return True, False, channel_info  # 上向きブレイク
        elif latest_close < lower_line - breakout_threshold:
            return False, True, channel_info  # 下向きブレイク
            
        return False, False, channel_info  # ブレイクなし

    except Exception as e:
        logging.error(f"Error in channel break detection: {str(e)}")
        return False, False, None

def analyze_channel_break(symbol, high_prices, low_prices, close_prices):
    """
    平行チャネルのブレイクアウトを分析
    """
    try:
        channel_up, channel_down, channel_info = detect_channel_break(
            high_prices, low_prices, close_prices
        )

        if channel_info is None:
            print(f"{symbol}: Channel analysis skipped due to insufficient data.")
            return "no_breakout", None

        if channel_up:
            print(f"{symbol}: Channel breakout UP detected. Direction: {channel_info['direction']}")
            return "channel_up", channel_info
        elif channel_down:
            print(f"{symbol}: Channel breakout DOWN detected. Direction: {channel_info['direction']}")
            return "channel_down", channel_info
        else:
            print(f"{symbol}: No channel breakout detected.")
            return "no_breakout", channel_info

    except Exception as e:
        print(f"{symbol}: Error in analyze_channel_break: {e}")
        return "error", None

# メイン関数内のトレード分析部分を修正
def analyze_technical_patterns(symbol, high_prices, low_prices, close_prices, volumes):
    """
    三角持ち合いを優先し、平行チャネルを条件付きで採用。
    """
    try:
        # 三角持ち合いのブレイクを分析
        triangle_result = analyze_triangle_break(symbol, high_prices, low_prices, close_prices)

        # 平行チャネルのブレイクを分析
        channel_up, channel_down, channel_info = detect_channel_break(high_prices, low_prices, close_prices)

        # 三角持ち合いを優先
        if triangle_result in ["triangle_up", "triangle_down"]:
            return triangle_result, None

        # 平行チャネルは条件付きで採用
        if channel_info:
            atr = calculate_atr(high_prices, low_prices, close_prices)
            avg_volume = sum(volumes) / len(volumes)

            # 平行チャネルの有効性をチェック（例: 出来高、ATR）
            if atr > 0.01 and avg_volume > 1.5 * min(volumes):
                if channel_up:
                    return "channel_up", channel_info
                elif channel_down:
                    return "channel_down", channel_info

        # パターンが検出されない場合
        return "no_pattern", None

    except Exception as e:
        logging.error(f"Error in analyze_technical_patterns for {symbol}: {e}")
        return "error", None

def analyze_volume_effect(close_prices, volumes):
    """
    ボリューム効果を分析して、買い圧または売り圧を判定します。
    """
    try:
        if len(close_prices) < 2 or len(volumes) < 2:
            return "neutral"

        recent_volume = volumes[-1]  # 最新の出来高
        average_volume = sum(volumes[:-1]) / len(volumes[:-1])  # 過去の平均出来高
        recent_price_change = close_prices[-1] - close_prices[-2]  # 直近の価格変化

        # ボリュームが増加しているかどうか
        if recent_volume > average_volume * 1.5:
            if recent_price_change > 0:
                return "buy_pressure"  # 買い圧
            elif recent_price_change < 0:
                return "sell_pressure"  # 売り圧

        # ボリュームが減少している場合の追加ロジック
        elif recent_volume < average_volume * 0.5:
            if recent_price_change > 0:
                return "low_volume_buy"  # 出来高が低いが価格上昇中
            elif recent_price_change < 0:
                return "low_volume_sell"  # 出来高が低いが価格下降中
            return "low_volume"  # 単なる低出来高

        return "neutral"
    except Exception as e:
        logging.error(f"Error analyzing volume effect: {e}")
        return "error"


def determine_candlestick_trend_with_threshold(close_prices, open_prices, threshold):
    """
    ローソク足の始値と終値を基にトレンドを判定し、
    始値と終値の変動率に基づいた閾値を適用します。
    """
    # 陽線と陰線のカウント
    up_count = sum(1 for i in range(len(close_prices)) if close_prices[i] > open_prices[i])
    down_count = sum(1 for  i in range(len(close_prices)) if close_prices[i] < open_prices[i])

    # 1本目の始値と4本目の終値を比較
    start_price = open_prices[0]
    end_price = close_prices[-1]
    change_rate = abs((end_price - start_price) / start_price)

    # トレンド判定
    if change_rate >= threshold:
        if up_count > down_count:
            return "uptrend"
        elif down_count > up_count:
            return "downtrend"
    return "sideways"



# RSI 計算
def calculate_rsi(close_prices, period=14):
    if len(close_prices) < period + 1:
        print(f"[DEBUG] Not enough data for RSI calculation. Required: {period}, Provided: {len(close_prices)}")
        return None
    delta = pd.Series(close_prices).diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

# ADX計算用関数
def calculate_adx(high_prices, low_prices, close_prices, period=14):
    try:
        # データ型をfloatに変換
        high_prices = list(map(float, high_prices))
        low_prices = list(map(float, low_prices))
        close_prices = list(map(float, close_prices))
        
        # TR (True Range) を計算
        tr_list = [
            max(high_prices[i] - low_prices[i], 
                abs(high_prices[i] - close_prices[i - 1]), 
                abs(low_prices[i] - close_prices[i - 1]))
            for i in range(1, len(high_prices))
        ]

        # +DM と -DM を計算
        plus_dm = [high_prices[i] - high_prices[i - 1] if high_prices[i] - high_prices[i - 1] > low_prices[i - 1] - low_prices[i] and high_prices[i] - high_prices[i - 1] > 0 else 0 for i in range(1, len(high_prices))]
        minus_dm = [low_prices[i - 1] - low_prices[i] if low_prices[i - 1] - low_prices[i] > high_prices[i] - high_prices[i - 1] and low_prices[i - 1] - low_prices[i] > 0 else 0 for i in range(1, len(low_prices))]

        # TR, +DM, -DM の移動平均を計算
        tr_sma = pd.Series(tr_list).rolling(window=period).mean().dropna()
        plus_dm_sma = pd.Series(plus_dm).rolling(window=period).mean().dropna()
        minus_dm_sma = pd.Series(minus_dm).rolling(window=period).mean().dropna()

        # +DI と -DI を計算
        plus_di = 100 * (plus_dm_sma / (tr_sma + 1e-10))
        minus_di = 100 * (minus_dm_sma / (tr_sma + 1e-10))

        # DX を計算
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100

        # ADX を計算
        adx = dx.rolling(window=period).mean().dropna()

        if not adx.empty:
            return adx.iloc[-1]  # 最新のADXを返す
        else:
            logging.warning("ADX calculation resulted in an empty series.")
            return None
    except Exception as e:
        logging.error(f"ADX calculation failed: {str(e)}")
        return None

# Bybitデータ取得後のADX計算部分を統合
def fetch_and_calculate_adx(symbol, interval="15", period=14):
    try:
        # Bybit APIからデータを取得する仮の関数（本番ではAPI呼び出しに置き換え）
        kline_data = fetch_kline_data(symbol, interval=interval)

        if not kline_data or len(kline_data) < period + 1:
            logging.error(f"Insufficient data returned for {symbol}: Expected {period + 1} got {len(kline_data) if kline_data else 0}")
            return None

        high_prices = [float(k[2]) for k in kline_data]
        low_prices = [float(k[3]) for k in kline_data]
        close_prices = [float(k[4]) for k in kline_data]

        adx_value = calculate_adx(high_prices, low_prices, close_prices, period=period)
        logging.info(f"Calculated ADX for {symbol}: {adx_value}")
        return adx_value
    except Exception as e:
        logging.error(f"Error in fetch_and_calculate_adx for {symbol}: {str(e)}")
        return None

def calculate_macd(close_prices):
    """
    MACDの計算
    """
    try:
        if len(close_prices) < 26:
            logging.warning(f"Not enough data for MACD calculation. Required: 26, Provided: {len(close_prices)}")
            return None, None

        short_ema = pd.Series(close_prices).ewm(span=12, adjust=False).mean()
        long_ema = pd.Series(close_prices).ewm(span=26, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        return macd_line.iloc[-1], signal_line.iloc[-1]
    except Exception as e:
        logging.error(f"Error in MACD calculation: {e}")
        return None, None

# ボリンジャーバンド計算
def calculate_bollinger_bands(close_prices, period=20):
    try:
        if len(close_prices) < period:
            raise ValueError("Not enough data for Bollinger Bands calculation.")
        sma = sum(close_prices[-period:]) / period
        squared_diff = [(price - sma) ** 2 for price in close_prices[-period:]]
        stddev = (sum(squared_diff) / period) ** 0.5
        upper_band = sma + (2 * stddev)
        lower_band = sma - (2 * stddev)
        return upper_band, sma, lower_band
    except Exception as e:
        print(f"[ERROR] Bollinger Bands calculation failed: {str(e)}")
        return None, None, None

def calculate_ema(data, period):
    """
    指数移動平均（EMA）を計算
    """
    ema = [sum(data[:period]) / period]  # 初期値
    multiplier = 2 / (period + 1)
    for price in data[period:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])
    return ema

# EMAクロス計算
def ema_crossover(prices):
    short_ema = calculate_ema(prices, 9)[-1]
    long_ema = calculate_ema(prices, 21)[-1]
    if short_ema > long_ema:
        return "bullish"  # 強気
    elif short_ema < long_ema:
        return "bearish"  # 弱気
    else:
        return "neutral"  # 中立

# ATR（平均真の範囲）計算
def calculate_atr(highs, lows, closes, period=14):
    tr = [max(h - l, abs(h - c), abs(l - c)) for h, l, c in zip(highs, lows, closes[:-1])]
    atr = sum(tr[:period]) / period  # 初期値
    for value in tr[period:]:
        atr = (atr * (period - 1) + value) / period
    return atr

# OBV（オンバランスボリューム）計算
def calculate_obv(close_prices, kline_data):
    """
    OBV（On-Balance Volume）を計算し、売り方向と買い方向の出来高を区別して返します。
    """
    try:
        if len(close_prices) < 2:
            print("[DEBUG] Not enough data for OBV calculation.")
            return None, None

        obv_buy = 0
        obv_sell = 0
        for i in range(1, len(close_prices)):
            volume = float(kline_data[i][5])  # 出来高
            if close_prices[i] > close_prices[i - 1]:
                obv_buy += volume  # 買い主導の出来高
            elif close_prices[i] < close_prices[i - 1]:
                obv_sell += volume  # 売り主導の出来高

        return obv_buy, obv_sell
    except Exception as e:
        print(f"[ERROR] OBV calculation failed: {str(e)}")
        return None, None


def calculate_stochastic(high_prices, low_prices, close_prices, period=14):
    if len(high_prices) < period or len(low_prices) < period or len(close_prices) < period:
        print("[DEBUG] Not enough data for Stochastic calculation.")
        return None, None

    highest_high = max(high_prices[-period:])
    lowest_low = min(low_prices[-period:])
    current_close = close_prices[-1]

    stoch_k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
    
    # Stoch D の計算に十分なデータがあるか確認
    if len(close_prices) < period + 3:
        print("[DEBUG] Not enough data for Stochastic D calculation.")
        return stoch_k, None  # Stoch K は計算できるが、D は計算不能

    # 過去3本の Stoch K 値で移動平均を計算
    stoch_k_values = [
        ((close_prices[i] - min(low_prices[i - period + 1:i + 1])) /
         (max(high_prices[i - period + 1:i + 1]) - min(low_prices[i - period + 1:i + 1]))) * 100
        for i in range(period - 1, len(close_prices))
    ]

    if len(stoch_k_values) < 3:
        print("[DEBUG] Not enough data for Stochastic D rolling average.")
        return stoch_k, None

    stoch_d = sum(stoch_k_values[-3:]) / 3  # 移動平均
    return stoch_k, stoch_d


def apply_trailing_stop(symbol, trailing_stop_distance):
    try:
        # Fetch position data
        endpoint = "/v5/position/list"
        params = {"category": "linear", "symbol": symbol}
        timestamp = str(get_server_timestamp())
        signature = create_signature(API_SECRET, params, timestamp)

        headers = {
            "X-BAPI-API-KEY": API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json"
        }

        response = requests.get(f"{BASE_URL}{endpoint}", headers=headers, params=params)

        if response.status_code == 200:
            positions = response.json().get("result", {}).get("list", [])
            for position in positions:
                size = float(position.get("size", 0))
                if size > 0:
                    # Set trailing stop
                    position_idx = position.get("positionIdx", 0)
                    payload = {
                        "category": "linear",
                        "symbol": symbol,
                        "positionIdx": position_idx,
                        "trailingStop": str(trailing_stop_distance),
                        "slTriggerBy": "LastPrice"
                    }

                    # Create signed request
                    endpoint = "/v5/position/trading-stop"
                    timestamp = str(get_server_timestamp())
                    signature = create_signature(API_SECRET, payload, timestamp)

                    headers = {
                        "X-BAPI-API-KEY": API_KEY,
                        "X-BAPI-TIMESTAMP": timestamp,
                        "X-BAPI-SIGN": signature,
                        "Content-Type": "application/json"
                    }

                    response = requests.post(f"{BASE_URL}{endpoint}", headers=headers, json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        if data.get("retCode") == 0:
                            logging.info(f"Successfully applied trailing stop for {symbol} with distance {trailing_stop_distance}")
                        else:
                            logging.warning(f"Failed to set trailing stop for {symbol}: {data.get('retMsg')}")
                    else:
                        logging.error(f"HTTP error when setting trailing stop for {symbol}: {response.text}")

        else:
            logging.error(f"Failed to fetch positions for {symbol}: {response.text}")

    except Exception as e:
        logging.error(f"Error applying trailing stop for {symbol}: {e}")


# トレード追跡ロジック (続き)
def update_trade(symbol, entry_price, direction):
    global active_trades
    try:
        stop_loss, take_profit = calculate_tp_sl(entry_price, direction)

        if symbol not in active_trades:
            active_trades[symbol] = {
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "direction": direction
            }

    except Exception as e:
        print(f"[ERROR] Error updating trade for {symbol}: {e}")

def detect_pressure(kline_data):
    """
    市場圧力（買い圧または売り圧）を検知する。
    """
    try:
        if len(kline_data) < 3:  # データが不足している場合
            return "neutral"

        # 必要なデータを取得
        close_prices = [float(k[4]) for k in kline_data[-3:]]
        open_prices = [float(k[1]) for k in kline_data[-3:]]
        volumes = [float(k[5]) for k in kline_data[-3:]]

        # ボリューム急増の判定
        avg_volume = sum(volumes[:-1]) / (len(volumes) - 1)  # 過去2本の平均
        latest_volume = volumes[-1]

        if (latest_volume > avg_volume * 2) and (close_prices[-1] > open_prices[-1]):  # 急増の閾値
            return "buy_pressure"
        elif (latest_volume > avg_volume * 2) and (close_prices[-1] < open_prices[-1]):  # 急増の閾値
            return "sell_pressure"

        return "neutral"

    except Exception as e:
        logging.error(f"Error in detect_pressure: {e}")
        return "neutral"
    
def cancel_order(order_id):
    """
    指定された注文IDの注文をキャンセルする。
    """
    try:
        endpoint = "/v5/order/cancel"
        url = f"{BASE_URL}{endpoint}"
        timestamp = str(get_server_timestamp())

        payload = {
            "orderId": order_id,
            "category": "linear"  # 必要に応じて変更
        }

        json_body = json.dumps(payload, separators=(',', ':'))
        origin_string = f"{timestamp}{API_KEY}{json_body}"
        signature = hmac.new(
            API_SECRET.encode("utf-8"),
            origin_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        headers = {
            "X-BAPI-API-KEY": API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json"
        }

        response = requests.post(url, headers=headers, data=json_body)
        if response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0:
                logging.info(f"Successfully cancelled order {order_id}.")
                return True
            else:
                logging.warning(f"Failed to cancel order {order_id}: {data.get('retMsg')}")
                return False
        else:
            logging.error(f"HTTP error when cancelling order {order_id}: {response.text}")
            return False
    except Exception as e:
        logging.error(f"Error in cancel_order for order {order_id}: {e}")
        return False


def cancel_expired_orders(symbol):
    """
    未約定の期限切れ注文、エントリー条件を満たさなくなった注文、
    および市場トレンドが転換した注文をキャンセルする。
    """
    try:
        # アクティブな注文を取得
        active_orders = fetch_active_orders(symbol)
        if not active_orders:
            logging.info(f"No active orders found for {symbol}.")
            return

        # 現在の時間
        current_time = time.time()

        for order in active_orders:
            order_time = order.get("created_time")
            order_direction = "long" if order.get("side") == "Buy" else "short"

            # 1. 時間経過によるキャンセル
            if order_time and (current_time - order_time > 3600):  # 1時間を超えている場合
                cancel_order(order["order_id"])
                logging.info(f"Cancelled expired order {order['order_id']} for {symbol} due to timeout.")
                continue

            # 2. 出来高急上昇の検知
            kline_data = fetch_historical_data(symbol, limit=20)
            if detect_volume_spike(kline_data, order_direction):
                cancel_order(order["order_id"])
                logging.info(f"Cancelled order {order['order_id']} for {symbol} due to volume spike in opposite direction.")
                continue

            # 3. 市場トレンドの転換によるキャンセル
            symbol_trend = analyze_symbol_trend(symbol)
            if not symbol_trend:
                logging.error(f"Failed to analyze trend for {symbol}")
                continue

            if order_direction == "long" and symbol_trend["15"] == "downtrend":
                cancel_order(order["order_id"])
                logging.info(f"Cancelled long order {order['order_id']} for {symbol} due to trend reversal.")
            elif order_direction == "short" and symbol_trend["15"] == "uptrend":
                cancel_order(order["order_id"])
                logging.info(f"Cancelled short order {order['order_id']} for {symbol} due to trend reversal.")

    except Exception as e:
        logging.error(f"Error in cancel_expired_orders for {symbol}: {e}")


def create_signed_headers_and_payload(payload, timestamp):
    """
    Create signed headers and payload for Bybit API requests.
    """
    json_body = json.dumps(payload, separators=(',', ':'))
    origin_string = f"{timestamp}{API_KEY}{json_body}"
    signature = hmac.new(
        API_SECRET.encode("utf-8"),
        origin_string.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-SIGN": signature,
        "Content-Type": "application/json"
    }
    return headers, signature

def calculate_tp_sl(entry_price, direction, leverage=5, tp_rate=0.3, sl_rate=0.1):
    """
    エントリー価格を基に利確・損切り価格を計算する関数。
    """
    try:
        sl_percent = sl_rate / leverage
        tp_percent = tp_rate / leverage

        if direction == "long":
            stop_loss = entry_price * (1 - sl_percent)  # 損切り価格はエントリー価格より低い
            take_profit = entry_price * (1 + tp_percent)  # 利確価格はエントリー価格より高い
        elif direction == "short":
            stop_loss = entry_price * (1 + sl_percent)  # 損切り価格はエントリー価格より高い
            take_profit = entry_price * (1 - tp_percent)  # 利確価格はエントリー価格より低い
        else:
            raise ValueError("Invalid direction specified. Use 'long' or 'short'.")

        # 小数点以下6桁に丸める
        return round(stop_loss, 6), round(take_profit, 6)
    except Exception as e:
        logging.error(f"Error in calculate_tp_sl: {e}")
        return None, None



# グローバルスコープでHEADERSを定義
def create_headers():
    timestamp = str(get_server_timestamp())
    signature = create_signature(API_SECRET, {}, timestamp)
    return {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-SIGN": signature,
        "Content-Type": "application/json"
    }

HEADERS = create_headers()

def fetch_order_history(symbol):
    """
    指定したシンボルの注文履歴を取得します。
    """
    try:
        endpoint = "/v5/order/history"  # APIの注文履歴エンドポイント
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": 50,  # 必要に応じて調整
        }
        response = requests.get(BASE_URL + endpoint, headers=HEADERS, params=params)
        data = response.json()

        if "result" in data and "list" in data["result"]:
            return data["result"]["list"]  # 注文履歴を返す
        else:
            logging.warning(f"No order history found for {symbol}")
            return []

    except Exception as e:
        logging.error(f"Error fetching order history for {symbol}: {str(e)}")
        return []



def fetch_market_data(symbol):
    """
    現在の価格と出来高を取得する関数。
    """
    endpoint = "/v5/market/tickers"
    url = f"{BASE_URL}{endpoint}"
    params = {"category": "linear", "symbol": symbol}
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0:
                result = data.get("result", {}).get("list", [])[0]  # 最初のデータを取得
                return {
                    "price": float(result["lastPrice"]),
                    "volume": float(result["volume24h"]),
                }
            else:
                print(f"[ERROR] API returned error: {data.get('retMsg')}")
        else:
            print(f"[ERROR] HTTP error: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Exception in fetch_market_data: {e}")
    return None

def fetch_historical_data(symbol, limit=50, interval="1"):
    """
    過去の価格と出来高データを取得する関数。
    他の関数のロジックに合わせて統一。
    """
    endpoint = "/v5/market/kline"
    url = f"{BASE_URL}{endpoint}"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0:
                kline_data = data.get("result", {}).get("list", [])
                if len(kline_data) < limit:
                    logging.warning(f"Insufficient data for {symbol}: Expected {limit}, got {len(kline_data)}")
                # 整形してデータを返す
                return {
                    "highs": [float(k[2]) for k in kline_data],
                    "lows": [float(k[3]) for k in kline_data],
                    "closes": [float(k[4]) for k in kline_data],
                }
            else:
                logging.error(f"API error for {symbol}: {data.get('retMsg')}")
        else:
            logging.error(f"HTTP error for {symbol}: {response.status_code}")
    except Exception as e:
        logging.error(f"Error in fetch_historical_data for {symbol}: {e}")
    return None


def make_api_request(method, endpoint, params=None, payload=None):
    """APIリクエストを実行する共通関数"""
    try:
        url = f"{BASE_URL}{endpoint}"
        timestamp = str(get_server_timestamp())
        
        if params:
            param_str = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        else:
            param_str = ''

        signature = create_signature(API_SECRET, params or payload or {}, timestamp)
        
        headers = {
            "X-BAPI-API-KEY": API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "X-BAPI-RECV-WINDOW": "5000",
            "Content-Type": "application/json"
        }

        if method.upper() == "GET":
            response = requests.get(url, params=params, headers=headers)
        else:
            response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            if result.get("retCode") == 0:
                return result
            else:
                logging.error(f"API error: {result.get('retMsg')}")
                return None
        else:
            logging.error(f"HTTP error {response.status_code}: {response.text}")
            return None

    except Exception as e:
        logging.error(f"Error in API request: {e}")
        return None

# fetch_active_orders関数を修正
def fetch_active_orders(symbol):
    """アクティブな注文を取得"""
    try:
        params = {
            "category": "linear",
            "symbol": symbol
        }
        result = make_api_request("GET", "/v5/order/realtime", params=params)
        if result and isinstance(result.get("result"), dict):
            return result["result"].get("list", [])
        return []
    except Exception as e:
        logging.error(f"Error fetching active orders for {symbol}: {e}")
        return []

def fetch_positions(symbol):
    """
    現在のポジション情報を取得。
    """
    try:
        endpoint = "/v5/position/list"
        params = {
            "category": "linear",
            "symbol": symbol,
        }
        
        # 署名とヘッダーの生成
        timestamp = str(get_server_timestamp())
        signature = create_signature(API_SECRET, params, timestamp)
        
        headers = {
            "X-BAPI-API-KEY": API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json"
        }

        response = requests.get(f"{BASE_URL}{endpoint}", headers=headers, params=params)
        if response.status_code == 200:  # 余分な括弧を削除
            data = response.json()
            if data["retCode"] == 0:
                return data["result"]["list"]
            else:
                logging.error(f"API error: {data['retMsg']}")
                return []
        else:
            logging.error(f"HTTP error: {response.status_code}")
            return []

    except Exception as e:
        logging.error(f"Error in fetch_positions: {e}")
        return []

def cancel_all_orders(symbol):
    """
    指定したシンボルの全ての未約定注文をキャンセル。
    """
    try:
        endpoint = "/v5/order/cancel-all"  # 正しいエンドポイント
        timestamp = str(get_server_timestamp())
        
        payload = {
            "category": "linear",
            "symbol": symbol,
        }
        
        json_body = json.dumps(payload)
        signature = create_signature(API_SECRET, payload, timestamp)
        
        headers = {
            "X-BAPI-API-KEY": API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json"
        }

        response = requests.post(f"{BASE_URL}{endpoint}", headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            if data["retCode"] == 0:
                logging.info(f"Successfully cancelled all orders for {symbol}")
                return True
            else:
                logging.error(f"API error: {data['retMsg']}")
                return False
        else:
            logging.error(f"HTTP error: {response.status_code}")
            return False

    except Exception as e:
        logging.error(f"Error in cancel_all_orders: {e}")
        return False

def set_trading_stop(
    symbol,
    position_idx,  # 0: 両方向, 1: ロング, 2: ショート
    entry_direction,
    sl_size=None,
    trailing_stop=None,
    btc_trend=None,
    obv_buy=None,
    obv_sell=None,
    close_prices=None,
):
    """
    トレイリングストップや損切りを動的に設定する。
    ポジション方向に応じて価格変動や出来高を警戒。

    Args:
        symbol (str): トレードするシンボル
        position_idx (int): ポジションインデックス (1: ロング, 2: ショート)
        entry_direction (str): エントリー方向 ("long" または "short")
        sl_size (float, optional): 損切り価格
        trailing_stop (float, optional): トレイリングストップ価格間隔
        btc_trend (str, optional): BTCトレンド ("uptrend", "downtrend", "sideways")
        obv_buy (float, optional): OBV買い方向の値
        obv_sell (float, optional): OBV売り方向の値
        close_prices (list, optional): 直近の終値データ
    """
    try:
        logging.info(f"Analyzing stop settings for {symbol}, Direction: {entry_direction}")
        
        # ショートトレード時の警戒条件
        if entry_direction == "short":
            if btc_trend == "uptrend":
                logging.warning(f"BTC is in an uptrend. Adjusting for potential losses in short position.")
                if trailing_stop:
                    trailing_stop *= 0.5  # トレイリングストップを狭める
            if obv_buy and obv_sell and obv_buy > obv_sell * 1.5:
                logging.warning(f"Buy pressure detected in {symbol}. Adjusting stop-loss.")
                if sl_size:
                    sl_size *= 1.1  # 損切りを緩める

        # ロングトレード時の警戒条件
        elif entry_direction == "long":
            if btc_trend == "downtrend":
                logging.warning(f"BTC is in a downtrend. Adjusting for potential losses in long position.")
                if trailing_stop:
                    trailing_stop *= 0.5  # トレイリングストップを狭める
            if obv_buy and obv_sell and obv_sell > obv_buy * 1.5:
                logging.warning(f"Sell pressure detected in {symbol}. Adjusting stop-loss.")
                if sl_size:
                    sl_size *= 1.1  # 損切りを緩める

        # 大きな価格変動への対応
        if close_prices and len(close_prices) >= 2:
            recent_change = abs(close_prices[-1] - close_prices[-2]) / close_prices[-2]
            if recent_change > 0.03:  # 3%以上の変動
                logging.info(f"High price volatility detected for {symbol}. Adjusting trailing stop.")
                if trailing_stop:
                    trailing_stop *= 1.2  # ボラティリティに応じて広げる

        # APIリクエストのペイロード作成
        payload = {
            "category": "linear",
            "symbol": symbol,
            "positionIdx": position_idx,
        }

        if sl_size is not None:
            payload["stopLoss"] = str(sl_size)
            payload["slTriggerBy"] = "LastPrice"
        if trailing_stop is not None:
            payload["trailingStop"] = str(trailing_stop)

        # APIリクエスト
        json_body = json.dumps(payload, separators=(',', ':'))
        timestamp = str(get_server_timestamp())
        signature = create_signature(API_SECRET, json_body, timestamp)
        headers = {
            "X-BAPI-API-KEY": API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json",
        }

        response = requests.post(f"{BASE_URL}/v5/position/trading-stop", headers=headers, json=payload)

        if response.status_code == 200:
            data = response.json()
            if data["retCode"] == 0:
                logging.info(f"Successfully set trading stop for {symbol}.")
                return True
            else:
                logging.error(f"API error: {data['retMsg']}")
                return False
        else:
            logging.error(f"HTTP error: {response.status_code}")
            logging.debug(f"Response: {response.text}")
            return False

    except Exception as e:
        logging.error(f"Error in set_trading_stop for {symbol}: {e}")
        return False


def detect_volume_spike(kline_data, direction):
    """
    出来高の急上昇を検知する。
    Args:
        kline_data: キャンドルデータ。
        direction: エントリー方向 ("long" または "short")。
    Returns:
        bool: 急上昇が検知されたか。
    """
    try:
        volumes = [float(k[5]) for k in kline_data]  # 出来高データ
        avg_volume = sum(volumes[:-1]) / len(volumes[:-1])  # 過去の平均
        recent_volume = volumes[-1]  # 最新の出来高
        
        # 急上昇条件（例: 平均の1.5倍以上）
        if direction == "long" and recent_volume > avg_volume * 1.5:
            return True  # ショート方向の売り圧
        elif direction == "short" and recent_volume > avg_volume * 1.5:
            return True  # ロング方向の買い圧

        return False
    except Exception as e:
        logging.error(f"Error detecting volume spike: {e}")
        return False

def monitor_active_trades():
    """
    アクティブなポジションと未約定注文を監視し、必要に応じて調整またはキャンセル。
    """
    try:
        for symbol in list(active_trades.keys()):
            # 現在のポジション情報を取得
            positions = fetch_positions(symbol)
            if not positions:
                continue

            for position in positions:
                size = float(position.get("size", 0))
                direction = position.get("direction", "long")  # "long" または "short"

                if size == 0:
                    # ポジションが既にクローズされている場合
                    if symbol in active_trades:
                        del active_trades[symbol]
                    continue

                # OBVやBTCトレンドを取得
                obv_buy, obv_sell = calculate_obv([...], [...])
                btc_trend = analyze_btc_trend()

                # トレーリングストップと損切りの動的設定
                set_trading_stop(
                    symbol=symbol,
                    position_idx=position.get("positionIdx"),
                    entry_direction=direction,
                    sl_size=position.get("stopLoss"),
                    trailing_stop=0.02,  # 初期値として2%を設定
                    btc_trend=btc_trend,
                    obv_buy=obv_buy,
                    obv_sell=obv_sell,
                    close_prices=[...]  # 最新の終値データ
                )

                # 出来高急上昇の検知
                kline_data = fetch_historical_data(symbol, limit=20)
                if detect_volume_spike(kline_data, direction):
                    close_position(symbol, direction)
                    logging.info(f"Closed position for {symbol} due to volume spike in opposite direction.")
                    continue

            # 未約定注文の管理
            active_orders = fetch_active_orders(symbol)
            if active_orders:
                for order in active_orders:
                    order_age = time.time() - float(order["createdTime"]) / 1000
                    if order_age > 3600:  # 1時間以上経過した注文
                        cancel_order(order["orderId"])
                        logging.info(f"Cancelled expired order {order['orderId']} for {symbol} due to timeout.")

                    # 逆方向の出来高急上昇によるキャンセル
                    if detect_volume_spike(fetch_historical_data(symbol, limit=20), "short" if order["side"] == "Buy" else "long"):
                        cancel_order(order["orderId"])
                        logging.info(f"Cancelled order {order['orderId']} for {symbol} due to volume spike in opposite direction.")

    except Exception as e:
        logging.error(f"Error in monitor_active_trades: {e}")




def detect_market_reversal(symbol):
    """
    急激な相場転換を検知。
    """
    try:
        # 15分足のデータを取得
        kline_data = fetch_kline_data(symbol, interval="15", limit=5)
        if not kline_data:
            return False

        close_prices = [float(k[4]) for k in kline_data]
        
        # 価格変動率の計算
        price_change = (close_prices[-1] - close_prices[0]) / close_prices[0] * 100
        
        # 急激な価格変動（3%以上）を検知
        return abs(price_change) >= 3

    except Exception as e:
        logging.error(f"Error in detect_market_reversal: {e}")
        return False

def close_position(symbol, direction):
    """
    ポジションを閉じる処理を行います。
    """
    try:
        side = "Sell" if direction == "long" else "Buy"
        print(f"[INFO] Closing position for {symbol} in direction {direction}. Side: {side}")

        # 実際のポジションクローズ処理
        endpoint = "/v5/order/close-position"
        url = f"{BASE_URL}{endpoint}"
        timestamp = str(get_server_timestamp())

        payload = {
            "category": "linear",
            "symbol": symbol,
            "side": side
        }

        signature = create_signature(API_SECRET, payload, timestamp)
        headers = {
            "X-BAPI-API-KEY": API_KEY,
            "X-BAPI-SIGN": signature,
            "X-BAPI-TIMESTAMP": timestamp,
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            print(f"[INFO] Successfully closed position for {symbol}")
            del active_trades[symbol]  # アクティブトレードから削除
        else:
            print(f"[ERROR] Failed to close position for {symbol}: {response.text}")
            log_error(symbol, response.text)

    except Exception as e:
        print(f"[ERROR] Exception closing position for {symbol}: {e}")
        log_error(symbol, str(e))

leverage = 5

def get_price_data(symbol):
    """
    市場データから高値、安値、終値を取得します。
    """
    try:
        data = fetch_historical_data(symbol, limit=50, interval="15")  # 必要なKlineデータを取得
        if not data:
            raise ValueError(f"Failed to fetch price data for {symbol}")
        return data["highs"], data["lows"], data["closes"]
    except Exception as e:
        print(f"[ERROR] Error in get_price_data for {symbol}: {e}")
        return None, None, None

def execute_trade(symbol, direction, wallet_balance, usdt_to_use=60):
    try:
        leverage = 5

        # 現在価格を取得
        market_price = fetch_market_price(symbol)
        if market_price is None:
            raise ValueError(f"Failed to fetch market price for {symbol}")

        # エントリー価格を計算
        entry_price = market_price * 0.99 if direction == "long" else market_price * 1.01

        # 利確・損切り価格を計算
        stop_loss, take_profit = calculate_tp_sl(entry_price, direction, leverage=leverage)

        # 数量を計算
        qty = (usdt_to_use * leverage) / entry_price
        qty = float(qty)  # 型変換を追加
        qty = round(qty, 2)  # 小数点以下2桁に丸める

        if qty <= 0:
            logging.warning(f"Invalid quantity calculated for {symbol}: {qty}")
            return None

        # フォーマット
        entry_price = f"{entry_price:.6f}"
        stop_loss = f"{stop_loss:.6f}"
        take_profit = f"{take_profit:.6f}"
        qty = f"{qty:.2f}"

        logging.debug(f"[DEBUG] Qty: {qty}, Entry Price: {entry_price}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")

        # サイドを決定
        side = "Buy" if direction == "long" else "Sell"

        # 注文を送信
        order_id = place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            leverage=leverage,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        if order_id:
            print(f"[INFO] Order placed successfully for {symbol}: Order ID {order_id}")
        else:
            print(f"[ERROR] Failed to place order for {symbol}")

        return order_id

    except Exception as e:
        logging.error(f"Error executing trade for {symbol}: {e}")
        return None



# データキャッシュの実装
class DataCache:
    def __init__(self, cache_time=60):  # 60秒のキャッシュ
        self.cache = {}
        self.cache_time = cache_time
    
    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_time:
                return data
        return None
    
    def set(self, key, data):
        self.cache[key] = (data, time.time())

# グローバルキャッシュインスタンス
market_cache = DataCache()

def fetch_symbol_info(symbol):
    """
    指定されたシンボルの詳細情報を取得します（精度、最小数量など）。
    """
    endpoint = "/v5/market/instruments-info"
    url = f"{BASE_URL}{endpoint}"
    params = {"category": "linear"}
    timestamp = get_server_timestamp()
    
    signature = create_signature(API_SECRET, params, timestamp)
    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": str(timestamp),
        "X-BAPI-SIGN": signature,
        "Content-Type": "application/json"
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        instruments = response.json().get("result", {}).get("list", [])
        for instrument in instruments:
            if instrument["symbol"] == symbol:
                return instrument
        raise ValueError(f"Symbol {symbol} not found in instruments info")
    else:
        raise Exception(f"Failed to fetch symbol info: {response.text}")


def calculate_precision(value, precision):
    """
    指定された精度で値を丸め込みます。
    """
    return round(value, precision)

def place_order(symbol, side, qty, leverage, entry_price, stop_loss, take_profit):
    try:
        endpoint = "/v5/order/create"
        url = f"{BASE_URL}{endpoint}"
        timestamp = str(get_server_timestamp())

        # 値を明示的に数値型に変換
        stop_loss = round(float(stop_loss), 6)
        take_profit = round(float(take_profit), 6)
        entry_price = round(float(entry_price), 6)
        qty = round(float(qty), 2)

        payload = {
            "category": "linear",
            "symbol": symbol,
            "side": side,  # "Buy" または "Sell"
            "orderType": "Limit",
            "qty": f"{qty:.2f}",
            "price": f"{entry_price:.6f}",
            "timeInForce": "GTC",
            "leverage": leverage,
            "stopLoss": f"{stop_loss:.6f}",
            "takeProfit": f"{take_profit:.6f}",
            "reduceOnly": False
        }

        # JSON文字列としてペイロードを生成
        json_body = json.dumps(payload, separators=(',', ':'))

        # Bybit署名ルールに従う
        origin_string = f"{timestamp}{API_KEY}{json_body}"
        signature = hmac.new(
            API_SECRET.encode("utf-8"),
            origin_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        headers = {
            "X-BAPI-API-KEY": API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json"
        }

        response = requests.post(url, headers=headers, data=json_body)
        logging.debug(f"[DEBUG] Payload: {payload}")
        logging.debug(f"[DEBUG] Response: {response.text}")

        if response.status_code == 200:
            result = response.json()
            if result.get("retCode") == 0:
                logging.info(f"[INFO] Order placed successfully for {symbol}")
                return result.get("orderId")
            else:
                raise ValueError(f"API Error: {result.get('retMsg')}")
        else:
            raise ValueError(f"HTTP Error {response.status_code}: {response.text}")

    except Exception as e:
        logging.error(f"Exception in place_order: {e}")
        log_error(symbol, str(e))
        return None


def close_all_positions():
    """
    アクティブなすべてのポジションをクローズする。
    """
    global active_trades
    for symbol, trade in active_trades.items():
        try:
            close_position(symbol, trade["direction"])
            logging.info(f"Closed position for {symbol}")
        except Exception as e:
            logging.error(f"Error closing position for {symbol}: {str(e)}")
    active_trades.clear()  # 全てのポジションがクローズされた後、リセット


# FRED APIキーを設定
FRED_API_KEY = "1959d19fa1d0f675b47de26fab974e1d"

def fetch_economic_events():
    """
    FRED APIから経済指標データを取得します。
    :return: 経済イベントリスト（名前と時刻を含む）
    """
    try:
        # FRED APIエンドポイント
        url = "https://api.stlouisfed.org/fred/releases"
        params = {
            "api_key": FRED_API_KEY,
            "file_type": "json",
        }
        response = requests.get(url, params=params)
        data = response.json()

        # データの整形
        if "releases" in data:
            events = [
                {"name": release["name"], "time": release["realtime_start"]}
                for release in data["releases"]
            ]
            return events
        else:
            logging.warning("No economic events retrieved from FRED API.")
            return []

    except Exception as e:
        logging.error(f"Error fetching economic events: {e}")
        return []

def check_economic_event_restrictions():
    """
    経済指標発表前後でエントリー制御を行います。
    """
    try:
        # 経済指標データを取得
        economic_events = fetch_economic_events()
        now = datetime.utcnow()  # 修正ポイント

        for event in economic_events:
            # 経済指標の時刻をUTCとして解析
            event_time = datetime.strptime(event["time"], "%Y-%m-%d")
            time_diff = (event_time - now).total_seconds() / 60  # 分単位で計算

            # 指標発表の30分前後はエントリー禁止
            if -30 <= time_diff <= 180:
                logging.warning(f"Entry restricted due to economic event: {event['name']}")
                return True

            # 指標発表の5分前は全ポジションをクローズ
            if 0 < time_diff <= 5:
                logging.warning(f"Closing positions due to imminent economic event: {event['name']}")
                close_all_positions()  # 事前定義のポジションクローズ関数
                return True

        return False  # 制限なし

    except Exception as e:
        logging.error(f"Error in check_economic_event_restrictions: {e}")
        return False

# モデル初期化を遅延読み込みに変更
class LazyTradingModelOptimizer:
    def __init__(self):
        self.data_file_path = r"C:\Users\fob92\OneDrive\デスクトップ\trading_data"
        self._ensure_data_directory()
        self._models_initialized = False
        self.trade_history = []
        self.performance_metrics = {}

def _ensure_data_directory(self):
    if not os.path.exists(self.data_file_path):
        os.makedirs(self.data_file_path)
        logging.info(f"Created directory: {self.data_file_path}")


    def _initialize_models(self):
        """必要になった時点でモデルを初期化"""
        if not self._models_initialized:
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.xgb_model = XGBClassifier(objective='binary:logistic', random_state=42)
            self.lstm_model = self._build_lstm_model()
            self._models_initialized = True

    def collect_trade_data(self, trade_info):
        """トレード実行時のみデータを収集"""
        self.trade_history.append(trade_info)
        
        # トレード履歴が一定数に達したら学習を実行
        if len(self.trade_history) >= 10:  # 10トレードごとに学習
            self._initialize_models()
            self.optimize_parameters()
            self.save_data()

    def save_data(self):
        """トレード結果のみを保存"""
        try:
            history_file = os.path.join(self.data_file_path, "trade_history.json")
            with open(history_file, 'w') as f:
                json.dump([{
                    'timestamp': trade['timestamp'].isoformat(),
                    'symbol': trade['symbol'],
                    'direction': trade['direction'],
                    'profit_loss': trade['profit_loss'],
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price']
                } for trade in self.trade_history], f)
        except Exception as e:
            logging.error(f"Error saving trade data: {e}")

# メイン関数内のML関連処理を修正
def main():
    try:
        monitored_symbols = get_top_volume_symbols()[2:500]
        logging.info(f"Monitoring {len(monitored_symbols)} symbols")

        # 注文履歴とエントリー中銘柄情報を冒頭で取得
        active_orders = {symbol: fetch_active_orders(symbol) for symbol in monitored_symbols}
        active_positions = {symbol: fetch_positions(symbol) for symbol in monitored_symbols}

        for symbol in monitored_symbols:
            try:
                # 未約定注文の取得
                orders = fetch_active_orders(symbol)
                if orders:
                    active_orders[symbol] = orders
                else:
                    logging.info(f"{symbol}: No active orders found. Skipping order-related management.")

                # アクティブポジションの取得
                positions = fetch_positions(symbol)
                if positions:
                    active_positions[symbol] = positions
                else:
                    logging.info(f"{symbol}: No active positions found. Skipping position-related management.")

            except Exception as e:
                logging.error(f"Error fetching data for {symbol}: {e}")
                continue

        # 未約定注文とポジションの初期管理
        for symbol in monitored_symbols:
            if symbol not in active_orders and symbol not in active_positions:
                logging.info(f"{symbol}: No active orders or positions. Skipping management.")
                continue

            cancel_expired_orders(symbol)

        monitor_active_trades()
        
        # キャッシュの初期化
        symbol_cache = {}
        btc_trend = None

        # 初回モデル更新フラグの記録
        update_model_flag()

        while True:
            try:
                # モデルの更新チェック
                if is_model_updated():
                    print("[INFO] Model updated. Restarting bot to apply changes.")
                    restart_bot()  # 再起動

                # BTCトレンドを取得
                btc_trend = analyze_btc_trend()
                
                # 経済指標による制限チェック
                if check_economic_event_restrictions():
                    logging.info("Trading restricted due to economic events")
                    continue

                # ウォレット残高の取得
                wallet_balance = get_wallet_balance()
                if not wallet_balance:
                    logging.error("Failed to fetch wallet balance")
                    continue

                logging.info(f"Current wallet balance: {wallet_balance} USDT")

                # 各シンボルの処理
                for symbol in monitored_symbols:
                    try:
                        print(f"Analyzing {symbol}:")

                        # キャッシュチェック
                        if symbol in symbol_cache:
                            cache_time, cached_data = symbol_cache[symbol]
                            if time.time() - cache_time < 300:  # 5分以内のデータは再利用
                                continue
                        
                        # 市場データの取得
                        kline_data = fetch_kline_data(symbol)
                        if not kline_data or len(kline_data) < 50:
                            continue

                        # データの整形
                        high_prices = [float(k[2]) for k in kline_data]
                        low_prices = [float(k[3]) for k in kline_data]
                        close_prices = [float(k[4]) for k in kline_data]
                        volumes = [float(k[5]) for k in kline_data]

                        # テクニカル指標の計算
                        rsi = calculate_rsi(close_prices)
                        adx = calculate_adx(high_prices, low_prices, close_prices)
                        macd_line, signal_line = calculate_macd(close_prices)
                        bollinger_upper, _, bollinger_lower = calculate_bollinger_bands(close_prices)
                        atr = calculate_atr(high_prices, low_prices, close_prices)
                        obv_buy, obv_sell = calculate_obv(close_prices, kline_data)

                        # ブレイクアウト分析
                        triangle_result = analyze_triangle_break(symbol, high_prices, low_prices, close_prices)
                        channel_up, channel_down, channel_info = detect_channel_break(high_prices, low_prices, close_prices)

                        # ブレイクアウト結果を出力
                        print(f"  Triangle Result: {triangle_result}")
                        print(f"  Channel Breakout - Up: {channel_up}, Down: {channel_down}")

                        # 指標を出力
                        print(f"  Indicators - RSI: {rsi}, ADX: {adx}")
                        print(f"  MACD Line: {macd_line}, Signal Line: {signal_line}")
                        print(f"  Bollinger Bands - Upper: {bollinger_upper}, Lower: {bollinger_lower}")
                        print(f"  ATR: {atr}, OBV - Buy: {obv_buy}, Sell: {obv_sell}")

                        # ブレイク方向の設定
                        if triangle_result == "triangle_up" or channel_up:
                            direction = "long"
                        elif triangle_result == "triangle_down" or channel_down:
                            direction = "short"
                        else:
                            logging.info(f"{symbol}: No valid breakout detected. Skipping.")
                            continue

                        # 矛盾チェック
                        if (direction == "short" and triangle_result == "triangle_up") or \
                           (direction == "long" and triangle_result == "triangle_down"):
                            logging.warning(f"{symbol}: Conflict between triangle and channel breakout. Skipping trade.")
                            continue

                        # エントリー判定とトレード実行
                        entry_signal, signal_direction = analyze_indicators(
                            rsi=rsi,
                            adx=adx,
                            macd_line=macd_line,
                            signal_line=signal_line,
                            close_price=close_prices[-1],
                            bollinger_upper=bollinger_upper,
                            bollinger_lower=bollinger_lower,
                            atr=atr,
                            obv_buy=obv_buy,
                            obv_sell=obv_sell,
                            btc_trend=btc_trend
                        )

                        # 未約定注文やポジションがない場合のみエントリー
                        if symbol in active_orders and symbol in active_positions:
                            logging.info(f"Skipping {symbol}: Active order or position exists.")
                            continue


                        # シグナル方向とブレイクアウト方向が一致する場合のみトレードを実行
                        if entry_signal and signal_direction == direction:
                            logging.info(f"{symbol}: Entry signal detected. Direction: {direction}")
                            trade_result = execute_trade(symbol, direction, wallet_balance, usdt_to_use=60)

                        if trade_result:
                            # トレード成功後のリスク管理
                            logging.info(f"Trade executed for {symbol}. Setting trading stop.")
                            set_trading_stop(
                            symbol=symbol,
                            position_idx=1 if direction == "long" else 2,
                            entry_direction=direction,
                            sl_size=None,  # 必要に応じて損切り価格を設定
                            trailing_stop=0.02,  # トレーリングストップ距離
                            btc_trend=btc_trend,
                            obv_buy=obv_buy,
                            obv_sell=obv_sell,
                            close_prices=close_prices
                            )

                            trade_info = {
                                'timestamp': datetime.now(),
                                'symbol': symbol,
                                'direction': direction,
                                'entry_price': close_prices[-1],
                                'indicators': {
                                'rsi': rsi,
                                'adx': adx,
                                'macd': macd_line,
                                    }
                                }
                            optimizer = get_model_optimizer()
                            optimizer.collect_trade_data(trade_info)
                        else:
                            logging.warning(f"{symbol}: Entry signal conflicts with breakout direction. Skipping trade.")



                        # キャッシュの更新
                        symbol_cache[symbol] = (time.time(), {
                            'kline_data': kline_data,
                            'indicators': {
                                'rsi': rsi,
                                'adx': adx,
                                'macd': (macd_line, signal_line)
                            }
                        })

                    except Exception as e:
                        logging.error(f"Error processing {symbol}: {str(e)}")
                        continue


            except Exception as e:
                logging.error(f"Error in main loop iteration: {str(e)}")
                time.sleep(300)

    except Exception as e:
        logging.error(f"Critical error in main loop: {str(e)}")
        raise


# グローバルインスタンスの作成を遅延化
model_optimizer = None

def get_model_optimizer():
    """必要な時にのみモデルを初期化"""
    global model_optimizer
    if model_optimizer is None:
        model_optimizer = LazyTradingModelOptimizer()
    return model_optimizer

if __name__ == "__main__":
    main()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(ws_handler())