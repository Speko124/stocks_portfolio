import datetime
import json
import urllib.request
from dataclasses import dataclass
from urllib.error import HTTPError

import investpy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf
from bs4 import BeautifulSoup as bs
from requests_html import HTMLSession


def parse_action(x):
    if x.lower() == "upgrade":
        return "positive"
    if x.lower() == "downgrade":
        return "negative"
    return "neutral"


def parse_rating(x):
    actions = [w.strip().lower() for w in x.split("→")]
    last_action = actions[-1]
    positive = ["buy", "peer perform", "strong buy", "outperform", "underweight"]
    neutral = ["hold", "neutral", "equal weight", "mkt perform", "in-line", "market perform"]
    negative = ["overweight", "underperform", "mkt outperform", "sell", "strong sell"]
    if last_action in positive:
        return "positive"
    if last_action in negative:
        return "negative"
    if last_action in neutral:
        return "neutral"


def price_parse(x):
    if not x:
        return None
    if type(x) == str:
        prices = [float(w.strip().replace("$", "")) for w in x.split("→")]
        return prices
    return x


def last_price(x):
    prices = price_parse(x)
    if not prices:
        return None
    if type(prices) == list:
        return prices[-1]
    return prices


def price_direction(x):
    prices = price_parse(x)
    if not prices:
        return None
    if type(x) != str:
        return x
    if len(prices) == 1:
        return "no change"
    if prices[0] > prices[1]:
        return "negative"
    if prices[0] < prices[1]:
        return "positive"
    return "no_change"


def process_finviz_analyst_rec(ticker):
    url = f'https://finviz.com/quote.ashx?t={ticker}&ty=c&ta=1&p=d'
    session = HTMLSession()
    r = session.get(url)
    soup = bs(r.content, 'html5lib')
    try:
        df = pd.read_html(str(soup.find('table', class_='fullview-ratings-outer')))[0]
    except (ConnectionError, ValueError):
        return pd.DataFrame([])
    df = df.iloc[range(1, len(df), 2)]
    df.columns = ['date', 'action', 'rating_institution', 'rating', 'price_target']
    df["action"] = df["action"].apply(parse_action)
    df["rating"] = df["rating"].apply(parse_rating)
    df["price_dir"] = df["price_target"].apply(price_direction)
    df["price_target"] = df["price_target"].apply(last_price)
    df["date"] = pd.to_datetime(df["date"])

    return df


def calc_finviz_agg_values(signals):
    pos = {"buy", "less_volatility", "oversold", "positive"}
    neg = {"sell", "volatile", "overbought", "negative"}
    neutral = {'neutral', 'hold', "no change"}
    counter = {}
    for s in signals:
        if s in pos:
            counter["p"] = counter.get("p", 0) + 1
        elif s in neg:
            counter["n"] = counter.get("n", 0) + 1
        elif s in neutral:
            counter["neutral"] = counter.get("natural", 0) + 1
        else:
            counter["neutral"] = counter.get("natural", 0) + 1
    p = counter.get("p", 0)
    n = counter.get("n", 0)
    neutral = counter.get("neutral", 0)
    _all = p + n + neutral
    if _all == 0:
        return None
    if _all == 1:
        _all = 3
    fin_score = p - n
    fin_score_conf = abs(fin_score) / _all
    if fin_score > 0:
        if fin_score_conf > 0.5:
            return "strong pos"
        return "pos"
    if fin_score == 0:
        return "neutral"
    if fin_score < 0:
        if fin_score_conf > 0.5:
            return "strong neg"
        return "neg"


def gen_fin_viz_recommendation(ticker, days_back=30):
    col_names = ["fin_act", "fin_rating", "fin_price_dir", "fin_price_tgt"]
    df = process_finviz_analyst_rec(ticker)
    if len(df) == 0:
        return col_names, [None, None, None, None]
    start_date = datetime.date.today() + datetime.timedelta(-days_back)
    df = df[df['date'] > pd.to_datetime(start_date)]
    if len(df) == 0:
        return col_names, [None, None, None, None]
    values = [
        calc_finviz_agg_values(df["action"].to_list()),
        calc_finviz_agg_values(df["rating"].to_list()),
        calc_finviz_agg_values(df["price_dir"].to_list()),
        round(df["price_target"].mean(), 2)
    ]
    return col_names, values


@dataclass
class StockData:
    ticker: str
    p_type: str
    price: float
    amount: int


class Portfolio(object):
    def __init__(self):
        self._assets = dict()
        self._hist_value = pd.DataFrame()
        self._tickers_hist = dict()

    def buy(self, ticker, p_type, amount, price):
        ticker = ticker.upper()
        if ticker in self._assets:
            asset = self._assets[ticker]
            self._assets[ticker].price = (asset.amount * asset.price + amount * price) / (asset.amount + amount)
            self._assets[ticker].amount = asset.amount + amount
        else:
            self._assets[ticker] = StockData(ticker=ticker, price=price, amount=amount, p_type=p_type)

    def sell(self, ticker, amount):
        ticker = ticker.upper()
        if ticker in self._assets:
            asset = self._assets[ticker]
            self._assets[ticker].amount = asset.amount - amount if (asset.amount - amount) >= 0 else 0
        else:
            raise ValueError("No such ticker in assets")

    def details(self):
        return pd.DataFrame(self._assets.values())

    def _calc_hist_value(self):
        prices = []
        for idx, i in enumerate(self._assets):
            asset = self._assets[i]
            ticker = asset.ticker
            product_type = asset.p_type
            amt = asset.amount
            search_result = investpy.search_quotes(text=ticker, products=[self._p_type_multi(product_type)],
                                                   countries=['united states'], n_results=1)
            raw = search_result.retrieve_recent_data()
            # update each ticker params
            self._tickers_hist[ticker] = raw.copy()[["Close", "Change Pct"]]
            self._tickers_hist[ticker]["gain_per_share"] = self._tickers_hist[ticker]["Close"] - asset.price
            self._tickers_hist[ticker]["gain_pct"] = round(
                self._tickers_hist[ticker]["gain_per_share"] * 100 / asset.price, 2)

            if idx == 0:
                dss = raw.index.values
                ds_parsed = [pd.to_datetime(str(i)).strftime('%Y-%m-%d') for i in dss]
                res_df = pd.DataFrame(ds_parsed, columns=["ds"])
            res_df[ticker] = ((raw["Close"] * amt).to_list())
            prices.append(asset.amount * asset.price)
        res_df["p_sum"] = res_df.sum(axis=1)
        res_df["p_buy_sum"] = sum(prices)
        res_df["gain"] = res_df["p_sum"] - res_df["p_buy_sum"]
        res_df["gain_ratio"] = (res_df["gain"] * 100) / res_df["p_buy_sum"]
        self._hist_value = res_df

    def get_hist_value(self):
        if len(self._hist_value) == 0:
            self.calc_hist()
        return self._hist_value

    def calc_hist(self):
        self._calc_hist_value()
        p_hist = self._hist_value
        res = [None]
        for i in range(1, len(p_hist)):
            curr_sum = p_hist.iloc[i]["p_sum"]
            prev_sum = p_hist.iloc[i - 1]["p_sum"]
            chg = round((curr_sum - prev_sum) * 100 / prev_sum, 2)
            res.append(chg)
        self._hist_value["p_chg"] = res

    def refresh(self):
        self.calc_hist()

    def compare_to(self, ticker, p_type):
        try:
            search_result = investpy.search_quotes(text=ticker, products=[self._p_type_multi(p_type)],
                                                   countries=['united states'], n_results=1)
        except (RuntimeError, ConnectionError, ValueError):
            raise ValueError(f"missing investing data {ticker} of type {p_type}")
        if len(self._hist_value) == 0:
            self.calc_hist()
        p_hist = self._hist_value[["ds", "p_sum", "p_chg"]].copy()
        raw = search_result.retrieve_recent_data()
        p_hist["p_chg_sum"] = p_hist["p_chg"].cumsum()
        p_hist["t_value"] = raw["Close"].to_list()
        p_hist["t_chg"] = raw["Change Pct"].to_list()
        p_hist["t_chg_sum"] = p_hist["t_chg"].cumsum()
        p_hist["diff"] = p_hist["p_chg"] - p_hist["t_chg"]
        p_hist["diff_sum"] = p_hist["diff"].cumsum()
        return p_hist

    def plot_compare_to(self, ticker, p_type):
        sns.set_theme(style="whitegrid")

        data = self.compare_to(ticker, p_type)[
            ["ds", "p_chg", "p_chg_sum", "t_chg", "t_chg_sum", "diff", "diff_sum"]]
        data = data.rename(columns={
            "p_chg": "protf_chg",
            "t_chg": f"{ticker}_chg",
            "p_chg_sum": "protf_chg_sum",
            "t_chg_sum": f"{ticker}_chg_sum",
        })

        fig, axes = plt.subplots(2, 2, sharex=True, figsize=(20, 10))
        fig.suptitle(f'Comparing Portfolio to {ticker} in the last M')
        axes[0][0].set_title('Daily Change')
        axes[0][1].set_title('Accumulated change (from the last M)')
        axes[1][0].set_title('Daily Change Difference (+ -> portfolio is better)')
        axes[1][1].set_title('Accumulated Change Difference (+ -> portfolio is better)')

        sns.lineplot(data=data[["protf_chg", f"{ticker}_chg"]], ax=axes[0][0], palette="rocket", linewidth=2.5, )
        sns.lineplot(data=data[["protf_chg_sum", f"{ticker}_chg_sum"]], ax=axes[0][1], palette="rocket",
                     linewidth=2.5, )
        sns.lineplot(data=data[["diff"]], ax=axes[1][0], palette="tab10", linewidth=2.5, )
        sns.lineplot(data=data[["diff_sum"]], ax=axes[1][1], palette="tab10", linewidth=2.5, )
        plt.show()

    def plot_hist(self):
        if len(self._hist_value) == 0:
            self.calc_hist()
        data = self._hist_value[["ds", "p_sum", "p_chg", "gain_ratio"]].copy()
        ds = data["ds"].iloc[-1]
        fig, axes = plt.subplots(1, 2, sharex=True, figsize=(30, 10))
        fig.suptitle(f'Portfolio performance last M, date - {ds}')
        axes[0].set_title('Daily Change')
        axes[1].set_title('Total Value')

        sns.lineplot(data=data[["p_chg", ]], ax=axes[0], palette="rocket", linewidth=2.5, )
        sns.lineplot(data=data[["p_sum"]], ax=axes[1], palette="tab10", linewidth=2.5, )
        plt.show()

    def basic_analysis(self):
        if len(self._hist_value) == 0:
            self.calc_hist()
        res = []
        col_names = ["ds", "ticker", "buy_price", "Close", "Change Pct", "gain_per_share", "gain_pct"]
        ds = self._hist_value.iloc[-1]["ds"]
        for t, df_hist in self._tickers_hist.items():
            row = [ds, t, self._assets[t].price]
            row.extend(df_hist.iloc[-1][["Close", "Change Pct", "gain_per_share", "gain_pct"]].to_list())
            res.append(row)
        res_df = pd.DataFrame(res, columns=col_names)
        res_df.columns = [c.lower().replace(" ", "_") for c in res_df.columns]
        return res_df.sort_values("change_pct", ascending=False)

    @staticmethod
    def _p_type_multi(p_type):
        if p_type == "index":
            return "indices"
        if p_type[-1] == 's':
            return p_type
        return p_type + "s"

    @staticmethod
    def _calc_agg_values(signals):
        pos = {"buy", "less_volatility", "oversold", "strong_buy"}
        neg = {"sell", "volatile", "overbought", "strong_sell"}
        neutral = {'neutral', 'hold'}
        counter = {}
        for s in signals:
            if s.lower() in pos:
                counter["p"] = counter.get("p", 0) + 1
            elif s.lower() in neg:
                counter["n"] = counter.get("n", 0) + 1
            elif s.lower() in neutral:
                counter["neutral"] = counter.get("natural", 0) + 1
            else:
                counter["neutral"] = counter.get("natural", 0) + 1
        p = counter.get("p", 0)
        n = counter.get("n", 0)
        neutral_count = counter.get("neutral", 0)
        fin_score = p - n
        if fin_score > 6:
            return "strong buy"
        if fin_score > 0:
            return "buy"
        if fin_score == 0:
            return "neutral"
        if fin_score < -6:
            return "strong sell"
        if fin_score < 0:
            return "sell"

    def _get_technical_indicators(self, ticker, p_type, interval="monthly", ret_type="list"):
        try:
            tech_values = investpy.technical_indicators(name=ticker, country='united states', product_type=p_type,
                                                        interval=interval)
        except (RuntimeError, ConnectionError, ValueError):
            search_result = investpy.search_quotes(text=ticker, products=[self._p_type_multi(p_type)],
                                                   countries=['united states'], n_results=1)
            tech_values = search_result.retrieve_technical_indicators(interval=interval)
        names = [f"tech_{interval}"]
        values = [self._calc_agg_values(tech_values["signal"])]
        return names, values

    def _get_moving_avg_final(self, ticker, interval="monthly", p_type='stock'):
        try:
            ma_values = investpy.moving_averages(name=ticker, country='united states', product_type=p_type,
                                                 interval=interval)
        except (RuntimeError, ConnectionError, ValueError):
            return None
        ma_list_signal = ma_values['sma_signal'].to_list() + ma_values['ema_signal'].to_list()
        return self._calc_agg_values(ma_list_signal)

    def _get_investing_tech_data_intervals(self, t, intervals, p_type):
        res_values = []
        res_names = []
        for i in intervals:
            n, v = self._get_technical_indicators(t, interval=i, ret_type="list", p_type=p_type)
            res_values += v
            res_names += n
            res_values.append(self._get_moving_avg_final(t, i, p_type))
            res_names.append(f"ma_{i}")
        return res_names, res_values

    @staticmethod
    def _gen_yf_recommendation(ticker):
        try:
            return yf.Ticker(ticker.upper()).info.get("recommendationKey", None)
        except (RuntimeError, ConnectionError, ValueError):
            return None

    @staticmethod
    def _gen_zacks_recommendation(ticker):
        ticker = ticker.upper()
        url = 'https://quote-feed.zacks.com/index?t=' + ticker
        try:
            j = json.loads(urllib.request.urlopen(url).read().decode())
        except HTTPError:
            return None
        res = j[ticker].get("zacks_rank_text", None)
        if type(res) == str:
            res = res.lower()
        return res

    def gen_recommendations(self):
        names = []
        res = []
        for a in self._assets.values():
            inves_n, inves_v = self._get_investing_tech_data_intervals(t=a.ticker,
                                                                       intervals=["daily", "weekly", "monthly"],
                                                                       p_type=a.p_type)
            f_n, f_v = gen_fin_viz_recommendation(a.ticker, days_back=30)
            ds = self._hist_value.iloc[-1]["ds"]
            names = ["ds", "ticker", "yf", "zacks"] + inves_n + f_n
            v = [
                    ds,
                    a.ticker.upper(),
                    self._gen_yf_recommendation(a.ticker),
                    self._gen_zacks_recommendation(a.ticker)] + \
                inves_v + f_v
            res.append(v)
        return pd.DataFrame(res, columns=names)
