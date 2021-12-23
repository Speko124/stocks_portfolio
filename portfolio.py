from dataclasses import dataclass
import seaborn as sns
import investpy
import matplotlib.pyplot as plt
import pandas as pd


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

    @staticmethod
    def _p_type_multi(p_type):
        if p_type == "index":
            return "indices"
        if p_type[-1] == 's':
            return p_type
        return p_type + "s"

    def buy(self, ticker, p_type, amount, price):
        if ticker in self._assets:
            asset = self._assets[ticker]
            self._assets[ticker].price = (asset.amount * asset.price + amount * price) / (asset.amount + amount)
            self._assets[ticker].amount = asset.amount + amount
        else:
            self._assets[ticker] = StockData(ticker=ticker, price=price, amount=amount, p_type=p_type)

    def sell(self, ticker, amount):
        if ticker in self._assets:
            asset = self._assets[ticker]
            self._assets[ticker].amount = asset.amount - amount if (asset.amount - amount) >= 0 else 0
        else:
            raise ValueError("No such ticker in assets")

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
        except:
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
        data = self._hist_value[["ds", "p_sum", "p_chg", "gain_ratio"]].copy()

        fig, axes = plt.subplots(1, 2, sharex=True, figsize=(30, 10))
        fig.suptitle('Portfolio performance last M')
        axes[0].set_title('Daily Change')
        axes[1].set_title('Total Value')

        sns.lineplot(data=data[["p_chg", ]], ax=axes[0], palette="rocket", linewidth=2.5, )
        sns.lineplot(data=data[["p_sum"]], ax=axes[1], palette="tab10", linewidth=2.5, )
        plt.show()

    def basic_analysis(self):
        if len(self._hist_value) == 0:
            self.calc_hist()
        res = []
        col_names = ["ticker", "buy_price", "Close", "Change Pct", "gain_per_share", "gain_pct"]
        for t, df_hist in self._tickers_hist.items():
            row = [t, self._assets[t].price]
            row.extend(df_hist.iloc[-1][["Close", "Change Pct", "gain_per_share", "gain_pct"]].to_list())
            res.append(row)
        return pd.DataFrame(res, columns=col_names).sort_values("gain_pct", ascending=False)
