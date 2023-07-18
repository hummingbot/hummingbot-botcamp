from decimal import Decimal

import pandas as pd
import pandas_ta as ta
import statsmodels.api as sm

from hummingbot.core.data_type.common import PriceType, TradeType
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.smart_components.position_executor.data_types import PositionConfig
from hummingbot.smart_components.position_executor.position_executor import PositionExecutor
from hummingbot.strategy.directional_strategy_base import DirectionalStrategyBase


class StatisticalArbitrage(DirectionalStrategyBase):
    """
    BotCamp Cohort #5 July 2023
    Design Template: https://github.com/hummingbot/hummingbot-botcamp/issues/48

    Description:
    Statistical Arbitrage strategy implementation based on the DirectionalStrategyBase.
    This strategy execute trades based on the Z-score values.
    When z-score indicates an entry signal. the left side will execute a long position and right side will execute a short position.
    When z-score indicates an exit signal. the left side will execute a short position and right side will execute a long position.
    """
    directional_strategy_name: str = "statistical_arbitrage"
    # Define the trading pair and exchange that we want to use and the csv where we are going to store the entries
    trading_pair: str = "APT-USDT"  # left side trading pair
    trading_pair_2: str = "MATIC-USDT"  # right side trading pair
    exchange: str = "binance_perpetual"
    order_amount_usd = Decimal("15")  # amount of order per side
    leverage = 10
    max_executors = 2
    max_hours_to_hold_position = 24
    length = 168  # length of spread/zscore calculation
    # candles parameters
    interval = "1h"
    max_records = 200

    # Configure the parameters for the position
    zscore_entry: int = -2
    zscore_entry_sl: int = -3
    zscore_exit: int = 2
    zscore_exit_sl: int = 3

    candles = [
        CandlesFactory.get_candle(connector=exchange,
                                  trading_pair=trading_pair,
                                  interval=interval, max_records=max_records),
        CandlesFactory.get_candle(connector=exchange,
                                  trading_pair=trading_pair_2,
                                  interval=interval, max_records=max_records),
    ]
    on_going_arbitrage = False
    last_signal = 0
    markets = {exchange: {trading_pair, trading_pair_2}}

    def on_tick(self):
        self.clean_and_store_executors()
        if self.is_perpetual:
            self.check_and_set_leverage()

        if self.all_candles_ready:
            signal = self.get_signal()
            if not self.on_going_arbitrage:
                position_configs = self.get_arbitrage_position_configs(signal)
                if position_configs:
                    self.on_going_arbitrage = True
                    self.last_signal = signal
                    for position_config in position_configs:
                        executor = PositionExecutor(strategy=self,
                                                    position_config=position_config)
                        self.active_executors.append(executor)
            else:
                if (self.last_signal == 1 and signal == -1) or (self.last_signal == -1 and signal == 1):
                    self.logger().info("Exit Arbitrage")
                    for executor in self.active_executors:
                        executor.early_stop()
                    self.on_going_arbitrage = False
                    self.last_signal = 0

    def get_arbitrage_position_configs(self, signal):
        trading_pair_1_amount, trading_pair_2_amount = self.get_order_amounts()
        if signal == 1:
            buy_config = PositionConfig(
                trading_pair=self.trading_pair,
                exchange=self.exchange,
                side=TradeType.BUY,
                amount=trading_pair_1_amount,
                leverage=self.leverage,
                time_limit=60 * 60 * self.max_hours_to_hold_position,
            )
            sell_config = PositionConfig(
                trading_pair=self.trading_pair_2,
                exchange=self.exchange,
                side=TradeType.SELL,
                amount=trading_pair_2_amount,
                leverage=self.leverage,
                time_limit=60 * 60 * self.max_hours_to_hold_position,
            )
            return [buy_config, sell_config]
        elif signal == -1:
            buy_config = PositionConfig(
                trading_pair=self.trading_pair_2,
                exchange=self.exchange,
                side=TradeType.BUY,
                amount=trading_pair_2_amount,
                leverage=self.leverage,
                time_limit=60 * 60 * self.max_hours_to_hold_position,
            )
            sell_config = PositionConfig(
                trading_pair=self.trading_pair,
                exchange=self.exchange,
                side=TradeType.SELL,
                amount=trading_pair_1_amount,
                leverage=self.leverage,
                time_limit=60 * 60 * self.max_hours_to_hold_position,
            )
            return [buy_config, sell_config]

    def get_order_amounts(self):
        base_quantized_1, usd_quantized_1 = self.get_order_amount_quantized_in_base_and_usd(self.trading_pair, self.order_amount_usd)
        base_quantized_2, usd_quantized_2 = self.get_order_amount_quantized_in_base_and_usd(self.trading_pair_2, self.order_amount_usd)
        if usd_quantized_2 > usd_quantized_1:
            base_quantized_2, usd_quantized_2 = self.get_order_amount_quantized_in_base_and_usd(self.trading_pair_2, usd_quantized_1)
        elif usd_quantized_1 > usd_quantized_2:
            base_quantized_1, usd_quantized_1 = self.get_order_amount_quantized_in_base_and_usd(self.trading_pair, usd_quantized_2)
        return base_quantized_1, base_quantized_2

    def get_order_amount_quantized_in_base_and_usd(self, trading_pair: str, order_amount_usd: Decimal):
        price = self.connectors[self.exchange].get_price_by_type(trading_pair, PriceType.MidPrice)
        amount_quantized = self.connectors[self.exchange].quantize_order_amount(trading_pair, order_amount_usd / price)
        return amount_quantized, amount_quantized * price

    def get_signal(self):

        candles_df = self.get_processed_df()
        z_score = candles_df.iat[-1, -1]

        # all execution are only on the left side trading pair
        if z_score < self.zscore_entry or z_score > self.zscore_exit_sl:
            return 1
        elif z_score < self.zscore_entry_sl or z_score > self.zscore_exit:
            return -1
        else:
            return 0

    def get_processed_df(self):
        candles_df_1 = self.candles[0].candles_df
        candles_df_2 = self.candles[1].candles_df

        # calculate the spread and z-score based on the candles of 2 trading pairs
        df = pd.merge(candles_df_1, candles_df_2, on="timestamp", how='inner', suffixes=('_1', '_2'))
        model = sm.OLS(df["close_1"], df["close_2"]).fit()

        if model is not None and model.params is not None and len(model.params) > 0:
            hedge_ratio = model.params[0]
        else:
            # Handle the case where model or model.params is None or empty
            hedge_ratio = None  # or perform an alternative action

        df["spread"] = df["close_1"] - (df["close_2"] * hedge_ratio)
        df["z_score"] = ta.zscore(df["spread"], length=self.length)

        return df

    def market_data_extra_info(self):
        """
        Provides additional information about the market data to the format status.
        Returns:
            List[str]: A list of formatted strings containing market data information.
        """
        lines = []
        columns_to_show = ["timestamp", "close_1", "close_2", "spread", "z_score"]
        candles_df = self.get_processed_df()
        lines.extend([f"Candles: {self.candles[0].name}-{self.candles[1].name} | Interval: {self.candles[0].interval}\n"])
        lines.extend(self.candles_formatted_list(candles_df, columns_to_show))
        return lines
