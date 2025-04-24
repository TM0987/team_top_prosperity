from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any, Tuple
import string
import jsonpickle
import numpy as np
import math


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    # Add new products
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBE = "DJEMBES"
    SQUID_INK = "SQUID_INK"
    # Volatility Smile Vouchers
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"
    SUGAR = "SUGAR"


PARAMS = {
    # Parameters from v1.py
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 10,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.25,  # Slightly stronger mean reversion
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "spike_trade_divisor":20,
        "autoregress_p": 2,
        "movingaverage_q":10,
        "priceWeight" :1,
        "MAweight": 0,
        "ARweight" :1,
        
    },
    # Parameters for basket1 strategy
    Product.PICNIC_BASKET1: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 2,
        "synthetic_weight": 0.04,
        "volatility_window_size": 10,
        "adverse_volatility": 0.1,
    },
    # Parameters for basket2 strategy
    Product.PICNIC_BASKET2: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 2,
        "synthetic_weight": 0.03,
        "volatility_window_size": 10,
        "adverse_volatility": 0.1,
    },
    # Parameters for volatility smile strategy
    "volatility_smile_strategy": {
        "underlying_product": Product.VOLCANIC_ROCK, # Assuming RESIN is the underlying
        "voucher_products": [
            Product.VOLCANIC_ROCK_VOUCHER_9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500,
        ],
        "strikes": {
            Product.VOLCANIC_ROCK_VOUCHER_9500: 9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 10500,
        },
        "risk_free_rate": 0.0, # Assuming 0 risk-free rate
        "volatility_fit_frequency": 100, # Fit smile every 20 timestamps
        "smile_fit_degree": 2, # Quadratic fit
        "volatility_data_points": 50, # Use last 50 points for fitting
        "trading_threshold_iv_diff": 0.001, # Trade if market IV differs from base IV by this absolute amount
        "max_order_size": 200, # Max size per order
        "min_time_to_expiration": 1/700.0, # Don't trade if less than 1 timestamp left
        "total_duration_timestamps": 8000000, # Using large number assuming single 7 day period for now
    },
    # Parameters for Magnificent Macarons strategy
    Product.MAGNIFICENT_MACARONS: {
        "critical_sunlight_index": 45,
        "min_profit_margin": 1,  # Minimum profit margin for arbitrage (NOW UNUSED)
        "position_limit": 75,
        "conversion_limit": 10,
        "sunlight_window": 5,  # Window for trend calculation
        # --- Market Making Params --- 
        "market_making_spread": 4,       # Spread around fair value for MM (Reset)
        "market_making_order_size": 25,  # Size of MM orders (Kept)
        # --- ADDED: Storage Cost Param ---
        "storage_cost_per_unit_per_ts": 0.1,
        # --- ADDED: Loss Prevention Param ---
        "loss_prevention_threshold": 0, # Exit if PnL hits this
        # --- ADDED: LR Fallback Param ---
        "lr_fair_value_fallback": 710, # Fallback FV if LR model fails/untrained
    },
}

# Define basket weights
BASKET1_WEIGHTS = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBE: 1,
}

BASKET2_WEIGHTS = {
    Product.CROISSANTS: 4,
    Product.JAMS: 2,
}

# Position limits
POSITION_LIMITS = {
    # Limits from v1.py
    Product.RAINFOREST_RESIN: 50,
    Product.KELP: 50,
    Product.SQUID_INK: 50,
    # New limits
    Product.CROISSANTS: 250,
    Product.JAMS: 350,
    Product.DJEMBE: 60,
    Product.PICNIC_BASKET1: 60,
    Product.PICNIC_BASKET2: 100,
    Product.VOLCANIC_ROCK: 400,
    # Voucher Limits
    Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
    Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
    Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
    Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
    Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
    Product.MAGNIFICENT_MACARONS: 75,
}

PARAMS_FILENAME = r"params.json"

class Trader:
    def __init__(self, params=None):
        
        loaded_params = None
        # Attempt to load params from the file
        # if os.path.exists(PARAMS_FILENAME):
        #     try:
        #         with open(PARAMS_FILENAME, 'r') as f:
        #             loaded_params = jsonpickle.decode(f.read())
        #         print(f"Trader: Loaded parameters from {PARAMS_FILENAME}")
        #     except Exception as e:
        #         print(f"Trader: Error loading {PARAMS_FILENAME}: {e}. Using defaults.")
        #         loaded_params = None
        
        # Use loaded params or fall back to defaults
        self.params = loaded_params if loaded_params is not None else PARAMS
        # Assuming POSITION_LIMITS are static for now, or load them similarly if needed
        self.position_limits = POSITION_LIMITS

        # Initialize state
        self.basket1_price_diffs = []
        self.basket2_price_diffs = []
        # State for volatility smile
        self.volatility_data = [] # List of (timestamp, moneyness, implied_vol)
        self.last_smile_coeffs = None
        self.last_base_iv = None # Store the IV at moneyness=0 from the last fit
        
        # Initialize delta hedging state
        self.portfolio_delta = 0.0 # Current net delta of the options portfolio
        self.last_hedged_timestamp = 0 # Timestamp of last hedge
        self.hedge_frequency = 10 # How often to recalculate and adjust delta hedge

        # Inside your Trader class
        self.olivia_squid_ink_last_event_ts = -1

        # --- State for NEW Macarons Strategy ---
        self.previous_sunlight_index = None # To detect crossing CSI
        self.macarons_sunlight_history = [] # For trend calculation
        self.in_long_sunlight_position = False # Are we holding the special long position?
        self.long_entry_price = None # Average entry price of the special long
        self.position_entry_timestamp = None # Timestamp of entry for storage cost calc

        # --- State for Macarons Linear Regression --- #
        self.sugar_price_history = []
        self.macaron_price_history = []
        self.macaron_lr_coeffs = None # Stores (slope, intercept)
        # --- Internal LR Control Attributes ---
        self.lr_min_data_points = 20 # Min points needed to fit
        self.lr_fit_frequency = 50 # How often to refit (timestamps)
        self.lr_history_limit = 100 # Max history points to keep

    # --- Methods from v1.py (KELP and RESIN) ---
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.position_limits[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            if best_ask in order_depth.sell_orders:
                best_ask_amount = -1 * order_depth.sell_orders[best_ask]

                if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                    if best_ask <= fair_value - take_width:
                        quantity = min(
                            best_ask_amount, position_limit - position
                        )
                        if quantity > 0:
                            orders.append(Order(product, best_ask, quantity))
                            buy_order_volume += quantity
                            order_depth.sell_orders[best_ask] += quantity
                            if order_depth.sell_orders[best_ask] >= 0:
                                del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            if best_bid in order_depth.buy_orders:
                best_bid_amount = order_depth.buy_orders[best_bid]

                if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                    if best_bid >= fair_value + take_width:
                        quantity = min(
                            best_bid_amount, position_limit + position
                        )
                        if quantity > 0:
                            orders.append(Order(product, best_bid, -1 * quantity))
                            sell_order_volume += quantity
                            order_depth.buy_orders[best_bid] -= quantity
                            if order_depth.buy_orders[best_bid] <= 0:
                                del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        position_limit = self.position_limits[product]
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        position_limit = self.position_limits[product]
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price if last_price != 0 else 0
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def detect_spike(self, current_price, price_history, spike_threshold=0.02):
        """
        Detect price spikes based on a percentage threshold.
        
        Args:
            current_price: The current mid-price.
            price_history: List of recent prices.
            spike_threshold: Percentage change to detect a spike.
            
        Returns:
            "up" if an upward spike is detected, "down" if a downward spike is detected, None otherwise.
        """
        if len(price_history) < 5:  # Ensure enough data for comparison
            return None

        recent_price = self.calculate_most_common_fair_value(price_history)
        price_change = (current_price - recent_price) / recent_price 
        

        if price_change > spike_threshold:
            return "up"
        elif price_change < -spike_threshold:
            return "down"
        return None

    def calculate_most_common_fair_value(self, price_history):
            """
            Calculate the most common fair value from the price history.
            
            Args:
                price_history: List of recent prices.
                
            Returns:
                The most common fair value (mode) or None if insufficient data.
            """
            if len(price_history) < 5:  # Need enough data for meaningful statistics
                return None

            # Calculate the mode (most common value)

            return int(sum(sorted((price_history), key=price_history.count, reverse=True)[:4])/5)
    

    def react_to_olivia_squid_ink(self, state: TradingState) -> List[Order]:
        """
        Reacts to Olivia's SQUID_INK trades based on current position:
        - If position is 0: Enters max position in Olivia's direction on the first trade of an event.
        - If position is non-zero: Liquidates the position to 0 upon detecting a *new* Olivia trade event.
        - Ignores subsequent trades of the same type within the same logical event (based on timestamp).

        Args:
            state: The current TradingState object containing market data.

        Returns:
            A list containing a single Order object for SQUID_INK if an action
            is taken, otherwise an empty list.
        """
        orders: List[Order] = []
        product = Product.SQUID_INK

        if product not in state.market_trades:
            return orders

        # Find the latest trade involving Olivia in the current batch
        latest_olivia_trade_in_batch = None
        latest_ts_in_batch = -1

        squid_ink_trades = state.market_trades[product]
        for trade in squid_ink_trades:
            is_olivia_trade = False
            action = None
            if trade.buyer == "Olivia":
                is_olivia_trade = True
                action = 'buy'
            elif trade.seller == "Olivia":
                is_olivia_trade = True
                action = 'sell'

            if is_olivia_trade and trade.timestamp > latest_ts_in_batch:
                latest_ts_in_batch = trade.timestamp
                latest_olivia_trade_in_batch = {'timestamp': trade.timestamp, 'action': action, 'price': trade.price, 'quantity': trade.quantity} # Store relevant info

        # If no Olivia trade found in this batch, do nothing
        if latest_olivia_trade_in_batch is None:
            return orders

        latest_olivia_ts = latest_olivia_trade_in_batch['timestamp']
        latest_olivia_action = latest_olivia_trade_in_batch['action']

        # If this trade's timestamp is not newer than the last event we processed, ignore it
        if latest_olivia_ts <= self.olivia_squid_ink_last_event_ts:
            # print(f"DEBUG: Ignoring Olivia trade at {latest_olivia_ts} as it's not newer than last event ts {self.olivia_squid_ink_last_event_ts}")
            return orders

        # --- This trade represents a new event (or part of one) we haven't processed ---

        current_position = state.position.get(product, 0)
        order_depth = state.order_depths.get(product)

        if not order_depth:
            print(f"Warning: No order depth for {product}. Cannot react.")
            return orders

        placed_order = False # Flag to track if we place an order in this call

        # --- Logic based on current position ---
        if current_position == 0:
            # --- Stage 1: Enter Max Position ---
            # Check if this new action is different from the last recorded one (if any)
            # This prevents re-entering if we just liquidated based on an opposite signal
            # Although, the timestamp check should mostly handle this. Redundant check? Maybe remove.
            # if self.olivia_squid_ink_last_event_action is not None:
            #     print(f"DEBUG: Position is 0, but last action was {self.olivia_squid_ink_last_event_action}. Holding off entry.")
            #     return orders # Avoid immediate re-entry after liquidation

            print(f"Detected New Olivia {latest_olivia_action} at {latest_olivia_ts}. Position=0. Entering max.")
            if latest_olivia_action == 'buy':
                volume_to_buy = POSITION_LIMITS[product]
                if volume_to_buy > 0 and order_depth.sell_orders:
                    price = min(order_depth.sell_orders.keys())
                    order = Order(product, price, volume_to_buy)
                    orders.append(order)
                    print(f"-> Placing BUY order for {volume_to_buy} at {price}")
                    placed_order = True
                elif volume_to_buy <= 0: print("-> Already at buy limit.")
                else: print(f"-> Warning: No sell orders for {product} to enter long.")

            elif latest_olivia_action == 'sell':
                volume_to_sell = POSITION_LIMITS[product]
                if volume_to_sell > 0 and order_depth.buy_orders:
                    price = max(order_depth.buy_orders.keys())
                    order = Order(product, price, -volume_to_sell)
                    orders.append(order)
                    print(f"-> Placing SELL order for {volume_to_sell} at {price}")
                    placed_order = True
                elif volume_to_sell <= 0: print("-> Already at sell limit.")
                else: print(f"-> Warning: No buy orders for {product} to enter short.")

        else: # current_position != 0
            # --- Stage 2: Liquidate Existing Position ---
            # Any new Olivia trade event triggers liquidation if we hold a position
            print(f"Detected New Olivia {latest_olivia_action} at {latest_olivia_ts}. Position={current_position}. Liquidating.")
            if current_position > 0: # Liquidate long
                volume_to_liquidate = current_position
                if order_depth.buy_orders:
                    price = max(order_depth.buy_orders.keys())
                    order = Order(product, price, -volume_to_liquidate)
                    orders.append(order)
                    print(f"-> Liquidating long: Placing SELL order for {volume_to_liquidate} at {price}")
                    placed_order = True
                else: print(f"-> Warning: No buy orders for {product} to liquidate long.")

            elif current_position < 0: # Liquidate short
                volume_to_liquidate = -current_position
                if order_depth.sell_orders:
                    price = min(order_depth.sell_orders.keys())
                    order = Order(product, price, volume_to_liquidate)
                    orders.append(order)
                    print(f"-> Liquidating short: Placing BUY order for {volume_to_liquidate} at {price}")
                    placed_order = True
                else: print(f"-> Warning: No sell orders for {product} to liquidate short.")

        # Update state *only if* we actually placed an order based on this event
        if placed_order:
            self.olivia_squid_ink_last_event_ts = latest_olivia_ts
            # Optionally store the action that caused this reaction, could be useful for debugging
            # self.olivia_squid_ink_last_event_action = latest_olivia_action if current_position == 0 else 'liquidate'

        return orders

   

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume


    # --- Methods for Picnic Basket Strategy ---
    # --- Methods for Picnic Basket Strategy ---
    # Updated to accept product argument
    def get_midprice(self, product: str, order_depth: OrderDepth) -> float | None: # Added type hint None
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        # Fetch adverse_volume based on product
        # Use .get() for safer access, provide default dict {} and default value 0
        adverse_volume = self.params.get(product, {}).get("adverse_volume", 0)

        filtered_ask_prices = [
            price
            for price, vol in order_depth.sell_orders.items()
            if abs(vol) >= adverse_volume
        ]
        filtered_bid_prices = [
            price
            for price, vol in order_depth.buy_orders.items()
            if abs(vol) >= adverse_volume
        ]
        best_filtered_ask = min(filtered_ask_prices) if filtered_ask_prices else None
        best_filtered_bid = max(filtered_bid_prices) if filtered_bid_prices else None

        if best_filtered_ask is not None and best_filtered_bid is not None:
            # Ensure filtered bid < filtered ask before averaging
            if best_filtered_bid < best_filtered_ask:
                return (best_filtered_ask + best_filtered_bid) / 2
            else:
                # If crossed, fall back to regular midprice or handle differently
                pass # Or return (best_bid + best_ask) / 2

        # Fallback midprice if filtered prices are not available or crossed
        if best_bid < best_ask:
             return (best_bid + best_ask) / 2
        else:
             # Handle crossed book case if necessary (e.g., return None or average)
             return None # Or potentially (best_bid + best_ask) / 2 if acceptable
    
    def get_basket1_fair_value(self, order_depths, current_position):
        croissants_midprice = self.get_midprice(Product.CROISSANTS, order_depths.get(Product.CROISSANTS))
        jams_midprice = self.get_midprice(Product.JAMS, order_depths.get(Product.JAMS))
        djembe_midprice = self.get_midprice(Product.DJEMBE, order_depths.get(Product.DJEMBE))
        basket1_midprice = self.get_midprice(Product.PICNIC_BASKET1, order_depths.get(Product.PICNIC_BASKET1))

        if not basket1_midprice:
            return None
        if not croissants_midprice or not jams_midprice or not djembe_midprice:
            return basket1_midprice

        synthetic_midprice = (
            croissants_midprice * BASKET1_WEIGHTS[Product.CROISSANTS] +
            jams_midprice * BASKET1_WEIGHTS[Product.JAMS] +
            djembe_midprice * BASKET1_WEIGHTS[Product.DJEMBE]
        )

        synthetic_weight = self.params[Product.PICNIC_BASKET1]["synthetic_weight"]
        
        if abs(current_position) > 0:
            position_limit_ratio = abs(current_position) / POSITION_LIMITS[Product.PICNIC_BASKET1]
            decay_factor = math.exp(-1 * position_limit_ratio)
            synthetic_weight *= decay_factor
        return (1 - synthetic_weight) * basket1_midprice + synthetic_weight * synthetic_midprice

    def get_basket2_fair_value(self, order_depths, current_position):
        croissants_midprice = self.get_midprice(Product.CROISSANTS, order_depths.get(Product.CROISSANTS))
        jams_midprice = self.get_midprice(Product.JAMS, order_depths.get(Product.JAMS))
        basket2_midprice = self.get_midprice(Product.PICNIC_BASKET2, order_depths.get(Product.PICNIC_BASKET2))

        if not basket2_midprice:
            return None
        if not croissants_midprice or not jams_midprice:
            return basket2_midprice

        synthetic_midprice = (
            croissants_midprice * BASKET2_WEIGHTS[Product.CROISSANTS] +
            jams_midprice * BASKET2_WEIGHTS[Product.JAMS]
        )

        synthetic_weight = self.params[Product.PICNIC_BASKET2]["synthetic_weight"]
        
        if abs(current_position) > 0:
            position_limit_ratio = abs(current_position) / POSITION_LIMITS[Product.PICNIC_BASKET2]
            decay_factor = math.exp(-1 * position_limit_ratio)
            synthetic_weight *= decay_factor
        return (1 - synthetic_weight) * basket2_midprice + synthetic_weight * synthetic_midprice



        order_depths = state.order_depths
        result = {}

        if is_buy_basket:  # Buy basket, Sell components
            # Basket Order
            basket_price = min(order_depths[Product.PICNIC_BASKET2].sell_orders.keys())
            result[Product.PICNIC_BASKET2] = [Order(Product.PICNIC_BASKET2, basket_price, volume)]


        else:  # Sell basket, Buy components
            # Basket Order
            basket_price = max(order_depths[Product.PICNIC_BASKET2].buy_orders.keys())
            result[Product.PICNIC_BASKET2] = [Order(Product.PICNIC_BASKET2, basket_price, -volume)]


        return result

    def trade_baskets(self, state: TradingState, basket) -> Dict[str, List[Order]]:
        """Main logic for basket arbitrage - Renamed from trade"""
        if basket == 1:
            
            basket1_position = (
                    state.position[Product.PICNIC_BASKET1]
                    if Product.PICNIC_BASKET1 in state.position
                    else 0
                )
            basket1_fair_value = self.get_basket1_fair_value(state.order_depths, basket1_position)
            
            basket1_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.PICNIC_BASKET1,
                state.order_depths[Product.PICNIC_BASKET1],
                basket1_fair_value,
                self.params[Product.PICNIC_BASKET1]["take_width"],
                basket1_position,
                self.params[Product.PICNIC_BASKET1]["prevent_adverse"],
                self.params[Product.PICNIC_BASKET1]["adverse_volume"],
            )
            
            basket1_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.PICNIC_BASKET1,
                state.order_depths[Product.PICNIC_BASKET1],
                basket1_fair_value,
                self.params[Product.PICNIC_BASKET1]["clear_width"],
                basket1_position,
                buy_order_volume,
                sell_order_volume,
            )
            basket1_make_orders, _, _ = self.make_orders(
                Product.PICNIC_BASKET1,
                state.order_depths[Product.PICNIC_BASKET1],
                basket1_fair_value,
                basket1_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.PICNIC_BASKET1]["disregard_edge"],
                self.params[Product.PICNIC_BASKET1]["join_edge"],
                self.params[Product.PICNIC_BASKET1]["default_edge"],
            )

            return basket1_make_orders + basket1_take_orders + basket1_clear_orders
        
        elif basket == 2:
            basket2_position = (
                    state.position[Product.PICNIC_BASKET2]
                    if Product.PICNIC_BASKET2 in state.position
                    else 0
                )
            basket2_fair_value = self.get_basket2_fair_value(state.order_depths, basket2_position)
            
            basket2_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.PICNIC_BASKET2,
                state.order_depths[Product.PICNIC_BASKET2],
                basket2_fair_value,
                self.params[Product.PICNIC_BASKET2]["take_width"],
                basket2_position,
                self.params[Product.PICNIC_BASKET2]["prevent_adverse"],
                self.params[Product.PICNIC_BASKET2]["adverse_volume"],
            )
            
            basket2_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.PICNIC_BASKET2,
                state.order_depths[Product.PICNIC_BASKET2],
                basket2_fair_value,
                self.params[Product.PICNIC_BASKET2]["clear_width"],
                basket2_position,
                buy_order_volume,
                sell_order_volume,
            )
            basket2_make_orders, _, _ = self.make_orders(
                Product.PICNIC_BASKET2,
                state.order_depths[Product.PICNIC_BASKET2],
                basket2_fair_value,
                basket2_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.PICNIC_BASKET2]["disregard_edge"],
                self.params[Product.PICNIC_BASKET2]["join_edge"],
                self.params[Product.PICNIC_BASKET2]["default_edge"],
            )

            return basket2_make_orders + basket2_take_orders + basket2_clear_orders


    # --- Volatility Smile Helper Functions ---

    def _get_midprice(self, product: str, state: TradingState) -> float | None:
        """Safely calculates the mid-price for a product."""
        if product not in state.order_depths:
            return None
        order_depth = state.order_depths[product]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def _cdf(self, x):
        """Cumulative distribution function for the standard normal distribution."""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def _black_scholes_price(self, S, K, T, r, sigma, is_call=True):
        """Calculate Black-Scholes option price."""
        if T <= 1e-9 or sigma <= 1e-9: # Avoid division by zero or instability near expiration/zero vol
             # Return intrinsic value if near expiration or vol is zero
             if is_call:
                 return max(0.0, S - K)
             else: # Put (not used here but for completeness)
                 return max(0.0, K - S)

        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if is_call:
            price = (S * self._cdf(d1) - K * math.exp(-r * T) * self._cdf(d2))
        else: # Put
            price = (K * math.exp(-r * T) * self._cdf(-d2) - S * self._cdf(-d1))
        return price

    def _implied_volatility(self, target_price, S, K, T, r, is_call=True, max_iter=100, tolerance=1e-5):
        """Calculate implied volatility using bisection method."""
        low_vol, high_vol = 0.0001, 5.0 # Reasonable volatility range bounds
        mid_vol = (low_vol + high_vol) / 2.0

        for _ in range(max_iter):
            price_at_mid = self._black_scholes_price(S, K, T, r, mid_vol, is_call)
            price_at_low = self._black_scholes_price(S, K, T, r, low_vol, is_call)

            if abs(price_at_mid - target_price) < tolerance:
                return mid_vol

            if (price_at_mid - target_price) * (price_at_low - target_price) < 0:
                 high_vol = mid_vol
            else:
                 low_vol = mid_vol
            mid_vol = (low_vol + high_vol) / 2.0

        # If not converged, return the midpoint of the last interval
        return mid_vol


    def _calculate_moneyness(self, S, K, T, sigma):
        """Calculate moneyness m_t = ln(S/K) / (sigma * sqrt(T))."""
        if sigma <= 1e-9 or T <= 1e-9:
            return 0.0 # Undefined or unstable, return neutral moneyness
        return math.log(S / K) / (sigma * math.sqrt(T))

    def _fit_smile_curve(self, volatility_data, degree):
        """Fits a polynomial curve to (moneyness, implied_vol) data."""
        if len(volatility_data) < degree + 1:
            return None # Not enough points to fit the curve

        timestamps, moneyness_values, implied_vols = zip(*volatility_data)
        
        # Filter out potential outliers or unstable values if needed (e.g., very high IVs)
        # For now, use all valid points
        valid_indices = [i for i, iv in enumerate(implied_vols) if iv is not None and iv > 1e-6 and iv < 5.0] # Basic filtering
        
        if len(valid_indices) < degree + 1:
             return None
             
        m_filtered = [moneyness_values[i] for i in valid_indices]
        iv_filtered = [implied_vols[i] for i in valid_indices]

        coeffs = np.polyfit(m_filtered, iv_filtered, degree)
        return np.poly1d(coeffs)

    def _calculate_time_to_expiration(self, timestamp: int, total_duration: int) -> float:
        """Calculates time to expiration T as a fraction of the total duration."""
        # Assuming expiration cycle repeats every total_duration timestamps
        timestamps_remaining = total_duration - (5000000 + (timestamp) % total_duration)
        # Return T as fraction of the period (e.g., if total duration is 1 year, T is in years)
        # Here, we treat the 700 timestamps as the full period. Black-Scholes T is typically annualized.
        # Let's keep T as a fraction of the 7-day cycle for consistency within the model.
        # If BS needs annualized T, we'd divide by (days_in_period * trading_days_year)
        # For now, using fraction of cycle: T = timestamps_remaining / total_duration
        T = max(0.0, timestamps_remaining / float(total_duration))
        return T

    def _option_delta(self, S, K, T, r, sigma, is_call=True):
        """
        Calculate the delta of an option using Black-Scholes formula.
        
        Args:
            S: The underlying price
            K: The strike price
            T: Time to expiration as a fraction of total duration
            r: Risk-free rate
            sigma: Implied volatility
            is_call: True for call options, False for put options
            
        Returns:
            The delta of the option (0 to 1 for calls, -1 to 0 for puts)
        """
        if T <= 1e-9 or sigma <= 1e-9:  # Avoid division by zero or instability
            # Return binary delta at expiration (0 or 1 for call, 0 or -1 for put)
            if is_call:
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
                
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        
        if is_call:
            return self._cdf(d1)  # Using our existing CDF function
        else:
            return self._cdf(d1) - 1.0  # Delta for puts: N(d1) - 1
    
    def _calculate_portfolio_delta(self, state, S, T, r, voucher_products, strikes, positions):
        """
        Calculate the net delta of the entire options portfolio.
        
        Args:
            state: TradingState object
            S: Current price of the underlying
            T: Time to expiration
            r: Risk-free rate
            voucher_products: List of option product names
            strikes: Dictionary mapping products to strike prices
            positions: Dictionary of current positions
            
        Returns:
            Net delta of the portfolio (positive means long delta exposure)
        """
        net_delta = 0.0
        
        for product in voucher_products:
            position = positions.get(product, 0)
            if position == 0:
                continue  # Skip products with no position
                
            K = strikes[product]
            
            # Get the implied volatility - either from current market or from our smile fit
            market_iv = None
            if product in state.order_depths:
                mid_price = self._get_midprice(product, state)
                if mid_price is not None:
                    try:
                        market_iv = self._implied_volatility(mid_price, S, K, T, r)
                    except Exception as e:
                        print(f"Error calculating IV for delta hedging: {e}")
            
            # Fall back to our base IV if market IV calculation failed
            if market_iv is None and self.last_base_iv is not None:
                market_iv = self.last_base_iv
            
            # If we still don't have a valid IV, use a conservative default
            if market_iv is None:
                market_iv = 0.3  # Conservative default
            
            # Calculate position delta (assuming all products are call options)
            is_call = True  # These are all call options
            option_delta = self._option_delta(S, K, T, r, market_iv, is_call)
            position_delta = option_delta * position
            net_delta += position_delta
            
            print(f"Product {product}: Position {position}, Delta {option_delta:.4f}, Position Delta {position_delta:.4f}")
            
        return net_delta
    
    def _generate_delta_hedge_orders(self, state, underlying, net_delta, hedge_ratio=1.0, max_hedge_size=80):
        """
        Generate orders to hedge delta exposure.
        
        Args:
            state: TradingState object
            underlying: The product to use for hedging (e.g., VOLCANIC_ROCK)
            net_delta: The net delta of the portfolio to hedge
            hedge_ratio: Portion of delta to hedge (1.0 = fully hedge)
            max_hedge_size: Maximum size of hedge order
            
        Returns:
            List of orders for the underlying product to hedge delta
        """
        if abs(net_delta) < 0.5:  # Don't hedge tiny delta exposures
            print(f"Net delta {net_delta:.4f} too small to hedge")
            return []
            
        # Convert delta to hedge order size (opposite sign)
        hedge_size = -round(net_delta * hedge_ratio)
        
        # Cap hedge size
        if abs(hedge_size) > max_hedge_size:
            hedge_size = max_hedge_size if hedge_size > 0 else -max_hedge_size
            
        # Check position limit
        current_position = state.position.get(underlying, 0)
        position_limit = self.position_limits[underlying]
        
        if hedge_size > 0:  # Buying
            max_allowed = position_limit - current_position
            hedge_size = min(hedge_size, max_allowed)
        else:  # Selling
            max_allowed = position_limit + current_position
            hedge_size = max(-max_allowed, hedge_size)
            
        # If after all caps hedge_size is too small, don't hedge
        if abs(hedge_size) < 1:
            return []
            
        # Get mid price for the underlying
        mid_price = self._get_midprice(underlying, state)
        if mid_price is None:
            print(f"Cannot hedge - no mid price for {underlying}")
            return []
            
        # Create hedge order
        hedge_price = round(mid_price)
        print(f"Hedging delta: {net_delta:.4f} with {hedge_size} units of {underlying} at price {hedge_price}")
        
        return [Order(underlying, hedge_price, hedge_size)]
    
    # --- Volatility Smile Trading Logic ---

    def trade_volatility_smile(self, state: TradingState) -> Dict[str, List[Order]]:
        """Implements the volatility smile trading strategy."""
        orders = {}
        smile_params = self.params["volatility_smile_strategy"]
        underlying = smile_params["underlying_product"]
        voucher_products = smile_params["voucher_products"]
        strikes = smile_params["strikes"]
        r = smile_params["risk_free_rate"]
        fit_freq = smile_params["volatility_fit_frequency"]
        fit_degree = smile_params["smile_fit_degree"]
        max_data_points = smile_params["volatility_data_points"]
        trading_threshold_iv_diff = smile_params["trading_threshold_iv_diff"]
        max_order_size = smile_params["max_order_size"]
        min_T = smile_params["min_time_to_expiration"]
        total_duration = smile_params["total_duration_timestamps"]

        # 1. Calculate Underlying Price (S) and Time to Expiration (T)
        S = self._get_midprice(underlying, state)
        if S is None:
            return {} # Cannot price options without underlying

        T = self._calculate_time_to_expiration(state.timestamp, total_duration)

        if T < min_T:
            return {} # Cannot trade near expiration

        # 2. Calculate and Store Implied Volatility and Moneyness for each Voucher
        current_m_iv = {}
        for product in voucher_products:
            mid_price = self._get_midprice(product, state)
            K = strikes[product]
            if mid_price is not None and S > 0:
                try:
                    # Ensure mid_price used for IV calculation is positive
                    if mid_price <= 0:
                         print(f"Timestamp {state.timestamp}: Warning - Mid price for {product} is non-positive ({mid_price}), skipping IV calc.")
                         current_m_iv[product] = (None, None)
                         continue

                    iv = self._implied_volatility(mid_price, S, K, T, r)
                    m = self._calculate_moneyness(S, K, T, iv)
                    self.volatility_data.append((state.timestamp, m, iv))
                    current_m_iv[product] = (m, iv)
                except (ValueError, OverflowError, ZeroDivisionError) as e:
                     print(f"Timestamp {state.timestamp}: Error calculating IV/Moneyness for {product}: {e}")
                     current_m_iv[product] = (None, None)
            else:
                 current_m_iv[product] = (None, None)


        # Keep only the most recent data points
        self.volatility_data = self.volatility_data[-max_data_points:]

        # 3. Fit Smile Curve Periodically
        if state.timestamp > 0 and state.timestamp % fit_freq == 0:
            smile_poly = self._fit_smile_curve(self.volatility_data, fit_degree)
            if smile_poly is not None:
                self.last_smile_coeffs = smile_poly.coeffs
                # Calculate and store base_iv = smile_poly(0)
                base_iv = smile_poly(0)
                # Ensure base_iv is positive before storing
                self.last_base_iv = max(1e-6, base_iv) # Store base IV for trading decisions, floor at small positive
                print(f"Timestamp {state.timestamp}: Fitted smile. Base IV: {self.last_base_iv:.4f}")
            else:
                 print(f"Timestamp {state.timestamp}: Failed to fit smile curve (insufficient data?).")
                 # Keep old coeffs and base IV? Or set to None? Let's keep old ones for now.
                 pass


        # 4. Generate Trades based on Deviation from Fitted Smile
        # Ensure we have a fitted smile and a base IV to compare against
        if self.last_smile_coeffs is not None and self.last_base_iv is not None:
            base_iv = self.last_base_iv # Use the stored base IV from the last fit

            for product in voucher_products:
                pos = state.position.get(product, 0)
                pos_limit = self.position_limits[product]
                market_mid_price = self._get_midprice(product, state)
                m, market_iv = current_m_iv.get(product, (None, None))
                K = strikes[product]

                if market_mid_price is None or m is None or market_iv is None:
                    continue # Skip if we don't have valid market data

                try:
                    # --- New Trading Logic: Compare market IV to base IV ---
                    product_orders = []
                    buy_signal = market_iv < base_iv - trading_threshold_iv_diff
                    sell_signal = market_iv > base_iv + trading_threshold_iv_diff

                    # Buy if market IV is significantly below base IV
                    if buy_signal:
                        available_buy_limit = pos_limit - pos
                        order_volume = min(max_order_size, available_buy_limit)
                        if order_volume > 0:
                            buy_price = math.ceil(market_mid_price) # Aggressive buy
                            print(f"Timestamp {state.timestamp}: BUY {product} - Market IV: {market_iv:.4f}, Base IV: {base_iv:.4f}, Vol: {order_volume}")
                            product_orders.append(Order(product, buy_price, order_volume))


                    # Sell if market IV is significantly above base IV
                    elif sell_signal:
                        available_sell_limit = pos_limit + pos # Position is negative for shorts
                        order_volume = min(max_order_size, available_sell_limit)
                        if order_volume > 0:
                            sell_price = math.floor(market_mid_price) # Aggressive sell
                            print(f"Timestamp {state.timestamp}: SELL {product} - Market IV: {market_iv:.4f}, Base IV: {base_iv:.4f}, Vol: {order_volume}")
                            product_orders.append(Order(product, sell_price, -order_volume))

                    if product_orders:
                       if product in orders:
                           orders[product].extend(product_orders)
                       else:
                           orders[product] = product_orders

                except (ValueError, OverflowError, ZeroDivisionError) as e:
                     print(f"Timestamp {state.timestamp}: Error during trading logic for {product}: {e}")

        # 5. Delta Hedging
        # Calculate and hedge if:
        # - We have new trades, or
        # - We haven't hedged in a while, or 
        # - It's the first timestamp
        is_first_timestamp = state.timestamp == 0
        has_new_trades = bool(orders)
        time_to_rehedge = (state.timestamp - self.last_hedged_timestamp) >= self.hedge_frequency
        
        if has_new_trades or time_to_rehedge or is_first_timestamp:
            print(f"Timestamp {state.timestamp}: Delta hedging check")
            
            # Calculate current portfolio delta
            net_delta = self._calculate_portfolio_delta(
                state, S, T, r, voucher_products, strikes, state.position
            )
            self.portfolio_delta = net_delta
            print(f"Timestamp {state.timestamp}: Current portfolio delta: {net_delta:.4f}")
            
            # Generate hedge orders
            hedge_ratio = 0.9  # Hedge 90% of delta to avoid over-hedging
            max_hedge_size = 40  # Maximum hedge size per order
            hedge_orders = self._generate_delta_hedge_orders(
                state, underlying, net_delta, hedge_ratio, max_hedge_size
            )
            
            # Add hedge orders to result
            if hedge_orders:
                if underlying in orders:
                    orders[underlying].extend(hedge_orders)
                else:
                    orders[underlying] = hedge_orders
                    
                self.last_hedged_timestamp = state.timestamp
                print(f"Timestamp {state.timestamp}: Delta hedge executed")

        return orders


    # --- Main Run Method ---
    def run(self, state: TradingState) -> Tuple[Dict[Product, List[Order]], int]:
        """
        Main trading logic
        """
        print(state.observations)
        # --- Load State from previous timestep (like round1/trader.py) ---
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
                # Restore strategy state if necessary
                self.basket1_price_diffs = traderObject.get('basket1_price_diffs', [])
                self.basket2_price_diffs = traderObject.get('basket2_price_diffs', [])
                # Restore volatility smile state
                self.volatility_data = traderObject.get('volatility_data', [])
                self.last_smile_coeffs = traderObject.get('last_smile_coeffs', None) # Load coeffs directly
                self.last_base_iv = traderObject.get('last_base_iv', None) # Load base IV
                # Restore delta hedging state
                self.portfolio_delta = traderObject.get('portfolio_delta', 0.0)
                self.last_hedged_timestamp = traderObject.get('last_hedged_timestamp', 0)
                # Restore basket price history state <<< ADDED
                self.basket1_synthetic_bids_history = traderObject.get('basket1_synthetic_bids_history', [])
                self.basket1_synthetic_asks_history = traderObject.get('basket1_synthetic_asks_history', [])
                self.basket2_synthetic_bids_history = traderObject.get('basket2_synthetic_bids_history', [])
                self.basket2_synthetic_asks_history = traderObject.get('basket2_synthetic_asks_history', [])
                # --- Load NEW Macarons State ---
                self.previous_sunlight_index = traderObject.get('previous_sunlight_index', None)
                self.macarons_sunlight_history = traderObject.get('macarons_sunlight_history', [])
                self.in_long_sunlight_position = traderObject.get('in_long_sunlight_position', False)
                self.long_entry_price = traderObject.get('long_entry_price', None)
                self.position_entry_timestamp = traderObject.get('position_entry_timestamp', None)
                # --- Load Macarons LR State ---
                self.sugar_price_history = traderObject.get('sugar_price_history', [])
                self.macaron_price_history = traderObject.get('macaron_price_history', [])
                self.macaron_lr_coeffs = traderObject.get('macaron_lr_coeffs', None)

                # Note: 'kelp_last_price' will be handled by kelp_fair_value method
            except Exception as e:
                print(f"Error decoding traderData: {e}")
                traderObject = {} # Reset state on error
                self.basket1_price_diffs = [] # Reset list state as well
                self.basket2_price_diffs = []
                self.volatility_data = []
                self.last_smile_coeffs = None
                self.last_base_iv = None
                self.portfolio_delta = 0.0
                self.last_hedged_timestamp = 0
                # Reset basket history state <<< ADDED
                self.basket1_synthetic_bids_history = []
                self.basket1_synthetic_asks_history = []
                self.basket2_synthetic_bids_history = []
                self.basket2_synthetic_asks_history = []
                self.basket1_price_diffs = [] # <<< ADDED BACK
                self.basket2_price_diffs = [] # <<< ADDED BACK
                # --- Reset NEW Macarons State --- #
                self.previous_sunlight_index = None
                self.macarons_sunlight_history = []
                self.in_long_sunlight_position = False
                self.long_entry_price = None
                self.position_entry_timestamp = None
                # --- Reset Macarons LR State ---
                self.sugar_price_history = []
                self.macaron_price_history = []
                self.macaron_lr_coeffs = None
        # --- End State Loading ---

        # --- Initialize Observations -- #
        # --- CORRECTED: Get Macaron obs without overwriting state.observations ---
        macaron_obs = None
        if hasattr(state.observations, 'conversionObservations') and Product.MAGNIFICENT_MACARONS in state.observations.conversionObservations:
            macaron_obs = state.observations.conversionObservations[Product.MAGNIFICENT_MACARONS]
            print(f"[Macarons] Found conversion obs.")
        else:
            print(f"[Macarons] No conversion obs found.")
        # ------------------------------------------------------------------------

        # --- Collect Data for LR --- #
        sugar_mid_price = self._get_midprice(Product.SUGAR, state) # Make sure SUGAR exists!
        macaron_mid_price = self._get_midprice(Product.MAGNIFICENT_MACARONS, state)

        if sugar_mid_price is not None and macaron_mid_price is not None:
            self.sugar_price_history.append(sugar_mid_price)
            self.macaron_price_history.append(macaron_mid_price)
            # Limit history size
            if len(self.sugar_price_history) > self.lr_history_limit:
                self.sugar_price_history.pop(0)
                self.macaron_price_history.pop(0)

        # --- Fit LR Model Periodically --- #
        if len(self.sugar_price_history) >= self.lr_min_data_points and \
           state.timestamp % self.lr_fit_frequency == 0:
            try:
                # Ensure lists are of equal length before fitting
                min_len = min(len(self.sugar_price_history), len(self.macaron_price_history))
                coeffs = np.polyfit(self.sugar_price_history[-min_len:], self.macaron_price_history[-min_len:], 1)
                self.macaron_lr_coeffs = coeffs
                print(f"MACARONS LR: Refitted. Coeffs (Slope, Intercept): {coeffs}")
            except Exception as e:
                print(f"MACARONS LR: Error fitting model: {e}")
                # Keep old coeffs if fit fails


        # Initialize result dictionary
        result: Dict[Product, List[Order]] = {}
        conversions = 0



        # # KELP and RAINFOREST_RESIN Logic
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            resin_params = self.params[Product.RAINFOREST_RESIN]
            rainforest_resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            rainforest_resin_od = state.order_depths[Product.RAINFOREST_RESIN]

            rainforest_resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    rainforest_resin_od, # Pass original
                    resin_params["fair_value"],
                    resin_params["take_width"],
                    rainforest_resin_position
                )
            )
            rainforest_resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    rainforest_resin_od, # Pass original (potentially modified by take_orders)
                    resin_params["fair_value"],
                    resin_params["clear_width"],
                    rainforest_resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            rainforest_resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                rainforest_resin_od, # Pass original (potentially modified)
                resin_params["fair_value"],
                rainforest_resin_position,
                buy_order_volume,
                sell_order_volume,
                resin_params["disregard_edge"],
                resin_params["join_edge"],
                resin_params["default_edge"],
                True, # manage_position=True
                resin_params["soft_position_limit"],
            )
            # Combine orders for the product
            all_resin_orders = rainforest_resin_take_orders + rainforest_resin_clear_orders + rainforest_resin_make_orders
            if all_resin_orders: # Only add if there are orders
                 result[Product.RAINFOREST_RESIN] = all_resin_orders


        if Product.KELP in self.params and Product.KELP in state.order_depths:
             # Ensure we have the product parameters
            kelp_params = self.params[Product.KELP]
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_od = state.order_depths[Product.KELP]
            # Pass the loaded traderObject for state persistence
            kelp_fair_value = self.kelp_fair_value(kelp_od, traderObject)

            if kelp_fair_value is not None: # Only trade if fair value is calculated
                kelp_take_orders, buy_order_volume, sell_order_volume = (
                    self.take_orders(
                        Product.KELP,
                        kelp_od, # Pass original
                        kelp_fair_value,
                        kelp_params["take_width"],
                        kelp_position,
                        kelp_params["prevent_adverse"],
                        kelp_params["adverse_volume"],
                    )
                )
                kelp_clear_orders, buy_order_volume, sell_order_volume = (
                    self.clear_orders(
                        Product.KELP,
                        kelp_od, # Pass original (potentially modified)
                        kelp_fair_value,
                        kelp_params["clear_width"],
                        kelp_position,
                        buy_order_volume,
                        sell_order_volume,
                    )
                )
                kelp_make_orders, _, _ = self.make_orders(
                    Product.KELP,
                    kelp_od, # Pass original (potentially modified)
                    kelp_fair_value,
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                    kelp_params["disregard_edge"],
                    kelp_params["join_edge"],
                    kelp_params["default_edge"],
                    # manage_position=False for KELP in v1
                )
                # Combine orders for the product
                all_kelp_orders = kelp_take_orders + kelp_clear_orders + kelp_make_orders
                if all_kelp_orders: # Only add if there are orders
                    result[Product.KELP] = all_kelp_orders

        # # SQUID_INK trading logic
        # # SQUID_INK trading logic
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            result[Product.SQUID_INK] = self.react_to_olivia_squid_ink(state)

        # Picnic Basket Arbitrage Logic
        if Product.PICNIC_BASKET1 in self.params and Product.PICNIC_BASKET1 in state.order_depths:
            basket1_orders = self.trade_baskets(state, basket=1)
            if basket1_orders:
                result[Product.PICNIC_BASKET1] = basket1_orders

        if Product.PICNIC_BASKET2 in self.params and Product.PICNIC_BASKET2 in state.order_depths:
            basket2_orders = self.trade_baskets(state, basket=2)
            if basket2_orders:
                result[Product.PICNIC_BASKET2] = basket2_orders

        # # Volatility Smile Trading Logic
        voucher_orders = self.trade_volatility_smile(state)
        for product, orders_list in voucher_orders.items():
            if orders_list: # Only add if there are orders
                if product in result:
                    result[product].extend(orders_list)
                else:
                    result[product] = orders_list

                # --- COMPLETELY REVISED Magnificent Macarons Strategy --- #
        
        if Product.MAGNIFICENT_MACARONS in state.order_depths:
            product = Product.MAGNIFICENT_MACARONS
            macaron_params = self.params[product]
            macaron_pos_limit = self.position_limits[product]
            macaron_orders = []
            trade_signal_fired = False

            current_sunlight = None
            if macaron_obs is not None:
                current_sunlight = macaron_obs.sunlightIndex
                print(f"MACARONS - Current Sunlight: {current_sunlight}")
                # Update sunlight history for trend calculation
                self.macarons_sunlight_history.append(current_sunlight)
                trend_window = macaron_params['sunlight_window']
                if len(self.macarons_sunlight_history) > trend_window:
                    self.macarons_sunlight_history.pop(0)
            else:
                print("MACARONS - No sunlight observation, cannot execute strategy.")

            if current_sunlight is not None: # Only proceed if we have sunlight data
                critical_sunlight = macaron_params['critical_sunlight_index']
                storage_cost = macaron_params['storage_cost_per_unit_per_ts']
                loss_threshold = macaron_params['loss_prevention_threshold']

                # Get current position and market prices
                current_position = state.position.get(product, 0)
                macaron_od = state.order_depths[product] # <<< ADDED DEFINITION
                best_ask_orderbook = min(macaron_od.sell_orders.keys()) if macaron_od.sell_orders else float('inf')
                best_bid_orderbook = max(macaron_od.buy_orders.keys()) if macaron_od.buy_orders else 0

                # Calculate current sunlight trend
                current_trend = 0
                if len(self.macarons_sunlight_history) >= 2:
                    # Simple difference over the window
                    current_trend = self.macarons_sunlight_history[-1] - self.macarons_sunlight_history[0]
                print(f"MACARONS - Trend: {current_trend:.2f}")

                # --- State Machine Logic ---
                if self.in_long_sunlight_position:
                    print(f"MACARONS - STATE: Holding Long Position. Entry: Px={self.long_entry_price}, Ts={self.position_entry_timestamp}")
                    exit_signal = False
                    exit_reason = ""

                    # Check Exit 1: Trend Reversal Down
                    if current_trend < 0:
                        exit_signal = True
                        exit_reason = "Trend Reversal Down"
                    
                    # Check Exit 2: Loss Prevention
                    if not exit_signal and self.long_entry_price is not None and self.position_entry_timestamp is not None and best_bid_orderbook > 0:
                        holding_duration = max(0, state.timestamp - self.position_entry_timestamp)
                        total_storage_cost = current_position * storage_cost * holding_duration
                        potential_sell_value = current_position * best_bid_orderbook
                        initial_buy_value = current_position * self.long_entry_price
                        potential_pnl = potential_sell_value - initial_buy_value - total_storage_cost
                        print(f"MACARONS - Holding PnL Check: SellVal={potential_sell_value:.0f}, BuyVal={initial_buy_value:.0f}, Storage={total_storage_cost:.2f}, PnL={potential_pnl:.2f}, Threshold={loss_threshold}")
                        if potential_pnl <= loss_threshold:
                            exit_signal = True
                            exit_reason = f"Loss Prevention (PnL {potential_pnl:.2f})"

                    # Execute Exit if triggered
                    if exit_signal:
                         print(f"MACARONS - EXIT SIGNAL ({exit_reason}). Selling {current_position} at {best_bid_orderbook}")
                         if best_bid_orderbook > 0 and current_position > 0:
                             sell_quantity = current_position # Sell entire position
                             macaron_orders.append(Order(product, best_bid_orderbook, -sell_quantity))
                             trade_signal_fired = True
                             # <<< MOVED STATE RESET HERE >>>
                             # Reset state only if exit order was placed
                             self.in_long_sunlight_position = False
                             self.long_entry_price = None
                             self.position_entry_timestamp = None
                         else:
                            print("MACARONS - Cannot exit, no bid or no position?")
                         # Reset state regardless of fill success  # <<< COMMENT INDICATING OLD BEHAVIOR (now removed)
                         #   self.in_long_sunlight_position = False
                         #   self.long_entry_price = None
                         #   self.position_entry_timestamp = None
                    else:
                         print("MACARONS - Holding long position, no exit signal.")

                else: # Not in the special long position
                    print("MACARONS - STATE: Not Holding Long Position.")
                    # Check Entry 1: Sunlight Crossing Up
                    if self.previous_sunlight_index is not None and self.previous_sunlight_index < critical_sunlight and current_sunlight >= critical_sunlight:
                        print(f"MACARONS - ENTRY SIGNAL (Sunlight crossed {critical_sunlight} upwards). Buying {macaron_pos_limit} at {best_ask_orderbook}")
                        if best_ask_orderbook < float('inf'):
                            buy_quantity = macaron_pos_limit - current_position # Buy up to full limit
                            if buy_quantity > 0:
                                macaron_orders.append(Order(product, best_ask_orderbook, buy_quantity))
                                # Attempt to set state assuming fill - PnL calc needs accurate entry price/time
                                self.in_long_sunlight_position = True
                                # Ideally, use execution reports for entry price/time, but approximate with order price/time for now
                                self.long_entry_price = best_ask_orderbook 
                                self.position_entry_timestamp = state.timestamp
                                trade_signal_fired = True
                            else:
                                print("MACARONS - Sunlight cross signal, but already at/above limit?")
                        else:
                            print("MACARONS - Sunlight cross signal, but no ask to buy from.")
                    
                    # Check Market Making (Only if high sunlight AND we didn't just enter)
                    elif current_sunlight >= critical_sunlight and not trade_signal_fired:
                        print("MACARONS - MARKET MAKING Signal (Sunlight High & Not Holding Long)")
                        
                        # --- Predict Fair Value using LR --- #
                        predicted_fair_value = None
                        current_sugar_price = self._get_midprice(Product.SUGAR, state) # Get current sugar price again

                        if self.macaron_lr_coeffs is not None and current_sugar_price is not None:
                            slope, intercept = self.macaron_lr_coeffs
                            predicted_fair_value = slope * current_sugar_price + intercept
                            print(f"MACARONS LR: Predicted FV = {slope:.2f} * {current_sugar_price:.2f} + {intercept:.2f} = {predicted_fair_value:.2f}")
                        else:
                             print(f"MACARONS LR: Cannot predict FV (coeffs: {self.macaron_lr_coeffs is not None}, sugar: {current_sugar_price is not None})")
                        
                        # Use prediction or fallback
                        fair_value_to_use = predicted_fair_value if predicted_fair_value is not None else macaron_params['lr_fair_value_fallback']
                        print(f"MACARONS MM: Using Fair Value: {fair_value_to_use:.2f}")
                        # ------------------------------------ #
                        
                        trade_signal_fired = True
                        # Fetch other MM parameters
                        # fair_value_guess = macaron_params['market_making_fair_value'] # Now using fair_value_to_use
                        spread = macaron_params['market_making_spread']
                        order_size = macaron_params['market_making_order_size']
                        # Calculate MM bid/ask using dynamic fair value
                        bid_price = round(fair_value_to_use - spread / 2)
                        ask_price = round(fair_value_to_use + spread / 2)

                        # Calculate volume limits
                        buy_volume_limit = macaron_pos_limit - current_position
                        sell_volume_limit = macaron_pos_limit + current_position

                        # Place MM Buy Order
                        if buy_volume_limit > 0:
                            buy_qty = min(order_size, buy_volume_limit)
                            print(f"MACARONS - Placing MM Buy Order: Price {bid_price}, Qty {buy_qty}")
                            macaron_orders.append(Order(product, bid_price, buy_qty))
                       
                        # Place MM Sell Order
                        if sell_volume_limit > 0:
                            sell_qty = min(order_size, sell_volume_limit)
                            print(f"MACARONS - Placing MM Sell Order: Price {ask_price}, Qty {-sell_qty}")
                            macaron_orders.append(Order(product, ask_price, -sell_qty))
                    
                    # Else (Low sunlight and not holding): Do nothing
                    elif current_sunlight < critical_sunlight:
                        print("MACARONS - Sunlight Low & Not Holding Long. Waiting for upward cross.")

            # Add macaron orders to the result
            if macaron_orders:
                if product in result:
                    result[product].extend(macaron_orders)
                else:
                    result[product] = macaron_orders

            # Update previous sunlight state for the next iteration
            self.previous_sunlight_index = current_sunlight

        # --- END REVISED Macarons Strategy --- #
            product = Product.MAGNIFICENT_MACARONS

            


        # Save state for next iteration
        # traderObject['basket1_price_diffs'] = self.basket1_price_diffs
        # traderObject['basket2_price_diffs'] = self.basket2_price_diffs
        # kelp_last_price is already saved within kelp_fair_value by calling it
        # Save volatility smile state
        traderObject['volatility_data'] = self.volatility_data
        traderObject['last_smile_coeffs'] = self.last_smile_coeffs
        traderObject['last_base_iv'] = self.last_base_iv
        # Save delta hedging state
        traderObject['portfolio_delta'] = self.portfolio_delta
        traderObject['last_hedged_timestamp'] = self.last_hedged_timestamp
        # --- Save NEW Macarons State --- #
        traderObject['previous_sunlight_index'] = self.previous_sunlight_index
        traderObject['macarons_sunlight_history'] = self.macarons_sunlight_history
        traderObject['in_long_sunlight_position'] = self.in_long_sunlight_position
        traderObject['long_entry_price'] = self.long_entry_price
        traderObject['position_entry_timestamp'] = self.position_entry_timestamp
        # --- Save Macarons LR State --- #
        traderObject['sugar_price_history'] = self.sugar_price_history
        traderObject['macaron_price_history'] = self.macaron_price_history
        traderObject['macaron_lr_coeffs'] = self.macaron_lr_coeffs


        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData

