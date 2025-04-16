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
    },
    # Parameters for basket1 strategy
    "basket1_strategy": {
        "zscore_threshold": 0,
        "min_profit_margin": 2, # Specific margin for basket 1 (if needed)
        "max_position_pct": 1, # Specific position % for basket 1
        "history_length": 1, # Specific history length for basket 1
    },
    # Parameters for basket2 strategy
    "basket2_strategy": {
        "zscore_threshold": 0.25,
        "min_profit_margin": 1, # Specific margin for basket 2
        "max_position_pct": 1, # Specific position % for basket 2
        "history_length": 25, # Specific history length for basket 2
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
    }
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
    # New limits
    Product.CROISSANTS: 250,
    Product.JAMS: 350,
    Product.DJEMBE: 60,
    Product.PICNIC_BASKET1: 60,
    Product.PICNIC_BASKET2: 100,
    # Voucher Limits
    Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
    Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
    Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
    Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
    Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
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

    def squid_ink_fair_value(self, order_depth: OrderDepth, price_history, p = 1) -> float:
        
        def AutoRegress(price_history, p = 1):
            if len(price_history) < p + 1:
                # Not enough data to calculate ARIMA
                return 0 if price_history else None
        
            # Calculate AR (autoregressive) component
            ar_terms = 0
            for i in range(1, p + 1):
                ar_terms += (price_history[-i] - price_history[-i - 1])

            return ar_terms / p if p > 0 else 0
        """
        Calculate the mid-price for SQUID_INK as the fair value.
        """
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2 + AutoRegress(price_history, p)
        return None

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
    def get_midprice(self, order_depth: OrderDepth) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def get_synthetic_basket1_price(self, order_depths) -> Tuple[float, float, float]:
        croissants_bid = max(order_depths[Product.CROISSANTS].buy_orders.keys()) if order_depths[Product.CROISSANTS].buy_orders else None
        croissants_ask = min(order_depths[Product.CROISSANTS].sell_orders.keys()) if order_depths[Product.CROISSANTS].sell_orders else None
        jams_bid = max(order_depths[Product.JAMS].buy_orders.keys()) if order_depths[Product.JAMS].buy_orders else None
        jams_ask = min(order_depths[Product.JAMS].sell_orders.keys()) if order_depths[Product.JAMS].sell_orders else None
        djembe_bid = max(order_depths[Product.DJEMBE].buy_orders.keys()) if order_depths[Product.DJEMBE].buy_orders else None
        djembe_ask = min(order_depths[Product.DJEMBE].sell_orders.keys()) if order_depths[Product.DJEMBE].sell_orders else None

        if None in [croissants_bid, croissants_ask, jams_bid, jams_ask, djembe_bid, djembe_ask]:
            return None, None, None

        synthetic_bid = (
            croissants_bid * BASKET1_WEIGHTS[Product.CROISSANTS] +
            jams_bid * BASKET1_WEIGHTS[Product.JAMS] +
            djembe_bid * BASKET1_WEIGHTS[Product.DJEMBE]
        )
        synthetic_ask = (
            croissants_ask * BASKET1_WEIGHTS[Product.CROISSANTS] +
            jams_ask * BASKET1_WEIGHTS[Product.JAMS] +
            djembe_ask * BASKET1_WEIGHTS[Product.DJEMBE]
        )
        synthetic_mid = (synthetic_bid + synthetic_ask) / 2 if synthetic_bid is not None and synthetic_ask is not None else None
        return synthetic_bid, synthetic_ask, synthetic_mid


    def get_synthetic_basket2_price(self, order_depths) -> Tuple[float, float, float]:
        croissants_bid = max(order_depths[Product.CROISSANTS].buy_orders.keys()) if order_depths[Product.CROISSANTS].buy_orders else None
        croissants_ask = min(order_depths[Product.CROISSANTS].sell_orders.keys()) if order_depths[Product.CROISSANTS].sell_orders else None
        jams_bid = max(order_depths[Product.JAMS].buy_orders.keys()) if order_depths[Product.JAMS].buy_orders else None
        jams_ask = min(order_depths[Product.JAMS].sell_orders.keys()) if order_depths[Product.JAMS].sell_orders else None

        if None in [croissants_bid, croissants_ask, jams_bid, jams_ask]:
            return None, None, None

        synthetic_bid = (
            croissants_bid * BASKET2_WEIGHTS[Product.CROISSANTS] +
            jams_bid * BASKET2_WEIGHTS[Product.JAMS]
        )
        synthetic_ask = (
            croissants_ask * BASKET2_WEIGHTS[Product.CROISSANTS] +
            jams_ask * BASKET2_WEIGHTS[Product.JAMS]
        )
        synthetic_mid = (synthetic_bid + synthetic_ask) / 2 if synthetic_bid is not None and synthetic_ask is not None else None
        return synthetic_bid, synthetic_ask, synthetic_mid

    def max_executable_volume(self, product: str, is_buy: bool, current_position: int) -> int:
        max_position = self.position_limits[product]
        # Determine which strategy parameters to use based on product
        if product == Product.PICNIC_BASKET1:
            strategy_params = self.params["basket1_strategy"]
        elif product == Product.PICNIC_BASKET2:
            strategy_params = self.params["basket2_strategy"]
        else:
            # Handle other products or default behavior if necessary
            # For now, assume it's only called for baskets, might need adjustment
            # Or use a default if no specific basket params found
             return max_position # Or some other default logic

        max_allowed = int(max_position * strategy_params["max_position_pct"])

        if is_buy:
            return max(0, max_allowed - current_position)
        else:  # sell
            return max(0, max_allowed + current_position)

    def get_basket1_component_volumes(self, basket_volume: int) -> Dict[str, int]:
        return {
            Product.CROISSANTS: basket_volume * BASKET1_WEIGHTS[Product.CROISSANTS],
            Product.JAMS: basket_volume * BASKET1_WEIGHTS[Product.JAMS],
            Product.DJEMBE: basket_volume * BASKET1_WEIGHTS[Product.DJEMBE],
        }

    def get_basket2_component_volumes(self, basket_volume: int) -> Dict[str, int]:
        return {
            Product.CROISSANTS: basket_volume * BASKET2_WEIGHTS[Product.CROISSANTS],
            Product.JAMS: basket_volume * BASKET2_WEIGHTS[Product.JAMS],
        }

    def check_component_liquidity(self, order_depths, component_volumes, is_buy: bool) -> int:
        max_volumes = []

        for product, target_volume in component_volumes.items():
            if target_volume == 0: continue # Skip if weight is zero
            depth = order_depths.get(product)
            if not depth: return 0 # Product not available

            if is_buy:  # We're buying components (looking at sell side of component book)
                if not depth.sell_orders: return 0
                available_volume = sum(abs(qty) for qty in depth.sell_orders.values())
                max_executable = available_volume // abs(target_volume) if abs(target_volume) > 0 else float('inf')
                max_volumes.append(max_executable)

            else:  # We're selling components (looking at buy side of component book)
                if not depth.buy_orders: return 0
                available_volume = sum(qty for qty in depth.buy_orders.values())
                max_executable = available_volume // abs(target_volume) if abs(target_volume) > 0 else float('inf')
                max_volumes.append(max_executable)

        return min(max_volumes) if max_volumes else 0

    def can_execute_basket1_arbitrage(self, state) -> Tuple[bool, int, bool]:
        order_depths = state.order_depths
        # Still need all products for synthetic price calculation
        required_products = [Product.PICNIC_BASKET1, Product.CROISSANTS, Product.JAMS, Product.DJEMBE]
        if not all(product in order_depths and order_depths[product] for product in required_products):
             return False, 0, False

        positions = state.position
        basket_position = positions.get(Product.PICNIC_BASKET1, 0)
        basket_midprice = self.get_midprice(order_depths[Product.PICNIC_BASKET1])
        synthetic_bid, synthetic_ask, synthetic_midprice = self.get_synthetic_basket1_price(order_depths)

        if None in [basket_midprice, synthetic_bid, synthetic_ask, synthetic_midprice]:
            return False, 0, False

        # Use basket1 specific history length
        history_length = self.params["basket1_strategy"]["history_length"]
        price_diff = basket_midprice - synthetic_midprice
        self.basket1_price_diffs.append(price_diff)
        if len(self.basket1_price_diffs) > history_length:
            self.basket1_price_diffs.pop(0)

        if len(self.basket1_price_diffs) >= 30: # Keep minimum lookback for std dev calculation
            mean_diff = np.mean(self.basket1_price_diffs)
            std_diff = np.std(self.basket1_price_diffs)
            zscore = (price_diff - mean_diff) / std_diff if std_diff > 0 else 0
        else:
            zscore = price_diff / 10 # Placeholder until enough history

        basket_best_bid = max(order_depths[Product.PICNIC_BASKET1].buy_orders.keys()) if order_depths[Product.PICNIC_BASKET1].buy_orders else 0
        basket_best_ask = min(order_depths[Product.PICNIC_BASKET1].sell_orders.keys()) if order_depths[Product.PICNIC_BASKET1].sell_orders else float('inf')

        # Use basket1 specific parameters
        zscore_thresh = self.params["basket1_strategy"]["zscore_threshold"]
        min_profit = self.params["basket1_strategy"]["min_profit_margin"]

        # Case 1: Sell basket, buy components (Basket overpriced)
        if zscore > zscore_thresh and basket_best_bid > synthetic_ask + min_profit:
            max_volume = self.max_executable_volume(Product.PICNIC_BASKET1, False, basket_position)
            basket_sell_volume = min(max_volume, order_depths[Product.PICNIC_BASKET1].buy_orders.get(basket_best_bid, 0))
            if basket_sell_volume <= 0: return False, 0, False
            # Check component liquidity for buying
            component_volumes = self.get_basket1_component_volumes(basket_sell_volume)
            component_max_volume = self.check_component_liquidity(order_depths, component_volumes, True) # True = buying components
            executable_volume = min(basket_sell_volume, component_max_volume)
            return executable_volume > 0, executable_volume, False # False = selling basket

        # Case 2: Buy basket, sell components (Basket underpriced)
        elif zscore < -zscore_thresh and basket_best_ask < synthetic_bid - min_profit:
            max_volume = self.max_executable_volume(Product.PICNIC_BASKET1, True, basket_position)
            basket_buy_volume = min(max_volume, abs(order_depths[Product.PICNIC_BASKET1].sell_orders.get(basket_best_ask, 0)))
            if basket_buy_volume <= 0: return False, 0, False
            # Check component liquidity for selling
            component_volumes = self.get_basket1_component_volumes(basket_buy_volume)
            component_max_volume = self.check_component_liquidity(order_depths, component_volumes, False) # False = selling components
            executable_volume = min(basket_buy_volume, component_max_volume)
            return executable_volume > 0, executable_volume, True # True = buying basket

        return False, 0, False


    def can_execute_basket2_arbitrage(self, state) -> Tuple[bool, int, bool]:
        order_depths = state.order_depths
        required_products = [Product.PICNIC_BASKET2, Product.CROISSANTS, Product.JAMS]
        if not all(product in order_depths and order_depths[product] for product in required_products):
             return False, 0, False

        positions = state.position
        basket_position = positions.get(Product.PICNIC_BASKET2, 0)
        basket_midprice = self.get_midprice(order_depths[Product.PICNIC_BASKET2])
        synthetic_bid, synthetic_ask, synthetic_midprice = self.get_synthetic_basket2_price(order_depths)

        if None in [basket_midprice, synthetic_bid, synthetic_ask, synthetic_midprice]:
            return False, 0, False

        # Use basket2 specific history length
        history_length = self.params["basket2_strategy"]["history_length"]
        price_diff = basket_midprice - synthetic_midprice
        self.basket2_price_diffs.append(price_diff)
        if len(self.basket2_price_diffs) > history_length:
            self.basket2_price_diffs.pop(0)

        if len(self.basket2_price_diffs) >= 30: # Keep minimum lookback for std dev calculation
            mean_diff = np.mean(self.basket2_price_diffs)
            std_diff = np.std(self.basket2_price_diffs)
            zscore = (price_diff - mean_diff) / std_diff if std_diff > 0 else 0
        else:
            zscore = price_diff / 10 # Placeholder

        basket_best_bid = max(order_depths[Product.PICNIC_BASKET2].buy_orders.keys()) if order_depths[Product.PICNIC_BASKET2].buy_orders else 0
        basket_best_ask = min(order_depths[Product.PICNIC_BASKET2].sell_orders.keys()) if order_depths[Product.PICNIC_BASKET2].sell_orders else float('inf')

        # Use basket2 specific parameters
        zscore_thresh = self.params["basket2_strategy"]["zscore_threshold"]
        min_profit = self.params["basket2_strategy"]["min_profit_margin"]

        # Case 1: Sell basket, buy components
        if zscore > zscore_thresh and basket_best_bid > synthetic_ask + min_profit:
             max_volume = self.max_executable_volume(Product.PICNIC_BASKET2, False, basket_position)
             basket_sell_volume = min(max_volume, order_depths[Product.PICNIC_BASKET2].buy_orders.get(basket_best_bid, 0))
             if basket_sell_volume <= 0: return False, 0, False
             component_volumes = self.get_basket2_component_volumes(basket_sell_volume)
             component_max_volume = self.check_component_liquidity(order_depths, component_volumes, True)
             executable_volume = min(basket_sell_volume, component_max_volume)
             return executable_volume > 0, executable_volume, False

        # Case 2: Buy basket, sell components
        elif zscore < -zscore_thresh and basket_best_ask < synthetic_bid - min_profit:
             max_volume = self.max_executable_volume(Product.PICNIC_BASKET2, True, basket_position)
             basket_buy_volume = min(max_volume, abs(order_depths[Product.PICNIC_BASKET2].sell_orders.get(basket_best_ask, 0)))
             if basket_buy_volume <= 0: return False, 0, False
             component_volumes = self.get_basket2_component_volumes(basket_buy_volume)
             component_max_volume = self.check_component_liquidity(order_depths, component_volumes, False)
             executable_volume = min(basket_buy_volume, component_max_volume)
             return executable_volume > 0, executable_volume, True

        return False, 0, False

    def execute_basket1_arbitrage(self, state, volume: int, is_buy_basket: bool) -> Dict[str, List[Order]]:
        order_depths = state.order_depths
        result = {}

        # Note: Check was performed considering components, but execution is basket only
        if is_buy_basket:  # Buy basket ONLY
            basket_price = min(order_depths[Product.PICNIC_BASKET1].sell_orders.keys())
            result[Product.PICNIC_BASKET1] = [Order(Product.PICNIC_BASKET1, basket_price, volume)]
            # No component orders placed

        else:  # Sell basket ONLY
            basket_price = max(order_depths[Product.PICNIC_BASKET1].buy_orders.keys())
            result[Product.PICNIC_BASKET1] = [Order(Product.PICNIC_BASKET1, basket_price, -volume)]
            # No component orders placed

        return result

    def execute_basket2_arbitrage(self, state, volume: int, is_buy_basket: bool) -> Dict[str, List[Order]]:
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

    def trade_baskets(self, state: TradingState) -> Dict[str, List[Order]]:
        """Main logic for basket arbitrage - Renamed from trade"""
        basket_result = {}

        # Try basket1 arbitrage
        can_execute_b1, volume_b1, is_buy_b1 = self.can_execute_basket1_arbitrage(state)
        if can_execute_b1 and volume_b1 > 0:
            b1_orders = self.execute_basket1_arbitrage(state, volume_b1, is_buy_b1)
            for product, orders in b1_orders.items():
                basket_result[product] = orders # Overwrite or add

        # Try basket2 arbitrage
        can_execute_b2, volume_b2, is_buy_b2 = self.can_execute_basket2_arbitrage(state)
        if can_execute_b2 and volume_b2 > 0:
            b2_orders = self.execute_basket2_arbitrage(state, volume_b2, is_buy_b2)
            for product, orders in b2_orders.items():
                if product in basket_result:
                    basket_result[product].extend(orders) # Extend if exists
                else:
                    basket_result[product] = orders # Add if new

        return basket_result

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
        timestamps_remaining = total_duration - (3000000 + (timestamp) % total_duration)
        # Return T as fraction of the period (e.g., if total duration is 1 year, T is in years)
        # Here, we treat the 700 timestamps as the full period. Black-Scholes T is typically annualized.
        # Let's keep T as a fraction of the 7-day cycle for consistency within the model.
        # If BS needs annualized T, we'd divide by (days_in_period * trading_days_year)
        # For now, using fraction of cycle: T = timestamps_remaining / total_duration
        T = max(0.0, timestamps_remaining / float(total_duration))
        return T

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


        return orders


    # --- Main Run Method ---
    def run(self, state: TradingState):
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
                # Note: 'kelp_last_price' will be handled by kelp_fair_value method
            except Exception as e:
                print(f"Error decoding traderData: {e}")
                traderObject = {} # Reset state on error
                self.basket1_price_diffs = [] # Reset list state as well
                self.basket2_price_diffs = []
                self.volatility_data = []
                self.last_smile_coeffs = None
                self.last_base_iv = None
        # --- End State Loading ---

        result = {}
        conversions = 0 # Assuming no conversions needed for now

        # KELP and RAINFOREST_RESIN Logic (from v1.py)
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            # Ensure we have the product parameters
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

        # SQUID_INK trading logic
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            squid_ink_position = (
                state.position[Product.SQUID_INK]
                if Product.SQUID_INK in state.position
                else 0
            )

            # Get current price and update price history
            best_ask = min(state.order_depths[Product.SQUID_INK].sell_orders.keys())
            best_bid = max(state.order_depths[Product.SQUID_INK].buy_orders.keys())
            current_price = (best_ask + best_bid) / 2

            if "squid_ink" not in traderObject:
                traderObject["squid_ink"] = {"price_history": [], "most_common_fair_value": None}

            traderObject["squid_ink"]["price_history"].append(current_price)

            # Calculate fair value using the autoregress function with dynamic p
            autoregress_p = self.params[Product.SQUID_INK].get("autoregress_p", 1)
            squid_ink_fair_value = self.squid_ink_fair_value(
                state.order_depths[Product.SQUID_INK],
                traderObject["squid_ink"]["price_history"],
                autoregress_p
            )

            # If fair value calculation fails, skip trading
            if squid_ink_fair_value is None:
                return

            # Keep only the last 20 prices (lookback window)
            if len(traderObject["squid_ink"]["price_history"]) > 20:
                traderObject["squid_ink"]["price_history"].pop(0)

            # Detect price spikes
            spike = self.detect_spike(current_price, traderObject["squid_ink"]["price_history"], 0.1)

            orders = []

            if spike == "up" and squid_ink_position > -50:
                # Use dynamic divisor for spike trades
                spike_trade_divisor = self.params[Product.SQUID_INK].get("spike_trade_divisor", 4)
                quantity = min(squid_ink_position + 50, 8) // spike_trade_divisor
                orders.append(Order(Product.SQUID_INK, best_bid, -quantity))
                print(f"Detected upward spike. Entering SHORT position for SQUID_INK at {best_bid}, quantity: {quantity}")

            elif spike == "down" and squid_ink_position < 50:
                # Use dynamic divisor for spike trades
                spike_trade_divisor = self.params[Product.SQUID_INK].get("spike_trade_divisor", 4)
                quantity = min(50 - squid_ink_position, 8) // spike_trade_divisor
                orders.append(Order(Product.SQUID_INK, best_ask, quantity))
                print(f"Detected downward spike. Entering LONG position for SQUID_INK at {best_ask}, quantity: {quantity}")

            else:
                # Market making logic
                spread = 1  # Configurable spread
                bid_price = round(squid_ink_fair_value - spread)
                ask_price = round(squid_ink_fair_value + spread)

                # Place buy order
                buy_quantity = min(50 - squid_ink_position, 10)  # Limit position size
                if buy_quantity > 0:
                    orders.append(Order(Product.SQUID_INK, bid_price, buy_quantity))
                    print(f"Placing BUY order for SQUID_INK at {bid_price}, quantity: {buy_quantity}")

                # Place sell order
                sell_quantity = min(squid_ink_position + 50, 10)  # Limit position size
                if sell_quantity > 0:
                    orders.append(Order(Product.SQUID_INK, ask_price, -sell_quantity))
                    print(f"Placing SELL order for SQUID_INK at {ask_price}, quantity: {sell_quantity}")

            result[Product.SQUID_INK] = orders

        # Picnic Basket Arbitrage Logic
        basket_orders = self.trade_baskets(state)
        for product, orders in basket_orders.items():
            if orders: # Only add if there are orders
                if product in result:
                    result[product].extend(orders)
                else:
                    result[product] = orders

        # Volatility Smile Trading Logic
        voucher_orders = self.trade_volatility_smile(state)
        for product, orders_list in voucher_orders.items():
            if orders_list: # Only add if there are orders
                if product in result:
                    result[product].extend(orders_list)
                else:
                    result[product] = orders_list


        # Save state for next iteration
        traderObject['basket1_price_diffs'] = self.basket1_price_diffs
        traderObject['basket2_price_diffs'] = self.basket2_price_diffs
        # kelp_last_price is already saved within kelp_fair_value by calling it
        # Save volatility smile state
        traderObject['volatility_data'] = self.volatility_data
        traderObject['last_smile_coeffs'] = self.last_smile_coeffs
        traderObject['last_base_iv'] = self.last_base_iv


        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData

