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
                # Note: 'kelp_last_price' will be handled by kelp_fair_value method
            except Exception as e:
                print(f"Error decoding traderData: {e}")
                traderObject = {} # Reset state on error
                self.basket1_price_diffs = [] # Reset list state as well
                self.basket2_price_diffs = []
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


        # Picnic Basket Arbitrage Logic
        basket_orders = self.trade_baskets(state)
        for product, orders in basket_orders.items():
            if orders: # Only add if there are orders
                if product in result:
                    result[product].extend(orders)
                else:
                    result[product] = orders

        # Save state for next iteration
        traderObject['basket1_price_diffs'] = self.basket1_price_diffs
        traderObject['basket2_price_diffs'] = self.basket2_price_diffs
        # kelp_last_price is already saved within kelp_fair_value by calling it

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData

