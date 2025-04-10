from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"


PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
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
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50, 
            Product.KELP: 50,
            Product.SQUID_INK: 50
        }

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
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
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
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
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
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

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
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None


    def calculate_z_score(self, price, price_history):
        """
        Calculate the Z-score of the current price relative to recent price history
        
        Args:
            price: Current price
            price_history: List of recent prices
            
        Returns:
            Z-score value or None if insufficient data
        """
        if len(price_history) < 5:  # Need enough data for meaningful statistics
            return None
        
        # Calculate moving average and standard deviation
        mean = sum(price_history) / len(price_history)
        variance = sum((x - mean) ** 2 for x in price_history) / len(price_history)
        std_dev = math.sqrt(variance)
        
        # Avoid division by zero
        if std_dev == 0:
            return 0
        
        # Calculate z-score
        z_score = (price - mean) / std_dev
        
        return z_score
    
    def squid_ink_fair_value(self, order_depth: OrderDepth, state: TradingState, traderObject) -> float:
        """
        Calculate fair value for SQUID_INK using Z-score based mean reversion
        """
        # Initialize product-specific data in trader object if not exists
        if "squid_ink" not in traderObject:
            traderObject["squid_ink"] = {
                "price_history": [],
                "last_z_score": None,
                "position_entry_price": None,
                "in_position": False,
                "position_side": None,  # "long" or "short"
                "stop_loss_level": None,
            }
        
        # Calculate mid price
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            
            # Add current price to history
            traderObject["squid_ink"]["price_history"].append(mid_price)
            
            # Keep only the last 20 prices (lookback window)
            if len(traderObject["squid_ink"]["price_history"]) > 20:
                traderObject["squid_ink"]["price_history"].pop(0)
            
            # Calculate z-score
            z_score = self.calculate_z_score(mid_price, traderObject["squid_ink"]["price_history"])
            traderObject["squid_ink"]["last_z_score"] = z_score
            
            # Use mid-price as fair value, Z-score will control entry/exit
            return mid_price
        
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
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
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

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        # RAINFOREST_RESIN trading logic
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            rainforest_resin_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            
            # Use static fair value for RAINFOREST_RESIN as defined in params
            rainforest_resin_fair_value = self.params[Product.RAINFOREST_RESIN]["fair_value"]
            
            rainforest_resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    rainforest_resin_fair_value,
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    rainforest_resin_position,
                )
            )
            rainforest_resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    rainforest_resin_fair_value,
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    rainforest_resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            rainforest_resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                rainforest_resin_fair_value,
                rainforest_resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                rainforest_resin_take_orders + rainforest_resin_clear_orders + rainforest_resin_make_orders
            )

        # KELP trading logic
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            kelp_fair_value = self.kelp_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            kelp_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["take_width"],
                    kelp_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            kelp_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                kelp_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = (
                kelp_take_orders + kelp_clear_orders + kelp_make_orders
            )
            
        # # SQUID_INK trading logic
        # if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
        #     squid_ink_position = (
        #         state.position[Product.SQUID_INK]
        #         if Product.SQUID_INK in state.position
        #         else 0
        #     )
            
        #     # Calculate fair value (mid price) for SQUID_INK
        #     squid_ink_fair_value = self.squid_ink_fair_value(
        #         state.order_depths[Product.SQUID_INK],
        #         state,
        #         traderObject
        #     )
            
        #     # If fair value calculation fails, use simple mid price
        #     if squid_ink_fair_value is None:
        #         best_ask = min(state.order_depths[Product.SQUID_INK].sell_orders.keys())
        #         best_bid = max(state.order_depths[Product.SQUID_INK].buy_orders.keys())
        #         squid_ink_fair_value = (best_ask + best_bid) / 2
                
        #     # Get z-score
        #     z_score = traderObject["squid_ink"].get("last_z_score", 0)
            
        #     orders = []
        #     buy_order_volume = 0
        #     sell_order_volume = 0
            
        #     # Check if we have a current position
        #     in_position = traderObject["squid_ink"].get("in_position", False)
        #     position_side = traderObject["squid_ink"].get("position_side", None)
        #     entry_price = traderObject["squid_ink"].get("position_entry_price", None)
        #     max_price = traderObject["squid_ink"].get("max_price", None)
        #     min_price = traderObject["squid_ink"].get("min_price", None)
            
        #     # Store previous z-score to detect reversal direction change
        #     prev_z_score = traderObject["squid_ink"].get("prev_z_score", z_score)
        #     traderObject["squid_ink"]["prev_z_score"] = z_score
            
        #     # Only initialize if we're in a position
        #     if in_position and entry_price is not None:
        #         if position_side == "long":
        #             # Initialize or update max price seen during this trade
        #             if max_price is None or squid_ink_fair_value > max_price:
        #                 traderObject["squid_ink"]["max_price"] = squid_ink_fair_value
        #                 max_price = squid_ink_fair_value
                        
        #             # Calculate current profit/loss
        #             profit_pct = (squid_ink_fair_value - entry_price) / entry_price
                    
        #             # AGGRESSIVE STOP LOSS CONDITIONS (any one triggers exit)
        #             stop_loss_triggered = False
                    
        #             # 1. Fixed stop loss - exit if we lose 0.5% from entry
        #             if squid_ink_fair_value <= entry_price * 0.995:
        #                 stop_loss_triggered = True
                        
        #             # 2. Trailing stop loss - exit if price drops 0.3% from highest seen
        #             if max_price is not None and squid_ink_fair_value <= max_price * 0.997:
        #                 stop_loss_triggered = True
                        
        #             # 3. Z-score reversal stop loss - exit if z-score movement reverses
        #             if z_score <= prev_z_score or z_score > 0:
        #                 stop_loss_triggered = True
                    
        #             if stop_loss_triggered:
        #                 # Use aggressive pricing to ensure immediate exit
        #                 # Find best bid (highest buy price available)
        #                 best_bid = max(state.order_depths[Product.SQUID_INK].buy_orders.keys())
                        
        #                 # Create a direct market order to sell our position immediately
        #                 if squid_ink_position > 0:
        #                     orders.append(Order(Product.SQUID_INK, best_bid, -squid_ink_position))
                            
        #                     # Reset position tracking
        #                     traderObject["squid_ink"]["in_position"] = False
        #                     traderObject["squid_ink"]["position_side"] = None
        #                     traderObject["squid_ink"]["position_entry_price"] = None
        #                     traderObject["squid_ink"]["max_price"] = None
        #                     traderObject["squid_ink"]["min_price"] = None
                
        #         elif position_side == "short":
        #             # Initialize or update min price seen during this trade
        #             if min_price is None or squid_ink_fair_value < min_price:
        #                 traderObject["squid_ink"]["min_price"] = squid_ink_fair_value
        #                 min_price = squid_ink_fair_value
                        
        #             # Calculate current profit/loss
        #             profit_pct = (entry_price - squid_ink_fair_value) / entry_price
                    
        #             # AGGRESSIVE STOP LOSS CONDITIONS (any one triggers exit)
        #             stop_loss_triggered = False
                    
        #             # 1. Fixed stop loss - exit if we lose 0.5% from entry
        #             if squid_ink_fair_value >= entry_price * 1.005:
        #                 stop_loss_triggered = True
                        
        #             # 2. Trailing stop loss - exit if price rises 0.3% from lowest seen
        #             if min_price is not None and squid_ink_fair_value >= min_price * 1.003:
        #                 stop_loss_triggered = True
                        
        #             # 3. Z-score reversal stop loss - exit if z-score movement reverses
        #             if z_score >= prev_z_score or z_score < 0:
        #                 stop_loss_triggered = True
                    
        #             if stop_loss_triggered:
        #                 # Use aggressive pricing to ensure immediate exit
        #                 # Find best ask (lowest sell price available)
        #                 best_ask = min(state.order_depths[Product.SQUID_INK].sell_orders.keys())
                        
        #                 # Create a direct market order to buy and cover our short position
        #                 if squid_ink_position < 0:
        #                     orders.append(Order(Product.SQUID_INK, best_ask, -squid_ink_position))  # Negative position requires positive order
                            
        #                     # Reset position tracking
        #                     traderObject["squid_ink"]["in_position"] = False
        #                     traderObject["squid_ink"]["position_side"] = None
        #                     traderObject["squid_ink"]["position_entry_price"] = None
        #                     traderObject["squid_ink"]["max_price"] = None
        #                     traderObject["squid_ink"]["min_price"] = None
            
        #     # ENTRY LOGIC - only if we're not already in a position
        #     if not in_position and z_score is not None:
        #         # LONG signal: Z-score below threshold indicates undervalued
        #         if z_score < -0.25 and squid_ink_position < 50:
        #             # Use take_orders to enter long position with aggressive pricing
        #             long_orders, buy_order_volume, sell_order_volume = self.take_orders(
        #                 Product.SQUID_INK,
        #                 state.order_depths[Product.SQUID_INK],
        #                 squid_ink_fair_value * 1.03,  # More aggressive - willing to pay 3% above mid
        #                 self.params[Product.SQUID_INK]["take_width"],
        #                 squid_ink_position,
        #                 self.params[Product.SQUID_INK]["prevent_adverse"],
        #                 self.params[Product.SQUID_INK]["adverse_volume"]
        #             )
                    
        #             # If orders were generated
        #             if long_orders:
        #                 orders.extend(long_orders)
        #                 # Record entry state
        #                 traderObject["squid_ink"]["in_position"] = True
        #                 traderObject["squid_ink"]["position_side"] = "long"
        #                 traderObject["squid_ink"]["position_entry_price"] = squid_ink_fair_value
        #                 traderObject["squid_ink"]["max_price"] = squid_ink_fair_value  # Initialize max price
        #                 traderObject["squid_ink"]["min_price"] = None  # Reset min price
                
        #         # SHORT signal: Z-score above threshold indicates overvalued
        #         elif z_score > 0.25 and squid_ink_position > -50:
        #             # Use take_orders to enter short position with aggressive pricing
        #             short_orders, buy_order_volume, sell_order_volume = self.take_orders(
        #                 Product.SQUID_INK,
        #                 state.order_depths[Product.SQUID_INK],
        #                 squid_ink_fair_value * 0.97,  # More aggressive - willing to sell 3% below mid
        #                 self.params[Product.SQUID_INK]["take_width"],
        #                 squid_ink_position,
        #                 self.params[Product.SQUID_INK]["prevent_adverse"],
        #                 self.params[Product.SQUID_INK]["adverse_volume"]
        #             )
                    
        #             # If orders were generated
        #             if short_orders:
        #                 orders.extend(short_orders)
        #                 # Record entry state
        #                 traderObject["squid_ink"]["in_position"] = True
        #                 traderObject["squid_ink"]["position_side"] = "short"
        #                 traderObject["squid_ink"]["position_entry_price"] = squid_ink_fair_value
        #                 traderObject["squid_ink"]["min_price"] = squid_ink_fair_value  # Initialize min price
        #                 traderObject["squid_ink"]["max_price"] = None  # Reset max price

        #     result[Product.SQUID_INK] = orders

        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData