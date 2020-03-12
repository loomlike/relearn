# TODO implement np.array interface into dispatch and delivery for efficiency

from collections import namedtuple
import random

import gym
from gym import spaces
import numpy as np
import pandas as pd


DeliveryType = namedtuple('DeliveryType', ['due', 'price', 'penalty'])
NEXT_DAY_DELIVERY = DeliveryType(due=2, price=1/2, penalty=1/4)
REGULAR_DELIVERY = DeliveryType(due=7, price=1/7, penalty=1/49)


class Item:
    # Made static to refer the value before init
    shape = 5
    columns = ('time', 'dest', 'due', 'price', 'penalty') 
    def __init__(self, t, dest, delivery_type):
        """
        Args:
            t (int): Ordered date (episode)
            dest (int): Delivery destination index
            delivery_type (DeliveryType): Delivery type
        """
        self.t = t
        self.dest = dest
        self.delivery_type = delivery_type

    def to_array(self):
        return np.array([
            float(self.t),
            self.dest,
            float(self.delivery_type.due),
            self.delivery_type.price,
            self.delivery_type.penalty,
        ])


class Warehouse:
    def __init__(self, capacity, delivery_map):
        """Logistics provider's distribution center

        Args:
            capacity (int): Warehouse capacity (the number of items)
            delivery_map (np.array[int]): Delivery cost map where indices of the list
                are the destination codes and values are the delivery costs.
        """
        self.capacity = capacity
        self.delivery_map = delivery_map
        
        self.inventory = np.array([], dtype=object)
        self.revenue = 0.0
        self.cost = 0.0
        
    def __len__(self):
        return len(self.inventory)

    def reset(self):
        self.inventory = np.array([], dtype=object)
        self.revenue = 0.0
        self.cost = 0.0

    def dispatch(self, t, items, cost=1.0):
        """Dispatch items from sellers to the logistics provider's warehouse.
        Also check the delivery date of the inventory items for the delay penalty.
        
        Args:
            t (int): Current date (episode)
            items (List[Item]): List of items
            cost (float): Dispatch cost
        
        Returns:
            inventory (np.array[Item]): Inventory items after the dispatch
            failed (np.array[Item]): Items that failed to dispatch because of the
                lack of capacity. These items will not be in the inventory.
            delay_penalty (float): Penalty for missing the delivery date.
        """
        delay_penalty = 0.0
        for item in self.inventory:
            if (t-item.t) > item.delivery_type.due:
                delay_penalty += item.delivery_type.penalty

        self.cost += cost + delay_penalty

        # TODO currently, no penalty for the failed dispatch
        failed = []
        over = len(self.inventory) + len(items) - self.capacity
        if over > 0:
            items, failed = items[:-over], items[-over:]

        self.inventory = np.concatenate((self.inventory, items))

        return self.inventory, np.array(failed), delay_penalty
    
    def deliver(self, item_ids):
        """Deliver items to the final destinations.
        Args:
            item_ids (List[int]): The indices of the inventory items to deliver

        TODO:
            failed_ids (np.array[Item]): Item ids that failed to deliver because of the
                lack of delivery capacity. These items will remain in the inventory.

        Returns:
            inventory (np.array[Item]): Inventory items after delivery
            revenue (float): Delivery revenue
            cost (float): Delivery cost 
        """
        # Get item ids in the inventory
        item_ids = np.array(item_ids)
        item_ids = item_ids[item_ids < len(self.inventory)]
        # Select item ids by delivery capacity
        # item_ids, failed_ids = item_ids[:self.delivery_capacity], item_ids[-self.delivery_capacity:]

        delivery = self.inventory[item_ids]
        self.inventory = np.delete(self.inventory, item_ids)

        # Calculate the revenue and cost from this delivery
        revenue = 0.0
        dest = set()
        for item in delivery:
            revenue += item.delivery_type.price
            dest.add(item.dest)
            
        # In current implementation, we ship the items of the same destination together. 
        cost = np.sum(self.delivery_map[list(dest)])
        
        self.revenue += revenue
        self.cost += cost
        
        return self.inventory, revenue, cost

    def to_array(self):
        """Return as np.array"""
        if len(self.inventory) == 0:
            return np.array([])

        state = np.array([item.to_array() for item in self.inventory])

        if len(self.inventory) > self.capacity:
            state = state[:self.capacity]
        elif len(self.inventory) < self.capacity:
            state = np.pad(
                state,
                ((0,self.capacity-len(self.inventory)),(0,0)),
                'constant'
            )

        return state
        
        
class SimpleLogistics(gym.Env):
    def __init__(
        self,
        T=10,
        capacity=100,
        num_locations=10,
        demand_fn=None,
        seed=None,
    ):
        """TODO: Environment description here
        Args:
            T (int): Maximum number of episodes
            capacity (int): Capacity of the warehouse
            num_locations (int): Number of locations to deliver items.
                Note - this is not the number of warehouses.
            demand_fn (callable): Demand generation function which returns the number
                of items (int). By default, we use Poisson(num_locations*2).
            seed (int): Random number seed
        """
        
        self.T = T
        self.t = 0  # Current time. Increase for each step.
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        delivery_map = np.random.rand(num_locations)
        self.warehouse = Warehouse(capacity, delivery_map)

        self.demand_fn = demand_fn
        
        # Action space: deliver n-th item or not
        self.action_space = spaces.MultiBinary(capacity)
        # Agent maps observations to actions to maximize reward
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(capacity, Item.shape),
            dtype=np.float32
        )
        
        # Other configurations
        self.delivery_types = (NEXT_DAY_DELIVERY, REGULAR_DELIVERY)

        self.reset()
        
    def reset(self, seed=None):
        self.t = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.warehouse.reset()
        # Initial inventory
        self.warehouse.dispatch(self.t, self.demand())
        
        return self.warehouse.to_array()

    def render(self):
        # TODO maybe render historical data as well based on the parameter
        # TODO choose between graph and array based on the parameter
        
        return pd.DataFrame(
            data=self.warehouse.to_array(),
            columns=Item.columns
        )
        
    def demand(self):
        """Generate item demands based on `demand_fn`"""
        if self.demand_fn is not None:
            num_items = self.demand_fn()
        else:
            num_items = np.random.poisson(len(self.warehouse.delivery_map)*2)

        return [
            Item(
                t=self.t,
                dest=random.randint(0, len(self.warehouse.delivery_map)-1),
                delivery_type=random.choice(self.delivery_types)
            ) for _ in range(num_items)
        ]
    
    def step(self, action):
        """Update the environment based on the agent step and return a reward.
        
        Args:
            action (List[int]): Agent action for the items in the inventory.
                E.g., deliver ith item if action[i] == 1
            
        Returns:
            state
            reward
            done
            info
        """
        self.t += 1
        
        # update state
        _, revenue, cost = self.warehouse.deliver(np.nonzero(action))
        _, failed, penalty = self.warehouse.dispatch(self.t, self.demand())

        total_revenue = self.warehouse.revenue
        total_cost = self.warehouse.cost

        reward = total_revenue - total_cost

        info = {
            "episode": self.t,
            "revenue": revenue,
            "penalty_and_cost": penalty+cost,
            "total_revenue": total_revenue,
            "total_cost": total_cost,
            "num_items_in_inventory": len(self.warehouse),
            "num_items_failed_to_dispatch": len(failed),
        }
        
        return self.warehouse.to_array(), reward, self.t==self.T, info
