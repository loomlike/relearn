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
    columns = ('t', 'dest', 'due', 'price', 'penalty')
    def __init__(self, t, dest, delivery_type):
        """
        Args:
            t (int): Item order time (step)
            dest (int): Delivery destination id
            delivery_type (DeliveryType): Delivery type including due, price, and penalty
        """
        self.t = t
        self.dest = dest
        self.due = delivery_type.due
        self.price = delivery_type.price
        self.penalty = delivery_type.penalty

    def to_array(self):
        return np.array([
            float(self.t),
            float(self.dest),
            float(self.due),
            self.price,
            self.penalty,
        ])


class Warehouse:
    columns = Item.columns + ('dest_cost', 'dest_cap', 't_left')

    def __init__(self, capacity, delivery_capacities, delivery_costs):
        """Logistics provider's distribution center

        Args:
            capacity (int): Warehouse capacity (the number of items).
            delivery_capacities (np.array[int]): Number of items that can be delivered to each destination at a time.
            delivery_costs (np.array[int]): Delivery cost to each destination.
                are the destination codes and values are the delivery costs.
        """
        assert len(delivery_capacities) == len(delivery_costs)

        self.capacity = capacity
        self.shape = (capacity, len(Warehouse.columns))
        self.delivery_capacities = delivery_capacities
        self.delivery_costs = delivery_costs

        self.reset()
        
    def __len__(self):
        return len(self.inventory)

    def reset(self):
        self.t = 0  # date (step)
        self.inventory = np.array([], dtype=object)
        self.revenue = 0.0
        self.cost = 0.0

    def update(self):
        """Update inventory status.
        """
        self.t += 1

    def dispatch(self, items, cost=1.0):
        """Dispatch items from sellers to the logistics provider's warehouse.
        Also check the delivery date of the inventory items for the delay penalty.
        
        Args:
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
            if (self.t-item.t) > item.due:
                delay_penalty += item.penalty

        self.cost += cost + delay_penalty

        # TODO currently, no penalty for the failed dispatch
        failed = []
        over = len(self.inventory) + len(items) - self.capacity
        if over > 0:
            items, failed = items[:-over], items[-over:]

        # Items that actually are dispatched
        self.inventory = np.concatenate((self.inventory, items))

        # Bug check. Maybe move to a test function later...
        assert len(self.inventory) <= self.capacity

        return self.inventory, np.array(failed), delay_penalty
    
    def deliver(self, p):
        """Deliver items to the final destinations.
        
        Args:
            p (List[float]): Probability of delivery for each item in the warehouse.
                E.g., [0.6, 0.9, 0.2] -> Deliver [item_1, item_0] (ordered by the probability)
                and not deliver [item_2], where the threshold is 0.5.

        Returns:
            inventory (np.array[Item]): Inventory items after delivery
            failed_ids (np.array[Item]): Item ids that failed to deliver because of the
                full delivery capacity. These items will remain in the inventory.
            revenue (float): Delivery revenue
            cost (float): Delivery cost 
        """
        # Sort by the probabilities
        i_p = list(zip(range(len(p)), p))
        i_p.sort(key=lambda x: -x[1])

        # Calculate the revenue and cost from this delivery
        revenue = 0.0
        delivered_ids = []
        failed_ids = []
        item_cnt = {}  # Number of items for each destination
        for i, p in i_p:
            if p < 0.5:
                break
            if i >= len(self.inventory):
                continue 

            item = self.inventory[i]
            if item_cnt.get(item.dest, 0) > self.delivery_capacities[item.dest]:
                # The truck is full
                failed_ids.append(i)
            else:
                # Get the item from inventory and load into the truck
                item_cnt[item.dest] = item_cnt.get(item.dest, 0) + 1
                delivered_ids.append(i)
                revenue += item.price

        # Update inventory after the delivery
        self.inventory = np.delete(self.inventory, delivered_ids)
       
        cost = np.sum(self.delivery_costs[list(item_cnt.keys())]) 
        self.revenue += revenue
        self.cost += cost
        
        return self.inventory, failed_ids, revenue, cost

    def to_array(self):
        """TODO adding dest_cost, dest_cap, t_left here is bug prone. Need refactor
        """
        # Remaining delivery capacity
        dest_cap = list(self.delivery_capacities)
        for item in self.inventory:
            dest_cap[item.dest] -= 1

        state = np.array([
            # Append ('dest_cost', 'dest_cap')
            np.concatenate([
                item.to_array(), [
                    self.delivery_costs[item.dest],
                    dest_cap[item.dest],
                    item.due-(self.t-item.t)
                ]
            ]) for item in self.inventory
        ])

        if len(self.inventory) < self.capacity:
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
        delivery_capacity=5,
        num_destinations=10,
        demand_fn=20,
        seed=None,
    ):
        """Logistics provider's item delivery simulation environment.
        Items are dispatched from arbitrary sellers to a warehouse and
        delivered to their destinations.

        Args:
            T (int): Maximum number of steps to run. I.e. steps per episode.
            capacity (int): Number of items that can be stored at the warehouse.
            delivery_capacity (int): Number of items that can be delivered to a destination at once.
            num_destinations (int): Number of locations this warehouse cover for delivery.
            demand_fn (Union[callable, int]): Demand generation function which returns the number of items.
                If an integer n is provided, use Poisson(n).
            seed (int): Random number seed
        """
        
        self.T = T
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Randomly initialize delivery cost to the destinations
        delivery_costs = np.random.rand(num_destinations)
        delivery_capacities = np.full(num_destinations, delivery_capacity)
        self.warehouse = Warehouse(capacity, delivery_capacities, delivery_costs)

        self.demand_fn = demand_fn

        # action: list of probabilities of delivery for the items
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(capacity,),
            dtype=np.float32
        )
        
        # Agent maps observations to actions to maximize reward
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.warehouse.shape,
            dtype=np.float32
        )
        
        # Other configurations
        self.delivery_types = (NEXT_DAY_DELIVERY, REGULAR_DELIVERY)

        self.reset()
        
    def reset(self, seed=None):        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.warehouse.reset()
        # Initial inventory
        self.warehouse.dispatch(self.demand())
        
        return self.warehouse.to_array()

    def render(self, mode='array'):
        """Render environment state
        
        Args:
            mode (str): One of ('array', 'df', 'plt').
        
        Returns:
            (object): Environment state, either np.array, pd.DataFrame,
                or matplotlib.plt.figure depending on the mode.
        """
        if mode == 'array':
            return self.warehouse.to_array()
        elif mode == 'df':    
            return pd.DataFrame(
                data=self.warehouse.to_array(),
                columns=Warehouse.columns
            )
        elif mode == 'plt':
            raise NotImplementedError("TODO")
        else:
            raise ValueError("Unknown mode: {}".format(mode))
        
    def demand(self):
        """Generate item demands based on `demand_fn`"""
        if isinstance(self.demand_fn, int):
            num_items = np.random.poisson(self.demand_fn)
        elif callable(self.demand_fn):
            num_items = self.demand_fn()
        else:
            raise ValueError("demand_fn should be a callable or an integer.")

        return [
            Item(
                t=self.warehouse.t,
                dest=random.randint(0, len(self.warehouse.delivery_costs)-1),
                delivery_type=random.choice(self.delivery_types),
            ) for _ in range(num_items)
        ]
    
    def step(self, action):
        """Update the environment based on the agent step and return a reward.
        
        Args:
            action (List[float]): List of probabilities to deliver the items in the warehouse.
            
        Returns:
            state
            reward
            done
            info
        """
        # t += 1
        self.warehouse.update()

        # update state
        _, failed_ids, revenue, cost = self.warehouse.deliver(action)
        _, failed, penalty = self.warehouse.dispatch(self.demand())

        total_revenue = self.warehouse.revenue
        total_cost = self.warehouse.cost

        reward = total_revenue - total_cost

        info = {
            "t": self.warehouse.t,
            "revenue": revenue,
            "penalty_and_cost": penalty+cost,
            "total_revenue": total_revenue,
            "total_cost": total_cost,
            "num_items_in_inventory": len(self.warehouse),
            "num_items_failed_to_dispatch": len(failed),
            "num_items_failed_to_deliver": len(failed_ids),
        }
        
        return self.warehouse.to_array(), reward, self.warehouse.t==self.T, info
