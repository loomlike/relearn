# TODO implement np.array interface into dispatch and delivery for efficiency

import gym
from gym import spaces
import numpy as np



DeliveryType = namedtuple('DeliveryType', ['due', 'price', 'penalty'])
NEXT_DAY_DELIVERY = DeliveryType(due=2, price=1/2, penalty=1/4)
REGULAR_DELIVERY = DeliveryType(due=7, price=1/7, penalty=1/49)


class Item:
    def __init__(self, t, delivery_cost, delivery_type):
        """
        Args:
            t (int): Ordered date (episode)
            delivery_cost (float): Delivery cost such as time and gas
            delivery_type (DeliveryType): Delivery type
        """
        self.t = t
        self.delivery_cost = delivery_cost
        self.delivery_type = delivery_type


class Warehouse:
    def __init__(self, delivery_capacity, capacity):
        """
        Args:
            delivery_capacity (int): Delivery capacity (the number of items) at a time
            capacity (int): Warehouse capacity (the number of items)
        """
        self.delivery_capacity = delivery_capacity
        self.capacity = capacity
        
        self.inventory = None

        # Total revenue and cost of this warehouse
        self.revenue = 0.0
        self.cost = 0.0
        
    def reset(self):
        self.inventory = np.array([], dtype=object)
        self.revenue = 0.0
        self.cost = 0.0
        
    def dispatch(self, t, items, cost=1.0):
        """Dispatch items from sellers to the logistics provider's warehouse.
        Also check the delivery date of the inventory items.
         which incur penalties. 
        
        Args:
            t (int): Current date (episode)
            items (List[Item]): List of items
            cost (float): Dispatch cost
        
        Returns:
            inventory (np.array[Item]): Inventory items after the dispatch
            failed (np.array[Item]): Items that failed to dispatch because of the
                lack of capacity. These items will not be in the inventory.
            penalty (float): Penalty. E.g. missed delivery date.
        """
        penalty = 0.0
        for item in self.inventory:
            if (t-item.t) > item.delivery_type.due:
                penalty += (t-item.t) * item.delivery_type.penalty

        self.cost += cost + penalty

        failed = []
        over = len(self.inventory) + len(items) - self.capacity
        if over > 0:
            items, failed = items[:-over], items[-over:]

        self.inventory = np.concatenate(self.inventory, items)

        return self.inventory, np.array(failed), penalty
    
    def deliver(self, item_ids):
        """Deliver items to the final destinations.
        Args:
            item_ids (List[int]): The indices of the inventory items to deliver

        Returns:
            inventory (np.array[Item]): Inventory items after delivery
            failed_ids (np.array[Item]): Item ids that failed to deliver because of the
                lack of delivery capacity. These items will remain in the inventory.
            profit (float): Delivery profit
            cost (float): Delivery cost 
        """
        # Get item ids in the inventory
        item_ids = np.array(item_ids)
        item_ids = item_ids[item_ids < len(self.inventory)]
        # Select item ids by delivery capacity
        item_ids, failed_ids = item_ids[:self.delivery_capacity], item_ids[-self.delivery_capacity:]

        delivery = self.inventory[item_ids]
        self.inventory = np.delete(self.inventory, item_ids)

        # Calculate profit and cost
        profit = 0.0
        cost = 0.0
        for item in delivery:
            profit += item.delivery_type.price
            cost += item.delivery_cost
            
        self.revenue += profit
        self.cost += cost
        
        return self.inventory, failed_ids, profit, cost

    def to_array(self):
        # TODO update to have delivery_cost, 'due', 'price', 'penalty (and maybe t as well)
        """2-dim np.array containing item delivery cost and due"""
        state = np.zeros((self.capacity, 2), dtype=np.float32)
        for i in range(len(self.inventory)):
            state[i][0] = float(self.inventory[i].delivery_cost)
            state[i][1] = float(self.inventory[i].delivery_due)
        return state
        
        
class SimpleLogistics(gym.Env):
    # TODO update Env based on the modified implementation...
    def __init__(
        self,
        T,
        warehouse,
        demand_fn=None
    ):
        """
        Args:
            T (int): Maximum number of episodes
            warehouse (Warehouse): Logistics provider's distribution center
            demand_fn (callable): Demand generation function. By default, use Poisson(delivery_capacity).
        """
        
        self.T = T
        self.t = 0  # Current time. Increase for each step.
        self.warehouse = warehouse
        if demand_fn is None:
            self.demand_fn = lambda: np.random.poisson(warehouse.delivery_capacity)
        else:
            self.demand_fn = demand_fn

        # Action space: deliver n-th item or not
        self.action_space = spaces.MultiBinary(self.warehouse.capacity)
        # Agent maps observations to actions to maximize reward
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.warehouse.to_array().shape,  
            dtype=np.float32
        )
        
        # Other configurations
        self.delivery_types = (NEXT_DAY_DELIVERY, REGULAR_DELIVERY)
        
    def reset(self):
        self.t = 0
        
        self.warehouse.reset()
        # initial dispatching items
        self.warehouse.dispatch(self.demand())
        
        return self.warehouse.to_array()

    def render(self):
        # TODO maybe render historical data as well based on the parameter
        # TODO choose between graph and array based on the parameter
        return self.warehouse.to_array()
    
    def demand(self):
        """Generate items w/ random delivery cost and type"""
        num_items = self.demand_fn()
        return [
            Item(
                t=self.t,
                delivery_cost=np.random.randint(1, 12+1),
                delivery_type=np.random.choice(self.delivery_types)
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
        cost = self.warehouse.deliver(np.nonzero(action))
        
        
        
#         action_obj = Action(self.supply_chain.warehouse_num)
#         action_obj.production_level = action[0]
#         action_obj.shippings_to_warehouses = action[1:]
#         self.state, reward, done = self.supply_chain.step(self.state, action_obj)
#         return self.state.to_array(), reward, done, {}
    

class Env(object):
    def __init__(self, T, warehouse, ):
        """
# Demand (List[Item]): Dynamically changing demands for item delivery
# Routes (List[int]): Dynamically changing costs (distance + traffic) for the routes between the distribution center to delivery destinations.
# Warehouse

            T (int): Episode duration
        """
        
        
        # TODO Other parameters, e.g.
        # maximum demand, units
        # maximum random demand variation, units 
        # time cost fn = 1/t
        
#         self.storage_capacities = np.fromfunction(lambda j: 10*(j+1), (self.warehouse_num + 1,), dtype=int)
#         self.storage_costs = np.fromfunction(lambda j: 2*(j+1), (self.warehouse_num + 1,), dtype=int)           # storage costs at the factory and each warehouse, dollars per unit
#         self.transporation_costs = np.fromfunction(lambda j: 5*(j+1), (self.warehouse_num,), dtype=int)       # transportation costs for each warehouse, dollars per unit
#         self.penalty_unit_cost = self.unit_price        
#         self.reset()



    def reset(self, demand_history_len = 4):
        self.demand_history = collections.deque(maxlen = demand_history_len)
        for i in range(demand_history_len):
            self.demand_history.append( np.zeros(self.warehouse_num) )
        self.t = 0

    # demand at time t at warehouse j
    def demand(self, j, t):
        return np.round(self.d_max/2 + self.d_max/2*np.sin(2*np.pi*(t + 2*j)/self.T*2) + np.random.randint(0, self.d_var))

    def initial_state(self):
        return State(self.warehouse_num, self.T, list(self.demand_history))

    def step(self, state, action):
        demands = np.fromfunction(lambda j: self.demand(j+1, self.t), (self.warehouse_num,), dtype=int)
    
        # calculating the reward (profit)
        total_revenue = self.unit_price * np.sum(demands)
        total_production_cost = self.unit_cost * action.production_level
        total_storage_cost = np.dot( self.storage_costs, np.maximum(state.stock_levels(), np.zeros(self.warehouse_num + 1)) )
        total_penalty_cost = - self.penalty_unit_cost * ( np.sum( np.minimum(state.warehouse_stock, np.zeros(self.warehouse_num)) )  +  min(state.factory_stock, 0))
        total_transportation_cost = np.dot( self.transporation_costs, action.shippings_to_warehouses )
        reward = total_revenue - total_production_cost - total_storage_cost - total_penalty_cost - total_transportation_cost

        # calculating the next state
        next_state = State(self.warehouse_num, self.T, self.t)
        next_state.factory_stock = min(state.factory_stock + action.production_level - np.sum(action.shippings_to_warehouses), self.storage_capacities[0]) 
        for w in range(self.warehouse_num):
            next_state.warehouse_stock[w] = min(state.warehouse_stock[w] + action.shippings_to_warehouses[w] - demands[w], self.storage_capacities[w+1])    
        next_state.demand_history = list(self.demand_history)

        self.t += 1
        self.demand_history.append(demands)

        return next_state, reward, self.t == self.T - 1
