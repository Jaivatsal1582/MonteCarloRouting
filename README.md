
# Route Clustering Algorithm - Monte Carlo Optimization

This project implements an advanced routing algorithm specifically designed for **clustered quick commerce delivery** in urban environments, providing optimal route planning for last-mile delivery services. Currently we are using haversine distance for simple demonstration purposes but its recommended to use realtime road distance matrix when implementing this algorithm.

## Algorithm Overview

The routing algorithm creates optimal delivery route clusters using a **two-tier approach**:

### **Tier 1: Exact Brute-Force (≤16 orders)**
- **150m proximity grouping** - Automatically groups orders in apartments/office buildings
- **Constraint validation** - Weight, package, and distance limits
- **Exhaustive optimization** - Evaluates all valid partition combinations
- **TSP route optimization** - Exact traveling salesman solution

### **Tier 2: Adaptive Hybrid Monte Carlo (>16 orders)**
- **400,000 simulation budget** with intelligent early termination
- **Angular rotation sampling** - 45° geographic sectors with rotation
- **Adaptive budget allocation** - 75% to better-performing sampler
- **Compactness optimization** - Minimizes cluster spread and single-order routes

## Why This Algorithm vs Commercial Solutions

### **Commercial Alternatives**
While enterprise solutions like **Google Maps Platform**, **HERE Routing API**, **Google OR-tools**, and **OSRM** offer sophisticated routing capabilities, this algorithm provides unique advantages:

### **Advantages of This Algorithm**
- **Portable & Self-Contained** - No external API dependencies or rate limits
- **Zero Cost** - No per-request fees or subscription costs
- **Instant Processing** - No network latency or API timeouts
- **Fully Customizable** - Modify constraints, scoring, and logic as needed
- **Enterprise Ready** - Deploy on-premise without data privacy concerns
- **Transparent Logic** - Full visibility into optimization decisions
- **Domain-Specific** - Optimized specifically for clustered urban delivery

### **Trade-offs**
- **Road Network**: Uses Haversine distance (straight-line) vs actual road routing
- **Traffic Data**: No real-time traffic integration
- **Advanced Features**: No turn restrictions, vehicle-specific routing, or live updates

### **Best Use Cases**
- **Quick Commerce** - Dense urban areas where straight-line approximation is acceptable
- **Internal Tools** - When you need full control and customization
- **Cost-Sensitive** - High-volume routing without per-request costs
- **Offline Capable** - No internet dependency for route optimization
- **Rapid Prototyping** - Fast iteration on routing logic and constraints

**Recommendation**: Use this algorithm for initial optimization and clustering, then optionally enhance with commercial APIs for final route refinement when needed.

---

### **1. Geographic Intelligence**
- **Angular sampling** naturally separates orders by city sectors (North, South, East, West)
- **Proximity grouping** handles dense urban areas (apartment complexes, office towers)
- **Bearing-based clustering** aligns with real street networks and traffic patterns

### **2. Constraint-Aware Optimization**
- **Vehicle capacity limits** prevent overloading (weight/packages)
- **Time slot optimization** matches cluster size to delivery windows
- **Distance constraints** ensure feasible routes within city limits

### **3. Scalable Performance**
- **Sub-second optimization** for small datasets (exact algorithm)
- **Linear scaling** for large datasets (Monte Carlo with early termination)
- **Memory efficient** - processes incrementally without storing all combinations

### **4. Real-World Adaptability**
- **Dynamic slot sizing** - 2.5 orders per hour delivery rate
- **Heavy order separation** - Isolates bulky items requiring special handling
- **Fallback mechanisms** - Graceful degradation for edge cases

## Algorithm Performance

| Dataset Size | Algorithm | Time Complexity | Typical Runtime | Quality |
|--------------|-----------|----------------|----------------|----------|
| ≤16 orders | Exact Brute-Force | O(n!) | <1 second | Optimal |
| 17-50 orders | Monte Carlo | O(k×n²) | 1-5 seconds | Near-optimal |
| 50+ orders | Monte Carlo + Early Stop | O(k×n²) | 2-10 seconds | High-quality |

*k = simulation budget (400,000), n = number of orders*

## Customizable Constraints

The algorithm supports easy modification of business constraints:

### **Capacity Constraints** (in `is_valid_cluster()`)
```python
# Current defaults for quick commerce
if total_weight > 50 or total_packages > 20:  # 50kg, 20 packages
    return False

# Example: Food delivery (lighter, more packages)
if total_weight > 25 or total_packages > 35:
    return False

# Example: Grocery delivery (heavier, fewer packages)
if total_weight > 80 or total_packages > 15:
    return False
```

### **Distance Constraints** (in `is_valid_cluster()`)
```python
# Current: 15km max between any two orders
if dist > 15:
    return False

# Example: Dense city center
if dist > 8:
    return False

# Example: Suburban delivery
if dist > 25:
    return False
```

### **Proximity Grouping** (in `group_by_proximity()`)
```python
# Current: 150m for apartment/office grouping
proximity_groups = self.group_by_proximity(orders, 0.15)

# Example: Tighter grouping for high-rise areas
proximity_groups = self.group_by_proximity(orders, 0.05)  # 50m

# Example: Looser grouping for suburban areas
proximity_groups = self.group_by_proximity(orders, 0.30)  # 300m
```

### **Delivery Rate** (in cluster sizing)
```python
# Current: 2.5 orders per hour
optimal_size = round(slot_hours * 2.5)

# Example: Experienced drivers
optimal_size = round(slot_hours * 3.5)

# Example: Complex deliveries (B2B)
optimal_size = round(slot_hours * 1.5)
```

### **Monte Carlo Budget** (in `perform_adaptive_hybrid_monte_carlo()`)
```python
# Current: 400k simulations
TOTAL_BUDGET = 400_000

# Example: Faster processing
TOTAL_BUDGET = 100_000

# Example: Higher quality (more time)
TOTAL_BUDGET = 1_000_000
```

## How the Algorithm Works

### **Phase 1: Preprocessing**
1. **Distance Matrix** - Precompute all pairwise Haversine distances
2. **Proximity Grouping** - Group orders within 150m (apartments/offices)
3. **Heavy Order Separation** - Isolate orders >50kg or >20 packages
4. **Constraint Validation** - Ensure all clusters meet capacity/distance limits

### **Phase 2: Algorithm Selection**
```
if orders ≤ 16:
    → Exact Brute-Force Algorithm
else:
    → Adaptive Hybrid Monte Carlo
```

### **Phase 3A: Exact Algorithm (Small Datasets)**
1. **Size Pattern Generation** - All valid cluster size combinations
2. **Partition Enumeration** - Generate all possible order groupings
3. **TSP Optimization** - Exact route distance for each cluster
4. **Global Optimization** - Find minimum total distance partition

### **Phase 3B: Monte Carlo Algorithm (Large Datasets)**

#### **Warmup Phase (50k simulations)**
- **Angular Sampler**: Divides city into 45° sectors, rotates for coverage
- **Normal Sampler**: Random partitioning with proximity bias
- **Convergence Analysis**: Measures improvement rate for each sampler

#### **Budget Allocation**
- **Better Sampler**: Gets 75% of remaining budget (262.5k simulations)
- **Worse Sampler**: Gets 25% of remaining budget (87.5k simulations)

#### **Exploitation Phase**
- **Early Termination**: Stops if no improvement for 150k iterations
- **Progress Tracking**: Reports best score and stagnation every 50k iterations
- **Global Best**: Maintains best solution across all samplers

### **Phase 4: Route Optimization**
1. **TSP Solving** - Optimal order sequence within each cluster
2. **Compactness Scoring** - Penalizes spread-out clusters
3. **Single-Order Penalty** - Discourages inefficient single-delivery routes

## Real-World Applications

### **Quick Commerce (10-30 min delivery)**
- **High order density** in urban areas
- **Small basket sizes** (2-5 items)
- **Time-critical** delivery windows
- **Multiple micro-fulfillment centers**

### **Food Delivery**
- **Restaurant clustering** by cuisine/location
- **Temperature-sensitive** items
- **Peak hour optimization** (lunch/dinner)
- **Driver capacity constraints**

### **Grocery Delivery**
- **Mixed item types** (frozen, fresh, dry)
- **Larger order sizes** (10-30 items)
- **Scheduled delivery slots**
- **Vehicle type optimization**

### **B2B Delivery**
- **Business hour constraints**
- **Bulk order handling**
- **Priority customer routing**
- **Multi-day planning**

## Files Structure

- `routing_algorithm.py` - Core algorithm implementation
- `test_monte_carlo.py` - Comprehensive testing with 30 orders
- `demo.py` - Multiple scenario demonstrations
- `orders_30.json` - Realistic test dataset (8-48km spread)
- `requirements.txt` - Python dependencies
- `*.png` - Generated visualization images

## Installation & Usage

### **Quick Start**
```bash
pip install -r requirements.txt
python3 test_monte_carlo.py  # Test with 30 spread-out orders
python3 demo.py             # Run multiple scenarios
```

### **Basic Usage**
```python
from routing_algorithm import RoutingAlgorithm, Order

# Initialize
station_lat, station_lon = 17.513678, 78.468307
router = RoutingAlgorithm(station_lat, station_lon)

# Create orders
orders = [
    Order("ORD001", 17.385234, 78.486789, 5.2, 2, "Charminar"),
    Order("ORD002", 17.594567, 78.123456, 6.3, 4, "Shamirpet"),
    # ... more orders
]

# Cluster orders
clusters = router.cluster_orders_with_algorithm(orders, slot_hours=3)

# Visualize
router.visualize_clusters(orders, clusters, "My Routes", "output.png")
```

## Generated Visualizations

The algorithm generates detailed visualizations showing:

1. **monte_carlo_30_orders.png** - Full Monte Carlo test with spread-out orders
2. **test_2h_slot.png** - 2-hour delivery window optimization
3. **test_3h_slot.png** - 3-hour delivery window optimization  
4. **test_4h_slot.png** - 4-hour delivery window optimization

Each visualization displays:
- **Red square**: Distribution center/station
- **Colored circles**: Orders grouped by delivery route
- **Numbers**: Delivery sequence within each route
- **Dashed lines**: Routes from station to first order
- **Solid lines**: Order-to-order connections

## Performance Benchmarks

### **Test Results (30 Orders, 8-48km spread)**
- **Algorithm**: Adaptive Hybrid Monte Carlo
- **Total Distance**: 354.78km across 8 clusters
- **Processing Time**: ~3-5 seconds
- **Budget Efficiency**: 37-40% of 400k simulations used
- **Early Termination**: Stops when no improvement for 150k iterations

### **Scalability**
- **10 orders**: <0.1 seconds (exact algorithm)
- **30 orders**: 3-5 seconds (Monte Carlo)
- **50 orders**: 5-10 seconds (Monte Carlo with early stop)
- **100+ orders**: 10-20 seconds (Monte Carlo with budget limits)

## Algorithm Comparison

| Feature | Greedy | Genetic Algorithm | Simulated Annealing | **This Algorithm** |
|---------|--------|-------------------|--------------------|-----------------|
| Small datasets | Fast, suboptimal | Slow, good | Medium, good | **Fast, optimal** |
| Large datasets | Fast, poor | Very slow | Slow, good | **Fast, near-optimal** |
| Constraint handling | Limited | Complex | Medium | **Native support** |
| Geographic awareness | None | Limited | None | **Angular sectors** |
| Real-time capable | Yes | No | Sometimes | **Yes** |
| Scalability | Linear | Exponential | Polynomial | **Linear with budget** |

## Key Innovations

1. **Hybrid Approach**: Combines exact optimization (small) with Monte Carlo (large)
2. **Geographic Intelligence**: Angular rotation naturally handles city sectors
3. **Adaptive Budget**: Dynamically allocates compute based on sampler performance
4. **Constraint Integration**: Native support for real-world delivery constraints
5. **Early Termination**: Stops when diminishing returns detected
6. **Proximity Awareness**: Handles dense urban delivery scenarios

## License

Implemented for educational and commercial routing optimization purposes.

---

**Perfect for**: Quick commerce, food delivery, grocery delivery, last-mile logistics, urban route optimization, and any clustered delivery scenario requiring fast, high-quality solutions.
=======
# MonteCarloRouting
Advanced Monte Carlo routing algorithm for clustered quick commerce delivery optimization with 400k simulation budget and adaptive hybrid approach.
>>>>>>> fe315a1bc3120721e5e940c073ca377f7c0686aa
