import numpy as np
import matplotlib.pyplot as plt
import math
import random
from itertools import combinations, permutations
from typing import List, Dict, Tuple, Optional
import time

class Order:
    def __init__(self, order_id: str, lat: float, lon: float, weight: float = 0, packages: int = 0, address: str = ""):
        self.order_id = order_id
        self.lat = lat
        self.lon = lon
        self.weight = weight
        self.packages = packages
        self.address = address
        self.dist_from_site = 0

class RoutingAlgorithm:
    def __init__(self, site_lat: float, site_lon: float):
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.colors = ['#ff4444', '#4444ff', '#44ff44', '#ff8844', '#8844ff', '#44ff88', 
                      '#ff4488', '#888888', '#ffff44', '#44ffff', '#ff6b6b', '#4ecdc4']
    
    def haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in km"""
        R = 6371  # Earth radius in km
        to_rad = lambda x: x * math.pi / 180
        d_lat = to_rad(lat2 - lat1)
        d_lon = to_rad(lon2 - lon1)
        a = (math.sin(d_lat/2)**2 + 
             math.cos(to_rad(lat1)) * math.cos(to_rad(lat2)) * math.sin(d_lon/2)**2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    def build_distance_matrix(self, orders: List[Order]) -> np.ndarray:
        """Build distance matrix using Haversine"""
        n = len(orders)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                d = self.haversine(orders[i].lat, orders[i].lon, orders[j].lat, orders[j].lon)
                matrix[i][j] = matrix[j][i] = d
        return matrix
    
    def group_by_proximity(self, orders: List[Order], max_distance: float = 0.15) -> List[List[Order]]:
        """Group orders by 150m proximity rule"""
        groups = []
        processed = set()
        
        for i, order in enumerate(orders):
            if i in processed:
                continue
                
            group = [order]
            processed.add(i)
            
            # Find all orders within max_distance of any order in current group
            changed = True
            while changed:
                changed = False
                for j, other_order in enumerate(orders):
                    if j in processed:
                        continue
                    
                    # Check if order j is within max_distance of any order in group
                    for group_order in group:
                        dist = self.haversine(other_order.lat, other_order.lon, 
                                            group_order.lat, group_order.lon)
                        if dist <= max_distance:
                            group.append(other_order)
                            processed.add(j)
                            changed = True
                            break
            
            groups.append(group)
        
        return groups
    
    def is_valid_cluster(self, cluster: List[Order]) -> bool:
        """Validate cluster constraints"""
        total_weight = sum(o.weight for o in cluster)
        total_packages = sum(o.packages for o in cluster)
        
        if total_weight > 50 or total_packages > 20:
            return False
        
        # Check 15km pair distance constraint
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                dist = self.haversine(cluster[i].lat, cluster[i].lon, 
                                    cluster[j].lat, cluster[j].lon)
                if dist > 15:
                    return False
        
        return True
    
    def exact_route_distance(self, cluster: List[Order]) -> float:
        """Calculate exact route distance using TSP"""
        if len(cluster) == 0:
            return 0
        if len(cluster) == 1:
            return self.haversine(self.site_lat, self.site_lon, cluster[0].lat, cluster[0].lon)
        
        best_distance = float('inf')
        
        # Try all permutations for small clusters
        for perm in permutations(cluster):
            distance = self.haversine(self.site_lat, self.site_lon, perm[0].lat, perm[0].lon)
            
            for i in range(len(perm) - 1):
                distance += self.haversine(perm[i].lat, perm[i].lon, 
                                         perm[i + 1].lat, perm[i + 1].lon)
            
            best_distance = min(best_distance, distance)
        
        return best_distance
    
    def get_valid_size_combinations(self, n: int, valid_sizes: List[int]) -> List[List[int]]:
        """Generate valid size combinations that sum to n"""
        result = []
        
        def backtrack(remaining, current):
            if remaining == 0:
                result.append(current[:])
                return
            
            for size in valid_sizes:
                if size <= remaining:
                    current.append(size)
                    backtrack(remaining - size, current)
                    current.pop()
        
        backtrack(n, [])
        
        # Deduplicate and sort patterns
        unique_patterns = []
        seen = set()
        
        for pattern in result:
            key = tuple(sorted(pattern))
            if key not in seen:
                seen.add(key)
                unique_patterns.append(list(key))
        
        return unique_patterns
    
    def score_partition(self, partition: List[List[Order]]) -> float:
        """Score a partition (sum of all cluster distances)"""
        total_score = 0
        
        for cluster in partition:
            if not self.is_valid_cluster(cluster):
                return float('inf')
            total_score += self.exact_route_distance(cluster)
        
        return total_score
    
    def perform_exact_brute_force_clustering(self, orders: List[Order], slot_hours: int = 3) -> List[List[Order]]:
        """Exact Brute-Force Clustering Algorithm (≤16 orders)"""
        print(f"🎯 EXACT ALGORITHM: Processing {len(orders)} orders with 150m proximity rule")
        start_time = time.time()
        TIME_LIMIT = 20  # 20 seconds
        
        # Step 1: Apply 150m proximity rule
        proximity_groups = self.group_by_proximity(orders, 0.15)
        print(f"📍 Proximity groups: {len(proximity_groups)} groups from {len(orders)} orders")
        
        # Step 2: Separate forced-alone groups (heavy/bulky orders)
        forced_clusters = []
        remaining_units = []
        
        for group in proximity_groups:
            has_heavy_order = any(order.weight > 50 or order.packages > 20 for order in group)
            if has_heavy_order:
                forced_clusters.append(group)
            else:
                remaining_units.append(group)
        
        print(f"⚖️ Result: {len(forced_clusters)} forced clusters, {len(remaining_units)} remaining units")
        
        if len(remaining_units) == 0:
            return forced_clusters
        
        # Step 3: Calculate total orders in remaining units
        total_orders = sum(len(unit) for unit in remaining_units)
        print(f"🔢 {total_orders} orders remaining for optimization")
        
        # Step 4: Get valid route sizes based on slot hours
        valid_sizes = [3, 4, 5, 6] if slot_hours == 2 else [4, 5, 6, 7]
        print(f"📏 Valid cluster sizes for {slot_hours}h slots: {valid_sizes}")
        
        # Step 5: Generate valid size combinations
        size_patterns = self.get_valid_size_combinations(total_orders, valid_sizes)
        print(f"🔀 Generated {len(size_patterns)} size patterns")
        
        best_score = float('inf')
        best_partition = None
        
        # Step 6: Try each size pattern (simplified for demo)
        for pattern in size_patterns[:3]:  # Limit to first 3 patterns for demo
            if time.time() - start_time > TIME_LIMIT:
                break
            
            # Generate simple partition for this pattern
            partition = self.generate_simple_partition(remaining_units, pattern)
            if partition:
                score = self.score_partition(partition)
                if score < best_score:
                    best_score = score
                    best_partition = partition
                    print(f"✨ New best score: {score:.3f}km")
        
        result = forced_clusters + (best_partition if best_partition else remaining_units)
        print(f"✅ EXACT RESULT: {len(result)} clusters, best score: {best_score:.3f}km")
        
        return result
    
    def generate_simple_partition(self, units: List[List[Order]], pattern: List[int]) -> Optional[List[List[Order]]]:
        """Generate a simple partition for the given pattern"""
        all_orders = [order for unit in units for order in unit]
        
        if len(all_orders) != sum(pattern):
            return None
        
        partition = []
        start_idx = 0
        
        for size in pattern:
            if start_idx + size <= len(all_orders):
                cluster = all_orders[start_idx:start_idx + size]
                partition.append(cluster)
                start_idx += size
        
        return partition
    

        
        partition = []
        start_idx = 0
        
        for size in pattern:
            if start_idx + size <= len(all_orders):
                cluster = all_orders[start_idx:start_idx + size]
                partition.append(cluster)
                start_idx += size
        
        return partition
    
    def cluster_orders_with_algorithm(self, orders: List[Order], slot_hours: int = 3) -> List[List[Order]]:
        """Main clustering function with algorithm selection"""
        optimal_size = round(slot_hours * 2.5)
        if len(orders) <= optimal_size:
            return [orders]
        
        distance_matrix = self.build_distance_matrix(orders)
        
        if len(orders) <= 16:
            print(f"🎯 Using EXACT algorithm for {len(orders)} orders")
            return self.perform_exact_brute_force_clustering(orders, slot_hours)
        else:
            print(f"🚀 Using MONTE CARLO algorithm for {len(orders)} orders")
            return self.perform_adaptive_hybrid_monte_carlo(orders, distance_matrix, slot_hours)
    
    def perform_adaptive_hybrid_monte_carlo(self, orders: List[Order], distance_matrix: np.ndarray, slot_hours: int = 3) -> List[List[Order]]:
        """Adaptive Hybrid Monte Carlo Algorithm"""
        TOTAL_BUDGET = 400_000
        WARMUP = 50_000
        REMAINING = TOTAL_BUDGET - WARMUP
        
        print(f"🚀 ADAPTIVE HYBRID: Starting with {len(orders)} orders, budget: {TOTAL_BUDGET:,}")
        
        global_best_score = float('inf')
        global_best_solution = None
        last_improvement_iteration = 0
        total_iterations = 0
        
        def record_best(partition):
            nonlocal global_best_score, global_best_solution, last_improvement_iteration, total_iterations
            score = self.evaluate_partition_with_compactness(partition, distance_matrix, orders)
            if score < global_best_score:
                global_best_score = score
                global_best_solution = partition
                last_improvement_iteration = total_iterations
                print(f"✨ NEW GLOBAL BEST: {score:.3f}km at iteration {total_iterations:,}")
            total_iterations += 1
            return score
        
        stats = {
            'angular': {'start': float('inf'), 'end': float('inf'), 'count': 0},
            'normal': {'start': float('inf'), 'end': float('inf'), 'count': 0}
        }
        
        print(f"📊 WARMUP PHASE: {WARMUP:,} simulations ({WARMUP//2:,} each)")
        
        # Phase 1: Warm-up - Angular sampler
        print(f"🔄 Angular Sampler Warmup...")
        for i in range(WARMUP // 2):
            partition = self.angular_rotated_monte_carlo_sampler(orders, distance_matrix, slot_hours)
            score = record_best(partition)
            if i == 0:
                stats['angular']['start'] = score
                print(f"🎯 Angular start score: {score:.3f}km")
            stats['angular']['end'] = min(stats['angular']['end'], score)
            stats['angular']['count'] += 1
        
        # Phase 1: Warm-up - Normal sampler
        print(f"🔄 Normal Sampler Warmup...")
        for i in range(WARMUP // 2):
            partition = self.random_partition(orders, math.ceil(len(orders) / round(slot_hours * 2.5)), round(slot_hours * 2.5))
            score = record_best(partition)
            if i == 0:
                stats['normal']['start'] = score
                print(f"🎯 Normal start score: {score:.3f}km")
            stats['normal']['end'] = min(stats['normal']['end'], score)
            stats['normal']['count'] += 1
        
        # Phase 2: Allocate budget
        angular_improvement = stats['angular']['start'] - stats['angular']['end']
        normal_improvement = stats['normal']['start'] - stats['normal']['end']
        
        print(f"📈 CONVERGENCE ANALYSIS:")
        print(f"   Angular: {stats['angular']['start']:.3f} → {stats['angular']['end']:.3f} (Δ{angular_improvement:.3f})")
        print(f"   Normal:  {stats['normal']['start']:.3f} → {stats['normal']['end']:.3f} (Δ{normal_improvement:.3f})")
        
        if angular_improvement > normal_improvement:
            angular_budget = int(REMAINING * 0.75)
            normal_budget = REMAINING - angular_budget
            print(f"🏆 Angular converges faster! Allocation: Angular {angular_budget:,} (75%), Normal {normal_budget:,} (25%)")
        else:
            normal_budget = int(REMAINING * 0.75)
            angular_budget = REMAINING - normal_budget
            print(f"🏆 Normal converges faster! Allocation: Normal {normal_budget:,} (75%), Angular {angular_budget:,} (25%)")
        
        # Phase 3: Exploitation with early termination
        print(f"⚡ EXPLOITATION PHASE: {REMAINING:,} simulations")
        STAGNATION_LIMIT = 150_000
        
        for i in range(angular_budget):
            partition = self.angular_rotated_monte_carlo_sampler(orders, distance_matrix, slot_hours)
            record_best(partition)
            
            if total_iterations - last_improvement_iteration > STAGNATION_LIMIT:
                print(f"🛑 Early termination: No improvement for {STAGNATION_LIMIT:,} iterations")
                break
            
            if i % 50_000 == 0 and i > 0:
                stagnation = total_iterations - last_improvement_iteration
                print(f"🔄 Angular progress: {(i/angular_budget)*100:.1f}% ({i:,}/{angular_budget:,}) - Best: {global_best_score:.3f}km (stagnant: {stagnation:,})")
        
        for i in range(normal_budget):
            if total_iterations - last_improvement_iteration > STAGNATION_LIMIT:
                break
            partition = self.random_partition(orders, math.ceil(len(orders) / round(slot_hours * 2.5)), round(slot_hours * 2.5))
            record_best(partition)
            
            if i % 50_000 == 0 and i > 0:
                stagnation = total_iterations - last_improvement_iteration
                print(f"🔄 Normal progress: {(i/normal_budget)*100:.1f}% ({i:,}/{normal_budget:,}) - Best: {global_best_score:.3f}km (stagnant: {stagnation:,})")
        
        print(f"🎉 ADAPTIVE HYBRID COMPLETE!")
        print(f"   Final best score: {global_best_score:.3f}km")
        print(f"   Total simulations: {total_iterations:,} / {TOTAL_BUDGET:,}")
        print(f"   Efficiency: {(total_iterations / TOTAL_BUDGET) * 100:.1f}% of budget used")
        print(f"   Final clusters: {len(global_best_solution) if global_best_solution else 0}")
        
        return global_best_solution if global_best_solution else self.fallback_clustering(orders, round(slot_hours * 2.5))
    
    def angular_rotated_monte_carlo_sampler(self, orders: List[Order], distance_matrix: np.ndarray, slot_hours: int) -> List[List[Order]]:
        """Angular Rotated Monte Carlo Sampler"""
        SLICE_WIDTH = 45
        ROTATION_STEP = 10
        rotation = random.randint(0, SLICE_WIDTH // ROTATION_STEP - 1) * ROTATION_STEP
        
        # 10% chance for exploration mode with relaxed angular constraint
        is_exploration_mode = random.random() < 0.1
        effective_slice_width = SLICE_WIDTH + 15 if is_exploration_mode else SLICE_WIDTH
        
        with_angles = []
        for order in orders:
            angle = (self.calculate_bearing(order.lat, order.lon) + rotation) % 360
            order_with_angle = order
            order_with_angle.angle = angle
            with_angles.append(order_with_angle)
        
        slices = {}
        for order in with_angles:
            slice_idx = int(order.angle // effective_slice_width)
            if slice_idx not in slices:
                slices[slice_idx] = []
            slices[slice_idx].append(order)
        
        routes = []
        for slice_orders in slices.values():
            if len(slice_orders) == 0:
                continue
            k = max(1, math.ceil(len(slice_orders) / round(slot_hours * 2.5)))
            sub_routes = self.random_partition(slice_orders, k, round(slot_hours * 2.5))
            routes.extend(sub_routes)
        
        return routes
    
    def calculate_bearing(self, lat: float, lon: float) -> float:
        """Calculate bearing from station to order"""
        d_lon = (lon - self.site_lon) * math.pi / 180
        lat1_rad = self.site_lat * math.pi / 180
        lat2_rad = lat * math.pi / 180
        y = math.sin(d_lon) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(d_lon)
        bearing = math.atan2(y, x) * 180 / math.pi
        return (bearing + 360) % 360
    
    def random_partition(self, orders: List[Order], k: int, max_size: int) -> List[List[Order]]:
        """Random partition with GRASP-style randomness"""
        if len(orders) == 0:
            return []
        
        # 70% sorted by distance, 30% shuffled
        if random.random() < 0.7:
            processed_orders = sorted(orders, key=lambda o: o.dist_from_site)
        else:
            processed_orders = orders[:]
            random.shuffle(processed_orders)
        
        partitions = [[] for _ in range(k)]
        
        for order in processed_orders:
            valid_routes = []
            
            for route_idx in range(k):
                route = partitions[route_idx]
                
                if len(route) >= max_size:
                    continue
                
                # Calculate score for this route
                proximity_score = 0
                if len(route) > 0:
                    avg_dist = sum(self.haversine(order.lat, order.lon, existing_order.lat, existing_order.lon) 
                                 for existing_order in route) / len(route)
                    proximity_score = 1 / (avg_dist + 0.1)
                
                balance_penalty = len(route) * 0.1
                score = proximity_score - balance_penalty
                
                valid_routes.append({'route_idx': route_idx, 'score': score})
            
            if len(valid_routes) == 0:
                # Force assign to least loaded route
                least_loaded = min(range(k), key=lambda i: len(partitions[i]))
                partitions[least_loaded].append(order)
            else:
                # GRASP-style randomness - pick randomly among top 2-3 routes
                valid_routes.sort(key=lambda x: x['score'], reverse=True)
                top_routes = valid_routes[:min(3, len(valid_routes))]
                selected_route = random.choice(top_routes)
                partitions[selected_route['route_idx']].append(order)
        
        return [cluster for cluster in partitions if len(cluster) > 0]
    
    def evaluate_partition_with_compactness(self, partition: List[List[Order]], distance_matrix: np.ndarray, orders: List[Order]) -> float:
        """Evaluate partition with compactness bonus"""
        total_distance = 0
        compactness_bonus = 0
        single_order_penalty = 0
        
        for cluster in partition:
            if len(cluster) == 1:
                single_order_penalty += 5.0
                continue
            
            cluster_distance = 0
            max_pair_dist = 0
            avg_pair_dist = 0
            pair_count = 0
            
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    dist = self.haversine(cluster[i].lat, cluster[i].lon, cluster[j].lat, cluster[j].lon)
                    cluster_distance += dist
                    max_pair_dist = max(max_pair_dist, dist)
                    avg_pair_dist += dist
                    pair_count += 1
            
            if pair_count > 0:
                avg_pair_dist /= pair_count
                compactness_bonus += max_pair_dist * 0.3 + avg_pair_dist * 0.2
            
            total_distance += cluster_distance
        
        return total_distance + compactness_bonus + single_order_penalty
    
    def fallback_clustering(self, orders: List[Order], optimal_size: int) -> List[List[Order]]:
        """Fallback clustering for edge cases"""
        print('Using fallback clustering for', len(orders), 'orders')
        
        clusters = []
        remaining = orders[:]
        
        # Sort by distance from site
        remaining.sort(key=lambda o: o.dist_from_site)
        
        while remaining:
            cluster = [remaining.pop(0)]
            
            # Add nearby orders up to optimal size
            while len(cluster) < optimal_size and remaining:
                best_idx = -1
                best_dist = float('inf')
                
                for i, order in enumerate(remaining):
                    # Check constraints
                    test_cluster = cluster + [order]
                    total_weight = sum(o.weight for o in test_cluster)
                    total_packages = sum(o.packages for o in test_cluster)
                    
                    if total_weight > 36 or total_packages > 10:
                        continue
                    
                    # Check max pair distance
                    violates_distance = False
                    for cluster_order in cluster:
                        dist = self.haversine(cluster_order.lat, cluster_order.lon, order.lat, order.lon)
                        if dist > 7:
                            violates_distance = True
                            break
                    
                    if violates_distance:
                        continue
                    
                    # Find closest valid order
                    avg_dist = sum(self.haversine(co.lat, co.lon, order.lat, order.lon) for co in cluster) / len(cluster)
                    
                    if avg_dist < best_dist:
                        best_dist = avg_dist
                        best_idx = i
                
                if best_idx != -1:
                    cluster.append(remaining.pop(best_idx))
                else:
                    break
            
            clusters.append(cluster)
        
        print('Fallback created', len(clusters), 'clusters')
        return clusters
    
    def visualize_clusters(self, orders: List[Order], clusters: List[List[Order]], title: str = "Route Clusters", save_path: str = None):
        """Visualize clusters and station using matplotlib"""
        plt.figure(figsize=(14, 10))
        
        # Plot station
        plt.scatter(self.site_lon, self.site_lat, c='red', s=300, marker='s', 
                   label='🏢 Station', zorder=5, edgecolors='black', linewidth=3)
        
        # Plot clusters
        for i, cluster in enumerate(clusters):
            color = self.colors[i % len(self.colors)]
            
            # Plot orders in cluster
            lats = [order.lat for order in cluster]
            lons = [order.lon for order in cluster]
            
            plt.scatter(lons, lats, c=color, s=120, alpha=0.8, 
                       label=f'Cluster {i+1} ({len(cluster)} orders)', zorder=3,
                       edgecolors='black', linewidth=1)
            
            # Add sequence numbers
            for j, order in enumerate(cluster):
                plt.annotate(str(j+1), (order.lon, order.lat), 
                           xytext=(0, 0), textcoords='offset points',
                           fontsize=10, fontweight='bold', color='white',
                           ha='center', va='center')
            
            # Draw route lines
            if len(cluster) > 0:
                # Connect station to first order
                plt.plot([self.site_lon, cluster[0].lon], [self.site_lat, cluster[0].lat], 
                        color=color, alpha=0.6, linewidth=2, linestyle='--', 
                        label=f'Route {i+1}' if i == 0 else "")
                
                # Connect orders in sequence
                for j in range(len(cluster) - 1):
                    plt.plot([cluster[j].lon, cluster[j+1].lon], 
                            [cluster[j].lat, cluster[j+1].lat],
                            color=color, alpha=0.6, linewidth=2)
        
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {save_path}")
        else:
            plt.savefig('route_clusters_visualization.png', dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: route_clusters_visualization.png")
        
        plt.close()
    
    def print_cluster_stats(self, clusters: List[List[Order]]):
        """Print detailed cluster statistics"""
        print(f"\n📊 CLUSTER STATISTICS:")
        print(f"Total clusters: {len(clusters)}")
        
        for i, cluster in enumerate(clusters):
            total_weight = sum(o.weight for o in cluster)
            total_packages = sum(o.packages for o in cluster)
            route_distance = self.exact_route_distance(cluster)
            
            print(f"\nCluster {i+1}:")
            print(f"  Orders: {len(cluster)}")
            print(f"  Weight: {total_weight:.1f}kg")
            print(f"  Packages: {total_packages}")
            print(f"  Route distance: {route_distance:.2f}km")
            
            for j, order in enumerate(cluster):
                print(f"    {j+1}. {order.order_id} ({order.weight:.1f}kg, {order.packages}pkg)")

# Example usage and demo
def create_sample_data():
    """Create sample order data for demonstration"""
    # Station coordinates (example: warehouse location)
    station_lat, station_lon = 17.513678, 78.468307
    
    # Generate sample orders around the station
    orders = []
    random.seed(42)  # For reproducible results
    
    order_data = [
        ("ORD001", 17.520, 78.470, 5.2, 2),
        ("ORD002", 17.515, 78.465, 3.8, 1),
        ("ORD003", 17.525, 78.475, 7.1, 3),
        ("ORD004", 17.510, 78.460, 4.5, 2),
        ("ORD005", 17.530, 78.480, 6.3, 4),
        ("ORD006", 17.505, 78.455, 2.9, 1),
        ("ORD007", 17.535, 78.485, 8.7, 5),
        ("ORD008", 17.518, 78.472, 3.2, 2),
        ("ORD009", 17.522, 78.468, 5.8, 3),
        ("ORD010", 17.512, 78.463, 4.1, 2),
        ("ORD011", 17.528, 78.478, 6.9, 4),
        ("ORD012", 17.508, 78.458, 3.7, 1),
        ("ORD013", 17.532, 78.482, 7.5, 3),
    ]
    
    for order_id, lat, lon, weight, packages in order_data:
        orders.append(Order(order_id, lat, lon, weight, packages, f"Address for {order_id}"))
    
    return station_lat, station_lon, orders

def main():
    """Main demonstration function"""
    print("🚀 Route Clustering Algorithm Demo")
    print("=" * 50)
    
    # Create sample data
    station_lat, station_lon, orders = create_sample_data()
    
    # Initialize routing algorithm
    router = RoutingAlgorithm(station_lat, station_lon)
    
    # Calculate distances from site
    for order in orders:
        order.dist_from_site = router.haversine(station_lat, station_lon, order.lat, order.lon)
    
    print(f"📍 Station: ({station_lat}, {station_lon})")
    print(f"📦 Total orders: {len(orders)}")
    
    # Perform clustering
    slot_hours = 3  # 3-hour delivery slot
    clusters = router.cluster_orders_with_algorithm(orders, slot_hours)
    
    # Print statistics
    router.print_cluster_stats(clusters)
    
    # Visualize results
    router.visualize_clusters(orders, clusters, 
                             f"Route Clusters - {len(clusters)} routes for {len(orders)} orders")

if __name__ == "__main__":
    main()