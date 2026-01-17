#!/usr/bin/env python3
"""
Test the Monte Carlo routing algorithm with 30 orders from JSON
"""

import json
from routing_algorithm import RoutingAlgorithm, Order

def load_orders_from_json(filename: str):
    """Load orders from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    station_lat = data['station']['latitude']
    station_lon = data['station']['longitude']
    
    orders = []
    for order_data in data['orders']:
        order = Order(
            order_data['orderId'],
            order_data['latitude'],
            order_data['longitude'],
            order_data['weight'],
            order_data['packages'],
            order_data['address']
        )
        orders.append(order)
    
    return station_lat, station_lon, orders

def test_monte_carlo_algorithm():
    """Test Monte Carlo algorithm with 30 orders"""
    print("🚀 Testing Monte Carlo Algorithm with 30 Orders")
    print("=" * 60)
    
    # Load orders from JSON
    station_lat, station_lon, orders = load_orders_from_json('orders_30.json')
    
    # Initialize routing algorithm
    router = RoutingAlgorithm(station_lat, station_lon)
    
    # Calculate distances from site
    for order in orders:
        order.dist_from_site = router.haversine(station_lat, station_lon, order.lat, order.lon)
    
    print(f"📍 Station: ({station_lat}, {station_lon})")
    print(f"📦 Total orders: {len(orders)}")
    print(f"📏 Order spread: {min(o.dist_from_site for o in orders):.2f}km - {max(o.dist_from_site for o in orders):.2f}km")
    
    # Test with 3-hour slot
    print(f"\n🎯 Testing with 3-hour delivery slot...")
    clusters = router.cluster_orders_with_algorithm(orders, slot_hours=3)
    
    # Print detailed statistics
    router.print_cluster_stats(clusters)
    
    # Calculate total metrics
    total_distance = sum(router.exact_route_distance(cluster) for cluster in clusters)
    total_weight = sum(sum(o.weight for o in cluster) for cluster in clusters)
    total_packages = sum(sum(o.packages for o in cluster) for cluster in clusters)
    
    print(f"\n📊 OVERALL METRICS:")
    print(f"Total clusters: {len(clusters)}")
    print(f"Total distance: {total_distance:.2f}km")
    print(f"Total weight: {total_weight:.1f}kg")
    print(f"Total packages: {total_packages}")
    print(f"Average cluster size: {len(orders)/len(clusters):.1f} orders")
    
    # Visualize results
    router.visualize_clusters(orders, clusters, 
                             f"Monte Carlo Algorithm - {len(clusters)} clusters for {len(orders)} orders",
                             "monte_carlo_30_orders.png")
    
    return clusters

def test_different_slot_hours():
    """Test with different slot hours"""
    print("\n" + "=" * 60)
    print("🕐 Testing Different Slot Hours")
    print("=" * 60)
    
    station_lat, station_lon, orders = load_orders_from_json('orders_30.json')
    router = RoutingAlgorithm(station_lat, station_lon)
    
    # Calculate distances from site
    for order in orders:
        order.dist_from_site = router.haversine(station_lat, station_lon, order.lat, order.lon)
    
    slot_hours_list = [2, 3, 4]
    
    for slot_hours in slot_hours_list:
        print(f"\n⏰ {slot_hours}-hour slot (optimal size: {round(slot_hours * 2.5)} orders/cluster)")
        clusters = router.cluster_orders_with_algorithm(orders, slot_hours=slot_hours)
        
        total_distance = sum(router.exact_route_distance(cluster) for cluster in clusters)
        avg_cluster_size = len(orders) / len(clusters)
        
        print(f"   Clusters: {len(clusters)}")
        print(f"   Total distance: {total_distance:.2f}km")
        print(f"   Avg cluster size: {avg_cluster_size:.1f} orders")
        
        # Save visualization
        router.visualize_clusters(orders, clusters,
                                 f"{slot_hours}h Slot - {len(clusters)} clusters",
                                 f"test_{slot_hours}h_slot.png")

def main():
    """Run all tests"""
    print("🧪 Monte Carlo Routing Algorithm Test Suite")
    print("Testing with 30 real orders from Hyderabad")
    
    # Test main algorithm
    clusters = test_monte_carlo_algorithm()
    
    # Test different slot hours
    test_different_slot_hours()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("📊 Generated visualizations:")
    print("  - monte_carlo_30_orders.png")
    print("  - test_2h_slot.png")
    print("  - test_3h_slot.png") 
    print("  - test_4h_slot.png")
    print("=" * 60)

if __name__ == "__main__":
    main()