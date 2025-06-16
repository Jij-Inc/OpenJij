#!/usr/bin/env python3
"""
Test script for parallel processing and local search functionality.
"""

import time
import numpy as np
import openjij as oj
from collections import Counter

def test_parallel_processing():
    """Test parallel processing with different num_threads settings."""
    print("=" * 60)
    print("Testing Parallel Processing (Batch-wise)")
    print("=" * 60)
    
    # Simple QUBO problem
    Q = {(0, 0): -1, (1, 1): -1, (0, 1): 2}
    
    # Test different configurations
    configurations = [
        {"num_reads": 10, "num_threads": 1},
        {"num_reads": 10, "num_threads": 2},
        {"num_reads": 10, "num_threads": 5},
        {"num_reads": 20, "num_threads": 4},
        {"num_reads": 30, "num_threads": 3},
    ]
    
    sampler = oj.SASampler()
    
    for config in configurations:
        print(f"\nTesting with num_reads={config['num_reads']}, num_threads={config['num_threads']}")
        
        start_time = time.time()
        response = sampler.sample_qubo(Q, num_sweeps=200, seed=42, **config)
        end_time = time.time()
        
        print(f"  Execution time: {end_time - start_time:.4f} seconds")
        print(f"  Number of samples: {len(response.samples())}")
        print(f"  Best energy: {min(response.data_vectors['energy']):.6f}")
        
        # Verify we got the expected number of samples
        assert len(response.samples()) == config['num_reads'], f"Expected {config['num_reads']} samples, got {len(response.samples())}"
        
    print("[OK] Parallel processing test passed!")

def test_local_search():
    """Test local search functionality."""
    print("\n" + "=" * 60)
    print("Testing Local Search Functionality")
    print("=" * 60)
    
    # Create a more complex QUBO problem where local search should make a difference
    # Max-Cut problem on a small graph
    Q = {
        (0, 0): -2, (1, 1): -2, (2, 2): -2, (3, 3): -2,
        (0, 1): 2, (0, 2): 2, (1, 3): 2, (2, 3): 2
    }
    
    sampler = oj.SASampler()
    
    # Test without local search
    print("\nTesting WITHOUT local search:")
    start_time = time.time()
    response_no_ls = sampler.sample_qubo(
        Q, 
        num_reads=20, 
        num_threads=2,
        num_sweeps=200, 
        local_search=False,
        seed=42
    )
    end_time = time.time()
    
    energies_no_ls = response_no_ls.data_vectors['energy']
    print(f"  Execution time: {end_time - start_time:.4f} seconds")
    print(f"  Best energy: {min(energies_no_ls):.6f}")
    print(f"  Average energy: {np.mean(energies_no_ls):.6f}")
    print(f"  Energy std: {np.std(energies_no_ls):.6f}")
    
    # Test with local search
    print("\nTesting WITH local search:")
    start_time = time.time()
    response_with_ls = sampler.sample_qubo(
        Q, 
        num_reads=20, 
        num_threads=2,
        num_sweeps=200, 
        local_search=True,
        seed=42
    )
    end_time = time.time()
    
    energies_with_ls = response_with_ls.data_vectors['energy']
    print(f"  Execution time: {end_time - start_time:.4f} seconds")
    print(f"  Best energy: {min(energies_with_ls):.6f}")
    print(f"  Average energy: {np.mean(energies_with_ls):.6f}")
    print(f"  Energy std: {np.std(energies_with_ls):.6f}")
    
    # Compare results
    print(f"\nComparison:")
    print(f"  Best energy improvement: {min(energies_no_ls) - min(energies_with_ls):.6f}")
    print(f"  Average energy improvement: {np.mean(energies_no_ls) - np.mean(energies_with_ls):.6f}")
    
    # Local search should generally produce better or equal results
    assert min(energies_with_ls) <= min(energies_no_ls), "Local search should not worsen the best solution"
    
    print("[OK] Local search test passed!")

def test_combined_features():
    """Test both parallel processing and local search together."""
    print("\n" + "=" * 60)
    print("Testing Combined Features (Parallel + Local Search)")
    print("=" * 60)
    
    # Random QUBO problem
    np.random.seed(42)
    n = 8
    Q = {}
    for i in range(n):
        Q[(i, i)] = np.random.uniform(-2, 0)
        for j in range(i+1, n):
            Q[(i, j)] = np.random.uniform(0, 2)
    
    sampler = oj.SASampler()
    
    # Test various combinations (reduced for speed)
    test_cases = [
        {"num_reads": 10, "num_threads": 1, "local_search": False},
        {"num_reads": 10, "num_threads": 1, "local_search": True},
        {"num_reads": 10, "num_threads": 2, "local_search": False},
        {"num_reads": 10, "num_threads": 2, "local_search": True},
    ]
    
    results = []
    
    for i, config in enumerate(test_cases):
        print(f"\nTest case {i+1}: {config}")
        
        start_time = time.time()
        response = sampler.sample_qubo(
            Q, 
            num_sweeps=200,
            seed=42,
            **config
        )
        end_time = time.time()
        
        energies = response.data_vectors['energy']
        result = {
            'config': config,
            'time': end_time - start_time,
            'best_energy': min(energies),
            'avg_energy': np.mean(energies),
            'num_samples': len(response.samples())
        }
        results.append(result)
        
        print(f"  Time: {result['time']:.4f}s")
        print(f"  Best energy: {result['best_energy']:.6f}")
        print(f"  Avg energy: {result['avg_energy']:.6f}")
        print(f"  Samples: {result['num_samples']}")
        
        # Verify we got the expected number of samples
        assert result['num_samples'] == config['num_reads']
    
    print("\n" + "=" * 40)
    print("Summary of Combined Features Test:")
    print("=" * 40)
    
    for i, result in enumerate(results):
        config = result['config']
        print(f"Case {i+1}: threads={config['num_threads']}, "
              f"reads={config['num_reads']}, "
              f"local_search={config['local_search']}")
        print(f"  -> Time: {result['time']:.4f}s, Best: {result['best_energy']:.6f}")
    
    print("[OK] Combined features test passed!")

def main():
    """Main test function."""
    print("OpenJij Parallel Processing and Local Search Test")
    print("Testing new features: batch-wise parallelization and greedy local search")
    
    try:
        test_parallel_processing()
        test_local_search()
        test_combined_features()
        
        print("\n" + "=" * 60)
        print("*** ALL TESTS PASSED! ***")
        print("[OK] Batch-wise parallel processing is working correctly")
        print("[OK] Greedy local search is working correctly")
        print("[OK] Combined features work well together")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n*** Test failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
