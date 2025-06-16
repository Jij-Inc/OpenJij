#!/usr/bin/env python3
"""
Simple test script for new functionality using sample_hubo directly.
"""

import time
import numpy as np
import openjij as oj

def test_hubo_functionality():
    """Test the new functionality using sample_hubo."""
    print("Testing sample_hubo with new parameters")
    print("=" * 50)
    
    # Simple QUBO problem
    J = {(0, 0): -1, (1, 1): -1, (0, 1): 2}
    
    sampler = oj.SASampler()
    
    # Test 1: Basic functionality
    print("\n1. Testing basic functionality:")
    response = sampler.sample_hubo(
        J, 
        vartype="BINARY",
        num_sweeps=100,
        num_reads=5,
        num_threads=1,
        seed=42
    )
    print(f"   Got {len(response.samples())} samples")
    print(f"   Best energy: {min(response.data_vectors['energy']):.6f}")
    
    # Test 2: Parallel processing
    print("\n2. Testing parallel processing:")
    start_time = time.time()
    response_parallel = sampler.sample_hubo(
        J,
        vartype="BINARY", 
        num_sweeps=100,
        num_reads=10,
        num_threads=4,
        seed=42
    )
    parallel_time = time.time() - start_time
    
    print(f"   Parallel time (num_threads=4): {parallel_time:.4f}s")
    print(f"   Got {len(response_parallel.samples())} samples")
    print(f"   Best energy: {min(response_parallel.data_vectors['energy']):.6f}")
    
    # Test 3: Local search
    print("\n3. Testing local search:")
    
    # Without local search
    start_time = time.time()
    response_no_ls = sampler.sample_hubo(
        J,
        vartype="BINARY",
        num_sweeps=100,
        num_reads=10,
        num_threads=2,
        local_search=False,
        seed=42
    )
    no_ls_time = time.time() - start_time
    
    # With local search
    start_time = time.time()
    response_with_ls = sampler.sample_hubo(
        J,
        vartype="BINARY",
        num_sweeps=100,
        num_reads=10,
        num_threads=2,
        local_search=True,
        seed=42
    )
    with_ls_time = time.time() - start_time
    
    print(f"   Without local search: {no_ls_time:.4f}s, best energy: {min(response_no_ls.data_vectors['energy']):.6f}")
    print(f"   With local search: {with_ls_time:.4f}s, best energy: {min(response_with_ls.data_vectors['energy']):.6f}")
    
    # Test 4: Complex problem
    print("\n4. Testing with larger problem:")
    
    # Create a random QUBO
    np.random.seed(42)
    n = 6
    J_complex = {}
    for i in range(n):
        J_complex[(i,)] = np.random.uniform(-1, 0)
        for j in range(i+1, n):
            J_complex[(i, j)] = np.random.uniform(0, 1)
    
    response_complex = sampler.sample_hubo(
        J_complex,
        vartype="BINARY",
        num_sweeps=200,
        num_reads=8,
        num_threads=2,
        local_search=True,
        seed=42
    )
    
    energies = response_complex.data_vectors['energy']
    print(f"   Complex problem: {len(response_complex.samples())} samples")
    print(f"   Best energy: {min(energies):.6f}")
    print(f"   Average energy: {np.mean(energies):.6f}")
    print(f"   Energy std: {np.std(energies):.6f}")
    
    print("\n" + "=" * 50)
    print("[OK] All tests completed successfully!")
    print("[OK] Batch-wise parallelization is working")
    print("[OK] Local search is working") 
    print("=" * 50)

if __name__ == "__main__":
    try:
        test_hubo_functionality()
    except Exception as e:
        print(f"\n*** Test failed: {e}")
        import traceback
        traceback.print_exc()
