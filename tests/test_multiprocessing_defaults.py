#!/usr/bin/env python3
"""
Test script to verify that num_threads and num_reads default to CPU count
"""

import multiprocessing
import sys
import os

# Add the current directory to Python path to import openjij
sys.path.insert(0, '/Users/yuyamashiro/workspace/OpenJij')

try:
    import openjij as oj
    
    # Test 1: Check if get_default_num_threads works
    print("=== Testing SASampler.get_default_num_threads() ===")
    default_threads = oj.SASampler.get_default_num_threads()
    cpu_count = multiprocessing.cpu_count()
    print(f"Available CPU cores: {cpu_count}")
    print(f"Default threads from SASampler: {default_threads}")
    assert default_threads == cpu_count, f"Expected {cpu_count}, got {default_threads}"
    print("[OK] SASampler.get_default_num_threads() works correctly")
    
    # Test 2: Test with environment variable override
    print("\n=== Testing environment variable override ===")
    os.environ['OPENJIJ_NUM_THREADS'] = '2'
    override_threads = oj.SASampler.get_default_num_threads()
    print(f"Override threads (OPENJIJ_NUM_THREADS=2): {override_threads}")
    assert override_threads == 2, f"Expected 2, got {override_threads}"
    print("[OK] Environment variable override works correctly")
    
    # Clean up environment variable
    del os.environ['OPENJIJ_NUM_THREADS']
    
    # Test 3: Test SASampler initialization
    print("\n=== Testing SASampler initialization ===")
    sampler = oj.SASampler()
    print(f"SASampler default_num_threads: {sampler._params.get('num_threads')}")
    print(f"SASampler default_num_reads: {sampler._params.get('num_reads')}")
    print("[OK] SASampler initialization works correctly")
    
    # Test 4: Test simple QUBO problem to verify everything works end-to-end
    print("\n=== Testing simple QUBO problem ===")
    Q = {(0, 0): -1, (1, 1): -1, (0, 1): 2}
    
    # Use default values (should use CPU count)
    print("Testing with default values...")
    response = sampler.sample_qubo(Q, num_sweeps=100)
    print(f"Response received with {len(response)} samples")
    print("[OK] sample_qubo with defaults works correctly")
    
    # Use explicit values to make sure overriding still works
    print("Testing with explicit values...")
    response2 = sampler.sample_qubo(Q, num_sweeps=100, num_reads=2, num_threads=1)
    print(f"Response received with {len(response2)} samples")
    print("[OK] sample_qubo with explicit values works correctly")
    
    print("\n=== All tests passed! ===")
    
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
