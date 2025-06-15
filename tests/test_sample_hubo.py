#!/usr/bin/env python3
"""
Test script to verify that sample_hubo method correctly uses CPU count defaults
"""

import multiprocessing
import sys
import os

# Add the current directory to Python path to import openjij
sys.path.insert(0, '/Users/yuyamashiro/workspace/OpenJij')

try:
    import openjij as oj
    
    # Test sample_hubo with None values
    print("=== Testing SASampler.sample_hubo with None values ===")
    
    sampler = oj.SASampler()
    hubo = {(0,): -1, (1,): -1, (0, 1): 2}  # Simple HUBO problem
    cpu_count = multiprocessing.cpu_count()
    
    # Call with None values - should use CPU count
    response = sampler.sample_hubo(
        J=hubo,
        vartype="BINARY",
        num_sweeps=100,
        num_reads=None,  # Should default to CPU count
        num_threads=None  # Should default to CPU count
    )
    
    print(f"Response received with {len(response)} samples")
    print(f"Available CPU cores: {cpu_count}")
    
    # For this test, we expect to get samples equal to the CPU count
    # since num_reads defaults to CPU count
    assert len(response) == cpu_count, f"Expected {cpu_count} samples, got {len(response)}"
    
    print("✓ SASampler.sample_hubo with None values works correctly")
    
    # Test sample_hubo with explicit values
    print("\n=== Testing SASampler.sample_hubo with explicit values ===")
    response2 = sampler.sample_hubo(
        J=hubo,
        vartype="BINARY",
        num_sweeps=100,
        num_reads=3,
        num_threads=1
    )
    
    print(f"Response received with {len(response2)} samples")
    assert len(response2) == 3, f"Expected 3 samples, got {len(response2)}"
    
    print("✓ SASampler.sample_hubo with explicit values works correctly")
    
    print("\n=== All sample_hubo tests passed! ===")
    
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
