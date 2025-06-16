#!/usr/bin/env python3
"""
Debug script to understand sample_hubo behavior
"""

import multiprocessing
import sys
import os

# Add the current directory to Python path to import openjij
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import openjij as oj
    
    print("=== Debugging SASampler.sample_hubo ===")
    
    sampler = oj.SASampler()
    print(f"SASampler._default_params: {sampler._default_params}")
    print(f"SASampler._params: {sampler._params}")
    print(f"Available via get_default_num_threads(): {sampler.get_default_num_threads()}")
    
    hubo = {(0,): -1.0, (1,): -1.0, (0, 1): 2.0}  # Simple HUBO problem
    cpu_count = multiprocessing.cpu_count()
    print(f"Available CPU cores: {cpu_count}")
    
    # Test with None values (should trigger new defaults)
    print("\n=== Testing None values ===")
    response = sampler.sample_hubo(
        J=hubo,
        vartype="BINARY",
        num_sweeps=100,
        num_reads=None,
        num_threads=None
    )
    
    print(f"Response samples: {len(response)}")
    print(f"Response info: {response.info}")
    
    # Test with specific updater to see path selection
    print("\n=== Testing with METROPOLIS updater ===")
    response2 = sampler.sample_hubo(
        J=hubo,
        vartype="BINARY",
        num_sweeps=100,
        num_reads=None,
        num_threads=None,
        updater="METROPOLIS"
    )
    
    print(f"Response2 samples: {len(response2)}")
    print(f"Response2 info: {response2.info}")
    
    # Test with k-local to see old path
    print("\n=== Testing with k-local updater (old path) ===")
    response3 = sampler.sample_hubo(
        J=hubo,
        vartype="BINARY",
        num_sweeps=100,
        num_reads=None,
        num_threads=None,
        updater="k-local"
    )
    
    print(f"Response3 samples: {len(response3)}")
    print(f"Response3 info: {response3.info}")
    
except Exception as e:
    print(f"Error during debugging: {e}")
    import traceback
    traceback.print_exc()
