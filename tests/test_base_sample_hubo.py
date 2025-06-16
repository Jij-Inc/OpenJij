#!/usr/bin/env python3
"""
Test script to verify that base_sample_hubo function correctly uses CPU count defaults
"""

import multiprocessing
import sys
import os

# Add the current directory to Python path to import openjij
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openjij.sampler.base_sa_sample_hubo import base_sample_hubo, _get_default_num_threads
    
    # Test 1: Check if _get_default_num_threads works
    print("=== Testing _get_default_num_threads() ===")
    default_threads = _get_default_num_threads()
    cpu_count = multiprocessing.cpu_count()
    print(f"Available CPU cores: {cpu_count}")
    print(f"Default threads from _get_default_num_threads: {default_threads}")
    
    # In CI environments, the actual available threads might be limited
    # So we check that default_threads is reasonable (>= 1 and <= cpu_count)
    assert default_threads >= 1, f"Default threads should be at least 1, got {default_threads}"
    assert default_threads <= cpu_count, f"Default threads {default_threads} should not exceed CPU count {cpu_count}"
    print(f"[OK] _get_default_num_threads() returns reasonable value: {default_threads}")
    
    # Test 2: Test base_sample_hubo with None values
    print("\n=== Testing base_sample_hubo with None values ===")
    hubo = {(0,): -1.0, (1,): -1.0, (0, 1): 2.0}  # Simple HUBO problem
    
    # Call with None values - should use CPU count or environment-limited value
    response = base_sample_hubo(
        hubo=hubo,
        vartype="BINARY",
        num_sweeps=100,
        num_reads=None,  # Should default to same as num_threads
        num_threads=None  # Should default to available threads
    )
    
    print(f"Response received with {len(response)} samples")
    print(f"Schedule info: {response.info.get('schedule', {})}")
    
    # Verify the response info contains expected values
    schedule_info = response.info.get('schedule', {})
    expected_threads = default_threads  # Use the actual default we got above
    expected_reads = default_threads    # Should be same as threads
    
    actual_threads = schedule_info.get('num_threads')
    actual_reads = schedule_info.get('num_reads')
    
    print(f"Expected threads: {expected_threads}, Actual: {actual_threads}")
    print(f"Expected reads: {expected_reads}, Actual: {actual_reads}")
    
    assert actual_threads == expected_threads, f"Expected threads {expected_threads}, got {actual_threads}"
    assert actual_reads == expected_reads, f"Expected reads {expected_reads}, got {actual_reads}"
    
    print("[OK] base_sample_hubo with None values works correctly")
    
    # Test 3: Test base_sample_hubo with explicit values
    print("\n=== Testing base_sample_hubo with explicit values ===")
    response2 = base_sample_hubo(
        hubo=hubo,
        vartype="BINARY",
        num_sweeps=100,
        num_reads=2,
        num_threads=1
    )
    
    schedule_info2 = response2.info.get('schedule', {})
    actual_threads2 = schedule_info2.get('num_threads')
    actual_reads2 = schedule_info2.get('num_reads')
    
    print(f"Explicit threads: {actual_threads2}, Explicit reads: {actual_reads2}")
    assert actual_threads2 == 1, f"Expected threads 1, got {actual_threads2}"
    assert actual_reads2 == 2, f"Expected reads 2, got {actual_reads2}"
    
    print("[OK] base_sample_hubo with explicit values works correctly")
    
    # Test 4: Test environment variable override
    print("\n=== Testing environment variable override ===")
    old_env = os.environ.get('OPENJIJ_NUM_THREADS')
    try:
        os.environ['OPENJIJ_NUM_THREADS'] = '2'
        from importlib import reload
        import openjij.sampler.base_sa_sample_hubo
        reload(openjij.sampler.base_sa_sample_hubo)
        from openjij.sampler.base_sa_sample_hubo import _get_default_num_threads
        
        env_threads = _get_default_num_threads()
        print(f"With OPENJIJ_NUM_THREADS=2, got: {env_threads}")
        assert env_threads == 2, f"Expected 2 from environment variable, got {env_threads}"
        print("[OK] Environment variable override works correctly")
        
    finally:
        # Restore original environment
        if old_env is not None:
            os.environ['OPENJIJ_NUM_THREADS'] = old_env
        elif 'OPENJIJ_NUM_THREADS' in os.environ:
            del os.environ['OPENJIJ_NUM_THREADS']
        # Reload to restore original state
        reload(openjij.sampler.base_sa_sample_hubo)
    
    print("\n=== All base_sample_hubo tests passed! ===")
    
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
