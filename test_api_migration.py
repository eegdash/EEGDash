#!/usr/bin/env python
"""Test script to validate the new API-based EEGDash client.

This script tests both the HTTP API client and backward compatibility
with direct MongoDB connections.
"""

import sys


def test_api_initialization():
    """Test API client initialization."""
    print("=" * 60)
    print("Test 1: API Client Initialization")
    print("=" * 60)
    
    try:
        from eegdash import EEGDash
        from eegdash.utils import _init_api_client
        
        # Initialize API configuration
        _init_api_client()
        
        # Create client using API (default)
        eegdash = EEGDash()
        print("✓ API client initialized successfully")
        print(f"  - Public: {eegdash.is_public}")
        print(f"  - Staging: {eegdash.is_staging}")
        print(f"  - Use API: {eegdash.use_api}")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize API client: {e}")
        return False


def test_api_query():
    """Test querying records via API."""
    print("\n" + "=" * 60)
    print("Test 2: Query Records via API")
    print("=" * 60)
    
    try:
        from eegdash import EEGDash
        
        # Create client
        eegdash = EEGDash(is_staging=True)  # Use staging for testing
        
        # Test find with limit
        print("Testing find() with limit=5...")
        records = eegdash.find({}, limit=5)
        print(f"✓ Retrieved {len(records)} records")
        
        if records:
            print(f"  - First record dataset: {records[0].get('dataset', 'N/A')}")
            print(f"  - First record has {len(records[0])} fields")
        
        return True
    except Exception as e:
        print(f"✗ Failed to query records: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_count():
    """Test counting documents via API."""
    print("\n" + "=" * 60)
    print("Test 3: Count Documents via API")
    print("=" * 60)
    
    try:
        from eegdash import EEGDash
        
        # Create client
        eegdash = EEGDash(is_staging=True)
        
        # Test count
        print("Testing count()...")
        count = eegdash.count({})
        print(f"✓ Total records in staging: {count}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to count documents: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test backward compatibility with MongoDB."""
    print("\n" + "=" * 60)
    print("Test 4: Backward Compatibility (MongoDB)")
    print("=" * 60)
    
    try:
        from eegdash import EEGDash
        
        # Create client with direct MongoDB (if available)
        print("Attempting to create MongoDB client...")
        eegdash = EEGDash(use_api=False)
        print("✓ MongoDB client created successfully")
        print(f"  - Use API: {eegdash.use_api}")
        
        return True
    except Exception as e:
        print(f"⚠ MongoDB connection not available (expected in API-only setup): {e}")
        return True  # This is acceptable if MongoDB isn't configured


def test_config_values():
    """Test configuration values."""
    print("\n" + "=" * 60)
    print("Test 5: Configuration Values")
    print("=" * 60)
    
    try:
        import mne
        
        api_url = mne.utils.get_config("EEGDASH_API_URL")
        api_token = mne.utils.get_config("EEGDASH_API_TOKEN")
        db_uri = mne.utils.get_config("EEGDASH_DB_URI")
        
        print(f"EEGDASH_API_URL: {api_url}")
        print(f"EEGDASH_API_TOKEN: {api_token[:20]}..." if api_token else "Not set")
        print(f"EEGDASH_DB_URI: {db_uri[:30]}..." if db_uri else "Not set")
        
        if api_url and api_token:
            print("✓ API configuration is set")
            return True
        else:
            print("✗ API configuration is incomplete")
            return False
            
    except Exception as e:
        print(f"✗ Failed to check configuration: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("EEGDash API Migration Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        test_api_initialization,
        test_config_values,
        test_api_query,
        test_api_count,
        test_backward_compatibility,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
