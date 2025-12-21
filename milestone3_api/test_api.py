"""
Simple test script for the Qwen Fine-tuned Model API.

This script tests both the health endpoint and the generation endpoint.

Usage:
    python test_api.py
    python test_api.py --url http://localhost:8000
"""

import requests
import json
import time
import argparse
from typing import Dict, Any


def test_health(base_url: str) -> bool:
    """Test the health endpoint."""
    print("\n" + "="*80)
    print("Testing Health Endpoint")
    print("="*80)

    try:
        response = requests.get(f"{base_url}/health", timeout=10)

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")

            if data.get("status") == "healthy" and data.get("model_loaded"):
                print("✓ Health check PASSED")
                return True
            else:
                print("✗ Health check FAILED: Model not loaded or unhealthy")
                return False
        else:
            print(f"✗ Health check FAILED: Status code {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"✗ Health check FAILED: {str(e)}")
        return False


def test_generation(base_url: str, prompt: str, max_new_tokens: int = 100) -> bool:
    """Test the generation endpoint."""
    print("\n" + "="*80)
    print("Testing Generation Endpoint")
    print("="*80)

    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }

    print(f"Prompt: {prompt}")
    print(f"Parameters: max_new_tokens={max_new_tokens}, temperature=0.7")
    print("\nSending request...")

    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/generate",
            json=payload,
            timeout=120  # 2 minutes timeout
        )
        end_time = time.time()

        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {end_time - start_time:.2f} seconds")

        if response.status_code == 200:
            data = response.json()
            print(f"\nGenerated Text:")
            print("-" * 80)
            print(data.get("generated_text", ""))
            print("-" * 80)
            print(f"\nTokens Generated: {data.get('num_tokens', 'N/A')}")
            print("✓ Generation test PASSED")
            return True
        else:
            print(f"✗ Generation test FAILED: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"✗ Generation test FAILED: {str(e)}")
        return False


def test_error_handling(base_url: str) -> bool:
    """Test error handling with invalid input."""
    print("\n" + "="*80)
    print("Testing Error Handling")
    print("="*80)

    # Test with empty prompt
    print("Testing with empty prompt...")
    try:
        response = requests.post(
            f"{base_url}/generate",
            json={"prompt": ""},
            timeout=10
        )

        if response.status_code == 422:  # Validation error
            print("✓ Empty prompt validation PASSED")
        else:
            print(f"✗ Empty prompt validation FAILED: Expected 422, got {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"✗ Error handling test FAILED: {str(e)}")
        return False

    # Test with invalid parameters
    print("Testing with invalid max_new_tokens...")
    try:
        response = requests.post(
            f"{base_url}/generate",
            json={"prompt": "test", "max_new_tokens": 1000},  # > 512 limit
            timeout=10
        )

        if response.status_code == 422:  # Validation error
            print("✓ Invalid parameter validation PASSED")
            return True
        else:
            print(f"✗ Invalid parameter validation FAILED: Expected 422, got {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"✗ Error handling test FAILED: {str(e)}")
        return False


def run_all_tests(base_url: str):
    """Run all tests and print summary."""
    print("\n" + "="*80)
    print("Qwen Fine-tuned Model API - Test Suite")
    print("="*80)
    print(f"API URL: {base_url}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {
        "health": False,
        "generation": False,
        "error_handling": False
    }

    # Test health endpoint
    results["health"] = test_health(base_url)

    if not results["health"]:
        print("\n⚠️  Health check failed. Skipping other tests.")
        print("Make sure the API is running and the model is loaded.")
        return results

    # Wait a moment for model to be fully ready
    print("\nWaiting 2 seconds before generation test...")
    time.sleep(2)

    # Test generation endpoint with simple prompt
    results["generation"] = test_generation(
        base_url,
        "What is the capital of France?",
        max_new_tokens=50
    )

    # Test with Alpaca format
    if results["generation"]:
        alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Explain what machine learning is in simple terms.

### Response:
"""
        print("\n" + "="*80)
        print("Testing with Alpaca Format Prompt")
        print("="*80)
        test_generation(base_url, alpaca_prompt, max_new_tokens=100)

    # Test error handling
    results["error_handling"] = test_error_handling(base_url)

    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)

    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 All tests passed! API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")

    print("="*80)

    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test the Qwen Fine-tuned Model API"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )

    args = parser.parse_args()

    # Check if API is reachable
    print("Checking if API is reachable...")
    try:
        response = requests.get(args.url, timeout=5)
        print(f"✓ API is reachable at {args.url}")
    except requests.exceptions.RequestException:
        print(f"✗ Cannot reach API at {args.url}")
        print("Make sure the API is running:")
        print("  - Local: python main.py")
        print("  - Docker: docker run -p 8000:8000 qwen-api")
        return

    # Run all tests
    run_all_tests(args.url)


if __name__ == "__main__":
    main()
