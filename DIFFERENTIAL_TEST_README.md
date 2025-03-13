# Keccak-256 Differential Test

This differential test verifies that the OpenCL and Metal implementations of the Keccak-256 hash function produce identical outputs for the same inputs.

## Purpose

The purpose of this test is to ensure that the Metal implementation correctly matches the behavior of the OpenCL implementation. This is important for ensuring consistent results across different GPU backends.

## How It Works

The test:

1. Sets up both OpenCL and Metal environments
2. Generates random test data (message and nonce)
3. Runs the same input through both implementations
4. Compares the outputs to ensure they match

If the outputs match, the test passes. If they don't match, the test fails and displays the differing outputs.

## Running the Test

To run the differential test:

```bash
./run_differential_test.sh
```

This script runs the test with both the OpenCL and Metal features enabled.

## Requirements

-   Both OpenCL and Metal must be available on your system
-   The test will automatically skip if either backend is not available

## Implementation Details

The test uses a simplified version of the hash computation that:

1. Sets up a single work item (thread) for each backend
2. Uses the same input data for both backends
3. Computes the full Keccak-256 hash using the CPU implementation to verify the results

This approach allows us to directly compare the outputs of the two implementations without the complexity of running the full mining algorithm.
