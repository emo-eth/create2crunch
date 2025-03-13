#!/bin/bash

# Ensure both OpenCL and Metal features are enabled
cargo test --features "opencl metal" -- differential_test::tests::test_opencl_vs_metal --nocapture 