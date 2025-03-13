# create2crunch

> A Rust program for finding salts that create gas-efficient Ethereum addresses via CREATE2.

Provide three arguments: a factory address (or contract that will call CREATE2), a caller address (for factory addresses that require it as a protection against frontrunning), and the keccak-256 hash of the initialization code of the contract that the factory will deploy.
(The example below references the `Create2Factory`'s address on one of the 21 chains where it has been deployed to.)

Live `Create2Factory` contracts can be found [here](https://blockscan.com/address/0x0000000000ffe8b47b3e2130213b802212439497).

For each efficient address found, the salt, resultant addresses, and value _(i.e. approximate rarity)_ will be written to `efficient_addresses.txt`. Verify that one of the salts actually results in the intended address before getting in too deep - ideally, the CREATE2 factory will have a view method for checking what address you'll get for submitting a particular salt. Be sure not to change the factory address or the init code without first removing any existing data to prevent the two salt types from becoming commingled.

This tool was originally built for use with [`Pr000xy`](https://github.com/0age/Pr000xy), including with [`Create2Factory`](https://github.com/0age/Pr000xy/blob/master/contracts/Create2Factory.sol) directly.

## How to Use

### Installation

1. **Prerequisites**:

    - Rust and Cargo (install via [rustup](https://rustup.rs/))
    - For OpenCL: OpenCL drivers for your GPU
    - For Metal: macOS with Metal-compatible GPU

2. **Clone the repository**:
    ```sh
    git clone https://github.com/0age/create2crunch
    cd create2crunch
    ```

### Building

#### CPU-only build

```sh
cargo build --release
```

#### With OpenCL support

```sh
cargo build --release --features opencl
```

#### With Metal support (macOS only)

```sh
cargo build --release --features metal
```

#### With both OpenCL and Metal support

```sh
cargo build --release --features "opencl metal"
```

### Running

The program requires the following arguments:

1. `factory_address`: The address of the contract that will call CREATE2 (hex without 0x prefix)
2. `calling_address`: The address of the caller of the factory contract (hex without 0x prefix)
3. `init_code_hash`: The keccak-256 hash of the bytecode that will be used to initialize the new contract (hex without 0x prefix)
4. `gpu_device` (optional): The GPU device ID to use (default: 255, which uses CPU)
5. `leading_zeroes_threshold` (optional): Minimum number of leading zero bytes to search for (default: 3)
6. `total_zeroes_threshold` (optional): Minimum number of total zero bytes to search for (default: 5)
7. `backend` (optional): GPU backend to use - "opencl", "metal", or "auto" (default: "auto")

#### Basic CPU Usage

```sh
cargo run --release -- --factory-address <factory_address> --calling-address <calling_address> --init-code-hash <init_code_hash>
```

#### Using OpenCL GPU

```sh
cargo run --release --features opencl -- --factory-address <factory_address> --calling-address <calling_address> --init-code-hash <init_code_hash> --gpu-device <gpu_device> --leading-zeroes-threshold <leading_zeroes> --total-zeroes-threshold <total_zeroes> --backend opencl
```

#### Using Metal GPU (macOS only)

```sh
cargo run --release --features metal -- --factory-address <factory_address> --calling-address <calling_address> --init-code-hash <init_code_hash> --gpu-device <gpu_device> --leading-zeroes-threshold <leading_zeroes> --total-zeroes-threshold <total_zeroes> --backend metal
```

#### Auto-detect Best Backend

```sh
cargo run --release --features "opencl metal" -- --factory-address <factory_address> --calling-address <calling_address> --init-code-hash <init_code_hash> --gpu-device <gpu_device> --leading-zeroes-threshold <leading_zeroes> --total-zeroes-threshold <total_zeroes> --backend auto
```

### Example Usage

Using the Create2Factory contract with OpenCL device 0:

```sh
export FACTORY="0x0000000000ffe8b47b3e2130213b802212439497"
export CALLER="0x1234567890123456789012345678901234567890"
export INIT_CODE_HASH="0x1234567890123456789012345678901234567890123456789012345678901234"
cargo run --release --features opencl -- --factory-address $FACTORY --calling-address $CALLER --init-code-hash $INIT_CODE_HASH --gpu-device 0 --leading-zeroes-threshold 3 --total-zeroes-threshold 5 --backend opencl
```

Using Metal on macOS:

```sh
cargo run --release --features metal -- --factory-address $FACTORY --calling-address $CALLER --init-code-hash $INIT_CODE_HASH --gpu-device 0 --leading-zeroes-threshold 3 --total-zeroes-threshold 5 --backend metal
```

### Comparing Performance

To compare the performance between Metal and OpenCL backends:

1. Run the program with the Metal backend and note the "rate" value (attempts per second):

    ```sh
    cargo run --release --features metal -- --factory-address 0000000000000000000000000000000000000000 --calling-address 0000000000000000000000000000000000000000 --init-code-hash 0000000000000000000000000000000000000000000000000000000000000000 --gpu-device 0 --leading-zeroes-threshold 3 --total-zeroes-threshold 5 --backend metal
    ```

2. Run the program with the OpenCL backend and compare the "rate" value:

    ```sh
    cargo run --release --features opencl -- --factory-address 0000000000000000000000000000000000000000 --calling-address 0000000000000000000000000000000000000000 --init-code-hash 0000000000000000000000000000000000000000000000000000000000000000 --gpu-device 0 --leading-zeroes-threshold 3 --total-zeroes-threshold 5 --backend opencl
    ```

3. The backend with the higher "rate" value is more efficient for your hardware.

### Tuning Performance

You can adjust the `WORK_SIZE` constant in `src/lib.rs` to optimize performance for your specific GPU. Higher values may improve performance on more powerful GPUs, but could cause issues on less powerful ones.

### Output

For each efficient address found, the program will:

1. Display it in the terminal
2. Append it to `efficient_addresses.txt` in the format: `<salt> => <address> => <value>`

The value represents the approximate rarity of the address based on the number of leading and total zero bytes.

### Monitoring

A simple monitoring tool is available to track progress:

```sh
python3 analysis.py
```

## GPU Support Details

### OpenCL

OpenCL support is available on most platforms and GPUs. The program will use the OpenCL device specified by the device ID parameter. To list available OpenCL devices on your system, you can use third-party tools like:

-   On Linux/macOS: `clinfo`
-   On Windows: GPU-Z or similar tools

### Metal (macOS)

Metal is Apple's GPU programming framework and is generally more efficient than OpenCL on macOS systems. To use Metal:

1. Make sure you're on macOS with a Metal-compatible GPU
2. Build with the Metal feature flag
3. Specify "metal" as the backend or use "auto" to let the program choose

On macOS, the auto-detect option will prefer Metal if available.

## Troubleshooting

### Common Issues

1. **"OpenCL support not enabled"**: Recompile with `--features opencl`
2. **"Metal support not enabled"**: Recompile with `--features metal` (macOS only)
3. **"Metal device index out of range"**: Specify a valid device ID (usually 0 for the default GPU)
4. **Performance issues**: Try adjusting the `WORK_SIZE` constant in `src/lib.rs`

### Debugging

If you encounter issues, you can run the program with the `RUST_BACKTRACE=1` environment variable to get more detailed error information:

```sh
RUST_BACKTRACE=1 cargo run --release --features metal -- <args>
```

PRs welcome!
