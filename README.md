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

1. `factory_address` (optional): The address of the contract that will call CREATE2 (hex without 0x prefix) [default: 4e59b44847b379578588920ca78fbf26c0b4956c]
2. `calling_address` (optional): The address of the caller of the factory contract (hex without 0x prefix) [default: 0000000000000000000000000000000000000000]
3. `init_code_hash`: The keccak-256 hash of the bytecode that will be used to initialize the new contract (hex without 0x prefix)
4. `gpu_device` (optional): The GPU device ID to use (default: 0)
5. `leading_zeroes` (optional): Minimum number of leading zero bytes for GPU filtering (default: 3)
6. `total_zeroes` (optional): Minimum number of total zero bytes for GPU filtering (default: 5)
7. `minimum_score` (optional): Minimum reward score for CPU filtering (default: 1000)
8. `backend` (optional): GPU backend to use - "opencl", "metal", or "metal2" (default: "opencl")

### Useful variables

```sh
export KEYLESS_CREATE2_FACTORY=0x4e59b44847b379578588920ca78fbf26c0b4956c
export IMMUTABLE_CREATE2_FACTORY=0x0000000000FFe8B47B3e2130213B802212439497
export PERMISSIONLESS_CALLER_ADDRESS=0x0000000000000000000000000000000000000000
export DUMMY_INIT_CODE_HASH=0x0000000000000000000000000000000000000000000000000000000000000000
```

#### Basic CPU Usage

```sh
cargo run --release -- --init-code-hash <init_code_hash>
```

#### Using OpenCL GPU

```sh
cargo run --release --features opencl -- --init-code-hash <init_code_hash> --gpu-device <gpu_device> --leading-zeroes <leading_zeroes> --total-zeroes <total_zeroes> --minimum-score <minimum_score> --backend opencl
```

#### Using Metal GPU (macOS only)

```sh
cargo run --release --features metal -- --init-code-hash <init_code_hash> --gpu-device <gpu_device> --leading-zeroes <leading_zeroes> --total-zeroes <total_zeroes> --minimum-score <minimum_score> --backend metal
```

#### Using Metal2 GPU (macOS only)

```sh
cargo run --release --features metal -- --init-code-hash <init_code_hash> --gpu-device <gpu_device> --leading-zeroes <leading_zeroes> --total-zeroes <total_zeroes> --minimum-score <minimum_score> --backend metal2
```

### Example Usage

Using the Create2Factory contract with OpenCL device 0 (using default factory and calling addresses):

```sh
export INIT_CODE_HASH="0x1234567890123456789012345678901234567890123456789012345678901234"
cargo run --release --features opencl -- --init-code-hash $INIT_CODE_HASH --gpu-device 0 --leading-zeroes 3 --total-zeroes 5 --minimum-score 132 --backend opencl
```

Using Metal on macOS (using default factory and calling addresses):

```sh
cargo run --release --features metal -- --init-code-hash $INIT_CODE_HASH --gpu-device 0 --leading-zeroes 3 --total-zeroes 5 --minimum-score 132 --backend metal
```

Using Metal2 on macOS (newer implementation, using default factory and calling addresses):

```sh
cargo run --release --features metal -- --init-code-hash $INIT_CODE_HASH --gpu-device 0 --leading-zeroes 3 --total-zeroes 5 --minimum-score 132 --backend metal2
```

### Comparing Performance

To compare the performance between Metal and OpenCL backends:

1. Run the program with the Metal backend and note the "rate" value (attempts per second):

    ```sh
    cargo run --release --features metal -- --init-code-hash 0000000000000000000000000000000000000000000000000000000000000000 --gpu-device 0 --leading-zeroes 3 --total-zeroes 5 --minimum-score 132 --backend metal
    ```

2. Run the program with the OpenCL backend and compare the "rate" value:

    ```sh
    cargo run --release --features opencl -- --init-code-hash 0000000000000000000000000000000000000000000000000000000000000000 --gpu-device 0 --leading-zeroes 3 --total-zeroes 5 --minimum-score 132 --backend opencl
    ```

3. Run the program with the Metal2 backend for potentially better performance:

    ```sh
    cargo run --release --features metal -- --init-code-hash 0000000000000000000000000000000000000000000000000000000000000000 --gpu-device 0 --leading-zeroes 3 --total-zeroes 5 --minimum-score 132 --backend metal2
    ```

4. The backend with the higher "rate" value is more efficient for your hardware.

### Tuning Performance

You can adjust the `WORK_SIZE` constant in `src/lib.rs` to optimize performance for your specific GPU. Higher values may improve performance on more powerful GPUs, but could cause issues on less powerful ones.

### Tuning Filtering Parameters

The two-stage filtering system allows you to tune both GPU and CPU filtering independently:

#### GPU Filtering (First Stage)

-   `--leading-zeroes`: Controls how many leading zero bytes are required (default: 3)
-   `--total-zeroes`: Controls how many total zero bytes are required (default: 5)
-   Lower values = more candidates pass GPU filtering = higher CPU load
-   Higher values = fewer candidates pass GPU filtering = lower CPU load

#### CPU Filtering (Second Stage)

-   `--minimum-score`: Controls the minimum reward score required (default: 132)
-   The reward score is calculated as: `leading_bytes³ + leading_nibbles² + total_zeroes`
-   Lower values = more addresses considered valid = more results
-   Higher values = fewer addresses considered valid = higher quality results

#### Example Tuning Scenarios

**High Performance, Lower Quality**:

```sh
--leading-zeroes 2 --total-zeroes 3 --minimum-score 500
```

**Balanced Performance and Quality**:

```sh
--leading-zeroes 3 --total-zeroes 5 --minimum-score 132
```

**Lower Performance, Higher Quality**:

```sh
--leading-zeroes 4 --total-zeroes 7 --minimum-score 2000
```

### Two-Stage Filtering System

The program uses a two-stage filtering approach for optimal performance and accuracy:

1. **GPU Stage (First Pass)**:

    - Uses `leading_zeroes` and `total_zeroes` thresholds for fast initial filtering
    - GPU shaders quickly filter out addresses that don't meet basic criteria
    - This reduces the number of candidates sent to the CPU

2. **CPU Stage (Second Pass)**:
    - Uses `minimum_score` threshold with sophisticated reward calculation
    - Applies the `LeadingNibbleReward` scoring system
    - Calculates score based on: `leading_bytes³ + leading_nibbles² + total_zeroes`
    - Only addresses meeting the minimum score are considered valid solutions

## Scoring

The scoring system evaluates the "rarity" or "efficiency" of Ethereum addresses based on their zero patterns. This helps identify addresses that are more valuable for gas optimization and aesthetic purposes.

### Scoring Algorithm

The program uses a `LeadingNibbleReward` scoring system, which calculates a score based on three key metrics:

1. **Leading Zero Bytes** (`leading_bytes`): The number of consecutive zero bytes at the beginning of the address
2. **Leading Zero Nibbles** (`leading_nibbles`): The number of consecutive zero nibbles (4-bit units) at the beginning of the address
3. **Total Zero Bytes** (`total_zeroes`): The total number of zero bytes in the entire address

The final score is calculated using the formula:

```
score = leading_bytes**3 + leading_nibbles**2 + total_zeroes
```

The longer the program runs, the more addresses it will find, with some being rarer (and thus more valuable) than others.

Addresses with many leading zeros are considered more aesthetically pleasing and can have:

-   **Brand value**: Memorable and distinctive addresses
-   **Social proof**: Demonstrates computational effort
-   **Collector appeal**: Rare addresses can be valuable to collectors

### Optimizing for Score

To find higher-scoring addresses, you can:

1. **Increase minimum score**: Use `--minimum-score 200` or higher for rare addresses
2. **Adjust filtering parameters**: Use higher `--leading-zeroes` and `--total-zeroes` values
3. **Run longer**: Higher scores become more likely with more attempts
4. **Use multiple GPUs**: Parallel processing increases the search rate

### Output

For each efficient address found, the program will:

1. Display it in the terminal
2. Append it to `efficient_addresses.txt` in the format: `<salt> => <address> => <score> (leading/total)`

The score represents the reward value calculated by the `LeadingNibbleReward` system, and the `(leading/total)` shows the leading and total zero byte counts for reference.

### Runtime Behavior

The program will run indefinitely until manually stopped, continuously searching for addresses that meet the specified criteria. You can:
