#![warn(unused_crate_dependencies, unreachable_pub)]
#![deny(unused_must_use, rust_2018_idioms)]

use alloy_primitives::{hex, Address, FixedBytes};
use byteorder::{BigEndian, ByteOrder, LittleEndian};
use clap::Parser;
use console::Term;
use fs4::FileExt;
#[cfg(feature = "opencl")]
use ocl::{Buffer, Context, Device, Kernel, MemFlags, Platform, ProQue, Program, Queue};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use separator::Separatable;
use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::io::{self, BufRead};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use terminal_size::{terminal_size, Height};
use tiny_keccak::{Hasher, Keccak};

mod reward;
pub use reward::Reward;

#[cfg(feature = "metal")]
pub mod metal_backend;

#[cfg(feature = "metal")]
pub mod optimize;

#[cfg(test)]
mod differential_test;

// Default work size
pub const WORK_SIZE: u64 = 1048576; // 2^20

const WORK_FACTOR: u128 = (WORK_SIZE as u128) / 1_000_000;
const CONTROL_CHARACTER: u8 = 0xff;
const MAX_INCREMENTER: u64 = 0xffffffffffff;

#[cfg(feature = "opencl")]
static KERNEL_SRC: &str = include_str!("./kernels/keccak256.cl");

/// GPU backend to use for computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// Use OpenCL for GPU computation
    OpenCL,
    /// Use Metal for GPU computation (macOS only)
    Metal,
}

impl std::str::FromStr for GpuBackend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "opencl" => Ok(GpuBackend::OpenCL),
            "metal" => Ok(GpuBackend::Metal),
            "auto" => Ok(detect_preferred_backend()),
            _ => Err(format!(
                "Invalid backend: {s}. Must be 'opencl', 'metal', or 'auto'"
            )),
        }
    }
}

/// Requires three hex-encoded arguments: the address of the contract that will
/// be calling CREATE2, the address of the caller of said contract *(assuming
/// the contract calling CREATE2 has frontrunning protection in place - if not
/// applicable to your use-case you can set it to the null address)*, and the
/// keccak-256 hash of the bytecode that is provided by the contract calling
/// CREATE2 that will be used to initialize the new contract. An additional set
/// of three optional values may be provided: a device to target for OpenCL GPU
/// search, a threshold for leading zeroes to search for, and a threshold for
/// total zeroes to search for.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Config {
    /// The address of the contract that will call CREATE2 (hex without 0x prefix)
    #[arg(long, value_parser = parse_eth_address)]
    pub factory_address: [u8; 20],

    /// The address of the caller of the factory contract (hex without 0x prefix)
    #[arg(long, value_parser = parse_eth_address)]
    pub calling_address: [u8; 20],

    /// The keccak-256 hash of the bytecode that will be used to initialize the new contract (hex without 0x prefix)
    #[arg(long, value_parser = parse_init_code_hash)]
    pub init_code_hash: [u8; 32],

    /// The GPU device ID to use (255 for CPU)
    #[arg(long, default_value = "255")]
    pub gpu_device: u8,

    /// Minimum number of leading zero bytes to search for
    #[arg(long, default_value = "3", value_parser = parse_leading_zeroes)]
    pub leading_zeroes_threshold: u8,

    /// Minimum number of total zero bytes to search for
    #[arg(long, default_value = "5", value_parser = parse_total_zeroes)]
    pub total_zeroes_threshold: u8,

    /// GPU backend to use - "opencl", "metal", or "auto"
    #[arg(long, default_value = "auto")]
    pub backend: GpuBackend,

    /// Run optimization mode to find optimal parameters (Metal only)
    #[arg(long)]
    pub optimize: bool,

    /// Use two-phase optimization strategy (faster but less comprehensive)
    #[arg(long)]
    pub two_phase: bool,

    /// Duration in seconds for each benchmark during optimization
    #[arg(long, default_value = "10")]
    pub benchmark_duration: u64,
}

/// Parse an Ethereum address from a hex string
fn parse_eth_address(s: &str) -> Result<[u8; 20], String> {
    let s = s.trim_start_matches("0x");
    let vec = hex::decode(s).map_err(|_| format!("Could not decode address: {s}"))?;
    let len = vec.len();
    vec.try_into()
        .map_err(|_| format!("Invalid address length: {}", len))
}

/// Parse an initialization code hash from a hex string
fn parse_init_code_hash(s: &str) -> Result<[u8; 32], String> {
    let s = s.trim_start_matches("0x");
    let vec =
        hex::decode(s).map_err(|_| format!("Could not decode initialization code hash: {s}"))?;
    let len = vec.len();
    vec.try_into()
        .map_err(|_| format!("Invalid initialization code hash length: {}", len))
}

/// Parse and validate leading zeroes threshold
fn parse_leading_zeroes(s: &str) -> Result<u8, String> {
    let value = s
        .parse::<u8>()
        .map_err(|_| format!("Invalid leading zeroes threshold: {s}"))?;
    if value > 20 {
        return Err(
            "Invalid value for leading zeroes threshold argument. (valid: 0..=20)".to_string(),
        );
    }
    Ok(value)
}

/// Parse and validate total zeroes threshold
fn parse_total_zeroes(s: &str) -> Result<u8, String> {
    let value = s
        .parse::<u8>()
        .map_err(|_| format!("Invalid total zeroes threshold: {s}"))?;
    if value > 20 && value != 255 {
        return Err(
            "Invalid value for total zeroes threshold argument. (valid: 0..=20 | 255)".to_string(),
        );
    }
    Ok(value)
}

/// Auto-detect the preferred backend based on the platform
#[cfg(target_os = "macos")]
fn detect_preferred_backend() -> GpuBackend {
    #[cfg(feature = "metal")]
    return GpuBackend::Metal;

    #[cfg(not(feature = "metal"))]
    return GpuBackend::OpenCL;
}

/// Auto-detect the preferred backend based on the platform
#[cfg(not(target_os = "macos"))]
fn detect_preferred_backend() -> GpuBackend {
    GpuBackend::OpenCL
}

/// Given a Config object with a factory address, a caller address, and a
/// keccak-256 hash of the contract initialization code, search for salts that
/// will enable the factory contract to deploy a contract to a gas-efficient
/// address via CREATE2.
///
/// The 32-byte salt is constructed as follows:
///   - the 20-byte calling address (to prevent frontrunning)
///   - a random 6-byte segment (to prevent collisions with other runs)
///   - a 6-byte nonce segment (incrementally stepped through during the run)
///
/// When a salt that will result in the creation of a gas-efficient contract
/// address is found, it will be appended to `efficient_addresses.txt` along
/// with the resultant address and the "value" (i.e. approximate rarity) of the
/// resultant address.
pub fn cpu(config: Config) -> Result<(), Box<dyn Error>> {
    // (create if necessary) and open a file where found salts will be written
    let file = output_file();

    // create object for computing rewards (relative rarity) for a given address
    let rewards = Reward::new();

    // begin searching for addresses
    loop {
        // header: 0xff ++ factory ++ caller ++ salt_random_segment (47 bytes)
        let mut header = [0; 47];
        header[0] = CONTROL_CHARACTER;
        header[1..21].copy_from_slice(&config.factory_address);
        header[21..41].copy_from_slice(&config.calling_address);
        header[41..].copy_from_slice(&FixedBytes::<6>::random()[..]);

        // create new hash object
        let mut hash_header = Keccak::v256();

        // update hash with header
        hash_header.update(&header);

        // iterate over a 6-byte nonce and compute each address
        (0..MAX_INCREMENTER)
            .into_par_iter() // parallelization
            .for_each(|salt| {
                let salt = salt.to_le_bytes();
                let salt_incremented_segment = &salt[..6];

                // clone the partially-hashed object
                let mut hash = hash_header.clone();

                // update with body and footer (total: 38 bytes)
                hash.update(salt_incremented_segment);
                hash.update(&config.init_code_hash);

                // hash the payload and get the result
                let mut res: [u8; 32] = [0; 32];
                hash.finalize(&mut res);

                // get the address that results from the hash
                let address = <&Address>::try_from(&res[12..]).unwrap();

                // count total and leading zero bytes
                let mut total = 0;
                let mut leading = 21;
                for (i, &b) in address.iter().enumerate() {
                    if b == 0 {
                        total += 1;
                    } else if leading == 21 {
                        // set leading on finding non-zero byte
                        leading = i;
                    }
                }

                // only proceed if there are at least three zero bytes
                if total < 3 {
                    return;
                }

                // look up the reward amount
                let key = leading * 20 + total;
                let reward_amount = rewards.get(&key);

                // only proceed if an efficient address has been found
                if reward_amount.is_none() {
                    return;
                }

                // get the full salt used to create the address
                let header_hex_string = hex::encode(header);
                let body_hex_string = hex::encode(salt_incremented_segment);
                let full_salt = format!("0x{}{}", &header_hex_string[42..], &body_hex_string);

                // display the salt and the address.
                let output = format!(
                    "{full_salt} => {address} => {}",
                    reward_amount.unwrap_or("0")
                );
                println!("{output}");

                // create a lock on the file before writing
                file.lock_exclusive().expect("Couldn't lock file.");

                // write the result to file
                writeln!(&file, "{output}")
                    .expect("Couldn't write to `efficient_addresses.txt` file.");

                // release the file lock
                FileExt::unlock(&file).expect("Couldn't unlock file.")
            });
    }
}

/// Given a Config object with a factory address, a caller address, a keccak-256
/// hash of the contract initialization code, and a device ID, search for salts
/// using OpenCL that will enable the factory contract to deploy a contract to a
/// gas-efficient address via CREATE2. This method also takes threshold values
/// for both leading zero bytes and total zero bytes - any address that does not
/// meet or exceed the threshold will not be returned. Default threshold values
/// are three leading zeroes or five total zeroes.
///
/// The 32-byte salt is constructed as follows:
///   - the 20-byte calling address (to prevent frontrunning)
///   - a random 4-byte segment (to prevent collisions with other runs)
///   - a 4-byte segment unique to each work group running in parallel
///   - a 4-byte nonce segment (incrementally stepped through during the run)
///
/// When a salt that will result in the creation of a gas-efficient contract
/// address is found, it will be appended to `efficient_addresses.txt` along
/// with the resultant address and the "value" (i.e. approximate rarity) of the
/// resultant address.
///
/// This method is still highly experimental and could almost certainly use
/// further optimization - contributions are more than welcome!
/// OpenCL GPU implementation for finding efficient Ethereum addresses
#[cfg(feature = "opencl")]
pub fn opencl_gpu(config: Config) -> Result<(), Box<dyn Error>> {
    println!(
        "Setting up experimental OpenCL miner using device {}...",
        config.gpu_device
    );

    // Get the work size from .env if available
    let work_size = get_work_size();
    println!("Using work size: {}", work_size);

    // (create if necessary) and open a file where found salts will be written
    let mut file = output_file();

    // create object for computing rewards (relative rarity) for a given address
    let rewards = Reward::new();

    // track how many addresses have been found
    let mut found = 0u64;

    // set up a controller for terminal output
    let term = Term::stdout();

    // Find the specified device
    let platform_id = 0;
    let device_id = config.gpu_device as usize;

    // Get platforms
    let platforms = Platform::list();

    if platforms.is_empty() {
        return Err("No OpenCL platforms found".into());
    }
    let platform = platforms[platform_id];

    // Get devices
    let devices = match Device::list_all(platform) {
        Ok(d) => d,
        Err(e) => return Err(format!("Failed to get OpenCL devices: {}", e).into()),
    };

    if devices.is_empty() {
        return Err("No OpenCL devices found".into());
    }
    if device_id >= devices.len() {
        return Err(format!("Device ID {} out of range", device_id).into());
    }
    let device = devices[device_id];

    // Create OpenCL context and queue
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    let queue = Queue::new(&context, device, None)?;

    // Prepare kernel source
    let kernel_src = mk_kernel_src(&config);

    // Build program
    let program = Program::builder()
        .devices(device)
        .src(kernel_src)
        .build(&context)?;

    // create a random number generator
    let mut rng = thread_rng();

    // determine the start time
    let start_time: f64 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    // the previous timestamp of printing to the terminal
    let mut previous_time = 0.0f64;

    // set up variables for tracking performance
    let mut rate = 0.0f64;
    let mut cumulative_nonce = 0u64;

    // begin searching for addresses
    loop {
        // construct the 4-byte message to hash
        let salt = FixedBytes::<4>::random();

        // Create a fixed-size message buffer (4 bytes) for the kernel
        let message_buffer = Buffer::<u8>::builder()
            .queue(queue.clone())
            .flags(MemFlags::READ_ONLY)
            .len(4)
            .copy_host_slice(&salt[..])
            .build()?;

        // reset nonce & initialize it to a random value for better distribution
        let nonce: [u32; 1] = rng.gen();
        let nonce_buffer = Buffer::<u32>::builder()
            .queue(queue.clone())
            .flags(MemFlags::READ_ONLY)
            .len(1)
            .copy_host_slice(&nonce[..])
            .build()?;

        // Create solutions buffer
        let solutions_buffer = Buffer::<u64>::builder()
            .queue(queue.clone())
            .flags(MemFlags::WRITE_ONLY)
            .len(1)
            .fill_val(0)
            .build()?;

        // Create and execute kernel
        let kernel = ocl::Kernel::builder()
            .program(&program)
            .name("hashMessage")
            .arg(&message_buffer)
            .arg(&nonce_buffer)
            .arg(&solutions_buffer)
            .build()?;

        // Run the kernel
        unsafe {
            kernel
                .cmd()
                .queue(&queue)
                .global_work_size([work_size])
                .enq()?;
        }

        // read the solution
        let mut solution = vec![0u64; 1];
        solutions_buffer.read(&mut solution).enq()?;

        // increment the cumulative nonce
        cumulative_nonce += 1;

        // update rate
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let current_time = now.as_secs() as f64;
        if current_time - start_time > 0.0 {
            rate = 1.0 / (current_time - start_time);
        }

        // only print to the terminal if enough time has passed
        if current_time - previous_time > 0.5 {
            previous_time = current_time;

            // clear the terminal screen
            let _ = term.clear_screen();

            // get the total runtime and parse into hours : minutes : seconds
            let total_runtime = current_time - start_time;
            let total_runtime_hrs = total_runtime as u64 / 3600;
            let total_runtime_mins = (total_runtime as u64 - total_runtime_hrs * 3600) / 60;
            let total_runtime_secs = total_runtime
                - (total_runtime_hrs * 3600) as f64
                - (total_runtime_mins * 60) as f64;

            // determine the number of attempts being made per second
            let work_rate: u128 = (work_size as u128) * cumulative_nonce as u128 / 1_000_000;

            // display information about the total runtime and work size
            let _ = term.write_line(&format!(
                "total runtime: {}:{:02}:{:02} ({} cycles)\t\t\t\
                 work size per cycle: {}",
                total_runtime_hrs,
                total_runtime_mins,
                total_runtime_secs,
                cumulative_nonce,
                work_size.separated_string(),
            ));

            // display information about the attempt rate and found solutions
            let _ = term.write_line(&format!(
                "rate: {:.2} million attempts per second\t\t\t\
                 total found this run: {}",
                work_rate as f64 * rate,
                found
            ));
        }

        // check if a solution was found
        if solution[0] != 0 {
            let solution_bytes = solution[0].to_le_bytes();

            let mut solution_message = [0; 85];
            solution_message[0] = CONTROL_CHARACTER;
            solution_message[1..21].copy_from_slice(&config.factory_address);
            solution_message[21..41].copy_from_slice(&config.calling_address);
            solution_message[41..45].copy_from_slice(&salt[..]);
            solution_message[45..53].copy_from_slice(&solution_bytes);
            solution_message[53..].copy_from_slice(&config.init_code_hash);

            // create new hash object
            let mut hash = Keccak::v256();

            // update with header
            hash.update(&solution_message);

            // hash the payload and get the result
            let mut res: [u8; 32] = [0; 32];
            hash.finalize(&mut res);

            // get the address that results from the hash
            let address = <&Address>::try_from(&res[12..]).unwrap();

            // count total and leading zero bytes
            let mut total = 0;
            let mut leading = 0;
            for (i, &b) in address.iter().enumerate() {
                if b == 0 {
                    total += 1;
                } else if leading == 0 {
                    // set leading on finding non-zero byte
                    leading = i;
                }
            }

            let key = leading * 20 + total;
            let reward = rewards.get(&key).unwrap_or("0");
            let output = format!(
                "0x{}{}{} => {} => {}",
                hex::encode(&config.calling_address),
                hex::encode(salt),
                hex::encode(solution_bytes),
                address,
                reward,
            );

            // display the solution
            let _ = term.write_line(&format!("{output} ({leading} / {total})"));

            // write the solution to the file
            if let Err(e) = writeln!(&mut file, "{}", output) {
                eprintln!("Error writing to file: {}", e);
            }

            // increment the found counter
            found += 1;
        }
    }
}

#[track_caller]
fn output_file() -> File {
    OpenOptions::new()
        .append(true)
        .create(true)
        .read(true)
        .open("efficient_addresses.txt")
        .expect("Could not create or open `efficient_addresses.txt` file.")
}

/// Creates the OpenCL kernel source code by populating the template with the
/// values from the Config object.
pub fn mk_kernel_src(config: &Config) -> String {
    let mut src = String::with_capacity(2048 + KERNEL_SRC.len());

    let factory = config.factory_address.iter();
    let caller = config.calling_address.iter();
    let hash = config.init_code_hash.iter();
    let hash = hash.enumerate().map(|(i, x)| (i + 52, x));
    for (i, x) in factory.chain(caller).enumerate().chain(hash) {
        writeln!(src, "#define S_{} {}u", i + 1, x).unwrap();
    }
    let lz = config.leading_zeroes_threshold;
    writeln!(src, "#define LEADING_ZEROES {lz}").unwrap();
    let tz = config.total_zeroes_threshold;
    writeln!(src, "#define TOTAL_ZEROES {tz}").unwrap();

    src.push_str(KERNEL_SRC);

    src
}

// Function to get the work size, either from .env or the default
pub fn get_work_size() -> u64 {
    if let Ok(lines) = read_lines(".env") {
        for line in lines.flatten() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split('=').collect();
            if parts.len() != 2 {
                continue;
            }

            let key = parts[0].trim();
            let value = parts[1].trim();

            if key == "OPTIMAL_WORK_SIZE" {
                if let Ok(size) = value.parse::<u64>() {
                    println!("Using optimal work size from .env: {}", size);
                    return size;
                }
            }
        }
    }

    // Return the default if no .env file or no valid work size found
    WORK_SIZE
}

// Helper function to read lines from a file
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
