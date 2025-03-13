use alloy_primitives::{Address, FixedBytes};
use clap::{Parser, ValueEnum};
use create2crunch::{Config, GpuBackend};
use std::error::Error;
use std::str::FromStr;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Factory address (20 bytes)
    #[arg(long)]
    factory_address: String,

    /// Calling address (20 bytes)
    #[arg(long)]
    calling_address: String,

    /// Init code hash (32 bytes)
    #[arg(long)]
    init_code_hash: String,

    /// GPU device ID to use
    #[arg(long, default_value_t = 0)]
    gpu_device: u32,

    /// Minimum number of leading zero bytes required
    #[arg(long, default_value_t = 1)]
    leading_zeroes: u8,

    /// Minimum number of total zero bytes required
    #[arg(long, default_value_t = 2)]
    total_zeroes: u8,

    /// GPU backend to use (opencl, metal, or metal2)
    #[arg(long, default_value = "opencl")]
    backend: String,

    /// Run optimization mode to find optimal parameters
    #[arg(long, default_value_t = false)]
    optimize: bool,

    /// Use two-phase optimization (faster but less thorough)
    #[arg(long, default_value_t = false)]
    two_phase: bool,

    /// Duration in seconds for each benchmark in optimization mode
    #[arg(long, default_value_t = 10)]
    benchmark_duration: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let args = Args::parse();

    // Convert hex strings to byte arrays
    let factory_address = parse_address(&args.factory_address)?;
    let calling_address = parse_address(&args.calling_address)?;
    let init_code_hash = parse_hash(&args.init_code_hash)?;

    // Get the work size (from .env if available)
    let work_size = create2crunch::get_work_size();
    println!("Using work size: {}", work_size);

    // Parse backend string
    let backend = match args.backend.to_lowercase().as_str() {
        "opencl" => GpuBackend::OpenCL,
        "metal" => GpuBackend::Metal,
        "metal2" => GpuBackend::Metal2,
        _ => return Err(format!("Unknown backend: {}", args.backend).into()),
    };

    // Create configuration
    let config = Config {
        factory_address,
        calling_address,
        init_code_hash,
        gpu_device: args.gpu_device as u8,
        leading_zeroes_threshold: args.leading_zeroes,
        total_zeroes_threshold: args.total_zeroes,
        backend,
        optimize: args.optimize,
        two_phase: args.two_phase,
        benchmark_duration: args.benchmark_duration,
    };

    // If optimization is requested, run the optimization process
    if config.optimize {
        #[cfg(feature = "metal")]
        {
            if config.two_phase {
                return create2crunch::optimize::two_phase_optimization(&config);
            } else {
                return create2crunch::optimize::grid_search(&config, config.benchmark_duration);
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            return Err(
                "Optimization requires Metal support. Recompile with --features metal".into(),
            );
        }
    }

    // Run the appropriate backend
    match config.backend {
        GpuBackend::OpenCL => {
            #[cfg(feature = "opencl")]
            {
                create2crunch::opencl_gpu(config)?;
            }
            #[cfg(not(feature = "opencl"))]
            {
                return Err(
                    "OpenCL support not compiled in. Recompile with --features opencl".into(),
                );
            }
        }
        GpuBackend::Metal => {
            #[cfg(feature = "metal")]
            {
                create2crunch::metal_gpu(config)?;
            }
            #[cfg(not(feature = "metal"))]
            {
                return Err(
                    "Metal support not compiled in. Recompile with --features metal".into(),
                );
            }
        }
        GpuBackend::Metal2 => {
            #[cfg(feature = "metal")]
            {
                create2crunch::metal_gpu2(config)?;
            }
            #[cfg(not(feature = "metal"))]
            {
                return Err(
                    "Metal support not compiled in. Recompile with --features metal".into(),
                );
            }
        }
    }

    Ok(())
}

// Helper function to parse a hex string into a 20-byte address
fn parse_address(hex_str: &str) -> Result<[u8; 20], Box<dyn Error>> {
    let hex_str = hex_str.trim_start_matches("0x");
    if hex_str.len() != 40 {
        return Err(format!("Invalid address length: {}", hex_str.len()).into());
    }

    let address = Address::from_str(hex_str)?;
    Ok(address.into())
}

// Helper function to parse a hex string into a 32-byte hash
fn parse_hash(hex_str: &str) -> Result<[u8; 32], Box<dyn Error>> {
    let hex_str = hex_str.trim_start_matches("0x");
    if hex_str.len() != 64 {
        return Err(format!("Invalid hash length: {}", hex_str.len()).into());
    }

    let hash = FixedBytes::<32>::from_str(hex_str)?;
    Ok(hash.into())
}
