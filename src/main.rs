use clap::Parser;
use create2crunch::{Config, GpuBackend};
use std::process;

fn main() {
    // Parse command-line arguments using clap
    let config = Config::parse();

    if config.gpu_device == 255 {
        if let Err(e) = create2crunch::cpu(config) {
            eprintln!("CPU application error: {e}");
            process::exit(1);
        }
    } else {
        match config.backend {
            GpuBackend::OpenCL => {
                #[cfg(feature = "opencl")]
                {
                    if let Err(e) = create2crunch::gpu(config) {
                        eprintln!("OpenCL GPU application error: {e}");
                        process::exit(1);
                    }
                }

                #[cfg(not(feature = "opencl"))]
                {
                    eprintln!("OpenCL support not enabled. Recompile with the 'opencl' feature.");
                    process::exit(1);
                }
            }
            GpuBackend::Metal => {
                #[cfg(feature = "metal")]
                {
                    if let Err(e) = create2crunch::metal_backend::metal_gpu(config) {
                        eprintln!("Metal GPU application error: {e}");
                        process::exit(1);
                    }
                }

                #[cfg(not(feature = "metal"))]
                {
                    eprintln!("Metal support not enabled. Recompile with the 'metal' feature.");
                    process::exit(1);
                }
            }
        }
    }
}
