use crate::Config;
use alloy_primitives::FixedBytes;
use metal::*;
use rand::{thread_rng, Rng};
use std::error::Error;
use std::fmt::Write as _;
use std::fs::File;
use std::io::Write as IoWrite;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

// Safety wrapper for ComputeCommandEncoder to prevent context leaks
struct SafeEncoder<'a> {
    encoder: &'a ComputeCommandEncoderRef,
    ended: bool,
}

impl<'a> SafeEncoder<'a> {
    fn new(encoder: &'a ComputeCommandEncoderRef) -> Self {
        SafeEncoder {
            encoder,
            ended: false,
        }
    }

    fn end_encoding(&mut self) {
        if !self.ended {
            self.encoder.end_encoding();
            self.ended = true;
        }
    }
}

impl<'a> Drop for SafeEncoder<'a> {
    fn drop(&mut self) {
        self.end_encoding();
    }
}

// Safety wrapper for BlitCommandEncoder to prevent context leaks
struct SafeBlitEncoder<'a> {
    encoder: &'a BlitCommandEncoderRef,
    ended: bool,
}

impl<'a> SafeBlitEncoder<'a> {
    fn new(encoder: &'a BlitCommandEncoderRef) -> Self {
        SafeBlitEncoder {
            encoder,
            ended: false,
        }
    }

    fn end_encoding(&mut self) {
        if !self.ended {
            self.encoder.end_encoding();
            self.ended = true;
        }
    }
}

impl<'a> Drop for SafeBlitEncoder<'a> {
    fn drop(&mut self) {
        self.end_encoding();
    }
}

// Struct to hold benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub work_size: u64,
    pub threadgroup_size: u64,
    pub parallel_buffers: usize,
    pub hash_rate: f64,
    pub solutions_found: u64,
    pub duration_seconds: f64,
}

// Function to benchmark a specific configuration
pub fn benchmark_configuration(
    config: &Config,
    work_size: u64,
    threadgroup_size: u64,
    parallel_buffers: usize,
    duration_seconds: u64,
) -> Result<BenchmarkResult, Box<dyn Error>> {
    println!(
        "Benchmarking: work_size={}, threadgroup_size={}, parallel_buffers={}",
        work_size, threadgroup_size, parallel_buffers
    );

    // Get the Metal device
    let devices = Device::all();
    if devices.is_empty() {
        return Err("No Metal devices found".into());
    }

    let device_index = config.gpu_device as usize;
    if device_index >= devices.len() {
        return Err(format!(
            "Metal device index {} out of range (max: {})",
            device_index,
            devices.len() - 1
        )
        .into());
    }

    let device = &devices[device_index];
    println!("Using Metal device: {}", device.name());

    // Create a command queue with maximum concurrency
    let command_queue = device.new_command_queue_with_max_command_buffer_count(64);

    // Compile the Metal kernel
    // let metal_src = crate::metal::kernel::mk_metal_src(config);
    let metal_src = crate::metal_backend::mk_metal_src(config);
    let options = CompileOptions::new();

    let library = device
        .new_library_with_source(&metal_src, &options)
        .map_err(|e| format!("Failed to compile Metal kernel: {}", e))?;

    let kernel_function = library
        .get_function("hashMessage", None)
        .map_err(|e| format!("Failed to get kernel function: {}", e))?;

    // Create a compute pipeline state
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel_function));

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .map_err(|e| format!("Failed to create pipeline state: {}", e))?;

    // Verify the threadgroup size is valid
    let max_threads = pipeline_state.max_total_threads_per_threadgroup();
    if threadgroup_size > max_threads {
        return Err(format!(
            "Threadgroup size {} exceeds maximum of {}",
            threadgroup_size, max_threads
        )
        .into());
    }

    // Calculate grid size based on work size
    let grid_size = MTLSize::new(work_size, 1, 1);
    let threadgroup_size = MTLSize::new(threadgroup_size, 1, 1);

    // Create a random number generator
    let mut rng = thread_rng();

    // Determine the optimal storage mode
    let resource_options = if device.has_unified_memory() {
        MTLResourceOptions::StorageModeShared
    } else {
        MTLResourceOptions::StorageModePrivate
    };

    // Create buffers
    let max_solutions: usize = 16; // Allow up to 16 solutions per batch

    // Create buffers with optimal alignment
    let message_buffer =
        device.new_buffer_with_data([0u8; 4].as_ptr() as *const _, 4, resource_options);

    let nonce_buffer =
        device.new_buffer_with_data([0u32; 1].as_ptr() as *const _, 4, resource_options);

    let solutions_buffer = device.new_buffer(
        8 * max_solutions as u64,
        MTLResourceOptions::StorageModeShared, // Always use shared for solutions to read back
    );

    // Create a counter buffer to track number of solutions found
    let counter_buffer = device.new_buffer_with_data(
        [0u32; 1].as_ptr() as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    // Create staging buffers for CPU-GPU transfers if using private storage
    let staging_message_buffer = if !device.has_unified_memory() {
        Some(device.new_buffer(4, MTLResourceOptions::StorageModeShared))
    } else {
        None
    };

    let staging_nonce_buffer = if !device.has_unified_memory() {
        Some(device.new_buffer(4, MTLResourceOptions::StorageModeShared))
    } else {
        None
    };

    // Track performance metrics
    let start_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let cumulative_nonce = Arc::new(Mutex::new(0u64));
    let solutions_found = Arc::new(Mutex::new(0u64));
    let end_time = start_time + duration_seconds as f64;

    // Begin benchmarking
    while SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
        < end_time
    {
        // Generate a random salt
        let salt = FixedBytes::<4>::random();

        // Update message buffer contents
        if let Some(staging) = &staging_message_buffer {
            let message_ptr = staging.contents() as *mut u8;
            unsafe {
                std::ptr::copy_nonoverlapping(salt.as_ptr(), message_ptr, 4);
            }
        } else {
            let message_ptr = message_buffer.contents() as *mut u8;
            unsafe {
                std::ptr::copy_nonoverlapping(salt.as_ptr(), message_ptr, 4);
            }
        }

        // Reset nonce & initialize it to a random value for better distribution
        let mut nonce: [u32; 1] = rng.gen();

        // Create multiple command buffers for pipelining
        let mut command_buffers = Vec::with_capacity(parallel_buffers);

        for _ in 0..parallel_buffers {
            // Update nonce buffer contents
            if let Some(staging) = &staging_nonce_buffer {
                let nonce_ptr = staging.contents() as *mut u32;
                unsafe {
                    *nonce_ptr = nonce[0];
                }
            } else {
                let nonce_ptr = nonce_buffer.contents() as *mut u32;
                unsafe {
                    *nonce_ptr = nonce[0];
                }
            }

            // Reset solutions counter
            let counter_ptr = counter_buffer.contents() as *mut u32;
            unsafe {
                *counter_ptr = 0;
            }

            // Create command buffer
            let command_buffer = command_queue.new_command_buffer();

            // Create compute command encoder with safety wrapper
            let mut compute_encoder =
                SafeEncoder::new(command_buffer.new_compute_command_encoder());

            // Set compute pipeline
            compute_encoder
                .encoder
                .set_compute_pipeline_state(&pipeline_state);

            // Copy from staging buffers if needed
            if let Some(staging) = &staging_message_buffer {
                let mut blit_encoder =
                    SafeBlitEncoder::new(command_buffer.new_blit_command_encoder());
                blit_encoder
                    .encoder
                    .copy_from_buffer(staging, 0, &message_buffer, 0, 4);
                blit_encoder.end_encoding();
            }

            if let Some(staging) = &staging_nonce_buffer {
                let mut blit_encoder =
                    SafeBlitEncoder::new(command_buffer.new_blit_command_encoder());
                blit_encoder
                    .encoder
                    .copy_from_buffer(staging, 0, &nonce_buffer, 0, 4);
                blit_encoder.end_encoding();
            }

            // Set buffers
            compute_encoder
                .encoder
                .set_buffer(0, Some(&message_buffer), 0);
            compute_encoder
                .encoder
                .set_buffer(1, Some(&nonce_buffer), 0);
            compute_encoder
                .encoder
                .set_buffer(2, Some(&solutions_buffer), 0);
            compute_encoder
                .encoder
                .set_buffer(3, Some(&counter_buffer), 0);

            // Dispatch threads
            compute_encoder
                .encoder
                .dispatch_threads(grid_size, threadgroup_size);

            // End encoding - will be called automatically by Drop, but we do it explicitly for clarity
            compute_encoder.end_encoding();

            // Commit command buffer
            command_buffer.commit();
            command_buffers.push(command_buffer);

            // Increment nonce for next buffer
            nonce[0] = nonce[0].wrapping_add(work_size as u32);
        }

        // Wait for all command buffers to complete and process results
        for buffer in command_buffers {
            buffer.wait_until_completed();

            // Increment the cumulative nonce
            {
                let mut cumulative = cumulative_nonce.lock().unwrap();
                *cumulative += 1;
            }

            // Check for solutions
            let counter_ptr = counter_buffer.contents() as *const u32;
            let num_solutions = unsafe { *counter_ptr };

            if num_solutions > 0 {
                // Process all solutions
                let solutions_ptr = solutions_buffer.contents() as *const u64;
                for i in 0..std::cmp::min(num_solutions as usize, max_solutions) {
                    let solution = unsafe { *solutions_ptr.add(i) };
                    if solution == 0 {
                        continue;
                    }

                    // Increment solutions found counter
                    let mut found_guard = solutions_found.lock().unwrap();
                    *found_guard += 1;
                }
            }
        }
    }

    // Calculate final metrics
    let end_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let actual_duration = end_time - start_time;
    let total_nonce = *cumulative_nonce.lock().unwrap();
    let total_hashes = total_nonce * work_size;
    let hash_rate = total_hashes as f64 / actual_duration / 1_000_000.0; // in millions of hashes per second

    let result = BenchmarkResult {
        work_size,
        threadgroup_size: threadgroup_size.width,
        parallel_buffers,
        hash_rate,
        solutions_found: *solutions_found.lock().unwrap(),
        duration_seconds: actual_duration,
    };

    println!(
        "Benchmark result: {:.2} MH/s, {} solutions found in {:.2}s",
        result.hash_rate, result.solutions_found, result.duration_seconds
    );

    Ok(result)
}

// Function to run a grid search over parameter space
pub fn grid_search(config: &Config, duration_seconds: u64) -> Result<(), Box<dyn Error>> {
    println!("Starting grid search optimization...");

    // Define parameter ranges based on the default work size
    let default_work_size = crate::get_work_size();

    // Define parameter ranges
    let work_sizes = [
        default_work_size / 16,
        default_work_size / 8,
        default_work_size / 4,
        default_work_size / 2,
        default_work_size,
        default_work_size * 2,
        default_work_size * 4,
    ];
    let threadgroup_sizes = [32, 64, 128, 256, 512, 1024];
    let parallel_buffers = [1, 2, 3, 4, 6, 8];

    // Track best configuration
    let mut best_result: Option<BenchmarkResult> = None;
    let mut results = Vec::new();

    // Grid search
    for &work_size in &work_sizes {
        for &threadgroup_size in &threadgroup_sizes {
            for &buffer_count in &parallel_buffers {
                match benchmark_configuration(
                    config,
                    work_size,
                    threadgroup_size,
                    buffer_count,
                    duration_seconds,
                ) {
                    Ok(result) => {
                        // Update best if improved
                        if best_result
                            .as_ref()
                            .is_none_or(|best| result.hash_rate > best.hash_rate)
                        {
                            println!(
                                "New best: {:.2} MH/s with work_size={}, threadgroup={}, buffers={}",
                                result.hash_rate, work_size, threadgroup_size, buffer_count
                            );
                            best_result = Some(result.clone());
                        }
                        results.push(result);
                    }
                    Err(e) => {
                        println!(
                            "Error benchmarking configuration (work_size={}, threadgroup={}, buffers={}): {}",
                            work_size, threadgroup_size, buffer_count, e
                        );
                    }
                }
            }
        }
    }

    // Print results summary
    println!("\nGrid Search Results Summary:");
    println!("----------------------------");

    // Sort results by hash rate (descending)
    results.sort_by(|a, b| b.hash_rate.partial_cmp(&a.hash_rate).unwrap());

    println!("| Rank | Work Size | Threadgroup | Buffers | Hash Rate (MH/s) |");
    println!("|------|-----------|------------|---------|------------------|");

    for (i, result) in results.iter().take(10).enumerate() {
        println!(
            "| {:4} | {:9} | {:10} | {:7} | {:16.2} |",
            i + 1,
            result.work_size,
            result.threadgroup_size,
            result.parallel_buffers,
            result.hash_rate
        );
    }

    if let Some(best) = best_result {
        println!("\nOptimal configuration:");
        println!("  WORK_SIZE = {}", best.work_size);
        println!("  Threadgroup Size = {}", best.threadgroup_size);
        println!("  Parallel Command Buffers = {}", best.parallel_buffers);
        println!("  Performance: {:.2} MH/s", best.hash_rate);

        // Save results to a file
        let mut file = std::fs::File::create("optimization_results.txt")?;
        use std::io::Write;
        writeln!(file, "Optimization Results")?;
        writeln!(file, "====================")?;
        writeln!(file, "Optimal configuration:")?;
        writeln!(file, "  WORK_SIZE = {}", best.work_size)?;
        writeln!(file, "  Threadgroup Size = {}", best.threadgroup_size)?;
        writeln!(
            file,
            "  Parallel Command Buffers = {}",
            best.parallel_buffers
        )?;
        writeln!(file, "  Performance: {:.2} MH/s", best.hash_rate)?;

        writeln!(file, "\nTop 10 Configurations:")?;
        for (i, result) in results.iter().take(10).enumerate() {
            writeln!(
                file,
                "{:2}. work_size={}, threadgroup={}, buffers={}: {:.2} MH/s",
                i + 1,
                result.work_size,
                result.threadgroup_size,
                result.parallel_buffers,
                result.hash_rate
            )?;
        }

        // Save optimal parameters to .env file
        let mut env_file = File::create(".env")?;
        writeln!(
            env_file,
            "# Optimal parameters from grid search optimization"
        )?;
        writeln!(env_file, "OPTIMAL_WORK_SIZE={}", best.work_size)?;
        writeln!(
            env_file,
            "OPTIMAL_THREADGROUP_SIZE={}",
            best.threadgroup_size
        )?;
        writeln!(
            env_file,
            "OPTIMAL_PARALLEL_BUFFERS={}",
            best.parallel_buffers
        )?;
        writeln!(env_file, "# Performance: {:.2} MH/s", best.hash_rate)?;
        println!("Optimal parameters saved to .env file");
    } else {
        println!("No valid configurations found!");
    }

    Ok(())
}

// Two-phase optimization: first coarse, then fine-tuned
pub fn two_phase_optimization(config: &Config) -> Result<(), Box<dyn Error>> {
    println!("Starting two-phase optimization...");

    // Get the default work size
    let default_work_size = crate::get_work_size();

    // Phase 1: Coarse grid search with shorter duration
    println!("\nPhase 1: Coarse grid search");
    println!("---------------------------");

    // Define coarse parameter ranges
    let work_sizes = [
        default_work_size / 8,
        default_work_size / 2,
        default_work_size,
        default_work_size * 2,
        default_work_size * 4,
        default_work_size * 8,
        default_work_size * 16,
        default_work_size * 32,
    ];
    let threadgroup_sizes = [64, 256, 1024];
    let parallel_buffers = [1, 4, 8, 16, 32, 64, 128];

    let mut best_work_size = 0;
    let mut best_threadgroup_size = 0;
    let mut best_parallel_buffers = 0;
    let mut best_hash_rate = 0.0;

    // Coarse grid search
    for &work_size in &work_sizes {
        for &threadgroup_size in &threadgroup_sizes {
            for &buffer_count in &parallel_buffers {
                match benchmark_configuration(
                    config,
                    work_size,
                    threadgroup_size,
                    buffer_count,
                    1, // shorter duration for phase 1
                ) {
                    Ok(result) => {
                        if result.hash_rate > best_hash_rate {
                            best_hash_rate = result.hash_rate;
                            best_work_size = work_size;
                            best_threadgroup_size = threadgroup_size;
                            best_parallel_buffers = buffer_count;
                        }
                    }
                    Err(e) => {
                        println!(
                            "Error in phase 1 (work_size={}, threadgroup={}, buffers={}): {}",
                            work_size, threadgroup_size, buffer_count, e
                        );
                    }
                }
            }
        }
    }

    println!(
        "\nPhase 1 best: {:.2} MH/s with work_size={}, threadgroup={}, buffers={}",
        best_hash_rate, best_work_size, best_threadgroup_size, best_parallel_buffers
    );

    // Phase 2: Fine-tuned search around the best configuration
    println!("\nPhase 2: Fine-tuned search");
    println!("-------------------------");

    // Define ranges around the best configuration
    let work_size_factor = 2;
    let fine_work_sizes = [
        best_work_size / work_size_factor,
        best_work_size,
        best_work_size * work_size_factor,
    ];

    let threadgroup_factor = 2;
    let fine_threadgroup_sizes = [
        std::cmp::max(32, best_threadgroup_size / threadgroup_factor),
        best_threadgroup_size,
        std::cmp::min(1024, best_threadgroup_size * threadgroup_factor),
    ];

    let fine_parallel_buffers = [
        std::cmp::max(1, best_parallel_buffers - 1),
        best_parallel_buffers,
        std::cmp::min(8, best_parallel_buffers + 1),
    ];

    let mut best_result: Option<BenchmarkResult> = None;
    let mut results = Vec::new();

    // Fine-tuned grid search
    for &work_size in &fine_work_sizes {
        for &threadgroup_size in &fine_threadgroup_sizes {
            for &buffer_count in &fine_parallel_buffers {
                match benchmark_configuration(
                    config,
                    work_size,
                    threadgroup_size,
                    buffer_count,
                    config.benchmark_duration, // longer duration for phase 2
                ) {
                    Ok(result) => {
                        if best_result
                            .as_ref()
                            .is_none_or(|best| result.hash_rate > best.hash_rate)
                        {
                            println!(
                                "New best: {:.2} MH/s with work_size={}, threadgroup={}, buffers={}",
                                result.hash_rate, work_size, threadgroup_size, buffer_count
                            );
                            best_result = Some(result.clone());
                        }
                        results.push(result);
                    }
                    Err(e) => {
                        println!(
                            "Error in phase 2 (work_size={}, threadgroup={}, buffers={}): {}",
                            work_size, threadgroup_size, buffer_count, e
                        );
                    }
                }
            }
        }
    }

    // Print results summary
    println!("\nTwo-Phase Optimization Results:");
    println!("------------------------------");

    // Sort results by hash rate (descending)
    results.sort_by(|a, b| b.hash_rate.partial_cmp(&a.hash_rate).unwrap());

    for (i, result) in results.iter().enumerate() {
        println!(
            "{:2}. work_size={}, threadgroup={}, buffers={}: {:.2} MH/s",
            i + 1,
            result.work_size,
            result.threadgroup_size,
            result.parallel_buffers,
            result.hash_rate
        );
    }

    if let Some(best) = best_result {
        println!("\nOptimal configuration:");
        println!("  WORK_SIZE = {}", best.work_size);
        println!("  Threadgroup Size = {}", best.threadgroup_size);
        println!("  Parallel Command Buffers = {}", best.parallel_buffers);
        println!("  Performance: {:.2} MH/s", best.hash_rate);

        // Save results to a file
        let mut file = std::fs::File::create("optimization_results.txt")?;
        use std::io::Write;
        writeln!(file, "Two-Phase Optimization Results")?;
        writeln!(file, "============================")?;
        writeln!(file, "Optimal configuration:")?;
        writeln!(file, "  WORK_SIZE = {}", best.work_size)?;
        writeln!(file, "  Threadgroup Size = {}", best.threadgroup_size)?;
        writeln!(
            file,
            "  Parallel Command Buffers = {}",
            best.parallel_buffers
        )?;
        writeln!(file, "  Performance: {:.2} MH/s", best.hash_rate)?;

        // Save optimal parameters to .env file
        let mut env_file = File::create(".env")?;
        writeln!(env_file, "# Optimal parameters from two-phase optimization")?;
        writeln!(env_file, "OPTIMAL_WORK_SIZE={}", best.work_size)?;
        writeln!(
            env_file,
            "OPTIMAL_THREADGROUP_SIZE={}",
            best.threadgroup_size
        )?;
        writeln!(
            env_file,
            "OPTIMAL_PARALLEL_BUFFERS={}",
            best.parallel_buffers
        )?;
        writeln!(env_file, "# Performance: {:.2} MH/s", best.hash_rate)?;
        println!("Optimal parameters saved to .env file");
    } else {
        println!("No valid configurations found!");
    }

    Ok(())
}
