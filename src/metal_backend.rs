use crate::{
    calculate_address_metrics, Config, LeadingNibbleReward, RewardTrait, CONTROL_CHARACTER,
};
use alloy_primitives::{hex, Address, FixedBytes};
use console::Term;
use metal::*;
use rand::thread_rng;
use separator::Separatable;
use std::collections::HashSet;
use std::error::Error;
use std::fmt::Write as _;
use std::io::prelude::*;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use terminal_size::{terminal_size, Height};
use tiny_keccak::{Hasher, Keccak};

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

static KERNEL_SRC: &str = include_str!("./kernels/keccak256.metal");

/// Function to generate Metal kernel source with the appropriate constants
pub fn mk_metal_src(config: &Config) -> String {
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

/// Metal GPU implementation for finding efficient Ethereum addresses
pub fn metal_gpu(config: Config) -> Result<(), Box<dyn Error>> {
    println!(
        "Setting up experimental Metal miner using device {}...",
        config.gpu_device
    );

    // Get the work size from .env if available
    let work_size = crate::get_work_size();
    println!("Using work size: {}", work_size);

    // (create if necessary) and open a file where found salts will be written
    Arc::new(crate::output_file());

    // create object for computing rewards (relative rarity) for a given address
    let rewards = Arc::new(LeadingNibbleReward::default());

    // track how many addresses have been found and information about them
    let found = Arc::new(Mutex::new(0u64));
    let found_list = Arc::new(Mutex::new(Vec::<String>::new()));

    // Track processed solutions to avoid duplicates
    let processed_solutions = Arc::new(Mutex::new(HashSet::<String>::new()));

    // set up a controller for terminal output
    let term = Arc::new(Term::stdout());

    // Get the default Metal device
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

    // Compile the Metal kernel with optimization options
    let metal_src = mk_metal_src(&config);
    let options = CompileOptions::new();

    let library = device
        .new_library_with_source(&metal_src, &options)
        .map_err(|e| format!("Failed to compile Metal kernel: {}", e))?;

    let kernel_function = library
        .get_function("hashMessage", None)
        .map_err(|e| format!("Failed to get kernel function: {}", e))?;

    // Create a compute pipeline state with options for better performance
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel_function));

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .map_err(|e| format!("Failed to create pipeline state: {}", e))?;

    // Determine optimal threadgroup size based on device capabilities
    let max_threads = pipeline_state.max_total_threads_per_threadgroup();
    let optimal_size = crate::get_optimal_threadgroup_size();
    let threadgroup_size = MTLSize::new(std::cmp::min(max_threads, optimal_size), 1, 1);

    println!(
        "Using threadgroup size: {}, max threads: {}",
        threadgroup_size.width, max_threads
    );

    // Calculate grid size based on work size, ensuring it's a multiple of threadgroup size
    let grid_size = MTLSize::new(work_size, 1, 1);

    // Global counter for base nonce to ensure uniqueness across command buffers
    // Using u32 since the Metal kernel only uses a 32-bit value for the second part of the nonce
    let global_nonce_counter = Arc::new(Mutex::new(0u32));

    // determine the start time
    let start_time: f64 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    // set up variables for tracking performance
    let rate = Arc::new(Mutex::new(0.0f64));
    let cumulative_nonce = Arc::new(Mutex::new(0u64));

    // the previous timestamp of printing to the terminal
    let previous_time = Arc::new(Mutex::new(0.0f64));

    // Create buffers once outside the main loop with optimized storage modes
    // Use private storage for better performance on discrete GPUs
    let resource_options = if device.has_unified_memory() {
        MTLResourceOptions::StorageModeShared
    } else {
        MTLResourceOptions::StorageModePrivate
    };

    // Create a larger buffer for multiple solutions
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

    // Create a separate thread for UI updates to avoid blocking mining
    // Note: We need to manually copy the config fields since it doesn't implement Clone
    let factory_address = config.factory_address;
    let calling_address = config.calling_address;
    let init_code_hash = config.init_code_hash;
    let leading_zeroes_threshold = config.leading_zeroes_threshold;
    let total_zeroes_threshold = config.total_zeroes_threshold;
    let minimum_score = config.minimum_score;
    let processed_solutions_clone = processed_solutions.clone();

    let term_clone = term.clone();
    let found_clone = found.clone();
    let found_list_clone = found_list.clone();
    let rate_clone = rate.clone();
    let cumulative_nonce_clone = cumulative_nonce.clone();
    let previous_time_clone = previous_time.clone();

    let ui_thread = thread::spawn(move || {
        loop {
            // Sleep to avoid consuming too much CPU
            thread::sleep(Duration::from_millis(500));

            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            let current_time = now.as_secs() as f64;

            // Only update UI if enough time has passed
            let mut prev_time = previous_time_clone.lock().unwrap();
            if current_time - *prev_time < 0.5 {
                continue;
            }
            *prev_time = current_time;

            // Clear the terminal screen
            if term_clone.clear_screen().is_err() {
                continue; // Skip this update if we can't clear the screen
            }

            // Get the total runtime and parse into hours : minutes : seconds
            let total_runtime = current_time - start_time;
            let total_runtime_hrs = total_runtime as u64 / 3600;
            let total_runtime_mins = (total_runtime as u64 - total_runtime_hrs * 3600) / 60;
            let total_runtime_secs = total_runtime
                - (total_runtime_hrs * 3600) as f64
                - (total_runtime_mins * 60) as f64;

            // Get current statistics
            let cumulative = *cumulative_nonce_clone.lock().unwrap();
            let current_rate = *rate_clone.lock().unwrap();

            // determine the number of attempts being made per second
            let work_rate: u128 = (work_size as u128) * cumulative as u128 / 1_000_000;

            // calculate the terminal height, defaulting to a height of ten rows
            let height = terminal_size().map(|(_w, Height(h))| h).unwrap_or(10);

            // display information about the total runtime and work size
            let _ = term_clone.write_line(&format!(
                "total runtime: {}:{:02}:{:02} ({} cycles)\t\t\t\
                 work size per cycle: {}",
                total_runtime_hrs,
                total_runtime_mins,
                total_runtime_secs,
                cumulative,
                work_size.separated_string(),
            ));

            // display information about the attempt rate and found solutions
            let _ = term_clone.write_line(&format!(
                "rate: {:.2} million attempts per second\t\t\t\
                 total found this run: {}",
                work_rate as f64 * current_rate,
                *found_clone.lock().unwrap()
            ));

            // display information about the optimized search strategy
            let _ = term_clone.write_line(&format!(
                "search strategy: 4-byte salt (buffer_id + random) | 8-byte nonce (hierarchical_thread_id + counter_nonce)\t\t\
                 threshold: {} leading or {} total zeroes (min score: {})",
                leading_zeroes_threshold, total_zeroes_threshold, minimum_score
            ));

            // display recently found solutions based on terminal height
            let rows = if height < 5 { 1 } else { height as usize - 4 };
            let found_list_guard = found_list_clone.lock().unwrap();
            let last_rows: Vec<String> =
                found_list_guard.iter().cloned().rev().take(rows).collect();
            let ordered: Vec<String> = last_rows.iter().cloned().rev().collect();
            let recently_found = &ordered.join("\n");
            let _ = term_clone.write_line(recently_found);
        }
    });

    // Number of parallel command buffers to use

    // begin searching for addresses
    loop {
        // Generate a completely unique random salt for each command buffer
        // This ensures different command buffers work on completely different salt spaces
        let salt = FixedBytes::<4>::random();

        // Update message buffer contents with the unique salt
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

        // Get a unique base nonce for this command buffer
        let base_nonce = {
            let mut counter = global_nonce_counter.lock().unwrap();
            *counter += 1;
            *counter
        };

        // Update nonce buffer with the sequential base nonce
        if let Some(staging) = &staging_nonce_buffer {
            let nonce_ptr = staging.contents() as *mut u32;
            unsafe {
                *nonce_ptr = base_nonce;
            }
        } else {
            let nonce_ptr = nonce_buffer.contents() as *mut u32;
            unsafe {
                *nonce_ptr = base_nonce;
            }
        }

        // Reset solutions counter
        let counter_ptr = counter_buffer.contents() as *mut u32;
        unsafe {
            *counter_ptr = 0;
        }

        // Create command buffer and encoder
        let command_buffer = command_queue.new_command_buffer();
        let mut compute_encoder = SafeEncoder::new(command_buffer.new_compute_command_encoder());

        // Set compute pipeline
        compute_encoder
            .encoder
            .set_compute_pipeline_state(&pipeline_state);

        // Copy from staging buffers if needed
        if let Some(staging) = &staging_message_buffer {
            let mut blit_encoder = SafeBlitEncoder::new(command_buffer.new_blit_command_encoder());
            blit_encoder
                .encoder
                .copy_from_buffer(staging, 0, &message_buffer, 0, 4);
            blit_encoder.end_encoding();
        }

        if let Some(staging) = &staging_nonce_buffer {
            let mut blit_encoder = SafeBlitEncoder::new(command_buffer.new_blit_command_encoder());
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

        // Instead of using a completion handler, we'll just wait for each command buffer
        // and process results immediately
        command_buffer.commit();

        // Wait for all command buffers to complete and process results
        command_buffer.wait_until_completed();

        // Increment the cumulative nonce
        {
            let mut cumulative = cumulative_nonce.lock().unwrap();
            *cumulative += 1;
        }

        // Update rate
        {
            let mut rate_val = rate.lock().unwrap();
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            let current_time = now.as_secs_f64();
            if current_time - start_time > 0.0 {
                *rate_val = 1.0 / (current_time - start_time);
            }
        }

        // Check for solutions
        let counter_ptr = counter_buffer.contents() as *const u32;
        let num_solutions = unsafe { *counter_ptr };

        if num_solutions > 0 {
            // Process all solutions
            let solutions_ptr = solutions_buffer.contents() as *const u64;

            // Track salts we've already processed in this batch
            let mut processed_salts = HashSet::new();
            let salt_hex = hex::encode(&salt[..]);

            for i in 0..std::cmp::min(num_solutions as usize, max_solutions) {
                let solution = unsafe { *solutions_ptr.add(i) };
                if solution == 0 {
                    continue;
                }

                let solution_bytes = solution.to_le_bytes();

                // Create a unique identifier for this solution
                let solution_id =
                    format!("{}{}", hex::encode(&salt[..]), hex::encode(solution_bytes));

                // Check if we've already processed this solution
                let mut processed_guard = processed_solutions.lock().unwrap();
                if processed_guard.contains(&solution_id) {
                    continue;
                }

                // Check if we've already processed a solution with this salt in this batch
                if processed_salts.contains(&salt_hex) {
                    continue;
                }

                // Add salt to processed salts for this batch
                processed_salts.insert(salt_hex.clone());

                // Add to processed solutions
                processed_guard.insert(solution_id);
                drop(processed_guard); // Release the lock early

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

                // calculate address metrics using helper function
                let (leading_bytes, leading_nibbles, total_zeroes) =
                    calculate_address_metrics(address);

                // Calculate the reward score for this address using pre-calculated values
                let reward_score =
                    rewards.get_from_counts(leading_bytes, leading_nibbles, total_zeroes);

                // Only process if the score meets the minimum threshold
                if reward_score >= config.minimum_score {
                    let output = format!(
                        "0x{}{}{} => {} => {}",
                        hex::encode(config.calling_address),
                        hex::encode(salt),
                        hex::encode(solution_bytes),
                        address,
                        reward_score,
                    );

                    let show = format!("{output} ({leading_bytes}/{total_zeroes})");

                    // Update found count and list
                    {
                        let mut found_list_guard = found_list.lock().unwrap();
                        found_list_guard.push(show.to_string());
                    }

                    // Write to file using a different approach
                    {
                        // Spawn a thread to handle file writing to avoid blocking the main thread
                        let output_to_write = output.clone();
                        thread::spawn(move || {
                            // Use OpenOptions to open the file in append mode
                            if let Ok(mut file) = std::fs::OpenOptions::new()
                                .append(true)
                                .open("efficient_addresses.txt")
                            {
                                // Write to the file
                                if let Err(e) = writeln!(file, "{}", output_to_write) {
                                    eprintln!("Error writing to file: {}", e);
                                }
                            } else {
                                eprintln!("Error opening file for writing");
                            }
                        });
                    }

                    // Increment found counter
                    let mut found_guard = found.lock().unwrap();
                    *found_guard += 1;
                }
            }
        }
    }
}
