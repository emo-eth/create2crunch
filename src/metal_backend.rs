use crate::{Config, Reward, CONTROL_CHARACTER, WORK_SIZE};
use alloy_primitives::{hex, Address, FixedBytes};
use byteorder::{BigEndian, ByteOrder, LittleEndian};
use console::Term;
use fs4::FileExt;
use metal::*;
use rand::{thread_rng, Rng};
use separator::Separatable;
use std::error::Error;
use std::fmt::Write as _;
use std::io::prelude::*;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use terminal_size::{terminal_size, Height};
use tiny_keccak::{Hasher, Keccak};

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

    // (create if necessary) and open a file where found salts will be written
    let file = Arc::new(crate::output_file());

    // create object for computing rewards (relative rarity) for a given address
    let rewards = Arc::new(Reward::new());

    // track how many addresses have been found and information about them
    let found = Arc::new(Mutex::new(0u64));
    let found_list = Arc::new(Mutex::new(Vec::<String>::new()));

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
    let thread_execution_width = pipeline_state.thread_execution_width();
    let threadgroup_size = MTLSize::new(std::cmp::min(max_threads, 256), 1, 1);

    println!("Using threadgroup size: {}", threadgroup_size.width);

    // Calculate grid size based on work size, ensuring it's a multiple of threadgroup size
    let grid_size = MTLSize::new(WORK_SIZE as u64, 1, 1);

    // create a random number generator
    let mut rng = thread_rng();

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
    let factory_address = config.factory_address.clone();
    let calling_address = config.calling_address.clone();
    let init_code_hash = config.init_code_hash.clone();
    let leading_zeroes_threshold = config.leading_zeroes_threshold;
    let total_zeroes_threshold = config.total_zeroes_threshold;

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
            if let Err(_) = term_clone.clear_screen() {
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
            let work_rate: u128 = (WORK_SIZE as u128) * cumulative as u128 / 1_000_000;

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
                WORK_SIZE.separated_string(),
            ));

            // display information about the attempt rate and found solutions
            let _ = term_clone.write_line(&format!(
                "rate: {:.2} million attempts per second\t\t\t\
                 total found this run: {}",
                work_rate as f64 * current_rate,
                *found_clone.lock().unwrap()
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
    let num_parallel_buffers = 3;

    // begin searching for addresses
    loop {
        // construct the 4-byte message to hash
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

        // reset nonce & initialize it to a random value for better distribution
        let mut nonce: [u32; 1] = rng.gen();

        // Create multiple command buffers for pipelining
        let mut command_buffers = Vec::with_capacity(num_parallel_buffers);

        for _ in 0..num_parallel_buffers {
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

            // Create compute command encoder
            let compute_encoder = command_buffer.new_compute_command_encoder();

            // Set compute pipeline
            compute_encoder.set_compute_pipeline_state(&pipeline_state);

            // Copy from staging buffers if needed
            if let Some(staging) = &staging_message_buffer {
                let blit_encoder = command_buffer.new_blit_command_encoder();
                blit_encoder.copy_from_buffer(&staging, 0, &message_buffer, 0, 4);
                blit_encoder.end_encoding();
            }

            if let Some(staging) = &staging_nonce_buffer {
                let blit_encoder = command_buffer.new_blit_command_encoder();
                blit_encoder.copy_from_buffer(&staging, 0, &nonce_buffer, 0, 4);
                blit_encoder.end_encoding();
            }

            // Set buffers
            compute_encoder.set_buffer(0, Some(&message_buffer), 0);
            compute_encoder.set_buffer(1, Some(&nonce_buffer), 0);
            compute_encoder.set_buffer(2, Some(&solutions_buffer), 0);
            compute_encoder.set_buffer(3, Some(&counter_buffer), 0);

            // Dispatch threads
            compute_encoder.dispatch_threads(grid_size, threadgroup_size);

            // End encoding
            compute_encoder.end_encoding();

            // Instead of using a completion handler, we'll just wait for each command buffer
            // and process results immediately
            command_buffer.commit();
            command_buffers.push(command_buffer);

            // Increment nonce for next buffer
            nonce[0] = nonce[0].wrapping_add(WORK_SIZE as u32);
        }

        // Wait for all command buffers to complete and process results
        for buffer in command_buffers {
            buffer.wait_until_completed();

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
                for i in 0..std::cmp::min(num_solutions as usize, max_solutions) {
                    let solution = unsafe { *solutions_ptr.add(i) };
                    if solution == 0 {
                        continue;
                    }

                    let solution_bytes = solution.to_le_bytes();

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

                    let show = format!("{output} ({leading} / {total})");

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
