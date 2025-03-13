use crate::{Config, Reward, CONTROL_CHARACTER};
use alloy_primitives::{hex, Address, FixedBytes};
use console::Term;
use metal::*;
use rand::{thread_rng, Rng, RngCore};
use std::collections::HashSet;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tiny_keccak::{Hasher, Keccak};

use super::safety_wrappers::{SafeBlitEncoder, SafeEncoder};
use super::setup::{initialize_metal, MetalContext};
use super::solution::SolutionProcessor;
use super::ui::UiManager;

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
    let file = Arc::new(crate::output_file());

    // create object for computing rewards (relative rarity) for a given address
    let rewards = Arc::new(Reward::new());

    // track how many addresses have been found and information about them
    let found = Arc::new(Mutex::new(0u64));
    let found_list = Arc::new(Mutex::new(Vec::<String>::new()));

    // Track processed solutions to avoid duplicates
    let processed_solutions = Arc::new(Mutex::new(HashSet::<String>::new()));

    // set up a controller for terminal output
    let term = Arc::new(Term::stdout());

    // Initialize Metal context
    let metal_ctx = initialize_metal(&config, work_size)?;

    // Set up UI manager
    let rate = Arc::new(Mutex::new(0.0f64));
    let cumulative_nonce = Arc::new(Mutex::new(0u64));

    let ui_manager = UiManager::new(
        term.clone(),
        found.clone(),
        found_list.clone(),
        rate.clone(),
        cumulative_nonce.clone(),
        config.clone(),
        work_size,
    );

    // Start UI thread
    let ui_thread = ui_manager.start_ui_thread();

    // Set up solution processor
    let solution_processor = SolutionProcessor::new(
        config.clone(),
        rewards.clone(),
        found.clone(),
        found_list.clone(),
    );

    let processed_solutions = solution_processor.get_processed_solutions();

    // Global counter for base nonce to ensure uniqueness across command buffers
    let global_nonce_counter = Arc::new(Mutex::new(0u32));

    // create a random number generator
    let mut rng = thread_rng();

    // determine the start time
    let start_time = ui_manager.get_start_time();

    // Number of parallel command buffers to use
    let num_parallel_buffers = 3;

    // Debug counter to force checking some addresses
    let debug_counter = Arc::new(Mutex::new(0u32));

    // begin searching for addresses
    loop {
        // Create multiple command buffers for pipelining
        let mut command_buffers = Vec::with_capacity(num_parallel_buffers);

        for buffer_idx in 0..num_parallel_buffers {
            // Generate a completely unique random salt for each command buffer
            // This ensures different command buffers work on completely different salt spaces
            // Include buffer_idx in the first byte to help with debugging and tracking
            let mut salt_bytes = [0u8; 4];
            salt_bytes[0] = buffer_idx as u8; // Use buffer index as first byte for tracking
            rng.fill_bytes(&mut salt_bytes[1..4]); // Fill the rest with random bytes
            let salt = FixedBytes::<4>::from_slice(&salt_bytes);

            // Update message buffer contents with the unique salt
            if let Some(staging) = &metal_ctx.staging_message_buffer {
                let message_ptr = staging.contents() as *mut u8;
                unsafe {
                    std::ptr::copy_nonoverlapping(salt.as_ptr(), message_ptr, 4);
                }
            } else {
                let message_ptr = metal_ctx.message_buffer.contents() as *mut u8;
                unsafe {
                    std::ptr::copy_nonoverlapping(salt.as_ptr(), message_ptr, 4);
                }
            }

            // Get a unique base nonce for this command buffer
            // Use simple sequential counter like in the original implementation
            let base_nonce = {
                let mut counter = global_nonce_counter.lock().unwrap();
                *counter = counter.wrapping_add(1);
                *counter
            };

            // Update nonce buffer with the base nonce
            if let Some(staging) = &metal_ctx.staging_nonce_buffer {
                let nonce_ptr = staging.contents() as *mut u32;
                unsafe {
                    *nonce_ptr = base_nonce;
                }
            } else {
                let nonce_ptr = metal_ctx.nonce_buffer.contents() as *mut u32;
                unsafe {
                    *nonce_ptr = base_nonce;
                }
            }

            // Reset solutions counter
            let counter_ptr = metal_ctx.counter_buffer.contents() as *mut u32;
            unsafe {
                *counter_ptr = 0;
            }

            // Create command buffer and encoder
            let command_buffer = metal_ctx.command_queue.new_command_buffer();
            let mut compute_encoder =
                SafeEncoder::new(&command_buffer.new_compute_command_encoder());

            // Set compute pipeline
            compute_encoder
                .encoder
                .set_compute_pipeline_state(&metal_ctx.pipeline_state);

            // Copy from staging buffers if needed
            if let Some(staging) = &metal_ctx.staging_message_buffer {
                let mut blit_encoder =
                    SafeBlitEncoder::new(&command_buffer.new_blit_command_encoder());
                blit_encoder
                    .encoder
                    .copy_from_buffer(&staging, 0, &metal_ctx.message_buffer, 0, 4);
                blit_encoder.end_encoding();
            }

            if let Some(staging) = &metal_ctx.staging_nonce_buffer {
                let mut blit_encoder =
                    SafeBlitEncoder::new(&command_buffer.new_blit_command_encoder());
                blit_encoder
                    .encoder
                    .copy_from_buffer(&staging, 0, &metal_ctx.nonce_buffer, 0, 4);
                blit_encoder.end_encoding();
            }

            // Set buffers
            compute_encoder
                .encoder
                .set_buffer(0, Some(&metal_ctx.message_buffer), 0);
            compute_encoder
                .encoder
                .set_buffer(1, Some(&metal_ctx.nonce_buffer), 0);
            compute_encoder
                .encoder
                .set_buffer(2, Some(&metal_ctx.solutions_buffer), 0);
            compute_encoder
                .encoder
                .set_buffer(3, Some(&metal_ctx.counter_buffer), 0);

            // Dispatch threads
            compute_encoder
                .encoder
                .dispatch_threads(metal_ctx.grid_size, metal_ctx.threadgroup_size);

            // End encoding
            compute_encoder.end_encoding();

            // Commit the command buffer
            command_buffer.commit();

            command_buffers.push((command_buffer, salt)); // Store the salt with the command buffer
        }

        // Process all command buffers in parallel
        for (buffer, salt) in command_buffers {
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

            // Check for solutions using direct pointer access
            let counter_ptr = metal_ctx.counter_buffer.contents() as *const u32;
            let num_solutions = unsafe { *counter_ptr };

            // Debug: Force check some addresses periodically
            let should_debug = {
                let mut debug_count = debug_counter.lock().unwrap();
                *debug_count += 1;
                *debug_count % 10 == 0 // Every 10th batch for more frequent debugging
            };

            if should_debug {
                // Try multiple test solutions to increase chances of finding one that meets criteria
                let test_solutions = [12345u64, 67890u64, 111111u64, 999999u64, 123456789u64];

                for test_solution in test_solutions {
                    println!(
                        "\nDEBUG: Testing solution with salt={} and nonce={}",
                        hex::encode(&salt[..]),
                        test_solution
                    );

                    // Manually construct and hash the message to see the resulting address
                    let solution_bytes = test_solution.to_le_bytes();

                    let mut solution_message = [0; 85];
                    solution_message[0] = CONTROL_CHARACTER;
                    solution_message[1..21].copy_from_slice(&config.factory_address);
                    solution_message[21..41].copy_from_slice(&config.calling_address);
                    solution_message[41..45].copy_from_slice(&salt[..]);
                    solution_message[45..53].copy_from_slice(&solution_bytes);
                    solution_message[53..].copy_from_slice(&config.init_code_hash);

                    // Create hash object
                    let mut hash = Keccak::v256();
                    hash.update(&solution_message);

                    // Get the result
                    let mut res = [0u8; 32];
                    hash.finalize(&mut res);

                    // Get the address
                    let address = <&Address>::try_from(&res[12..]).unwrap();
                    let address_hex = hex::encode(address.as_slice());

                    // Count zeroes (bytes)
                    let mut total_zeroes_bytes = 0;
                    let mut leading_zeroes_bytes = 0;
                    let mut still_leading_bytes = true;

                    for &byte in address.iter() {
                        if byte == 0 {
                            total_zeroes_bytes += 1;
                            if still_leading_bytes {
                                leading_zeroes_bytes += 1;
                            }
                        } else {
                            still_leading_bytes = false;
                        }
                    }

                    // Count zeroes (nibbles/hex digits)
                    let mut total_zeroes_nibbles = 0;
                    let mut leading_zeroes_nibbles = 0;
                    let mut still_leading_nibbles = true;

                    for c in address_hex.chars() {
                        if c == '0' {
                            total_zeroes_nibbles += 1;
                            if still_leading_nibbles {
                                leading_zeroes_nibbles += 1;
                            }
                        } else {
                            still_leading_nibbles = false;
                        }
                    }

                    println!("DEBUG: Generated address: {}", address);
                    println!(
                        "DEBUG: Zeroes (bytes): Leading={}, Total={}",
                        leading_zeroes_bytes, total_zeroes_bytes
                    );
                    println!(
                        "DEBUG: Zeroes (nibbles/hex): Leading={}, Total={}",
                        leading_zeroes_nibbles, total_zeroes_nibbles
                    );
                    println!(
                        "DEBUG: Thresholds: Leading={}, Total={}",
                        config.leading_zeroes_threshold, config.total_zeroes_threshold
                    );

                    // Check if it meets criteria (using nibbles)
                    let meets_leading =
                        leading_zeroes_nibbles >= config.leading_zeroes_threshold as usize;
                    let meets_total =
                        total_zeroes_nibbles >= config.total_zeroes_threshold as usize;
                    println!(
                        "DEBUG: Meets criteria (nibbles): Leading={}, Total={}",
                        meets_leading, meets_total
                    );

                    // Now process through the normal solution processor
                    let result = solution_processor.process_solution(&salt[..], test_solution);
                    println!("DEBUG: Solution processing result: {}\n", result);
                }

                // Also check if the kernel is finding any solutions at all
                if num_solutions > 0 {
                    println!(
                        "DEBUG: Kernel found {} solutions but they may not meet criteria",
                        num_solutions
                    );

                    // Examine all solutions
                    let solutions_ptr = metal_ctx.solutions_buffer.contents() as *const u64;
                    for i in 0..std::cmp::min(num_solutions as usize, metal_ctx.max_solutions) {
                        let solution = unsafe { *solutions_ptr.add(i) };
                        if solution == 0 {
                            continue;
                        }

                        // Print detailed information about each solution
                        println!("DEBUG: Kernel solution {}: {}", i, solution);

                        // Manually check if this solution would meet criteria
                        let solution_bytes = solution.to_le_bytes();

                        let mut solution_message = [0; 85];
                        solution_message[0] = CONTROL_CHARACTER;
                        solution_message[1..21].copy_from_slice(&config.factory_address);
                        solution_message[21..41].copy_from_slice(&config.calling_address);
                        solution_message[41..45].copy_from_slice(&salt[..]);
                        solution_message[45..53].copy_from_slice(&solution_bytes);
                        solution_message[53..].copy_from_slice(&config.init_code_hash);

                        // Create hash object
                        let mut hash = Keccak::v256();
                        hash.update(&solution_message);

                        // Get the result
                        let mut res = [0u8; 32];
                        hash.finalize(&mut res);

                        // Get the address
                        let address = <&Address>::try_from(&res[12..]).unwrap();
                        let address_hex = hex::encode(address.as_slice());

                        // Count zeroes (bytes)
                        let mut total_zeroes_bytes = 0;
                        let mut leading_zeroes_bytes = 0;
                        let mut still_leading_bytes = true;

                        for &byte in address.iter() {
                            if byte == 0 {
                                total_zeroes_bytes += 1;
                                if still_leading_bytes {
                                    leading_zeroes_bytes += 1;
                                }
                            } else {
                                still_leading_bytes = false;
                            }
                        }

                        // Count zeroes (nibbles/hex digits)
                        let mut total_zeroes_nibbles = 0;
                        let mut leading_zeroes_nibbles = 0;
                        let mut still_leading_nibbles = true;

                        for c in address_hex.chars() {
                            if c == '0' {
                                total_zeroes_nibbles += 1;
                                if still_leading_nibbles {
                                    leading_zeroes_nibbles += 1;
                                }
                            } else {
                                still_leading_nibbles = false;
                            }
                        }

                        println!("DEBUG: Kernel solution {} address: {}", i, address);
                        println!(
                            "DEBUG: Kernel solution {} zeroes (nibbles): Leading={}, Total={}",
                            i, leading_zeroes_nibbles, total_zeroes_nibbles
                        );

                        // Check if it meets criteria (using nibbles)
                        let meets_leading =
                            leading_zeroes_nibbles >= config.leading_zeroes_threshold as usize;
                        let meets_total =
                            total_zeroes_nibbles >= config.total_zeroes_threshold as usize;
                        println!(
                            "DEBUG: Kernel solution {} meets criteria: Leading={}, Total={}",
                            i, meets_leading, meets_total
                        );
                    }
                }
            }

            if num_solutions > 0 {
                println!("Found {} solutions in this batch!", num_solutions);

                // Process all solutions
                let solutions_ptr = metal_ctx.solutions_buffer.contents() as *const u64;

                for i in 0..std::cmp::min(num_solutions as usize, metal_ctx.max_solutions) {
                    let solution = unsafe { *solutions_ptr.add(i) };
                    if solution == 0 {
                        continue;
                    }

                    println!("Processing solution: {}", solution);

                    // Create a unique identifier for this solution
                    let solution_id = format!(
                        "{}{}",
                        hex::encode(&salt[..]),
                        hex::encode(&solution.to_le_bytes())
                    );

                    // Check if we've already processed this solution
                    let mut processed_guard = processed_solutions.lock().unwrap();
                    if processed_guard.contains(&solution_id) {
                        println!("Solution already processed: {}", solution_id);
                        continue;
                    }

                    // Add to processed solutions
                    processed_guard.insert(solution_id.clone());
                    drop(processed_guard); // Release the lock early

                    // IMPORTANT: The kernel constructs the nonce differently than we do in our validation
                    // We need to use the solution value directly as provided by the kernel
                    // This ensures we're validating the exact same hash that the kernel computed
                    let result = solution_processor.process_solution(&salt[..], solution);

                    if result {
                        println!("Successfully processed solution: {}", solution_id);
                    } else {
                        // If the solution failed validation, let's try to understand why
                        println!("Solution did not meet criteria: {}", solution_id);

                        // Let's print the raw solution bytes for debugging
                        let solution_bytes = solution.to_le_bytes();
                        println!("Raw solution bytes: {}", hex::encode(&solution_bytes));

                        // Construct the message exactly as the kernel would
                        let mut solution_message = [0; 85];
                        solution_message[0] = CONTROL_CHARACTER;
                        solution_message[1..21].copy_from_slice(&config.factory_address);
                        solution_message[21..41].copy_from_slice(&config.calling_address);
                        solution_message[41..45].copy_from_slice(&salt[..]);
                        solution_message[45..53].copy_from_slice(&solution_bytes);
                        solution_message[53..].copy_from_slice(&config.init_code_hash);

                        println!("Message being hashed: {}", hex::encode(&solution_message));

                        // Create hash object
                        let mut hash = Keccak::v256();
                        hash.update(&solution_message);

                        // Get the result
                        let mut res = [0u8; 32];
                        hash.finalize(&mut res);

                        println!("Hash result: {}", hex::encode(&res));

                        // Get the address
                        let address = <&Address>::try_from(&res[12..]).unwrap();

                        // Count zeroes
                        let mut total_zeroes = 0;
                        let mut leading_zeroes = 0;
                        let mut still_leading = true;

                        for &byte in address.iter() {
                            if byte == 0 {
                                total_zeroes += 1;
                                if still_leading {
                                    leading_zeroes += 1;
                                }
                            } else {
                                still_leading = false;
                            }
                        }

                        println!("Failed solution address: {}", address);
                        println!(
                            "Failed solution zeroes: Leading={}, Total={}",
                            leading_zeroes, total_zeroes
                        );
                        println!(
                            "Thresholds: Leading={}, Total={}",
                            config.leading_zeroes_threshold, config.total_zeroes_threshold
                        );

                        // Check if it meets criteria
                        let meets_leading =
                            leading_zeroes >= config.leading_zeroes_threshold as usize;
                        let meets_total = total_zeroes >= config.total_zeroes_threshold as usize;
                        println!(
                            "Should meet criteria: Leading={}, Total={}",
                            meets_leading, meets_total
                        );
                    }
                }
            }
        }
    }
}
