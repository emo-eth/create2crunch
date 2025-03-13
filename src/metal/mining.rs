use crate::{Config, Reward};
use alloy_primitives::FixedBytes;
use console::Term;
use metal::*;
use rand::{thread_rng, Rng, RngCore};
use std::collections::HashSet;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

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

    // Create separate counter buffers for each command buffer
    let mut counter_buffers = Vec::with_capacity(num_parallel_buffers);
    let mut solutions_buffers = Vec::with_capacity(num_parallel_buffers);
    let mut message_buffers = Vec::with_capacity(num_parallel_buffers);
    let mut nonce_buffers = Vec::with_capacity(num_parallel_buffers);

    for _ in 0..num_parallel_buffers {
        // Create a new counter buffer for each command buffer
        let counter_buffer = metal_ctx.device.new_buffer_with_data(
            &[0u32] as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            metal_ctx.resource_options,
        );

        // Create a new solutions buffer for each command buffer
        let solutions_buffer = metal_ctx.device.new_buffer(
            (std::mem::size_of::<u64>() * metal_ctx.max_solutions) as u64,
            metal_ctx.resource_options,
        );

        // Create a new message buffer for each command buffer
        let message_buffer = metal_ctx.device.new_buffer(
            4, // 4 bytes for the salt
            metal_ctx.resource_options,
        );

        // Create a new nonce buffer for each command buffer
        let nonce_buffer = metal_ctx.device.new_buffer_with_data(
            &[0u32] as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            metal_ctx.resource_options,
        );

        counter_buffers.push(counter_buffer);
        solutions_buffers.push(solutions_buffer);
        message_buffers.push(message_buffer);
        nonce_buffers.push(nonce_buffer);
    }

    // begin searching for addresses
    loop {
        // Create multiple command buffers for pipelining
        let mut command_buffers = Vec::with_capacity(num_parallel_buffers);

        for buffer_idx in 0..num_parallel_buffers {
            // Generate a completely unique random salt for each command buffer
            let mut salt_bytes = [0u8; 4];
            rng.fill_bytes(&mut salt_bytes);

            // Incorporate buffer index into the salt to ensure uniqueness across parallel buffers
            salt_bytes[0] = buffer_idx as u8;

            let salt = FixedBytes::<4>::from_slice(&salt_bytes);

            // Update message buffer contents with the unique salt
            let contents = message_buffers[buffer_idx].contents();
            // Safety: We know the buffer is at least 4 bytes in size and is properly aligned
            unsafe {
                let message_slice = std::slice::from_raw_parts_mut(contents as *mut u8, 4);
                message_slice.copy_from_slice(&salt[..]);
            }

            // Get a unique base nonce for this command buffer
            let base_nonce = {
                let mut counter = global_nonce_counter.lock().unwrap();
                *counter = counter.wrapping_add(1);
                (*counter << 16) | (rng.gen::<u32>() & 0xFFFF)
            };

            // Update nonce buffer with the base nonce
            let contents = nonce_buffers[buffer_idx].contents();
            // Safety: We know the buffer is at least 4 bytes in size and is properly aligned for u32
            unsafe {
                let nonce_slice = std::slice::from_raw_parts_mut(contents as *mut u32, 1);
                nonce_slice[0] = base_nonce;
            }

            // Reset solutions counter for this buffer
            let contents = counter_buffers[buffer_idx].contents();
            // Safety: We know the buffer is at least 4 bytes in size and is properly aligned for u32
            unsafe {
                let counter_slice = std::slice::from_raw_parts_mut(contents as *mut u32, 1);
                counter_slice[0] = 0;
            }

            // Create command buffer and encoder
            let command_buffer = metal_ctx.command_queue.new_command_buffer();
            let mut compute_encoder =
                SafeEncoder::new(command_buffer.new_compute_command_encoder());

            // Set compute pipeline
            compute_encoder
                .encoder
                .set_compute_pipeline_state(&metal_ctx.pipeline_state);

            // Set buffers - use the buffer-specific message and nonce buffers
            compute_encoder
                .encoder
                .set_buffer(0, Some(&message_buffers[buffer_idx]), 0);
            compute_encoder
                .encoder
                .set_buffer(1, Some(&nonce_buffers[buffer_idx]), 0);
            compute_encoder
                .encoder
                .set_buffer(2, Some(&solutions_buffers[buffer_idx]), 0);
            compute_encoder
                .encoder
                .set_buffer(3, Some(&counter_buffers[buffer_idx]), 0);

            // Dispatch threads
            compute_encoder
                .encoder
                .dispatch_threads(metal_ctx.grid_size, metal_ctx.threadgroup_size);

            // End encoding
            compute_encoder.end_encoding();

            // Commit the command buffer
            command_buffer.commit();

            command_buffers.push((command_buffer, salt, buffer_idx)); // Store the salt and buffer index with the command buffer
        }

        // validate the command buffers do not all start with 02

        // Wait for all command buffers to complete and process results
        let mut command_buffers_to_process = command_buffers;

        for (buffer_idx, (buffer, salt, original_idx)) in
            command_buffers_to_process.into_iter().enumerate()
        {
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

            // Check for solutions using the buffer's own counter
            let contents = counter_buffers[original_idx].contents();
            // Safety: We know the buffer is at least 4 bytes in size and is properly aligned for u32
            let num_solutions = unsafe {
                let counter_slice = std::slice::from_raw_parts(contents as *const u32, 1);
                counter_slice[0]
            };

            if num_solutions > 0 {
                // Process all solutions from this buffer's solutions buffer
                let contents = solutions_buffers[original_idx].contents();
                // Safety: We know the buffer is large enough to hold the solutions and is properly aligned for u64
                unsafe {
                    let solutions_slice = std::slice::from_raw_parts(
                        contents as *const u64,
                        std::cmp::min(num_solutions as usize, metal_ctx.max_solutions),
                    );

                    for &solution in solutions_slice {
                        if solution == 0 {
                            continue;
                        }

                        // Process the solution using the salt that was used to generate this solution
                        solution_processor.process_solution(&salt[..], solution);
                    }
                }
            }
        }
    }
}
