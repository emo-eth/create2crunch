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
use std::time::{SystemTime, UNIX_EPOCH};
use terminal_size::{terminal_size, Height};
use tiny_keccak::{Hasher, Keccak};

static KERNEL_SRC: &str = include_str!("./kernels/keccak256.metal");

/// Function to generate Metal kernel source with the appropriate constants
fn mk_metal_src(config: &Config) -> String {
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
    let file = crate::output_file();

    // create object for computing rewards (relative rarity) for a given address
    let rewards = Reward::new();

    // track how many addresses have been found and information about them
    let mut found: u64 = 0;
    let mut found_list: Vec<String> = vec![];

    // set up a controller for terminal output
    let term = Term::stdout();

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

    // Create a command queue
    let command_queue = device.new_command_queue();

    // Compile the Metal kernel
    let metal_src = mk_metal_src(&config);
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

    // Determine optimal threadgroup size
    let threadgroup_size = MTLSize::new(
        std::cmp::min(pipeline_state.max_total_threads_per_threadgroup(), 256),
        1,
        1,
    );

    // Calculate grid size based on work size
    let grid_size = MTLSize::new(WORK_SIZE as u64, 1, 1);

    // create a random number generator
    let mut rng = thread_rng();

    // determine the start time
    let start_time: f64 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    // set up variables for tracking performance
    let mut rate: f64 = 0.0;
    let mut cumulative_nonce: u64 = 0;

    // the previous timestamp of printing to the terminal
    let mut previous_time: f64 = 0.0;

    // the last work duration in milliseconds
    let mut work_duration_millis: u64 = 0;

    // Create buffers once outside the main loop
    let message_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let nonce_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let solutions_buffer = device.new_buffer(8, MTLResourceOptions::StorageModeShared);

    // begin searching for addresses
    loop {
        // construct the 4-byte message to hash
        let salt = FixedBytes::<4>::random();

        // Update message buffer contents
        let message_ptr = message_buffer.contents() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(salt.as_ptr(), message_ptr, 4);
        }

        // reset nonce & initialize it to a random value for better distribution
        let mut nonce: [u32; 1] = rng.gen();
        let mut view_buf = [0; 8];

        // Update nonce buffer contents
        let nonce_ptr = nonce_buffer.contents() as *mut u32;
        unsafe {
            *nonce_ptr = nonce[0];
        }

        // repeatedly enqueue kernel to search for new addresses
        loop {
            // Reset solutions buffer
            let solutions_ptr = solutions_buffer.contents() as *mut u64;
            unsafe {
                *solutions_ptr = 0;
            }

            // Create command buffer
            let command_buffer = command_queue.new_command_buffer();

            // Create compute command encoder
            let compute_encoder = command_buffer.new_compute_command_encoder();

            // Set compute pipeline
            compute_encoder.set_compute_pipeline_state(&pipeline_state);

            // Set buffers
            compute_encoder.set_buffer(0, Some(&message_buffer), 0);
            compute_encoder.set_buffer(1, Some(&nonce_buffer), 0);
            compute_encoder.set_buffer(2, Some(&solutions_buffer), 0);

            // Dispatch threads
            compute_encoder.dispatch_threads(grid_size, threadgroup_size);

            // End encoding
            compute_encoder.end_encoding();

            // Commit command buffer and wait for completion
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // calculate the current time
            let mut now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            let current_time = now.as_secs() as f64;

            // we don't want to print too fast
            let print_output = current_time - previous_time > 0.99;
            previous_time = current_time;

            // clear the terminal screen
            if print_output {
                term.clear_screen()?;

                // get the total runtime and parse into hours : minutes : seconds
                let total_runtime = current_time - start_time;
                let total_runtime_hrs = total_runtime as u64 / 3600;
                let total_runtime_mins = (total_runtime as u64 - total_runtime_hrs * 3600) / 60;
                let total_runtime_secs = total_runtime
                    - (total_runtime_hrs * 3600) as f64
                    - (total_runtime_mins * 60) as f64;

                // determine the number of attempts being made per second
                let work_rate: u128 = (WORK_SIZE as u128) * cumulative_nonce as u128 / 1_000_000;
                if total_runtime > 0.0 {
                    rate = 1.0 / total_runtime;
                }

                // fill the buffer for viewing the properly-formatted nonce
                LittleEndian::write_u64(&mut view_buf, (nonce[0] as u64) << 32);

                // calculate the terminal height, defaulting to a height of ten rows
                let height = terminal_size().map(|(_w, Height(h))| h).unwrap_or(10);

                // display information about the total runtime and work size
                term.write_line(&format!(
                    "total runtime: {}:{:02}:{:02} ({} cycles)\t\t\t\
                     work size per cycle: {}",
                    total_runtime_hrs,
                    total_runtime_mins,
                    total_runtime_secs,
                    cumulative_nonce,
                    WORK_SIZE.separated_string(),
                ))?;

                // display information about the attempt rate and found solutions
                term.write_line(&format!(
                    "rate: {:.2} million attempts per second\t\t\t\
                     total found this run: {}",
                    work_rate as f64 * rate,
                    found
                ))?;

                // display information about the current search criteria
                term.write_line(&format!(
                    "current search space: {}xxxxxxxx{:08x}\t\t\
                     threshold: {} leading or {} total zeroes",
                    hex::encode(salt),
                    BigEndian::read_u64(&view_buf),
                    config.leading_zeroes_threshold,
                    config.total_zeroes_threshold
                ))?;

                // display recently found solutions based on terminal height
                let rows = if height < 5 { 1 } else { height as usize - 4 };
                let last_rows: Vec<String> = found_list.iter().cloned().rev().take(rows).collect();
                let ordered: Vec<String> = last_rows.iter().cloned().rev().collect();
                let recently_found = &ordered.join("\n");
                term.write_line(recently_found)?;
            }

            // increment the cumulative nonce (does not reset after a match)
            cumulative_nonce += 1;

            // record the start time of the work
            let work_start_time_millis = now.as_secs() * 1000 + now.subsec_nanos() as u64 / 1000000;

            // sleep for 98% of the previous work duration to conserve CPU
            if work_duration_millis != 0 {
                std::thread::sleep(std::time::Duration::from_millis(
                    work_duration_millis * 980 / 1000,
                ));
            }

            // Read the solution from the buffer
            let solution_ptr = solutions_buffer.contents() as *const u64;
            let solution = unsafe { *solution_ptr };

            // record the end time of the work and compute how long the work took
            now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            work_duration_millis = (now.as_secs() * 1000 + now.subsec_nanos() as u64 / 1000000)
                - work_start_time_millis;

            // if a solution is found, end the loop
            if solution != 0 {
                break;
            }

            // if no solution has yet been found, increment the nonce
            nonce[0] += 1;

            // Update the nonce buffer with the incremented nonce value
            let nonce_ptr = nonce_buffer.contents() as *mut u32;
            unsafe {
                *nonce_ptr = nonce[0];
            }
        }

        // Read the solution from the buffer
        let solution_ptr = solutions_buffer.contents() as *const u64;
        let solution = unsafe { *solution_ptr };

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
            hex::encode(config.calling_address),
            hex::encode(salt),
            hex::encode(solution_bytes),
            address,
            reward,
        );

        let show = format!("{output} ({leading} / {total})");
        found_list.push(show.to_string());

        file.lock_exclusive().expect("Couldn't lock file.");

        writeln!(&file, "{output}").expect("Couldn't write to `efficient_addresses.txt` file.");

        FileExt::unlock(&file).expect("Couldn't unlock file.");
        found += 1;
    }
}
