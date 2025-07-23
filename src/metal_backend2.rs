use crate::{
    calculate_address_metrics, Config, LeadingNibbleReward, RewardTrait, CONTROL_CHARACTER,
};
use alloy_primitives::{hex, Address, FixedBytes};
use console::Term;
use metal::*;
use rand::{thread_rng, Rng, RngCore};
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

/// Structure to hold all Metal buffers
struct BufferSet {
    message_buffer: Buffer,
    nonce_buffer: Buffer,
    solutions_buffer: Buffer,
    counter_buffer: Buffer,
    staging_message_buffer: Option<Buffer>,
    staging_nonce_buffer: Option<Buffer>,
    max_solutions: usize,
}

/// Structure to hold shared mining state
struct MiningState {
    found: Arc<Mutex<u64>>,
    found_list: Arc<Mutex<Vec<String>>>,
    processed_solutions: Arc<Mutex<HashSet<String>>>,
    rate: Arc<Mutex<f64>>,
    cumulative_nonce: Arc<Mutex<u64>>,
    previous_time: Arc<Mutex<f64>>,
    global_nonce_counter: Arc<Mutex<u32>>,
    start_time: f64,
}

impl MiningState {
    fn new() -> Self {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        MiningState {
            found: Arc::new(Mutex::new(0u64)),
            found_list: Arc::new(Mutex::new(Vec::<String>::new())),
            processed_solutions: Arc::new(Mutex::new(HashSet::<String>::new())),
            rate: Arc::new(Mutex::new(0.0f64)),
            cumulative_nonce: Arc::new(Mutex::new(0u64)),
            previous_time: Arc::new(Mutex::new(0.0f64)),
            global_nonce_counter: Arc::new(Mutex::new(0u32)),
            start_time,
        }
    }
}

/// Structure to hold pipeline resources
struct PipelineResources {
    device: Device,
    command_queue: CommandQueue,
    pipeline_state: ComputePipelineState,
    threadgroup_size: MTLSize,
    grid_size: MTLSize,
}

/// Setup Metal device based on configuration
fn setup_metal_device(config: &Config) -> Result<Device, Box<dyn Error>> {
    println!(
        "Setting up experimental Metal miner using device {}...",
        config.gpu_device
    );

    // Get all Metal devices
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

    // Clone the device to avoid lifetime issues
    let device = devices[device_index].clone();
    println!("Using Metal device: {}", device.name());

    Ok(device)
}

/// Create compute pipeline for Metal GPU
fn create_compute_pipeline(
    device: &Device,
    config: &Config,
) -> Result<PipelineResources, Box<dyn Error>> {
    // Get the work size from .env if available
    let work_size = crate::get_work_size();
    println!("Using work size: {}", work_size);

    // Create a command queue with maximum concurrency
    let command_queue = device.new_command_queue_with_max_command_buffer_count(64);

    // Compile the Metal kernel with optimization options
    let metal_src = mk_metal_src(config);
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

    // Calculate grid size based on work size
    let grid_size = MTLSize::new(work_size, 1, 1);

    Ok(PipelineResources {
        device: device.clone(),
        command_queue,
        pipeline_state,
        threadgroup_size,
        grid_size,
    })
}

/// Create all necessary buffers for Metal GPU
fn create_buffers(device: &Device, max_solutions: usize) -> BufferSet {
    // Use private storage for better performance on discrete GPUs
    let resource_options = if device.has_unified_memory() {
        MTLResourceOptions::StorageModeShared
    } else {
        MTLResourceOptions::StorageModePrivate
    };

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

    BufferSet {
        message_buffer,
        nonce_buffer,
        solutions_buffer,
        counter_buffer,
        staging_message_buffer,
        staging_nonce_buffer,
        max_solutions,
    }
}

/// Update buffer contents with new salt and nonce
fn update_buffer_contents(buffers: &BufferSet, salt: &FixedBytes<4>, base_nonce: u32) {
    // Update message buffer contents with the unique salt
    if let Some(staging) = &buffers.staging_message_buffer {
        let message_ptr = staging.contents() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(salt.as_ptr(), message_ptr, 4);
        }
    } else {
        let message_ptr = buffers.message_buffer.contents() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(salt.as_ptr(), message_ptr, 4);
        }
    }

    // Update nonce buffer with the base nonce
    if let Some(staging) = &buffers.staging_nonce_buffer {
        let nonce_ptr = staging.contents() as *mut u32;
        unsafe {
            *nonce_ptr = base_nonce;
        }
    } else {
        let nonce_ptr = buffers.nonce_buffer.contents() as *mut u32;
        unsafe {
            *nonce_ptr = base_nonce;
        }
    }

    // Reset solutions counter
    let counter_ptr = buffers.counter_buffer.contents() as *mut u32;
    unsafe {
        *counter_ptr = 0;
    }
}

/// Create and configure a command buffer for computation
fn create_command_buffer(
    command_queue: &CommandQueue,
    pipeline_state: &ComputePipelineState,
    buffers: &BufferSet,
    grid_size: MTLSize,
    threadgroup_size: MTLSize,
) -> CommandBuffer {
    // Create command buffer
    let command_buffer = command_queue.new_command_buffer().to_owned();

    // Create compute encoder
    let compute_encoder = command_buffer.new_compute_command_encoder();

    // Set compute pipeline
    compute_encoder.set_compute_pipeline_state(pipeline_state);

    // Copy from staging buffers if needed
    if let Some(staging) = &buffers.staging_message_buffer {
        let blit_encoder = command_buffer.new_blit_command_encoder();
        blit_encoder.copy_from_buffer(staging, 0, &buffers.message_buffer, 0, 4);
        blit_encoder.end_encoding();
    }

    if let Some(staging) = &buffers.staging_nonce_buffer {
        let blit_encoder = command_buffer.new_blit_command_encoder();
        blit_encoder.copy_from_buffer(staging, 0, &buffers.nonce_buffer, 0, 4);
        blit_encoder.end_encoding();
    }

    // Set buffers
    compute_encoder.set_buffer(0, Some(&buffers.message_buffer), 0);
    compute_encoder.set_buffer(1, Some(&buffers.nonce_buffer), 0);
    compute_encoder.set_buffer(2, Some(&buffers.solutions_buffer), 0);
    compute_encoder.set_buffer(3, Some(&buffers.counter_buffer), 0);

    // Dispatch threads
    compute_encoder.dispatch_threads(grid_size, threadgroup_size);

    // End encoding
    compute_encoder.end_encoding();

    command_buffer
}

/// Process solutions found by the GPU
fn process_solutions(
    buffers: &BufferSet,
    salt: &FixedBytes<4>,
    config: &Config,
    rewards: &Arc<LeadingNibbleReward>,
    state: &MiningState,
) {
    // Check for solutions
    let counter_ptr = buffers.counter_buffer.contents() as *const u32;
    let num_solutions = unsafe { *counter_ptr };

    if num_solutions > 0 {
        // Process all solutions
        let solutions_ptr = buffers.solutions_buffer.contents() as *const u64;

        // Track salts we've already processed in this batch
        let mut processed_salts = HashSet::new();
        let salt_hex = hex::encode(&salt[..]);

        for i in 0..std::cmp::min(num_solutions as usize, buffers.max_solutions) {
            let solution = unsafe { *solutions_ptr.add(i) };
            if solution == 0 {
                continue;
            }

            // Convert the 64-bit solution to bytes (8 bytes total)
            let solution_bytes = solution.to_le_bytes();

            // Create a unique identifier for this solution
            let solution_id = format!("{}{}", hex::encode(&salt[..]), hex::encode(solution_bytes));

            // Check if we've already processed this solution
            let mut processed_guard = state.processed_solutions.lock().unwrap();
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
            let (leading_bytes, leading_nibbles, total_zeroes) = calculate_address_metrics(address);

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
                    let mut found_list_guard = state.found_list.lock().unwrap();
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
                let mut found_guard = state.found.lock().unwrap();
                *found_guard += 1;
            }
        }
    }
}

/// Spawn UI thread for displaying mining statistics
fn spawn_ui_thread(
    config: &Config,
    state: &MiningState,
    term: Arc<Term>,
    work_size: u64,
    threadgroup_size: u64,
) -> thread::JoinHandle<()> {
    // Clone necessary state for the UI thread
    let term_clone = term.clone();
    let found_clone = state.found.clone();
    let found_list_clone = state.found_list.clone();
    let rate_clone = state.rate.clone();
    let cumulative_nonce_clone = state.cumulative_nonce.clone();
    let previous_time_clone = state.previous_time.clone();
    let start_time = state.start_time;

    // Copy config values needed for display
    let leading_zeroes_threshold = config.leading_zeroes_threshold;
    let total_zeroes_threshold = config.total_zeroes_threshold;
    let minimum_score = config.minimum_score;

    thread::spawn(move || {
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
                "[Metal2] search strategy: 4-byte salt (random) | 4-byte sequential nonce\t\t\
                 threadgroup size: {}, threshold: {} leading or {} total zeroes (min score: {})",
                threadgroup_size, leading_zeroes_threshold, total_zeroes_threshold, minimum_score
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
    })
}

/// Generate a unique salt
fn generate_salt(rng: &mut impl RngCore) -> FixedBytes<4> {
    let mut salt_bytes = [0u8; 4];
    rng.fill_bytes(&mut salt_bytes);
    FixedBytes::<4>::from_slice(&salt_bytes)
}

/// Generate a unique base nonce for a command buffer
fn generate_base_nonce(
    global_nonce_counter: &Arc<Mutex<u32>>,
    rng: &mut impl RngCore,
    threadgroup_size: u64,
) -> u32 {
    let mut counter = global_nonce_counter.lock().unwrap();

    // Increment the counter to ensure uniqueness
    *counter = counter.wrapping_add(1);

    // Simply return the counter as the base nonce
    *counter
}

/// Update mining statistics after a command buffer completes
fn update_mining_stats(state: &MiningState) {
    // Increment the cumulative nonce
    {
        let mut cumulative = state.cumulative_nonce.lock().unwrap();
        *cumulative += 1;
    }

    // Update rate
    {
        let mut rate_val = state.rate.lock().unwrap();
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let current_time = now.as_secs_f64();
        if current_time - state.start_time > 0.0 {
            *rate_val = 1.0 / (current_time - state.start_time);
        }
    }
}

/// Metal GPU implementation for finding efficient Ethereum addresses
pub fn metal_gpu(config: Config) -> Result<(), Box<dyn Error>> {
    // (create if necessary) and open a file where found salts will be written
    Arc::new(crate::output_file());

    println!("Using Metal2 implementation...");

    // Get the work size from .env if available
    let work_size = crate::get_work_size();

    // Setup Metal device
    let device = setup_metal_device(&config)?;

    // Create compute pipeline
    let pipeline_resources = create_compute_pipeline(&device, &config)?;

    // Create buffers
    let max_solutions: usize = 16; // Allow up to 16 solutions per batch
    let buffers = create_buffers(&device, max_solutions);

    // Initialize mining state
    let state = MiningState::new();

    // Create object for computing rewards (relative rarity) for a given address
    let rewards = Arc::new(LeadingNibbleReward::default());

    // Set up a controller for terminal output
    let term = Arc::new(Term::stdout());

    // Spawn UI thread
    let _ui_thread = spawn_ui_thread(
        &config,
        &state,
        term,
        work_size,
        pipeline_resources.threadgroup_size.width as u64,
    );

    // Create a random number generator
    let mut rng = thread_rng();

    // Begin searching for addresses
    loop {
        // Generate a completely unique random salt
        let salt = generate_salt(&mut rng);

        // Generate a unique base nonce
        let base_nonce = generate_base_nonce(
            &state.global_nonce_counter,
            &mut rng,
            pipeline_resources.threadgroup_size.width as u64,
        );

        // Update buffer contents
        update_buffer_contents(&buffers, &salt, base_nonce);

        // Create and configure command buffer
        let command_buffer = create_command_buffer(
            &pipeline_resources.command_queue,
            &pipeline_resources.pipeline_state,
            &buffers,
            pipeline_resources.grid_size,
            pipeline_resources.threadgroup_size,
        );

        // Commit command buffer
        command_buffer.commit();

        // Wait for command buffer to complete
        command_buffer.wait_until_completed();

        // Update mining statistics
        update_mining_stats(&state);

        // Process solutions
        process_solutions(&buffers, &salt, &config, &rewards, &state);
    }
}
