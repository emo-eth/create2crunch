use alloy_primitives::FixedBytes;
use byteorder::{ByteOrder, LittleEndian};
use rand::{thread_rng, Rng};
use std::error::Error;

#[cfg(not(feature = "opencl"))]
fn is_opencl_available() -> bool {
    false
}

// Helper function to check if Metal is available
#[cfg(feature = "metal")]
fn is_metal_available() -> bool {
    use metal::Device;

    Device::system_default().is_some()
}

#[cfg(not(feature = "metal"))]
fn is_metal_available() -> bool {
    false
}

// Run a single hash computation using OpenCL
#[cfg(feature = "opencl")]
fn run_single_hash_opencl(
    config: &crate::Config,
    message: &[u8],
    nonce_high: u32,
) -> Result<[u8; 32], Box<dyn Error>> {
    use ocl::{Buffer, Context, Device, MemFlags, Platform, Program, Queue};

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
    let kernel_src = crate::mk_kernel_src(config);

    // Build program
    let program = Program::builder()
        .devices(device)
        .src(kernel_src)
        .build(&context)?;

    // Create a fixed-size message buffer (4 bytes) for the kernel
    // If message is shorter, pad with zeros; if longer, truncate
    let mut fixed_message = [0u8; 4];
    let copy_len = std::cmp::min(message.len(), 4);
    fixed_message[..copy_len].copy_from_slice(&message[..copy_len]);

    // Create buffers
    let message_buffer = Buffer::<u8>::builder()
        .queue(queue.clone())
        .flags(MemFlags::READ_ONLY)
        .len(4)
        .copy_host_slice(&fixed_message)
        .build()?;

    // Create nonce buffer with the u32 nonce
    let nonce_buffer = Buffer::<u32>::builder()
        .queue(queue.clone())
        .flags(MemFlags::READ_ONLY)
        .len(1)
        .copy_host_slice(&[nonce_high])
        .build()?;

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

    // Run with a single work item for testing
    unsafe {
        // Use explicit queue and global work size
        kernel
            .cmd()
            .queue(&queue)
            .global_work_size(1) // Use a single work item for testing
            .enq()?;
    }

    // Read result
    let mut solution = vec![0u64; 1];
    solutions_buffer.read(&mut solution).enq()?;

    // For the solution, we need to combine the thread ID (0 in our case) with the nonce
    // to form a u64 value similar to how it's done in the main implementation
    let full_nonce = solution[0];

    // Compute the full hash using the solution
    let hash = compute_full_hash(config, message, full_nonce)?;

    Ok(hash)
}

#[cfg(feature = "metal")]
use metal::*;

#[cfg(feature = "metal")]
// Safety wrapper for ComputeCommandEncoder to prevent context leaks
struct SafeEncoder<'a> {
    encoder: &'a ComputeCommandEncoderRef,
    ended: bool,
}

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
impl<'a> Drop for SafeEncoder<'a> {
    fn drop(&mut self) {
        self.end_encoding();
    }
}

// Run a single hash computation using Metal
#[cfg(feature = "metal")]
fn run_single_hash_metal(
    config: &crate::Config,
    message: &[u8],
    nonce_high: u32,
) -> Result<[u8; 32], Box<dyn Error>> {
    // Get the default device
    let device = Device::system_default().unwrap();

    // Create a command queue
    let command_queue = device.new_command_queue();

    // Prepare kernel source
    let kernel_src = crate::metal_backend::mk_metal_src(config);

    // Create Metal library and function
    let options = CompileOptions::new();
    let library = device.new_library_with_source(&kernel_src, &options)?;
    let function = library.get_function("hashMessage", None)?;

    // Create pipeline state
    let pipeline_state = device.new_compute_pipeline_state_with_function(&function)?;

    // Create a fixed-size message buffer (4 bytes) for the kernel
    // If message is shorter, pad with zeros; if longer, truncate
    let mut fixed_message = [0u8; 4];
    let copy_len = std::cmp::min(message.len(), 4);
    fixed_message[..copy_len].copy_from_slice(&message[..copy_len]);

    // Create buffers
    let message_buffer = device.new_buffer_with_data(
        fixed_message.as_ptr() as *const _,
        fixed_message.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create nonce buffer with the u32 nonce
    let nonce_buffer = device.new_buffer_with_data(
        &nonce_high as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let solutions_buffer = device.new_buffer(
        std::mem::size_of::<u64>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create a counter buffer for solution count
    let counter_buffer = device.new_buffer_with_data(
        [0u32; 1].as_ptr() as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Clear solutions buffer
    let zeros = [0u64; 1];
    let solutions_ptr = solutions_buffer.contents() as *mut u64;
    unsafe {
        std::ptr::copy_nonoverlapping(zeros.as_ptr(), solutions_ptr, 1);
    }

    // Create command buffer and encoder
    let command_buffer = command_queue.new_command_buffer();
    let compute_encoder = command_buffer.new_compute_command_encoder();
    let mut safe_encoder = SafeEncoder::new(&compute_encoder);

    // Set pipeline state and buffers
    safe_encoder
        .encoder
        .set_compute_pipeline_state(&pipeline_state);
    safe_encoder.encoder.set_buffer(0, Some(&message_buffer), 0);
    safe_encoder.encoder.set_buffer(1, Some(&nonce_buffer), 0);
    safe_encoder
        .encoder
        .set_buffer(2, Some(&solutions_buffer), 0);
    safe_encoder.encoder.set_buffer(3, Some(&counter_buffer), 0);

    // Dispatch a single thread for testing
    let thread_group_size = MTLSize::new(1, 1, 1);
    let thread_groups_per_grid = MTLSize::new(1, 1, 1);

    safe_encoder
        .encoder
        .dispatch_thread_groups(thread_groups_per_grid, thread_group_size);
    safe_encoder.end_encoding();

    // Commit and wait
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read result
    let solution_ptr = solutions_buffer.contents() as *const u64;
    let solution = unsafe { *solution_ptr };

    // Compute the full hash using the solution
    let hash = compute_full_hash(config, message, solution)?;

    Ok(hash)
}

// Compute the full Keccak-256 hash using the CPU implementation
fn compute_full_hash(
    config: &crate::Config,
    message: &[u8],
    nonce: u64,
) -> Result<[u8; 32], Box<dyn Error>> {
    use tiny_keccak::{Hasher, Keccak};

    // Prepare the message with the same format as in the kernels
    let mut buffer = [0u8; 200]; // Keccak sponge size

    // Control character
    buffer[0] = 0xff;

    // Factory and caller addresses
    for (i, &byte) in config.factory_address.iter().enumerate() {
        buffer[i + 1] = byte;
    }

    for (i, &byte) in config.calling_address.iter().enumerate() {
        buffer[i + 21] = byte;
    }

    // Message - handle variable length
    let msg_len = message.len();
    if msg_len > 0 {
        let copy_len = std::cmp::min(msg_len, 4); // Limit to 4 bytes for now to match kernel
        for i in 0..copy_len {
            buffer[41 + i] = message[i];
        }
    }

    // Nonce
    let mut nonce_bytes = [0u8; 8];
    LittleEndian::write_u64(&mut nonce_bytes, nonce);
    for i in 0..8 {
        buffer[45 + i] = nonce_bytes[i];
    }

    // Init code hash
    for (i, &byte) in config.init_code_hash.iter().enumerate() {
        buffer[53 + i] = byte;
    }

    // Padding (as in the kernels)
    buffer[85] = 0x01;
    buffer[135] = 0x80;

    // Compute hash
    let mut keccak = Keccak::v256();
    let mut hash = [0u8; 32];
    keccak.update(&buffer[..136]); // Only include up to the padding
    keccak.finalize(&mut hash);

    Ok(hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, GpuBackend};

    // Helper function to create a test config
    fn create_test_config() -> Config {
        Config {
            factory_address: [0x11; 20], // Dummy factory address
            calling_address: [0x22; 20], // Dummy caller address
            init_code_hash: [0x33; 32],  // Dummy init code hash
            gpu_device: 0,               // First GPU device
            leading_zeroes_threshold: 1, // Low threshold for testing
            total_zeroes_threshold: 2,   // Low threshold for testing
            backend: GpuBackend::OpenCL, // Will be overridden in tests
            optimize: false,             // Not running in optimization mode
            two_phase: false,            // Not using two-phase optimization
            benchmark_duration: 1,       // Default benchmark duration
        }
    }

    // Test that OpenCL and Metal implementations produce the same results
    #[test]
    #[cfg(all(feature = "opencl", feature = "metal"))]
    fn test_opencl_vs_metal() {
        // Skip test if either backend is not available
        if !is_opencl_available() || !is_metal_available() {
            println!("Skipping test_opencl_vs_metal: One or both backends not available");
            return;
        }

        let mut config = create_test_config();

        // Number of test iterations
        const TEST_ITERATIONS: usize = 10;

        println!("Running {} test iterations", TEST_ITERATIONS);

        for iteration in 0..TEST_ITERATIONS {
            // Generate random test data
            let mut rng = thread_rng();

            // Generate a random 4-byte salt (similar to lib.rs)
            let salt = FixedBytes::<4>::random();

            // Generate a random 4-byte nonce
            let nonce: u32 = rng.gen();

            println!(
                "Iteration {}/{}: Testing with salt: {:?}, nonce: {}",
                iteration + 1,
                TEST_ITERATIONS,
                salt,
                nonce
            );

            // Run OpenCL implementation
            config.backend = GpuBackend::OpenCL;
            let opencl_result = run_single_hash_opencl(&config, &salt[..], nonce)
                .expect("OpenCL hash computation failed");

            // Run Metal implementation
            config.backend = GpuBackend::Metal;
            let metal_result = run_single_hash_metal(&config, &salt[..], nonce)
                .expect("Metal hash computation failed");

            // Compare results
            assert_eq!(
                opencl_result, metal_result,
                "OpenCL and Metal implementations produced different results!\nOpenCL: {:?}\nMetal: {:?}",
                opencl_result, metal_result
            );

            println!(
                "Iteration {}/{}: OpenCL and Metal implementations match! Result: {:?}",
                iteration + 1,
                TEST_ITERATIONS,
                opencl_result
            );
        }

        println!(
            "All {} test iterations passed successfully!",
            TEST_ITERATIONS
        );
    }

    // Helper function to check if OpenCL is available
    #[cfg(feature = "opencl")]
    fn is_opencl_available() -> bool {
        use ocl::{Device, Platform};

        let platforms = Platform::list();
        if platforms.is_empty() {
            return false;
        }

        for platform in platforms {
            match Device::list_all(platform) {
                Ok(devices) => {
                    if !devices.is_empty() {
                        return true;
                    }
                }
                Err(_) => continue,
            }
        }
        false
    }
}
