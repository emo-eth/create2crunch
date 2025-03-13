use super::kernel::mk_metal_src;
use crate::Config;
use metal::*;
use std::error::Error;

pub struct MetalContext {
    pub device: Device,
    pub command_queue: CommandQueue,
    pub pipeline_state: ComputePipelineState,
    pub threadgroup_size: MTLSize,
    pub grid_size: MTLSize,
    pub message_buffer: Buffer,
    pub nonce_buffer: Buffer,
    pub solutions_buffer: Buffer,
    pub counter_buffer: Buffer,
    pub staging_message_buffer: Option<Buffer>,
    pub staging_nonce_buffer: Option<Buffer>,
    pub resource_options: MTLResourceOptions,
    pub max_solutions: usize,
}

pub fn initialize_metal(config: &Config, work_size: u64) -> Result<MetalContext, Box<dyn Error>> {
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

    let device = devices[device_index].clone();
    println!("Using Metal device: {}", device.name());

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
    let thread_execution_width = pipeline_state.thread_execution_width();
    let threadgroup_size = MTLSize::new(std::cmp::min(max_threads, 256), 1, 1);

    println!("Using threadgroup size: {}", threadgroup_size.width);

    // Calculate grid size based on work size, ensuring it's a multiple of threadgroup size
    let grid_size = MTLSize::new(work_size, 1, 1);

    // Create buffers with optimized storage modes
    let resource_options = if device.has_unified_memory() {
        MTLResourceOptions::StorageModeShared
    } else {
        MTLResourceOptions::StorageModePrivate
    };

    // Maximum solutions per batch
    let max_solutions: usize = 16;

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

    Ok(MetalContext {
        device,
        command_queue,
        pipeline_state,
        threadgroup_size,
        grid_size,
        message_buffer,
        nonce_buffer,
        solutions_buffer,
        counter_buffer,
        staging_message_buffer,
        staging_nonce_buffer,
        resource_options,
        max_solutions,
    })
}
