# Implementation Plan for Metal Compute Backend

## Approach

We'll implement a Metal compute backend as an alternative to the existing OpenCL implementation, focusing specifically on the compute operations needed for Keccak-256 hashing and applying targeted optimizations based on our workload characteristics.

## Implementation Steps

### 1. Project Structure Changes

1. **Add Metal Dependency**:

    ```toml
    [dependencies]
    ocl = { version = "0.19", optional = true }
    metal = { version = "0.31", optional = true }

    [features]
    default = ["opencl"]
    opencl = ["dep:ocl"]
    metal = ["dep:metal"]
    ```

2. **Create Metal Backend Module**:

    - Create `src/metal_backend.rs` with a `metal_gpu` function that parallels the existing `gpu` function
    - Implement Metal-specific error handling

3. **Update Config Structure**:

    ```rust
    pub enum GpuBackend {
        OpenCL,
        Metal,
    }

    pub struct Config {
        // existing fields
        pub backend: GpuBackend,
    }
    ```

### 2. Metal Compute Kernel Implementation

1. **Create Metal Compute Kernel**:

    - Create `src/kernels/keccak256.metal`
    - Translate the Keccak-256 implementation from OpenCL C to Metal language
    - Focus on maintaining the same algorithm while adapting to Metal's syntax

2. **Key Translation Points**:

    - Replace OpenCL kernel attributes with Metal attributes (`__kernel` → `kernel`)
    - Update buffer access qualifiers (`__constant` → `constant`)
    - Use appropriate address space qualifiers (`thread` for thread-local data)
    - Update thread identification (`get_global_id(0)` → `thread_position_in_grid`)

3. **Metal-Specific Optimizations**:
    - Use `thread` qualifier for sponge buffer to keep it in registers
    - Minimize divergent execution paths in the kernel
    - Consider using Metal's SIMD types for bit operations

### 3. Optimized Metal Compute Pipeline Implementation

1. **Device Setup**:

    ```rust
    let devices = Device::all();
    let device = &devices[config.gpu_device as usize];
    let command_queue = device.new_command_queue();
    ```

2. **Compute Kernel Compilation**:

    ```rust
    let metal_src = mk_metal_src(&config);
    let library = device.new_library_with_source(&metal_src, &CompileOptions::new())?;
    let kernel_function = library.get_function("hashMessage", None)?;

    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel_function));
    let pipeline_state = device.new_compute_pipeline_state(&pipeline_state_descriptor)?;
    ```

3. **Optimized Buffer Management**:

    ```rust
    // Create buffers once outside the main loop
    let message_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let nonce_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let solutions_buffer = device.new_buffer(8, MTLResourceOptions::StorageModeShared);

    // In the loop, update buffer contents instead of creating new buffers
    let salt = FixedBytes::<4>::random();
    let message_ptr = message_buffer.contents() as *mut u8;
    unsafe {
        std::ptr::copy_nonoverlapping(salt.as_ptr(), message_ptr, 4);
    }
    ```

4. **Optimized Thread Dispatching**:

    ```rust
    // Query the device's recommended work group size
    let threadgroup_size = metal::MTLSize::new(
        min(pipeline_state.max_total_threads_per_threadgroup(), 256),
        1,
        1
    );

    // Calculate grid size based on work size and threadgroup size
    let grid_size = metal::MTLSize::new(
        WORK_SIZE as u64,
        1,
        1
    );

    compute_encoder.dispatch_threads(grid_size, threadgroup_size);
    ```

5. **Efficient Command Execution**:

    ```rust
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

    // Commit command buffer and wait for completion (synchronous execution is appropriate for our workload)
    command_buffer.commit();
    command_buffer.wait_until_completed();
    ```

6. **Direct Memory Access for Results**:

    ```rust
    // Read results directly from shared memory
    let solution_ptr = solutions_buffer.contents() as *const u32;
    let solution = unsafe { std::slice::from_raw_parts(solution_ptr, 2) };

    // Process solutions as in the OpenCL implementation
    ```

### 4. Integration with Main Code

1. **Backend Selection Logic**:

    ```rust
    fn main() -> Result<(), Box<dyn Error>> {
        let config = Config::new(std::env::args())?;

        match config.backend {
            GpuBackend::OpenCL => {
                #[cfg(feature = "opencl")]
                return crate::gpu(config);

                #[cfg(not(feature = "opencl"))]
                return Err("OpenCL support not enabled".into());
            }
            GpuBackend::Metal => {
                #[cfg(feature = "metal")]
                return crate::metal_backend::metal_gpu(config);

                #[cfg(not(feature = "metal"))]
                return Err("Metal support not enabled".into());
            }
        }
    }
    ```

2. **Auto-detection for macOS**:

    ```rust
    #[cfg(target_os = "macos")]
    fn detect_preferred_backend() -> GpuBackend {
        #[cfg(feature = "metal")]
        return GpuBackend::Metal;

        #[cfg(not(feature = "metal"))]
        return GpuBackend::OpenCL;
    }

    #[cfg(not(target_os = "macos"))]
    fn detect_preferred_backend() -> GpuBackend {
        GpuBackend::OpenCL
    }
    ```

### 5. Testing and Validation

1. **Functionality Testing**:

    - Verify that the Metal implementation produces the same results as OpenCL
    - Test with different input parameters and thresholds

2. **Performance Testing**:

    - Compare performance between Metal and OpenCL on the same Apple hardware
    - Measure hash rate, power consumption, and solution discovery rate

3. **Error Handling**:
    - Test error cases such as invalid device selection
    - Verify graceful fallback to OpenCL when Metal is not available

## Implementation Priorities

Based on our performance evaluation, we should prioritize these optimizations:

1. **Buffer Reuse**: Create buffers once and update contents rather than creating new buffers in each iteration
2. **Appropriate Storage Modes**: Use `StorageModeShared` for our small, frequently accessed buffers
3. **Optimized Thread Dispatching**: Adjust work group sizes based on the specific GPU
4. **Kernel Optimization**: Focus on optimizing the Keccak-256 implementation for Metal's architecture

We should not spend time implementing:

1. **Asynchronous Execution**: Not beneficial for our sequential checking pattern
2. **Multiple Command Buffers/Queues**: Unnecessary for our single-operation workload
