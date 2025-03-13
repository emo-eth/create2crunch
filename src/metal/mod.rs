// Create a new module structure
pub mod kernel;
mod mining;
mod safety_wrappers;
mod setup;
mod solution;
mod ui;

// Re-export the main function
pub use mining::metal_gpu;
