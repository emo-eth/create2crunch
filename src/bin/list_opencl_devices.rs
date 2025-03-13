use ocl::{Device, Platform};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Get all platforms
    let platforms = Platform::list();

    if platforms.is_empty() {
        println!("No OpenCL platforms found!");
        return Ok(());
    }

    println!("Found {} OpenCL platform(s):", platforms.len());

    for (platform_idx, platform) in platforms.iter().enumerate() {
        println!("Platform {}: {}", platform_idx, platform.name()?);
        println!("  Vendor: {}", platform.vendor()?);
        println!("  Version: {}", platform.version()?);

        // Get devices for this platform
        match Device::list_all(*platform) {
            Ok(devices) => {
                if devices.is_empty() {
                    println!("  No devices found for this platform");
                } else {
                    println!("  Found {} device(s):", devices.len());

                    for (device_idx, device) in devices.iter().enumerate() {
                        println!("    Device {}: {}", device_idx, device.name()?);
                        println!("      Vendor: {}", device.vendor()?);
                        println!("      Version: {}", device.version()?);
                        println!("      Available: {}", device.is_available()?);
                    }
                }
            }
            Err(e) => println!("  Error listing devices: {}", e),
        }
        println!();
    }

    Ok(())
}
