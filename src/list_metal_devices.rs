use metal::Device;

fn main() {
    println!("Available Metal devices:");
    println!("------------------------");

    let devices = Device::all();

    for (i, device) in devices.iter().enumerate() {
        println!("Device ID: {}", i);
        println!("  Name: {}", device.name());
        println!("  Registry ID: 0x{:x}", device.registry_id());
        println!("  Headless: {}", device.is_headless());
        println!("  Low Power: {}", device.is_low_power());
        println!("  Removable: {}", device.is_removable());
        println!();
    }

    println!("For create2crunch, use the Device ID as the --gpu-device parameter.");
    println!("Example: --gpu-device 0 for the first device");
}
