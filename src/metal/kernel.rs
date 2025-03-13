use crate::Config;
use alloy_primitives::hex;
use std::fmt::Write as _;

static KERNEL_SRC: &str = include_str!("../kernels/keccak256.metal");

/// Function to generate Metal kernel source with the appropriate constants
pub(crate) fn mk_metal_src(config: &Config) -> String {
    let mut src = String::with_capacity(2048 + KERNEL_SRC.len());

    // Print debug information about the configuration
    println!("Generating Metal kernel with the following configuration:");
    println!(
        "  Leading zeroes threshold: {}",
        config.leading_zeroes_threshold
    );
    println!(
        "  Total zeroes threshold: {}",
        config.total_zeroes_threshold
    );
    println!(
        "  Factory address: 0x{}",
        hex::encode(&config.factory_address)
    );
    println!(
        "  Calling address: 0x{}",
        hex::encode(&config.calling_address)
    );
    println!(
        "  Init code hash: 0x{}",
        hex::encode(&config.init_code_hash)
    );

    let factory = config.factory_address.iter();
    let caller = config.calling_address.iter();
    let hash = config.init_code_hash.iter();
    let hash = hash.enumerate().map(|(i, x)| (i + 52, x));
    for (i, x) in factory.chain(caller).enumerate().chain(hash) {
        writeln!(src, "#define S_{} {}u", i + 1, x).unwrap();
    }

    // Set the thresholds for the kernel
    writeln!(
        src,
        "#define LEADING_ZEROES {}",
        config.leading_zeroes_threshold
    )
    .unwrap();
    writeln!(
        src,
        "#define TOTAL_ZEROES {}",
        config.total_zeroes_threshold
    )
    .unwrap();

    // Add a debug flag to enable more verbose kernel output
    writeln!(src, "#define DEBUG 1").unwrap();

    // Add a comment to explain how the kernel checks for solutions
    writeln!(
        src,
        "// The kernel checks for solutions by counting zero bytes in the address"
    )
    .unwrap();
    writeln!(
        src,
        "// An address has 20 bytes, and we check if it has enough leading or total zero bytes"
    )
    .unwrap();
    writeln!(
        src,
        "// Leading zeroes: consecutive zero bytes at the start of the address"
    )
    .unwrap();
    writeln!(
        src,
        "// Total zeroes: total number of zero bytes in the address"
    )
    .unwrap();

    src.push_str(KERNEL_SRC);

    src
}
