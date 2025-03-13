use crate::{Config, Reward, CONTROL_CHARACTER};
use alloy_primitives::{hex, Address};
use std::collections::HashSet;
use std::fmt::Write as _;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::thread;
use tiny_keccak::{Hasher, Keccak};

pub(super) struct SolutionProcessor {
    config: Config,
    rewards: Arc<Reward>,
    found: Arc<Mutex<u64>>,
    found_list: Arc<Mutex<Vec<String>>>,
    processed_solutions: Arc<Mutex<HashSet<String>>>,
}

impl SolutionProcessor {
    pub(super) fn new(
        config: Config,
        rewards: Arc<Reward>,
        found: Arc<Mutex<u64>>,
        found_list: Arc<Mutex<Vec<String>>>,
    ) -> Self {
        SolutionProcessor {
            config,
            rewards,
            found,
            found_list,
            processed_solutions: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    pub(super) fn get_processed_solutions(&self) -> Arc<Mutex<HashSet<String>>> {
        self.processed_solutions.clone()
    }

    pub(super) fn process_solution(&self, salt: &[u8], solution: u64) -> bool {
        // Convert the 64-bit solution to bytes (8 bytes total)
        let solution_bytes = solution.to_le_bytes();

        // Create a unique identifier for this solution
        let solution_id = format!("{}{}", hex::encode(salt), hex::encode(solution_bytes));

        // Check if we've already processed this solution
        let mut processed_guard = self.processed_solutions.lock().unwrap();
        if processed_guard.contains(&solution_id) {
            return false;
        }

        // Add to processed solutions
        processed_guard.insert(solution_id);
        drop(processed_guard); // Release the lock early

        let mut solution_message = [0; 85];
        solution_message[0] = CONTROL_CHARACTER;
        solution_message[1..21].copy_from_slice(&self.config.factory_address);
        solution_message[21..41].copy_from_slice(&self.config.calling_address);
        solution_message[41..45].copy_from_slice(salt);
        solution_message[45..53].copy_from_slice(&solution_bytes);
        solution_message[53..].copy_from_slice(&self.config.init_code_hash);

        // create new hash object
        let mut hash = Keccak::v256();

        // update with header
        hash.update(&solution_message);

        // hash the payload and get the result
        let mut res: [u8; 32] = [0; 32];
        hash.finalize(&mut res);

        // get the address that results from the hash
        let address = <&Address>::try_from(&res[12..]).unwrap();

        // Count zero bytes in the address (20 bytes)
        let mut total_zeroes = 0;
        let mut leading_zeroes = 0;
        let mut still_leading = true;

        for &byte in address.iter() {
            if byte == 0 {
                total_zeroes += 1;
                if still_leading {
                    leading_zeroes += 1;
                }
            } else {
                still_leading = false;
            }
        }

        // Verify this is actually a solution according to our criteria
        let meets_leading_criteria =
            leading_zeroes >= self.config.leading_zeroes_threshold as usize;
        let meets_total_criteria = total_zeroes >= self.config.total_zeroes_threshold as usize;

        // Only process if it meets either criteria
        if meets_leading_criteria || meets_total_criteria {
            // Use the correct leading count for key and display
            let key = leading_zeroes * 20 + total_zeroes;
            let reward = self.rewards.get(&key).unwrap_or("0");

            // Extract buffer index from the first byte of the salt
            let buffer_idx = salt[0];

            let output = format!(
                "0x{}{}{} => {} => {} (buffer {})",
                hex::encode(self.config.calling_address),
                hex::encode(salt),
                hex::encode(solution_bytes),
                address,
                reward,
                buffer_idx,
            );

            let show = format!("{output} ({leading_zeroes} / {total_zeroes})");

            // Update found list
            {
                let mut found_list_guard = self.found_list.lock().unwrap();
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
            let mut found_guard = self.found.lock().unwrap();
            *found_guard += 1;

            return true;
        }

        false
    }
}
