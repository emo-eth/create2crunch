use alloy_primitives::Address;

pub trait RewardTrait {
    fn get(&self, addr: &Address) -> u64;
    fn get_from_counts(&self, leading_bytes: u64, leading_nibbles: u64, total_zeroes: u64) -> u64;
}

pub struct LeadingNibbleReward {}

impl RewardTrait for LeadingNibbleReward {
    fn get(&self, addr: &Address) -> u64 {
        let mut leading_bytes: u64 = 0;
        let mut still_leading = true;
        let mut leading_nibbles: u64 = 0;
        let mut total_zeroes = 0;

        for &byte in addr.iter() {
            if byte == 0 {
                total_zeroes += 1;
            }
            if byte < 16 {
                leading_nibbles += 1;
            }
            if still_leading {
                leading_bytes += 1;
                leading_nibbles += 1;
                if byte != 0 {
                    still_leading = false;
                }
            }
        }
        return leading_bytes.pow(3) + leading_nibbles.pow(2) + total_zeroes as u64;
    }

    fn get_from_counts(&self, leading_bytes: u64, leading_nibbles: u64, total_zeroes: u64) -> u64 {
        return leading_bytes.pow(3) + leading_nibbles.pow(2) + total_zeroes;
    }
}

impl Default for LeadingNibbleReward {
    fn default() -> Self {
        Self {}
    }
}
