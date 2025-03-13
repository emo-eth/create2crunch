use crate::Config;
use console::Term;
use separator::Separatable;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use terminal_size::{terminal_size, Height};

pub(super) struct UiManager {
    term: Arc<Term>,
    found: Arc<Mutex<u64>>,
    found_list: Arc<Mutex<Vec<String>>>,
    rate: Arc<Mutex<f64>>,
    cumulative_nonce: Arc<Mutex<u64>>,
    previous_time: Arc<Mutex<f64>>,
    start_time: f64,
    config: Config,
    work_size: u64,
}

impl UiManager {
    pub(super) fn new(
        term: Arc<Term>,
        found: Arc<Mutex<u64>>,
        found_list: Arc<Mutex<Vec<String>>>,
        rate: Arc<Mutex<f64>>,
        cumulative_nonce: Arc<Mutex<u64>>,
        config: Config,
        work_size: u64,
    ) -> Self {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        UiManager {
            term,
            found,
            found_list,
            rate,
            cumulative_nonce,
            previous_time: Arc::new(Mutex::new(0.0f64)),
            start_time,
            config,
            work_size,
        }
    }

    pub(super) fn get_start_time(&self) -> f64 {
        self.start_time
    }

    pub(super) fn get_rate(&self) -> Arc<Mutex<f64>> {
        self.rate.clone()
    }

    pub(super) fn get_cumulative_nonce(&self) -> Arc<Mutex<u64>> {
        self.cumulative_nonce.clone()
    }

    pub(super) fn get_previous_time(&self) -> Arc<Mutex<f64>> {
        self.previous_time.clone()
    }

    pub(super) fn start_ui_thread(&self) -> thread::JoinHandle<()> {
        let term_clone = self.term.clone();
        let found_clone = self.found.clone();
        let found_list_clone = self.found_list.clone();
        let rate_clone = self.rate.clone();
        let cumulative_nonce_clone = self.cumulative_nonce.clone();
        let previous_time_clone = self.previous_time.clone();
        let start_time = self.start_time;
        let work_size = self.work_size;
        let leading_zeroes_threshold = self.config.leading_zeroes_threshold;
        let total_zeroes_threshold = self.config.total_zeroes_threshold;

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
                    "search strategy: 4-byte salt (buffer_id + random) | 8-byte nonce (hierarchical_thread_id + counter_nonce)\t\t\
                     threshold: {} leading or {} total zeroes",
                    leading_zeroes_threshold,
                    total_zeroes_threshold
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
}
