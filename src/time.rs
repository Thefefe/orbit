use std::time::{Duration, Instant};

pub struct Time {
    current_frame: Option<Instant>,
    delta_duration: Option<Duration>,
    elapsed_time: Duration,
}

impl Default for Time {
    fn default() -> Self {
        Self {
            current_frame: None,
            delta_duration: None,
            elapsed_time: Duration::ZERO,
        }
    }
}

impl Time {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn delta(&self) -> Duration {
        self.delta_duration.unwrap_or(Duration::ZERO)
    }

    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.elapsed_time
    }

    pub fn update_now(&mut self) {
        self.delta_duration = self.current_frame.map(|frame| frame.elapsed());
        self.current_frame = Some(Instant::now());
        self.elapsed_time += self.delta();
    }
}