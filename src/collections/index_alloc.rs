use std::sync::{atomic::{AtomicU32, Ordering}, Mutex};

pub struct IndexAllocator {
    index_counter: AtomicU32,
    freed_indices: Mutex<Vec<u32>>,
}

impl IndexAllocator {
    pub fn new(start: u32) -> Self {
        Self {
            index_counter: AtomicU32::new(start),
            freed_indices: Mutex::new(Vec::new()),
        }
    }

    pub fn alloc(&self) -> u32 {
        if let Some(free_index) = self.freed_indices.lock().unwrap().pop() {
            free_index
        } else {
            self.index_counter.fetch_add(1, Ordering::Relaxed)
        }
    }

    pub fn free(&self, index: u32) {
        self.freed_indices.lock().unwrap().push(index);
    }
}