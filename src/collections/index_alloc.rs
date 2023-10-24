use std::sync::{
    atomic::{AtomicU32, Ordering},
    Mutex,
};

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
        let mut free_indices = self.freed_indices.lock().unwrap();
        if let Some(free_index) = free_indices.pop() {
            free_index
        } else {
            self.index_counter.fetch_add(1, Ordering::SeqCst)
        }
    }

    #[track_caller]
    pub fn free(&self, index: u32) {
        let mut free_indices = self.freed_indices.lock().unwrap();
        debug_assert!(!free_indices.contains(&index)); // freeing freed index
        free_indices.push(index);
    }
}
