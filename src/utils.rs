use std::{cell::Cell, ops::RangeBounds, path::Path, sync::MutexGuard};

pub fn div_ceil(lhs: usize, rhs: usize) -> usize {
    let d = lhs / rhs;
    let r = lhs % rhs;
    if r > 0 && rhs > 0 {
        d + 1
    } else {
        d
    }
}

pub fn aligned_size(size: usize, align: usize) -> usize {
    div_ceil(size, align) * align
}

pub fn load_spv(path: impl AsRef<Path>) -> std::io::Result<Vec<u32>> {
    let mut file = std::fs::File::open(path)?;
    ash::util::read_spv(&mut file)
}

pub struct OptionDefaultIterator<I, T> {
    iter: Option<I>,
    default: T,
}

impl<I> OptionDefaultIterator<I, I::Item>
where
    I: Iterator,
    I::Item: Clone,
{
    pub fn new(iter: Option<I>, default: I::Item) -> Self {
        Self { iter, default }
    }
}

impl<I> Iterator for OptionDefaultIterator<I, I::Item>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(iter) = &mut self.iter {
            iter.next()
        } else {
            Some(self.default.clone())
        }
    }
}

// workaround for unstable `impl !Sync`/`Send`
// should be used with `PhantomData`
pub type Unsync = Cell<()>;
pub type Unsend = MutexGuard<'static, ()>;

pub fn init_logger(log_to_file: bool) {
    let target = if log_to_file {
        let file =
            Box::new(std::fs::OpenOptions::new().create(true).truncate(true).write(true).open("log.txt").unwrap());

        env_logger::Target::Pipe(file)
    } else {
        env_logger::Target::Stdout
    };

    env_logger::builder().parse_filters("panic,orbit,vulkan=debug").target(target).init();
    log_panics::init();
}

pub fn range_bounds_to_base_count(bounds: impl RangeBounds<u32>, min_bound: u32, max_bound: u32) -> (u32, u32) {
    let base = match bounds.start_bound() {
        std::ops::Bound::Included(bound) => *bound,
        std::ops::Bound::Excluded(bound) => *bound + 1,
        std::ops::Bound::Unbounded => 0,
    }
    .clamp(min_bound, max_bound);

    // exclusive
    let end_bound = match bounds.end_bound() {
        std::ops::Bound::Included(bound) => *bound - 1,
        std::ops::Bound::Excluded(bound) => *bound,
        std::ops::Bound::Unbounded => max_bound,
    }
    .clamp(min_bound, max_bound);

    (base, end_bound - base)
}


#[derive(Debug, Clone, Copy)]
pub struct StructuredDataBuilder<const SIZE: usize> {
    pub constants: [u8; SIZE],
    pub byte_cursor: usize,
}

impl<const SIZE: usize> StructuredDataBuilder<SIZE> {
    pub fn new() -> Self {
        Self {
            constants: [0; SIZE],
            byte_cursor: 0,
        }
    }

    #[inline(always)]
    pub fn reamaining_byte(&self) -> usize {
        128 - self.byte_cursor
    }

    #[track_caller]
    #[inline(always)]
    pub fn push_bytes_with_align(&mut self, bytes: &[u8], align: usize) {
        let padding = self.byte_cursor % align;
        debug_assert!(padding + bytes.len() < self.reamaining_byte());

        let offset = self.byte_cursor + padding;
        self.constants[offset..offset + bytes.len()].copy_from_slice(bytes);
        self.byte_cursor += padding + bytes.len();
    }

    pub fn bytes(&self) -> &[u8] {
        &self.constants[0..self.byte_cursor]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_size_test() {
        assert_eq!(aligned_size(8, 16), 16);
        assert_eq!(aligned_size(16, 16), 16);
        assert_eq!(aligned_size(0, 16), 0);
        assert_eq!(aligned_size(24, 16), 32);
        assert_eq!(aligned_size(20, 16), 32);
    }

    #[test]
    fn range_bounds_to_base_count_test() {
        assert_eq!(range_bounds_to_base_count(0..3, 0, 3), (0, 3));
        assert_eq!(range_bounds_to_base_count(1..3, 0, 3), (1, 2));

        // min/max bounds
        assert_eq!(range_bounds_to_base_count(0..10, 0, 3), (0, 3));
        assert_eq!(range_bounds_to_base_count(0..3, 1, 3), (1, 2));
        assert_eq!(range_bounds_to_base_count(0..10, 1, 3), (1, 2));

        // unbounded
        assert_eq!(range_bounds_to_base_count(..3, 0, 3), (0, 3));
        assert_eq!(range_bounds_to_base_count(1.., 0, 3), (1, 2));
        assert_eq!(range_bounds_to_base_count(.., 1, 3), (1, 2));
    }
}
