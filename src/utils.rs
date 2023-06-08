use std::{cell::Cell, sync::MutexGuard};

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

pub fn load_spv(path: &str) -> std::io::Result<Vec<u32>> {
    let mut file = std::fs::File::open(path)?;
    ash::util::read_spv(&mut file)
}

// workaround for unstable `impl !Sync`/`Send`
// should be used with `PhantomData`
pub type Unsync = Cell<()>;
pub type Unsend = MutexGuard<'static, ()>;

pub fn init_logger() {
    log_panics::init();
    env_logger::builder()
        .parse_filters("orbit,vulkan=warn")
        .target(env_logger::Target::Stdout)
        .init();
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn aligned_size_test() {
        assert_eq!(aligned_size(8, 16), 16);
        assert_eq!(aligned_size(16, 16), 16);
        assert_eq!(aligned_size(0, 16), 0);
        assert_eq!(aligned_size(24, 16), 32);
        assert_eq!(aligned_size(20, 16), 32);
    }
}