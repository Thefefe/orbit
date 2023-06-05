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

pub fn init_logger() -> Result<(), fern::InitError> {
    log_panics::init();
    use fern::colors::*;
    use std::fs;

    let log_file = fs::OpenOptions::new().write(true).create(true).truncate(true).open("log.txt")?;

    let colors_level = ColoredLevelConfig::new()
        .error(Color::Red)
        .warn(Color::Yellow)
        .info(Color::Blue)
        .debug(Color::BrightBlack)
        .trace(Color::White);

    let color_line = format!("\x1B[{}m", Color::White.to_fg_str());

    #[rustfmt::skip]
    fern::Dispatch::new()
        .level(log::LevelFilter::Trace)
        .chain(
            fern::Dispatch::new()
                .level(log::LevelFilter::Debug)
                .format(move |out, message, record| {
                    out.finish(format_args!(
    "{color_line}[{color_level}{target}{color_line}][{color_level}{level}{color_line}]{color_level} {message}\x1B[0m",
                        color_level = format_args!("\x1B[{}m", colors_level.get_color(&record.level()).to_fg_str()),
                        target = record.target().split("::").next().unwrap_or(""),
                        level = record.level(),
                        message = message,
                    ));
                })
                .chain(std::io::stdout()),
        )
        .chain(
            fern::Dispatch::new()
                .format(move |out, message, record| {
                    out.finish(format_args!(
                        "[{target}][{level}] {message}",
                        target = record.target(),
                        level = record.level(),
                        message = message,
                    ));
                })
                .chain(log_file),
        )
        .apply()?;

    Ok(())
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