// Panics reachable from inside the guarded closure reach the guard's `Drop`
// (through emulated unwinding), so the unresolved symbol remains in use.

// build-fail
// normalize-stderr-test "\S*/lib/rustlib/" -> "$$SYSROOT/lib/rustlib/"

use spirv_std::spirv;

// Minimal version of what `#[no_panic::no_panic]` expands to: a guard that is
// only dropped while unwinding, referencing a symbol that is never defined.
macro_rules! no_panic {
    ($symbol:literal, $body:expr) => {{
        struct Guard;
        unsafe extern "C" {
            #[link_name = $symbol]
            fn trigger() -> !;
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                unsafe { trigger() }
            }
        }
        let guard = Guard;
        let result = (move || $body)();
        core::mem::forget(guard);
        result
    }};
}

#[spirv(compute(threads(64)))]
pub fn main(
    #[spirv(global_invocation_id)] id: spirv_std::glam::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] buf: &mut [u32; 4],
) {
    let i = id.x as usize;
    let a = *buf;
    buf[0] = no_panic!("detected panic: index", a[i]);
    let x = id.y;
    buf[1] = no_panic!(
        "detected panic: unwrap",
        (if x > 3 { Some(x) } else { None }).unwrap()
    );
}
