// Unresolved symbols are only errors if they're still used after optimizations,
// so `no_panic` can detect the absence of reachable panics.

// build-pass

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
    let x = id.x;
    let a = *buf;
    buf[0] = no_panic!("no panic: wrapping_add", x.wrapping_add(1));
    buf[1] = no_panic!("no panic: const index", a[2]);
    buf[2] = no_panic!("no panic: checked_add", x.checked_add(1).unwrap_or(0));
}
