// Checks that are implied by dominating branches (e.g. a bounds check after an
// explicit one, or in a loop over a range that is in bounds) are removed, which
// allows proving the absence of panics in common cases.

// build-pass
// compile-flags: -C opt-level=3 -C codegen-units=1

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
    buf[0] = no_panic!("no panic: masked index", a[i & 3]);
    buf[1] = no_panic!("no panic: checked index", if i < a.len() { a[i] } else { 0 });
    buf[2] = no_panic!("no panic: range loop", {
        let mut sum = 0u32;
        for j in 0..4 {
            sum = sum.wrapping_add(a[j]);
        }
        sum
    });
}
