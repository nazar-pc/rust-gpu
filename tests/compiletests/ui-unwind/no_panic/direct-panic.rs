// A `panic!` (with formatting) called directly while a value that needs dropping
// is alive (as opposed to e.g. from a closure, like `no_panic`) also unwinds.
// FIXME: panic site markers are only created for panics in inlined callees, so
// there is no "possible panic at" note here.

// build-fail

use spirv_std::spirv;

fn guarded(x: u32) -> u32 {
    struct Guard;
    unsafe extern "C" {
        #[link_name = "detected panic: direct"]
        fn trigger() -> !;
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            unsafe { trigger() }
        }
    }
    let guard = Guard;
    if x == 3 {
        panic!("x is {}", x);
    }
    core::mem::forget(guard);
    x
}

#[spirv(compute(threads(64)))]
pub fn main(
    #[spirv(global_invocation_id)] id: spirv_std::glam::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] buf: &mut [u32; 4],
) {
    buf[0] = guarded(id.x);
}
