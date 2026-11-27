// Unresolved symbols that aren't used (after optimizations) aren't errors.

// build-pass

use spirv_std::spirv;

unsafe extern "C" {
    fn missing() -> u32;
}

// Being `pub` keeps this function (and its use of `missing`) around,
// until the linker removes it (as it's not called from any entry-points).
pub fn never_called() -> u32 {
    unsafe { missing() }
}

#[spirv(compute(threads(64)))]
pub fn main(
    #[spirv(global_invocation_id)] id: spirv_std::glam::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] buf: &mut [u32; 4],
) {
    buf[0] = id.x;
}
