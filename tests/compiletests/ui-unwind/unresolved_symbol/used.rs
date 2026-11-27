// Unresolved symbols that remain used (after optimizations) are errors.

// build-fail

use spirv_std::spirv;

unsafe extern "C" {
    #[link_name = "missing_but_used"]
    fn missing() -> u32;
}

#[spirv(compute(threads(64)))]
pub fn main(
    #[spirv(global_invocation_id)] id: spirv_std::glam::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] buf: &mut [u32; 4],
) {
    if id.x == 3 {
        buf[0] = unsafe { missing() };
    }
}
