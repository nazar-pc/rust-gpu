// Cleanups (e.g. `Drop` impls) run when a panic unwinds, before exiting the
// invocation (here, only on the path with a panic, as the guard is forgotten).

// build-pass
// compile-flags: -C llvm-args=--disassemble-entry=main

use spirv_std::spirv;

struct WriteOnDrop<'a>(&'a mut u32);

impl Drop for WriteOnDrop<'_> {
    fn drop(&mut self) {
        *self.0 = 42;
    }
}

#[spirv(compute(threads(64)))]
pub fn main(
    #[spirv(global_invocation_id)] id: spirv_std::glam::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] out: &mut u32,
) {
    let guard = WriteOnDrop(out);
    if id.x == 3 {
        panic!();
    }
    core::mem::forget(guard);
}
