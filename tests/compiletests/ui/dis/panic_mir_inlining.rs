#![crate_name = "panic_mir_inlining"]

// Test that `format_args!` in panics is still decoded into `debugPrintf` with
// MIR inlining (which must not inline the `fmt::Arguments` constructors).

// build-pass
// compile-flags: -Zinline-mir=on
// compile-flags: -C target-feature=+ext:SPV_KHR_non_semantic_info
// compile-flags: -C llvm-args=--abort-strategy=debug-printf
// compile-flags: -C llvm-args=--disassemble
//
// normalize-stderr-test "; (SPIR-V|Generator: rspirv|Version: 1\.\d+|Bound: \d+)\n" -> ""
// normalize-stderr-test "\n\W*OpCapability VulkanMemoryModel" -> ""
// normalize-stderr-test "\n\W*OpSource .*" -> ""
// normalize-stderr-test "\n\W*OpExtension .SPV_KHR_vulkan_memory_model." -> ""
// normalize-stderr-test "OpMemoryModel Logical Vulkan" -> "OpMemoryModel Logical Simple"
// normalize-stderr-test "\S*/lib/rustlib/" -> "$$SYSROOT/lib/rustlib/"
// normalize-stderr-test "ui/dis/" -> "$$DIR/"

use spirv_std::spirv;

fn check(x: u32, y: u32) {
    if x >= y {
        panic!("x ({x}) must be less than y ({y})");
    }
}

#[spirv(fragment)]
pub fn main(#[spirv(flat)] x: u32, #[spirv(flat)] y: u32) {
    check(x, y);
}
