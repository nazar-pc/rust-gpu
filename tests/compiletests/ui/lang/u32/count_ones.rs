// Test all trailing and leading zeros. No need to test ones, they just call the zero variant with !value

// build-pass
// compile-flags: -C target-feature=+Int8,+Int16,+Int64

use spirv_std::spirv;

#[spirv(fragment)]
pub fn count_ones_u8(
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] buffer: &u8,
    out: &mut u32,
) {
    *out = u8::count_ones(*buffer);
}

#[spirv(fragment)]
pub fn count_ones_u16(
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] buffer: &u16,
    out: &mut u32,
) {
    *out = u16::count_ones(*buffer);
}

#[spirv(fragment)]
pub fn count_ones_u32(
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] buffer: &u32,
    out: &mut u32,
) {
    *out = u32::count_ones(*buffer);
}

#[spirv(fragment)]
pub fn count_ones_u64(
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] buffer: &u64,
    out: &mut u32,
) {
    *out = u64::count_ones(*buffer);
}

#[spirv(fragment)]
pub fn count_ones_i8(
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] buffer: &i8,
    out: &mut u32,
) {
    *out = i8::count_ones(*buffer);
}

#[spirv(fragment)]
pub fn count_ones_i16(
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] buffer: &i16,
    out: &mut u32,
) {
    *out = i16::count_ones(*buffer);
}

#[spirv(fragment)]
pub fn count_ones_i32(
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] buffer: &i32,
    out: &mut u32,
) {
    *out = i32::count_ones(*buffer);
}

#[spirv(fragment)]
pub fn count_ones_i64(
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] buffer: &i64,
    out: &mut u32,
) {
    *out = i64::count_ones(*buffer);
}
