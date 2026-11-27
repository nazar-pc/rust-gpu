//! This algorithm is not intended to be an optimization, it is rather for legalization.
//! Specifically, spir-v disallows things like a `StorageClass::Function` pointer to a
//! `StorageClass::Input` pointer. Our frontend definitely allows it, though, this is like taking a
//! `&Input<T>` in a function! So, we inline all functions that take these "illegal" pointers, then
//! run mem2reg (see mem2reg.rs) on the result to "unwrap" the Function pointer.

use super::apply_rewrite_rules;
use super::ipo::CallGraph;
use super::simple_passes::outgoing_edges;
use super::{get_name, get_names};
use crate::custom_decorations::SpanRegenerator;
use crate::custom_insts::{self, CustomInst, CustomOp};
use rspirv::dr::{Block, Function, Instruction, Module, ModuleHeader, Operand};
use rspirv::spirv::{FunctionControl, Op, StorageClass, Word};
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet};
use rustc_errors::ErrorGuaranteed;
use rustc_session::Session;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::iter;
use std::mem;

// FIXME(eddyb) this is a bit silly, but this keeps being repeated everywhere.
fn next_id(header: &mut ModuleHeader) -> Word {
    let result = header.bound;
    header.bound += 1;
    result
}

pub fn inline(sess: &Session, module: &mut Module) -> super::Result<()> {
    // This algorithm gets real sad if there's recursion - but, good news, SPIR-V bans recursion
    deny_recursion_in_module(sess, module)?;

    // Compute the call-graph that will drive (inside-out, aka bottom-up) inlining.
    let (call_graph, func_id_to_idx) = CallGraph::collect_with_func_id_to_idx(module);

    let custom_ext_inst_set_import = module
        .ext_inst_imports
        .iter()
        .find(|inst| {
            assert_eq!(inst.class.opcode, Op::ExtInstImport);
            inst.operands[0].unwrap_literal_string() == &custom_insts::CUSTOM_EXT_INST_SET[..]
        })
        .map(|inst| inst.result_id.unwrap());

    let legal_globals = LegalGlobal::gather_from_module(module);

    let strings: FxHashMap<Word, String> = module
        .debug_string_source
        .iter()
        .filter(|inst| inst.class.opcode == Op::String)
        .map(|inst| {
            (
                inst.result_id.unwrap(),
                inst.operands[0].unwrap_literal_string().to_string(),
            )
        })
        .collect();
    let u32_constants: FxHashMap<Word, u32> = module
        .types_global_values
        .iter()
        .filter(|inst| inst.class.opcode == Op::Constant)
        .filter_map(|inst| match inst.operands[..] {
            [Operand::LiteralBit32(x)] => Some((inst.result_id.unwrap(), x)),
            _ => None,
        })
        .collect();

    let unwinding_abort_kind_ids = module
        .debug_string_source
        .iter()
        .filter(|inst| {
            inst.class.opcode == Op::String
                && inst.operands[0].unwrap_literal_string() == custom_insts::ABORT_KIND_PANIC
        })
        .map(|inst| inst.result_id.unwrap())
        .collect();

    let header = module.header.as_mut().unwrap();

    // FIXME(eddyb) clippy false positive (separate `map` required for borrowck).
    #[allow(clippy::map_unwrap_or)]
    let mut inliner = Inliner {
        op_type_void_id: module
            .types_global_values
            .iter()
            .find(|inst| inst.class.opcode == Op::TypeVoid)
            .map(|inst| inst.result_id.unwrap())
            .unwrap_or_else(|| {
                let id = next_id(header);
                let inst = Instruction::new(Op::TypeVoid, None, Some(id), vec![]);
                module.types_global_values.push(inst);
                id
            }),

        custom_ext_inst_set_import: custom_ext_inst_set_import.unwrap_or_else(|| {
            let id = next_id(header);
            let inst = Instruction::new(
                Op::ExtInstImport,
                None,
                Some(id),
                vec![Operand::LiteralString(
                    custom_insts::CUSTOM_EXT_INST_SET.to_string(),
                )],
            );
            module.ext_inst_imports.push(inst);
            id
        }),

        func_id_to_idx,

        id_to_name: module
            .debug_names
            .iter()
            .filter(|inst| inst.class.opcode == Op::Name)
            .map(|inst| {
                (
                    inst.operands[0].unwrap_id_ref(),
                    inst.operands[1].unwrap_literal_string(),
                )
            })
            .collect(),

        cached_op_strings: FxHashMap::default(),

        header,
        debug_string_source: &mut module.debug_string_source,
        annotations: &mut module.annotations,

        legal_globals,

        // NOTE(eddyb) this is needed because our custom `Abort` instructions get
        // lowered to a simple `OpReturn` in entry-points, but that requires that
        // they get inlined all the way up to the entry-points in the first place.
        functions_that_may_abort: module
            .functions
            .iter()
            .filter_map(|func| {
                let custom_ext_inst_set_import = custom_ext_inst_set_import?;
                func.blocks
                    .iter()
                    .any(|block| match &block.instructions[..] {
                        [.., last_normal_inst, terminator_inst]
                            if last_normal_inst.class.opcode == Op::ExtInst
                                && last_normal_inst.operands[0].unwrap_id_ref()
                                    == custom_ext_inst_set_import
                                && CustomOp::decode_from_ext_inst(last_normal_inst)
                                    == CustomOp::Abort =>
                        {
                            assert_eq!(terminator_inst.class.opcode, Op::Unreachable);
                            true
                        }

                        _ => false,
                    })
                    .then_some(func.def_id().unwrap())
            })
            .collect(),

        inlined_dont_inlines_to_cause_and_callers: FxIndexMap::default(),

        unwinding_abort_kind_ids,

        strings,
        u32_constants,
        panic_site_markers: FxIndexMap::default(),
    };

    let mut functions: Vec<_> = mem::take(&mut module.functions)
        .into_iter()
        .map(Ok)
        .collect();

    let mut mem2reg_pointer_to_pointee = FxHashMap::default();
    let mut mem2reg_constants = FxHashMap::default();
    {
        let mut u32 = None;
        for inst in &module.types_global_values {
            match inst.class.opcode {
                Op::TypePointer => {
                    mem2reg_pointer_to_pointee
                        .insert(inst.result_id.unwrap(), inst.operands[1].unwrap_id_ref());
                }
                Op::TypeInt
                    if inst.operands[0].unwrap_literal_bit32() == 32
                        && inst.operands[1].unwrap_literal_bit32() == 0 =>
                {
                    assert!(u32.is_none());
                    u32 = Some(inst.result_id.unwrap());
                }
                Op::Constant if u32.is_some() && inst.result_type == u32 => {
                    let value = inst.operands[0].unwrap_literal_bit32();
                    mem2reg_constants.insert(inst.result_id.unwrap(), value);
                }
                _ => {}
            }
        }
    }

    // Inline functions in post-order (aka inside-out aka bottom-up) - that is,
    // callees are processed before their callers, to avoid duplicating work.
    for func_idx in call_graph.post_order() {
        let mut function = mem::replace(&mut functions[func_idx], Err(FuncIsBeingInlined)).unwrap();
        inliner.inline_fn(&mut function, &functions);

        // Any remaining calls (that were not inlined) can't unwind, as only
        // functions that may (transitively) `Abort` can unwind, and those are
        // always inlined (see also `CustomInst::CallUnwound`).
        inliner.fold_call_unwound_to_false(&mut function);

        fuse_trivial_branches(&mut function);

        super::duplicates::DebuginfoDeduplicator {
            custom_ext_inst_set_import,
        }
        .remove_duplicate_debuginfo_in_function(&mut function);

        {
            super::simple_passes::block_ordering_pass(&mut function);
            // NOTE emulated unwinding can make landing pads unreachable,
            // and `block_ordering_pass` removes them, but not `OpPhi` operands.
            remove_phi_operands_from_missing_blocks(&mut function);

            // NOTE emulated unwinding (see `CustomInst::CallUnwound`)
            // can remove `Abort`s (by turning them into landing pad jumps),
            // and landing pads may become unreachable, so the original
            // `functions_that_may_abort` is only an over-approximation.
            if let Some(func_id) = function.def_id()
                && inliner.functions_that_may_abort.contains(&func_id)
                && !inliner.function_may_abort(&function)
            {
                inliner.functions_that_may_abort.remove(&func_id);
            }

            // Note: mem2reg requires functions to be in RPO order (i.e. block_ordering_pass)
            super::mem2reg::mem2reg(
                inliner.header,
                &mut module.types_global_values,
                &mem2reg_pointer_to_pointee,
                &mem2reg_constants,
                &mut function,
            );
            super::destructure_composites::destructure_composites(&mut function);
        }

        functions[func_idx] = Ok(function);
    }

    module.functions = functions.into_iter().map(|func| func.unwrap()).collect();

    let Inliner {
        id_to_name,
        inlined_dont_inlines_to_cause_and_callers,
        panic_site_markers,
        op_type_void_id,
        ..
    } = inliner;

    let mut span_regen = SpanRegenerator::new(sess.source_map(), module);
    for (callee_id, (cause, callers)) in inlined_dont_inlines_to_cause_and_callers {
        let callee_name = get_name(&id_to_name, callee_id);

        // HACK(eddyb) `libcore` hides panics behind `#[inline(never)]` `fn`s,
        // making this too noisy and useless (since it's an impl detail).
        if cause == "panicking" && callee_name.starts_with("core::") {
            continue;
        }

        let callee_span = span_regen
            .src_loc_for_id(callee_id)
            .and_then(|src_loc| span_regen.src_loc_to_rustc(src_loc))
            .unwrap_or_default();
        sess.dcx()
            .struct_span_warn(
                callee_span,
                format!("`#[inline(never)]` function `{callee_name}` has been inlined"),
            )
            .with_note(format!("inlining was required due to {cause}"))
            .with_note(format!(
                "called from {}",
                callers
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &caller_id)| {
                        // HACK(eddyb) avoid showing too many names.
                        match i.cmp(&4) {
                            Ordering::Less => {
                                Some(format!("`{}`", get_name(&id_to_name, caller_id)))
                            }
                            Ordering::Equal => Some(format!("and {} more", callers.len() - i)),
                            Ordering::Greater => None,
                        }
                    })
                    .collect::<SmallVec<[_; 5]>>()
                    .join(", ")
            ))
            .emit();
    }
    drop(span_regen);

    if !panic_site_markers.is_empty() {
        define_panic_site_markers(module, op_type_void_id, panic_site_markers);
    }

    Ok(())
}

// https://stackoverflow.com/a/53995651
fn deny_recursion_in_module(sess: &Session, module: &Module) -> super::Result<()> {
    let func_to_index: FxHashMap<Word, usize> = module
        .functions
        .iter()
        .enumerate()
        .map(|(index, func)| (func.def_id().unwrap(), index))
        .collect();
    let mut discovered = vec![false; module.functions.len()];
    let mut finished = vec![false; module.functions.len()];
    let mut has_recursion = None;
    for index in 0..module.functions.len() {
        if !discovered[index] && !finished[index] {
            visit(
                sess,
                module,
                index,
                &mut discovered,
                &mut finished,
                &mut has_recursion,
                &func_to_index,
            );
        }
    }

    fn visit(
        sess: &Session,
        module: &Module,
        current: usize,
        discovered: &mut Vec<bool>,
        finished: &mut Vec<bool>,
        has_recursion: &mut Option<ErrorGuaranteed>,
        func_to_index: &FxHashMap<Word, usize>,
    ) {
        discovered[current] = true;

        for next in calls(&module.functions[current], func_to_index) {
            if discovered[next] {
                let names = get_names(module);
                let current_name = get_name(&names, module.functions[current].def_id().unwrap());
                let next_name = get_name(&names, module.functions[next].def_id().unwrap());
                *has_recursion = Some(sess.dcx().err(format!(
                    "module has recursion, which is not allowed: `{current_name}` calls `{next_name}`"
                )));
                break;
            }

            if !finished[next] {
                visit(
                    sess,
                    module,
                    next,
                    discovered,
                    finished,
                    has_recursion,
                    func_to_index,
                );
            }
        }

        discovered[current] = false;
        finished[current] = true;
    }

    fn calls<'a>(
        func: &'a Function,
        func_to_index: &'a FxHashMap<Word, usize>,
    ) -> impl Iterator<Item = usize> + 'a {
        func.all_inst_iter()
            .filter(|inst| inst.class.opcode == Op::FunctionCall)
            .map(move |inst| {
                *func_to_index
                    .get(&inst.operands[0].id_ref_any().unwrap())
                    .unwrap()
            })
    }

    match has_recursion {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

/// Any type/const/global variable, which is "legal" (i.e. can be kept in SPIR-V).
///
/// For the purposes of the inliner, a legal global cannot:
/// - refer to any illegal globals
/// - (if a type) refer to any pointer types
///   - this rules out both pointers in composites, and pointers to pointers
///     (the latter itself *also* rules out variables containing pointers)
enum LegalGlobal {
    TypePointer(StorageClass),
    TypeNonPointer,
    Const,
    Variable,
}

impl LegalGlobal {
    fn gather_from_module(module: &Module) -> FxHashMap<Word, Self> {
        let mut legal_globals = FxHashMap::<_, Self>::default();
        for inst in &module.types_global_values {
            let global = match inst.class.opcode {
                Op::TypePointer => Self::TypePointer(inst.operands[0].unwrap_storage_class()),
                Op::Variable => Self::Variable,
                op if op.is_type() => Self::TypeNonPointer,
                op if op.is_constant() => Self::Const,

                // FIXME(eddyb) should this be `unreachable!()`?
                _ => continue,
            };
            let legal_result_type = match inst.result_type {
                Some(result_type_id) => matches!(
                    (&global, legal_globals.get(&result_type_id)),
                    (Self::Variable, Some(Self::TypePointer(_)))
                        | (Self::Const, Some(Self::TypeNonPointer))
                ),
                None => matches!(global, Self::TypePointer(_) | Self::TypeNonPointer),
            };
            let legal_operands = inst.operands.iter().all(|operand| match operand {
                Operand::IdRef(id) => matches!(
                    legal_globals.get(id),
                    Some(Self::TypeNonPointer | Self::Const)
                ),

                // NOTE(eddyb) this assumes non-ID operands are always legal.
                _ => operand.id_ref_any().is_none(),
            });
            if legal_result_type && legal_operands {
                legal_globals.insert(inst.result_id.unwrap(), global);
            }
        }
        legal_globals
    }

    fn legal_as_fn_param_ty(&self) -> bool {
        match *self {
            Self::TypePointer(storage_class) => matches!(
                storage_class,
                StorageClass::UniformConstant
                    | StorageClass::Function
                    | StorageClass::Private
                    | StorageClass::Workgroup
                    | StorageClass::AtomicCounter
            ),
            Self::TypeNonPointer => true,

            // FIXME(eddyb) should this be an `unreachable!()`?
            Self::Const | Self::Variable => false,
        }
    }

    fn legal_as_fn_ret_ty(&self) -> bool {
        #[allow(clippy::match_same_arms)]
        match *self {
            Self::TypePointer(_) => false,
            Self::TypeNonPointer => true,

            // FIXME(eddyb) should this be an `unreachable!()`?
            Self::Const | Self::Variable => false,
        }
    }
}

/// Helper type which encapsulates all the information about one specific call.
#[derive(Copy, Clone)]
struct CallSite<'a> {
    caller: &'a Function,
    call_inst: &'a Instruction,
}

fn has_dont_inline(function: &Function) -> bool {
    let def = function.def.as_ref().unwrap();
    let control = def.operands[0].unwrap_function_control();
    control.contains(FunctionControl::DONT_INLINE)
}

/// Helper error type for `should_inline` (see its doc comment).
#[derive(Copy, Clone, PartialEq, Eq)]
struct MustInlineToLegalize(&'static str);

/// Returns `Ok(true)`/`Err(MustInlineToLegalize(_))` if `callee` should/must be
/// inlined (either in general, or specifically from `call_site`, if provided).
///
/// The distinction made here is that `Err(MustInlineToLegalize(cause))` is
/// very much *not* a heuristic, and inlining is *mandatory* due to `cause`
/// (usually illegal signature/arguments, but also the panicking mechanism).
//
// FIXME(eddyb) the causes here are not fine-grained enough.
fn should_inline(
    legal_globals: &FxHashMap<Word, LegalGlobal>,
    functions_that_may_abort: &FxHashSet<Word>,
    callee: &Function,
    call_site: CallSite<'_>,
) -> Result<bool, MustInlineToLegalize> {
    let callee_def = callee.def.as_ref().unwrap();
    let callee_control = callee_def.operands[0].unwrap_function_control();

    if functions_that_may_abort.contains(&callee.def_id().unwrap()) {
        return Err(MustInlineToLegalize("panicking"));
    }

    let ret_ty = legal_globals
        .get(&callee_def.result_type.unwrap())
        .ok_or(MustInlineToLegalize("illegal return type"))?;
    if !ret_ty.legal_as_fn_ret_ty() {
        return Err(MustInlineToLegalize("illegal (pointer) return type"));
    }

    for (i, param) in callee.parameters.iter().enumerate() {
        let param_ty = legal_globals
            .get(param.result_type.as_ref().unwrap())
            .ok_or(MustInlineToLegalize("illegal parameter type"))?;
        if !param_ty.legal_as_fn_param_ty() {
            return Err(MustInlineToLegalize("illegal (pointer) parameter type"));
        }

        // If the call isn't passing a legal pointer argument (a "memory object",
        // i.e. an `OpVariable` or one of the caller's `OpFunctionParameter`s),
        // then inlining is required to have a chance at producing legal SPIR-V.
        //
        // FIXME(eddyb) rewriting away the pointer could be another alternative.
        if let LegalGlobal::TypePointer(_) = param_ty {
            let ptr_arg = call_site.call_inst.operands[i + 1].unwrap_id_ref();
            match legal_globals.get(&ptr_arg) {
                Some(LegalGlobal::Variable) => {}

                // FIXME(eddyb) should some constants (undef/null) be allowed?
                Some(_) => return Err(MustInlineToLegalize("illegal (pointer) argument")),

                None => {
                    let mut caller_param_and_var_ids = call_site
                        .caller
                        .parameters
                        .iter()
                        .chain(
                            call_site.caller.blocks[0]
                                .instructions
                                .iter()
                                .filter(|caller_inst| {
                                    // HACK(eddyb) this only avoids scanning the
                                    // whole entry block for `OpVariable`s, so
                                    // it can overapproximate debuginfo insts.
                                    let may_be_debuginfo = matches!(
                                        caller_inst.class.opcode,
                                        Op::Line | Op::NoLine | Op::ExtInst
                                    );
                                    !may_be_debuginfo
                                })
                                .take_while(|caller_inst| caller_inst.class.opcode == Op::Variable),
                        )
                        .map(|caller_inst| caller_inst.result_id.unwrap());

                    if !caller_param_and_var_ids.any(|id| ptr_arg == id) {
                        return Err(MustInlineToLegalize("illegal (pointer) argument"));
                    }
                }
            }
        }
    }

    Ok(callee_control.contains(FunctionControl::INLINE))
}

/// Helper error type for `Inliner`'s `functions` field, indicating a `Function`
/// was taken out of its slot because it's being inlined.
#[derive(Debug)]
struct FuncIsBeingInlined;

// Steps:
// Move OpVariable decls
// Rewrite return
// Renumber IDs
// Insert blocks

struct Inliner<'a, 'b> {
    /// ID of `OpExtInstImport` for our custom "extended instruction set"
    /// (see `crate::custom_insts` for more details).
    custom_ext_inst_set_import: Word,

    op_type_void_id: Word,

    /// Map from each function's ID to its index in `functions`.
    func_id_to_idx: FxHashMap<Word, usize>,

    /// Pre-collected `OpName`s, that can be used to find any function's name
    /// during inlining (to be able to generate debuginfo that uses names).
    id_to_name: FxHashMap<Word, &'a str>,

    /// `OpString` cache (for deduplicating `OpString`s for the same string).
    //
    // FIXME(eddyb) currently this doesn't reuse existing `OpString`s, but since
    // this is mostly for inlined callee names, it's expected almost no overlap
    // exists between existing `OpString`s and new ones, anyway.
    cached_op_strings: FxHashMap<&'a str, Word>,

    header: &'b mut ModuleHeader,
    debug_string_source: &'b mut Vec<Instruction>,
    annotations: &'b mut Vec<Instruction>,

    legal_globals: FxHashMap<Word, LegalGlobal>,
    functions_that_may_abort: FxHashSet<Word>,
    inlined_dont_inlines_to_cause_and_callers: FxIndexMap<Word, (&'static str, FxIndexSet<Word>)>,
    // rewrite_rules: FxHashMap<Word, Word>,
    /// IDs of `OpString`s used as the `kind` of `Abort`s that unwind (i.e. for
    /// `panic!`s, when emulating unwinding, see `CustomInst::CallUnwound`).
    unwinding_abort_kind_ids: FxHashSet<Word>,

    /// Pre-collected `OpString`s and 32-bit `OpConstant`s (for describing panics).
    strings: FxHashMap<Word, String>,
    u32_constants: FxHashMap<Word, u32>,

    /// "Panic site markers": functions (by description) called in place of the
    /// `Abort`s of `panic!`s which unwind to a landing pad (see
    /// `import_export_link::PANIC_SITE_MARKER_NAME_PREFIX` for more details).
    panic_site_markers: FxIndexMap<String, Word>,
}

impl Inliner<'_, '_> {
    fn id(&mut self) -> Word {
        next_id(self.header)
    }

    /// Applies all rewrite rules to the decorations in the header.
    fn apply_rewrite_for_decorations(&mut self, rewrite_rules: &FxHashMap<Word, Word>) {
        // NOTE(siebencorgie): We don't care *what* decoration we rewrite atm.
        // AFAIK there is no case where keeping decorations on inline wouldn't be valid.
        for annotation_idx in 0..self.annotations.len() {
            let inst = &self.annotations[annotation_idx];
            if let [Operand::IdRef(target), ..] = inst.operands[..]
                && let Some(&rewritten_target) = rewrite_rules.get(&target)
            {
                // Copy decoration instruction and push it.
                let mut cloned_inst = inst.clone();
                cloned_inst.operands[0] = Operand::IdRef(rewritten_target);
                self.annotations.push(cloned_inst);
            }
        }
    }

    fn inline_fn(
        &mut self,
        function: &mut Function,
        functions: &[Result<Function, FuncIsBeingInlined>],
    ) {
        let mut block_idx = 0;
        while block_idx < function.blocks.len() {
            // If we successfully inlined a block, then repeat processing on the same block, in
            // case the newly inlined block has more inlined calls.
            // TODO: This is quadratic
            if !self.inline_block(function, block_idx, functions) {
                // TODO(eddyb) skip past the inlined callee without rescanning it.
                block_idx += 1;
            }
        }
    }

    fn inline_block(
        &mut self,
        caller: &mut Function,
        block_idx: usize,
        functions: &[Result<Function, FuncIsBeingInlined>],
    ) -> bool {
        // Find the first inlined OpFunctionCall
        let call = caller.blocks[block_idx]
            .instructions
            .iter()
            .enumerate()
            .filter(|(_, inst)| inst.class.opcode == Op::FunctionCall)
            .filter_map(|(index, inst)| {
                // NOTE panic site markers are created during inlining.
                let &callee_idx = self
                    .func_id_to_idx
                    .get(&inst.operands[0].id_ref_any().unwrap())?;
                Some((index, inst, functions[callee_idx].as_ref().unwrap()))
            })
            .find(|(_, inst, f)| {
                let call_site = CallSite {
                    caller,
                    call_inst: inst,
                };
                match should_inline(
                    &self.legal_globals,
                    &self.functions_that_may_abort,
                    f,
                    call_site,
                ) {
                    Ok(inline) => inline,
                    Err(MustInlineToLegalize(cause)) => {
                        if has_dont_inline(f) {
                            self.inlined_dont_inlines_to_cause_and_callers
                                .entry(f.def_id().unwrap())
                                .or_insert_with(|| (cause, Default::default()))
                                .1
                                .insert(caller.def_id().unwrap());
                        }
                        true
                    }
                }
            });
        let (call_index, call_inst, callee) = match call {
            None => return false,
            Some(call) => call,
        };

        // Emulated unwinding: find the landing pad (if this is an `invoke`).
        let invoke_landing_pad_and_normal_dest =
            self.find_invoke_landing_pad_and_normal_dest(&caller.blocks[block_idx], call_index);

        // Propagate "may abort" from callee to caller (i.e. as aborts get inlined).
        if self
            .functions_that_may_abort
            .contains(&callee.def_id().unwrap())
        {
            self.functions_that_may_abort
                .insert(caller.def_id().unwrap());
        }

        let mut maybe_call_result_phi = {
            let ty = call_inst.result_type.unwrap();
            if ty == self.op_type_void_id {
                None
            } else {
                Some(Instruction::new(
                    Op::Phi,
                    Some(ty),
                    Some(call_inst.result_id.unwrap()),
                    vec![],
                ))
            }
        };

        // Get the debug "source location" instruction that applies to the call.
        let custom_ext_inst_set_import = self.custom_ext_inst_set_import;
        let call_debug_src_loc_inst = caller.blocks[block_idx].instructions[..call_index]
            .iter()
            .rev()
            .find_map(|inst| {
                Some(match inst.class.opcode {
                    Op::Line => Some(inst),
                    Op::NoLine => None,
                    Op::ExtInst
                        if inst.operands[0].unwrap_id_ref() == custom_ext_inst_set_import =>
                    {
                        match CustomOp::decode_from_ext_inst(inst) {
                            CustomOp::SetDebugSrcLoc => Some(inst),
                            CustomOp::ClearDebugSrcLoc => None,
                            _ => return None,
                        }
                    }
                    _ => return None,
                })
            })
            .flatten();

        // Rewrite parameters to arguments
        let call_arguments = call_inst
            .operands
            .iter()
            .skip(1)
            .map(|op| op.id_ref_any().unwrap());
        let callee_parameters = callee.parameters.iter().map(|inst| {
            assert!(inst.class.opcode == Op::FunctionParameter);
            inst.result_id.unwrap()
        });
        let mut rewrite_rules = callee_parameters.zip(call_arguments).collect();

        let return_jump = self.id();
        // Rewrite OpReturns of the callee.
        let mut inlined_callee_blocks = self.get_inlined_blocks(
            callee,
            call_debug_src_loc_inst,
            maybe_call_result_phi.as_mut(),
            return_jump,
        );
        // Clone the IDs of the callee, because otherwise they'd be defined multiple times if the
        // fn is inlined multiple times.
        self.add_clone_id_rules(&mut rewrite_rules, &inlined_callee_blocks);
        apply_rewrite_rules(&rewrite_rules, &mut inlined_callee_blocks);
        self.apply_rewrite_for_decorations(&rewrite_rules);

        // Emulated unwinding: redirect all (unwinding) `Abort`s in the callee
        // to the landing pad (as they would otherwise propagate to the caller).
        let mut unwinding_inlined_callee_block_indices = SmallVec::<[usize; 4]>::new();
        if let Some((landing_pad, _)) = invoke_landing_pad_and_normal_dest {
            for (i, block) in inlined_callee_blocks.iter_mut().enumerate() {
                if self.try_redirect_unwinding_abort(block, landing_pad) {
                    unwinding_inlined_callee_block_indices.push(i);
                }
            }
        }

        if let Some(call_result_phi) = &mut maybe_call_result_phi {
            // HACK(eddyb) new IDs should be generated earlier, to avoid pushing
            // callee IDs to `call_result_phi.operands` only to rewrite them here.
            for op in &mut call_result_phi.operands {
                if let Some(id) = op.id_ref_any_mut()
                    && let Some(&rewrite) = rewrite_rules.get(id)
                {
                    *id = rewrite;
                }
            }

            // HACK(eddyb) this special-casing of the single-return case is
            // really necessary for passes like `mem2reg` which are not capable
            // of skipping through the extraneous `OpPhi`s on their own.
            if let [returned_value, _return_block] = &call_result_phi.operands[..] {
                let call_result_id = call_result_phi.result_id.unwrap();
                let returned_value_id = returned_value.unwrap_id_ref();

                maybe_call_result_phi = None;

                // HACK(eddyb) this is a conservative approximation of all the
                // instructions that could potentially reference the call result.
                let reaching_insts = {
                    let (pre_call_blocks, call_and_post_call_blocks) =
                        caller.blocks.split_at_mut(block_idx);
                    (pre_call_blocks.iter_mut().flat_map(|block| {
                        block
                            .instructions
                            .iter_mut()
                            .take_while(|inst| inst.class.opcode == Op::Phi)
                    }))
                    .chain(
                        call_and_post_call_blocks
                            .iter_mut()
                            .flat_map(|block| &mut block.instructions),
                    )
                };
                for reaching_inst in reaching_insts {
                    for op in &mut reaching_inst.operands {
                        if let Some(id) = op.id_ref_any_mut()
                            && *id == call_result_id
                        {
                            *id = returned_value_id;
                        }
                    }
                }
            }
        }

        // Split the block containing the `OpFunctionCall` into pre-call vs post-call.
        let pre_call_block_idx = block_idx;
        #[expect(unused)]
        let block_idx: usize; // HACK(eddyb) disallowing using the unrenamed variable.
        let mut post_call_block_insts = caller.blocks[pre_call_block_idx]
            .instructions
            .split_off(call_index + 1);

        // Emulated unwinding: if the inlined callee returns, it didn't unwind.
        if let Some((_, normal_dest)) = invoke_landing_pad_and_normal_dest {
            self.replace_call_unwound_with_false(&mut post_call_block_insts, normal_dest);
        }

        // HACK the label of the pre-call block is needed later, and the
        // callee's (fused) entry block also ends up using the same label.
        let pre_call_block_label_id = caller.blocks[pre_call_block_idx].label_id().unwrap();
        let unwinding_pred_label_ids: SmallVec<[Word; 4]> = unwinding_inlined_callee_block_indices
            .iter()
            .map(|&i| {
                if i == 0 {
                    pre_call_block_label_id
                } else {
                    inlined_callee_blocks[i].label_id().unwrap()
                }
            })
            .collect();

        // pop off OpFunctionCall
        let call = caller.blocks[pre_call_block_idx]
            .instructions
            .pop()
            .unwrap();
        assert!(call.class.opcode == Op::FunctionCall);

        // Insert non-entry inlined callee blocks just after the pre-call block.
        let non_entry_inlined_callee_blocks = inlined_callee_blocks.drain(1..);
        let num_non_entry_inlined_callee_blocks = non_entry_inlined_callee_blocks.len();
        caller.blocks.splice(
            (pre_call_block_idx + 1)..(pre_call_block_idx + 1),
            non_entry_inlined_callee_blocks,
        );

        if let Some(call_result_phi) = maybe_call_result_phi {
            // Add the `OpPhi` for the call result value, after the inlined function.
            post_call_block_insts.insert(0, call_result_phi);
        }

        // Insert the post-call block, after all the inlined callee blocks.
        {
            let post_call_block_idx = pre_call_block_idx + num_non_entry_inlined_callee_blocks + 1;
            let post_call_block = Block {
                label: Some(Instruction::new(Op::Label, None, Some(return_jump), vec![])),
                instructions: post_call_block_insts,
            };
            caller.blocks.insert(post_call_block_idx, post_call_block);

            // Adjust any `OpPhi`s in the (caller) targets of the original call block,
            // to refer to post-call block (the new source of those CFG edges).
            rewrite_phi_sources(
                caller.blocks[pre_call_block_idx].label_id().unwrap(),
                &mut caller.blocks,
                post_call_block_idx,
            );
        }

        // Fuse the inlined callee entry block into the pre-call block.
        // This is okay because it's illegal to branch to the first BB in a function.
        {
            // NOTE(eddyb) `OpExtInst`s have a result ID, even if unused, and
            // it has to be unique, so this allocates new IDs as-needed.
            let instantiate_debuginfo = |this: &mut Self, inst: &Instruction| {
                let mut inst = inst.clone();
                if let Some(id) = &mut inst.result_id {
                    *id = this.id();
                }
                inst
            };

            let custom_inst_to_inst = |this: &mut Self, inst: CustomInst<_>| {
                Instruction::new(
                    Op::ExtInst,
                    Some(this.op_type_void_id),
                    Some(this.id()),
                    [
                        Operand::IdRef(this.custom_ext_inst_set_import),
                        Operand::LiteralExtInstInteger(inst.op() as u32),
                    ]
                    .into_iter()
                    .chain(inst.into_operands())
                    .collect(),
                )
            };

            // Return the subsequence of `insts` made from `OpVariable`s, and any
            // debuginfo instructions (which may apply to them), while removing
            // *only* `OpVariable`s from `insts` (and keeping debuginfo in both).
            let mut steal_vars = |insts: &mut Vec<Instruction>| {
                // HACK(eddyb) this duplicates some code from `get_inlined_blocks`,
                // but that will be removed once the inliner is refactored to be
                // inside-out instead of outside-in (already finished in a branch).
                let mut enclosing_inlined_frames = SmallVec::<[_; 8]>::new();
                let mut current_debug_src_loc_inst = None;
                let mut vars_and_debuginfo_range = 0..0;
                while vars_and_debuginfo_range.end < insts.len() {
                    let inst = &insts[vars_and_debuginfo_range.end];
                    match inst.class.opcode {
                        Op::Line => current_debug_src_loc_inst = Some(inst),
                        Op::NoLine => current_debug_src_loc_inst = None,
                        Op::ExtInst
                            if inst.operands[0].unwrap_id_ref()
                                == self.custom_ext_inst_set_import =>
                        {
                            match CustomOp::decode_from_ext_inst(inst) {
                                CustomOp::SetDebugSrcLoc => current_debug_src_loc_inst = Some(inst),
                                CustomOp::ClearDebugSrcLoc => current_debug_src_loc_inst = None,
                                CustomOp::PushInlinedCallFrame => {
                                    enclosing_inlined_frames
                                        .push((current_debug_src_loc_inst.take(), inst));
                                }
                                CustomOp::PopInlinedCallFrame => {
                                    if let Some((callsite_debug_src_loc_inst, _)) =
                                        enclosing_inlined_frames.pop()
                                    {
                                        current_debug_src_loc_inst = callsite_debug_src_loc_inst;
                                    }
                                }
                                CustomOp::Abort | CustomOp::CallUnwound => break,
                            }
                        }
                        Op::Variable => {}
                        _ => break,
                    }
                    vars_and_debuginfo_range.end += 1;
                }

                // `vars_and_debuginfo_range.end` indicates where `OpVariable`s
                // end and other instructions start (modulo debuginfo), but to
                // split the block in two, both sides of the "cut" need "repair":
                // - the variables are missing "inlined call frames" pops, that
                //   may happen later in the block, and have to be synthesized
                // - the non-variables are missing "inlined call frames" pushes,
                //   that must be recreated to avoid ending up with dangling pops
                //
                // FIXME(eddyb) this only collects to avoid borrow conflicts,
                // between e.g. `enclosing_inlined_frames` and mutating `insts`,
                // but also between different uses of `self`.
                let all_pops_after_vars: SmallVec<[_; 8]> = enclosing_inlined_frames
                    .iter()
                    .map(|_| custom_inst_to_inst(self, CustomInst::PopInlinedCallFrame))
                    .collect();
                let all_repushes_before_non_vars: SmallVec<[_; 8]> =
                    (enclosing_inlined_frames.into_iter().flat_map(
                        |(callsite_debug_src_loc_inst, push_inlined_call_frame_inst)| {
                            (callsite_debug_src_loc_inst.into_iter())
                                .chain([push_inlined_call_frame_inst])
                        },
                    ))
                    .chain(current_debug_src_loc_inst)
                    .map(|inst| instantiate_debuginfo(self, inst))
                    .collect();

                let vars_and_debuginfo =
                    insts.splice(vars_and_debuginfo_range, all_repushes_before_non_vars);
                let repaired_vars_and_debuginfo = vars_and_debuginfo.chain(all_pops_after_vars);

                // FIXME(eddyb) collecting shouldn't be necessary but this is
                // nested in a closure, and `splice` borrows the original `Vec`.
                repaired_vars_and_debuginfo.collect::<SmallVec<[_; 8]>>()
            };

            let [mut inlined_callee_entry_block]: [_; 1] =
                inlined_callee_blocks.try_into().unwrap();

            // Move the `OpVariable`s of the callee to the caller.
            let callee_vars_and_debuginfo =
                steal_vars(&mut inlined_callee_entry_block.instructions);
            self.insert_opvariables(&mut caller.blocks[0], callee_vars_and_debuginfo);

            caller.blocks[pre_call_block_idx]
                .instructions
                .append(&mut inlined_callee_entry_block.instructions);

            // Adjust any `OpPhi`s in the (inlined callee) targets of the
            // inlined callee entry block, to refer to the pre-call block
            // (the new source of those CFG edges).
            rewrite_phi_sources(
                inlined_callee_entry_block.label_id().unwrap(),
                &mut caller.blocks,
                pre_call_block_idx,
            );
        }

        // Emulated unwinding: the landing pad's predecessor (the pre-call block)
        // was replaced by all the inlined callee blocks that unwind (if any).
        if let Some((landing_pad, _)) = invoke_landing_pad_and_normal_dest {
            let landing_pad_block = caller
                .blocks
                .iter_mut()
                .find(|b| b.label_id() == Some(landing_pad))
                .unwrap();
            replace_phi_source(
                landing_pad_block,
                pre_call_block_label_id,
                &unwinding_pred_label_ids,
            );
        }

        true
    }

    /// Emulated unwinding (see `CustomInst::CallUnwound`): if the `OpFunctionCall`
    /// at `block.instructions[call_index]` is an `invoke` (i.e. followed by its
    /// `CallUnwound`, and an `OpBranchConditional` using it), return the
    /// landing pad (unwind destination) and the normal (return) destination.
    fn find_invoke_landing_pad_and_normal_dest(
        &self,
        block: &Block,
        call_index: usize,
    ) -> Option<(Word, Word)> {
        let call_id = block.instructions[call_index].result_id.unwrap();
        let call_unwound_id = block.instructions[call_index + 1..]
            .iter()
            .find(|inst| {
                inst.class.opcode == Op::ExtInst
                    && inst.operands[0].unwrap_id_ref() == self.custom_ext_inst_set_import
                    && CustomOp::decode_from_ext_inst(inst) == CustomOp::CallUnwound
                    && inst.operands[2].unwrap_id_ref() == call_id
            })?
            .result_id
            .unwrap();
        let terminator = block.instructions.last().unwrap();
        assert_eq!(terminator.class.opcode, Op::BranchConditional);
        assert_eq!(terminator.operands[0].unwrap_id_ref(), call_unwound_id);
        Some((
            terminator.operands[1].unwrap_id_ref(),
            terminator.operands[2].unwrap_id_ref(),
        ))
    }

    /// Emulated unwinding (see `CustomInst::CallUnwound`): replace the only
    /// `CallUnwound` in `insts` with `false` (i.e. remove it, and replace the
    /// `OpBranchConditional` using it, the terminator, with `normal_dest`).
    ///
    /// Returns `false` (without making any changes) if there's no `CallUnwound`.
    fn replace_call_unwound_with_false(
        &self,
        insts: &mut Vec<Instruction>,
        normal_dest: Word,
    ) -> bool {
        let Some(call_unwound_idx) = insts.iter().position(|inst| {
            inst.class.opcode == Op::ExtInst
                && inst.operands[0].unwrap_id_ref() == self.custom_ext_inst_set_import
                && CustomOp::decode_from_ext_inst(inst) == CustomOp::CallUnwound
        }) else {
            return false;
        };
        let call_unwound = insts.remove(call_unwound_idx);
        let terminator = insts.last_mut().unwrap();
        assert_eq!(terminator.class.opcode, Op::BranchConditional);
        assert_eq!(
            terminator.operands[0].unwrap_id_ref(),
            call_unwound.result_id.unwrap()
        );
        assert_eq!(terminator.operands[2].unwrap_id_ref(), normal_dest);
        *terminator = Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(normal_dest)]);
        true
    }

    /// Emulated unwinding (see `CustomInst::CallUnwound`): replace every
    /// `CallUnwound` in `function` with `false` (and remove the corresponding
    /// landing pad `OpPhi` operands), for calls that were not inlined.
    fn fold_call_unwound_to_false(&self, function: &mut Function) {
        for block_idx in 0..function.blocks.len() {
            let landing_pad_and_normal_dest = match function.blocks[block_idx].instructions.last() {
                Some(terminator) if terminator.class.opcode == Op::BranchConditional => (
                    terminator.operands[1].unwrap_id_ref(),
                    terminator.operands[2].unwrap_id_ref(),
                ),
                _ => continue,
            };
            let (landing_pad, normal_dest) = landing_pad_and_normal_dest;
            if self.replace_call_unwound_with_false(
                &mut function.blocks[block_idx].instructions,
                normal_dest,
            ) {
                let label_id = function.blocks[block_idx].label_id().unwrap();
                let landing_pad_block = function
                    .blocks
                    .iter_mut()
                    .find(|b| b.label_id() == Some(landing_pad))
                    .unwrap();
                replace_phi_source(landing_pad_block, label_id, &[]);
            }
        }
    }

    /// Emulated unwinding (see `CustomInst::CallUnwound`): if `block` ends in
    /// an unwinding `Abort`, replace it with a jump to `landing_pad`.
    fn try_redirect_unwinding_abort(&mut self, block: &mut Block, landing_pad: Word) -> bool {
        let insts = &mut block.instructions;
        if insts.last().map(|inst| inst.class.opcode) != Some(Op::Unreachable) {
            return false;
        }
        let terminator_idx = insts.len() - 1;
        let custom_op = |inst: &Instruction| {
            (inst.class.opcode == Op::ExtInst
                && inst.operands[0].unwrap_id_ref() == self.custom_ext_inst_set_import)
                .then(|| CustomOp::decode_from_ext_inst(inst))
        };
        let abort_idx = insts[..terminator_idx].iter().rposition(|inst| {
            let is_debuginfo = matches!(inst.class.opcode, Op::Line | Op::NoLine)
                || custom_op(inst).is_some_and(|op| op.is_debuginfo());
            !is_debuginfo
        });
        let Some(abort_idx) = abort_idx else {
            return false;
        };
        let abort = &insts[abort_idx];
        if custom_op(abort) != Some(CustomOp::Abort) {
            return false;
        }
        let CustomInst::Abort {
            kind,
            message_debug_printf,
        } = CustomInst::decode(abort)
        else {
            unreachable!()
        };
        if !self
            .unwinding_abort_kind_ids
            .contains(&kind.unwrap_id_ref())
        {
            return false;
        }

        // Describe the panic (location and message), for the panic site marker,
        // preferring the innermost location outside of the standard library
        // (e.g. the indexing expression, instead of the bounds check failure),
        // by tracking the "inlined call frames" (like `spirt_passes::controlflow`).
        let mut current_location = None;
        let mut call_stack = SmallVec::<[_; 8]>::new();
        for inst in &insts[..abort_idx] {
            match (inst.class.opcode, custom_op(inst)) {
                (Op::Line, _) => {
                    current_location = Some((
                        inst.operands[0].unwrap_id_ref(),
                        inst.operands[1].unwrap_literal_bit32(),
                        inst.operands[2].unwrap_literal_bit32(),
                    ));
                }
                (Op::NoLine, _) | (_, Some(CustomOp::ClearDebugSrcLoc)) => {
                    current_location = None;
                }
                (_, Some(CustomOp::SetDebugSrcLoc)) => {
                    let CustomInst::SetDebugSrcLoc {
                        file,
                        line_start,
                        col_start,
                        ..
                    } = CustomInst::decode(inst)
                    else {
                        unreachable!()
                    };
                    current_location = match (
                        self.u32_constants.get(&line_start.unwrap_id_ref()),
                        self.u32_constants.get(&col_start.unwrap_id_ref()),
                    ) {
                        (Some(&line), Some(&col)) => Some((file.unwrap_id_ref(), line, col)),
                        _ => None,
                    };
                }
                (_, Some(CustomOp::PushInlinedCallFrame)) => {
                    call_stack.push(current_location.take());
                }
                (_, Some(CustomOp::PopInlinedCallFrame)) => {
                    current_location = call_stack.pop().flatten();
                }
                _ => {}
            }
        }
        let location = iter::once(current_location)
            .chain(call_stack.into_iter().rev())
            .flatten()
            .filter_map(|(file, line, col)| {
                // FIXME see `spirt_passes::controlflow` for the column
                // numbering convention mismatch.
                Some((self.strings.get(&file)?.as_str(), line, col + 1))
            })
            .reduce(|innermost, (file, line, col)| {
                let is_std = |file: &str| file.contains("/rustlib/src/rust/library/");
                if is_std(innermost.0) && !is_std(file) {
                    (file, line, col)
                } else {
                    innermost
                }
            })
            .map(|(file, line, col)| format!("{file}:{line}:{col}"));
        let message = message_debug_printf
            .first()
            .and_then(|fmt_str| self.strings.get(&fmt_str.unwrap_id_ref()))
            .map(|fmt_str| fmt_str.replace("%%", "%"));
        let description = match (location, message) {
            (Some(location), Some(message)) => format!("{location}: {message}"),
            (Some(location), None) => location,
            (None, Some(message)) => message,
            (None, None) => "<unknown location>".to_string(),
        };
        let marker = if let Some(&marker) = self.panic_site_markers.get(&description) {
            marker
        } else {
            let marker = self.id();
            self.panic_site_markers.insert(description, marker);
            marker
        };
        let call_result_id = self.id();

        let insts = &mut block.instructions;
        insts[abort_idx] = Instruction::new(
            Op::FunctionCall,
            Some(self.op_type_void_id),
            Some(call_result_id),
            vec![Operand::IdRef(marker)],
        );
        *insts.last_mut().unwrap() =
            Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(landing_pad)]);
        true
    }

    /// Returns `true` iff `function` contains any `Abort`s.
    fn function_may_abort(&self, function: &Function) -> bool {
        function.blocks.iter().any(|block| {
            block.instructions.iter().any(|inst| {
                inst.class.opcode == Op::ExtInst
                    && inst.operands[0].unwrap_id_ref() == self.custom_ext_inst_set_import
                    && CustomOp::decode_from_ext_inst(inst) == CustomOp::Abort
            })
        })
    }

    fn add_clone_id_rules(&mut self, rewrite_rules: &mut FxHashMap<Word, Word>, blocks: &[Block]) {
        for block in blocks {
            for inst in block.label.iter().chain(&block.instructions) {
                if let Some(result_id) = inst.result_id {
                    let new_id = self.id();
                    let old = rewrite_rules.insert(result_id, new_id);
                    assert!(old.is_none());
                }
            }
        }
    }

    fn get_inlined_blocks(
        &mut self,
        callee: &Function,
        call_debug_src_loc_inst: Option<&Instruction>,
        mut maybe_call_result_phi: Option<&mut Instruction>,
        return_jump: Word,
    ) -> Vec<Block> {
        let Self {
            custom_ext_inst_set_import,
            op_type_void_id,
            ..
        } = *self;

        // Prepare the debuginfo insts to prepend/append to every block.
        // FIXME(eddyb) this could be more efficient if we only used one pair of
        // `{Push,Pop}InlinedCallFrame` for the whole inlined callee, but there
        // is no way to hint the SPIR-T CFG (re)structurizer that it should keep
        // the entire callee in one region - a SPIR-T inliner wouldn't have this
        // issue, as it would require a fully structured callee.
        let callee_name = self
            .id_to_name
            .get(&callee.def_id().unwrap())
            .copied()
            .unwrap_or("");
        let callee_name_id = *self
            .cached_op_strings
            .entry(callee_name)
            .or_insert_with(|| {
                let id = next_id(self.header);
                self.debug_string_source.push(Instruction::new(
                    Op::String,
                    None,
                    Some(id),
                    vec![Operand::LiteralString(callee_name.to_string())],
                ));
                id
            });
        let mut mk_debuginfo_prefix_and_suffix = || {
            // NOTE(eddyb) `OpExtInst`s have a result ID, even if unused, and
            // it has to be unique (same goes for the other instructions below).
            let instantiate_debuginfo = |this: &mut Self, inst: &Instruction| {
                let mut inst = inst.clone();
                if let Some(id) = &mut inst.result_id {
                    *id = this.id();
                }
                inst
            };
            let custom_inst_to_inst = |this: &mut Self, inst: CustomInst<_>| {
                Instruction::new(
                    Op::ExtInst,
                    Some(op_type_void_id),
                    Some(this.id()),
                    [
                        Operand::IdRef(custom_ext_inst_set_import),
                        Operand::LiteralExtInstInteger(inst.op() as u32),
                    ]
                    .into_iter()
                    .chain(inst.into_operands())
                    .collect(),
                )
            };

            (
                (call_debug_src_loc_inst.map(|inst| instantiate_debuginfo(self, inst)))
                    .into_iter()
                    .chain([custom_inst_to_inst(
                        self,
                        CustomInst::PushInlinedCallFrame {
                            callee_name: Operand::IdRef(callee_name_id),
                        },
                    )]),
                [custom_inst_to_inst(self, CustomInst::PopInlinedCallFrame)],
            )
        };

        let mut blocks = callee.blocks.clone();
        for block in &mut blocks {
            let mut terminator = block.instructions.pop().unwrap();

            // HACK(eddyb) strip trailing debuginfo (as it can't impact terminators).
            while let Some(last) = block.instructions.last() {
                let can_remove = match last.class.opcode {
                    Op::Line | Op::NoLine => true,
                    Op::ExtInst => {
                        last.operands[0].unwrap_id_ref() == custom_ext_inst_set_import
                            && matches!(
                                CustomOp::decode_from_ext_inst(last),
                                CustomOp::SetDebugSrcLoc | CustomOp::ClearDebugSrcLoc
                            )
                    }
                    _ => false,
                };
                if can_remove {
                    block.instructions.pop();
                } else {
                    break;
                }
            }

            if let Op::Return | Op::ReturnValue = terminator.class.opcode {
                if Op::ReturnValue == terminator.class.opcode {
                    let return_value = terminator.operands[0].id_ref_any().unwrap();
                    let call_result_phi = maybe_call_result_phi.as_deref_mut().unwrap();
                    call_result_phi.operands.extend([
                        Operand::IdRef(return_value),
                        Operand::IdRef(block.label_id().unwrap()),
                    ]);
                } else {
                    assert!(maybe_call_result_phi.is_none());
                }
                terminator =
                    Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(return_jump)]);
            }

            let num_phis = block
                .instructions
                .iter()
                .take_while(|inst| inst.class.opcode == Op::Phi)
                .count();

            // HACK(eddyb) avoid adding debuginfo to otherwise-empty blocks.
            if block.instructions.len() > num_phis {
                let (debuginfo_prefix, debuginfo_suffix) = mk_debuginfo_prefix_and_suffix();
                // Insert the prefix debuginfo instructions after `OpPhi`s,
                // which sadly can't be covered by them.
                block
                    .instructions
                    .splice(num_phis..num_phis, debuginfo_prefix);
                // Insert the suffix debuginfo instructions before the terminator,
                // which sadly can't be covered by them.
                block.instructions.extend(debuginfo_suffix);
            }

            block.instructions.push(terminator);
        }

        blocks
    }

    fn insert_opvariables(&self, block: &mut Block, insts: impl IntoIterator<Item = Instruction>) {
        // HACK(eddyb) this isn't as efficient as it could be in theory, but it's
        // very important to make sure sure to never insert new instructions in
        // the middle of debuginfo (as it would be affected by it).
        let mut inlined_frames_depth = 0usize;
        let mut outermost_has_debug_src_loc = false;
        let mut last_debugless_var_insertion_point_candidate = None;
        for (i, inst) in block.instructions.iter().enumerate() {
            last_debugless_var_insertion_point_candidate =
                (inlined_frames_depth == 0 && !outermost_has_debug_src_loc).then_some(i);

            let changed_has_debug_src_loc = match inst.class.opcode {
                Op::Line => true,
                Op::NoLine => false,
                Op::ExtInst
                    if inst.operands[0].unwrap_id_ref() == self.custom_ext_inst_set_import =>
                {
                    match CustomOp::decode_from_ext_inst(inst) {
                        CustomOp::SetDebugSrcLoc => true,
                        CustomOp::ClearDebugSrcLoc => false,
                        CustomOp::PushInlinedCallFrame => {
                            inlined_frames_depth += 1;
                            continue;
                        }
                        CustomOp::PopInlinedCallFrame => {
                            inlined_frames_depth = inlined_frames_depth.saturating_sub(1);
                            continue;
                        }
                        CustomOp::Abort | CustomOp::CallUnwound => break,
                    }
                }
                Op::Variable => continue,
                _ => break,
            };

            if inlined_frames_depth == 0 {
                outermost_has_debug_src_loc = changed_has_debug_src_loc;
            }
        }

        // HACK(eddyb) fallback to inserting at the start, which should be correct.
        // FIXME(eddyb) some level of debuginfo repair could prevent needing this.
        let i = last_debugless_var_insertion_point_candidate.unwrap_or(0);
        block.instructions.splice(i..i, insts);
    }
}

fn fuse_trivial_branches(function: &mut Function) {
    let all_preds = compute_preds(&function.blocks);
    'outer: for (dest_block, mut preds) in all_preds.iter().enumerate() {
        // Don't fuse branches into blocks with `OpPhi`s.
        let any_phis = function.blocks[dest_block]
            .instructions
            .iter()
            .filter(|inst| {
                // These are the only instructions that are allowed before `OpPhi`.
                !matches!(inst.class.opcode, Op::Line | Op::NoLine)
            })
            .take_while(|inst| inst.class.opcode == Op::Phi)
            .next()
            .is_some();
        if any_phis {
            continue;
        }

        // if there's two trivial branches in a row, the middle one might get inlined before the
        // last one, so when processing the last one, skip through to the first one.
        let pred = loop {
            if preds.len() != 1 || preds[0] == dest_block {
                continue 'outer;
            }
            let pred = preds[0];
            if !function.blocks[pred].instructions.is_empty() {
                break pred;
            }
            preds = &all_preds[pred];
        };
        let pred_insts = &function.blocks[pred].instructions;
        if pred_insts.last().unwrap().class.opcode == Op::Branch {
            let mut dest_insts = mem::take(&mut function.blocks[dest_block].instructions);
            let pred_insts = &mut function.blocks[pred].instructions;
            pred_insts.pop(); // pop the branch
            pred_insts.append(&mut dest_insts);

            // Adjust any `OpPhi`s in the targets of the original block, to refer
            // to the sole predecessor (the new source of those CFG edges).
            rewrite_phi_sources(
                function.blocks[dest_block].label_id().unwrap(),
                &mut function.blocks,
                pred,
            );
        }
    }
    function.blocks.retain(|b| !b.instructions.is_empty());
}

fn compute_preds(blocks: &[Block]) -> Vec<Vec<usize>> {
    let mut result = vec![vec![]; blocks.len()];
    for (source_idx, source) in blocks.iter().enumerate() {
        for dest_id in outgoing_edges(source) {
            let dest_idx = blocks
                .iter()
                .position(|b| b.label_id().unwrap() == dest_id)
                .unwrap();
            result[dest_idx].push(source_idx);
        }
    }
    result
}

/// Helper for adjusting `OpPhi` source label IDs, when the terminator of the
/// `original_label_id`-labeled block got moved to `blocks[original_block_idx]`.
fn rewrite_phi_sources(original_label_id: Word, blocks: &mut [Block], new_block_idx: usize) {
    let new_label_id = blocks[new_block_idx].label_id().unwrap();

    // HACK(eddyb) can't keep `blocks` borrowed, the loop needs mutable access.
    let target_ids: SmallVec<[_; 4]> = outgoing_edges(&blocks[new_block_idx]).collect();

    for target_id in target_ids {
        let target_block = blocks
            .iter_mut()
            .find(|b| b.label_id().unwrap() == target_id)
            .unwrap();
        let phis = target_block
            .instructions
            .iter_mut()
            .filter(|inst| {
                // These are the only instructions that are allowed before `OpPhi`.
                !matches!(inst.class.opcode, Op::Line | Op::NoLine)
            })
            .take_while(|inst| inst.class.opcode == Op::Phi);
        for phi in phis {
            for value_and_source_id in phi.operands.chunks_mut(2) {
                let source_id = value_and_source_id[1].id_ref_any_mut().unwrap();
                if *source_id == original_label_id {
                    *source_id = new_label_id;
                    break;
                }
            }
        }
    }
}

/// Replace every `OpPhi` operand pair in `block` that uses `original_source_id`
/// as the source (predecessor) block, with one pair per `new_source_ids` (all
/// using the same value), or remove it, if `new_source_ids` is empty.
fn replace_phi_source(block: &mut Block, original_source_id: Word, new_source_ids: &[Word]) {
    let phis = block
        .instructions
        .iter_mut()
        .filter(|inst| {
            // These are the only instructions that are allowed before `OpPhi`.
            !matches!(inst.class.opcode, Op::Line | Op::NoLine)
        })
        .take_while(|inst| inst.class.opcode == Op::Phi);
    for phi in phis {
        let old_operands = mem::take(&mut phi.operands);
        for value_and_source_id in old_operands.chunks(2) {
            let [value, source_id] = value_and_source_id else {
                unreachable!()
            };
            if source_id.unwrap_id_ref() == original_source_id {
                for &new_source_id in new_source_ids {
                    phi.operands
                        .extend([value.clone(), Operand::IdRef(new_source_id)]);
                }
            } else {
                phi.operands.extend([value.clone(), source_id.clone()]);
            }
        }
    }
}

/// Remove `OpPhi` operand pairs whose source (predecessor) block is missing
/// from `function` (e.g. because it was removed for being unreachable).
fn remove_phi_operands_from_missing_blocks(function: &mut Function) {
    let label_ids: FxHashSet<Word> = function
        .blocks
        .iter()
        .filter_map(|b| b.label_id())
        .collect();
    for block in &mut function.blocks {
        let phis = block
            .instructions
            .iter_mut()
            .filter(|inst| !matches!(inst.class.opcode, Op::Line | Op::NoLine))
            .take_while(|inst| inst.class.opcode == Op::Phi);
        for phi in phis {
            let old_operands = mem::take(&mut phi.operands);
            for value_and_source_id in old_operands.chunks(2) {
                if label_ids.contains(&value_and_source_id[1].unwrap_id_ref()) {
                    phi.operands.extend(value_and_source_id.iter().cloned());
                }
            }
        }
    }
}

/// Define the functions for all the panic site markers (see `Inliner`'s field).
fn define_panic_site_markers(
    module: &mut Module,
    op_type_void_id: Word,
    panic_site_markers: FxIndexMap<String, Word>,
) {
    let header = module.header.as_mut().unwrap();
    let existing_fn_type_id = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeFunction
                && inst.operands[..] == [Operand::IdRef(op_type_void_id)]
        })
        .map(|inst| inst.result_id.unwrap());
    let fn_type_id = existing_fn_type_id.unwrap_or_else(|| {
        let id = next_id(header);
        module.types_global_values.push(Instruction::new(
            Op::TypeFunction,
            None,
            Some(id),
            vec![Operand::IdRef(op_type_void_id)],
        ));
        id
    });
    for (description, marker) in panic_site_markers {
        let label_id = next_id(module.header.as_mut().unwrap());
        let mut function = Function::new();
        function.def = Some(Instruction::new(
            Op::Function,
            Some(op_type_void_id),
            Some(marker),
            vec![
                Operand::FunctionControl(FunctionControl::DONT_INLINE),
                Operand::IdRef(fn_type_id),
            ],
        ));
        function.end = Some(Instruction::new(Op::FunctionEnd, None, None, vec![]));
        function.blocks.push(Block {
            label: Some(Instruction::new(Op::Label, None, Some(label_id), vec![])),
            instructions: vec![Instruction::new(Op::Return, None, None, vec![])],
        });
        module.functions.push(function);
        module.debug_names.push(Instruction::new(
            Op::Name,
            None,
            None,
            vec![
                Operand::IdRef(marker),
                Operand::LiteralString(format!(
                    "{}{description}",
                    super::import_export_link::PANIC_SITE_MARKER_NAME_PREFIX
                )),
            ],
        ));
    }
}
