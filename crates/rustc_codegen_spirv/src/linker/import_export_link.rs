use super::Result;
use crate::custom_decorations::{CustomDecoration, ZombieDecoration};
use rspirv::dr::{Block, Instruction, Module, Operand};
use rspirv::spirv::{Capability, Decoration, FunctionControl, LinkageType, Op, Word};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_session::Session;

pub fn run(opts: &super::Options, sess: &Session, module: &mut Module) -> Result<()> {
    let (rewrite_rules, killed_parameters, unresolved_fn_imports) =
        find_import_export_pairs_and_killed_params(sess, module)?;
    define_unresolved_fn_imports_as_stubs(module, unresolved_fn_imports);
    kill_linkage_instructions(opts, module, &rewrite_rules);
    import_kill_annotations_and_debug(module, &rewrite_rules, &killed_parameters);
    replace_all_uses_with(module, &rewrite_rules);
    Ok(())
}

/// Function imports that have no matching export, but which can be "defined"
/// as stubs (see `define_unresolved_fn_imports_as_late_zombies`), instead of
/// being reported as errors right away (i.e. even if they end up unused).
struct UnresolvedFnImport {
    id: Word,
    name: String,
}

/// Rewrite rules (from imports to exports), killed parameters (of imports),
/// and unresolved function imports (see `UnresolvedFnImport`).
type ImportExportPairs = (FxHashMap<u32, u32>, FxHashSet<u32>, Vec<UnresolvedFnImport>);

fn find_import_export_pairs_and_killed_params(
    sess: &Session,
    module: &Module,
) -> Result<ImportExportPairs> {
    let type_map = get_type_map(module);
    let fn_parameters = fn_parameters(module);

    // Map from name -> (definition, type)
    let mut exports = FxHashMap::default();
    // Rules to rewrite the module with
    let mut rewrite_rules = FxHashMap::default();
    let mut killed_parameters = FxHashSet::default();

    // HACK(eddyb) collect all zombies, and anything that transitively mentions
    // them (or is "infected"), to ignore them when doing type checks, as they
    // will either be removed later, or become an error of their own.
    let mut zombie_infected: FxHashSet<Word> = ZombieDecoration::decode_all(module)
        .map(|(z, _)| z)
        .collect();
    for inst in module.global_inst_iter() {
        if let Some(result_id) = inst.result_id
            && !zombie_infected.contains(&result_id)
        {
            let mut id_operands = inst.operands.iter().filter_map(|o| o.id_ref_any());
            // NOTE(eddyb) this takes advantage of the fact that the module
            // is ordered def-before-use (with the minor exception of forward
            // references for recursive data, which are not fully supported),
            // to be able to propagate "zombie infection" in one pass.
            if id_operands.any(|id| zombie_infected.contains(&id)) {
                zombie_infected.insert(result_id);
            }
        }
    }

    // First, collect all the exports.
    for annotation in &module.annotations {
        let (id, name) = match get_linkage_inst(annotation) {
            Some((id, name, LinkageType::Export)) => (id, name),
            _ => continue,
        };
        let type_id = *type_map.get(&id).expect("Unexpected op");
        if exports.insert(name, (id, type_id)).is_some() {
            return Err(sess
                .dcx()
                .err(format!("Multiple exports found for {name:?}")));
        }
    }
    // Pointer types can't be passed to/returned from functions in general (and
    // forced inlining would remove all calls to stub definitions, see below).
    let pointer_types: FxHashSet<Word> = module
        .types_global_values
        .iter()
        .filter(|inst| inst.class.opcode == Op::TypePointer)
        .filter_map(|inst| inst.result_id)
        .collect();
    let fn_type_has_pointers = |fn_type: Word| {
        module
            .types_global_values
            .iter()
            .find(|inst| inst.class.opcode == Op::TypeFunction && inst.result_id == Some(fn_type))
            .is_none_or(|inst| {
                inst.operands
                    .iter()
                    .any(|op| pointer_types.contains(&op.unwrap_id_ref()))
            })
    };

    let mut any_err = None;
    let mut unresolved_fn_imports = vec![];
    // Then, collect all the imports, and create the rewrite rules.
    for annotation in &module.annotations {
        let (import_id, name) = match get_linkage_inst(annotation) {
            Some((id, name, LinkageType::Import)) => (id, name),
            _ => continue,
        };
        let (export_id, export_type) = match exports.get(name) {
            None => {
                // NOTE unresolved functions are only reported if they
                // remain used after optimizations (like linkers for other targets,
                // which only see what survived optimizations, and which e.g.
                // `no_panic` relies on, to detect reachable panics).
                // This is limited to `-C panic=unwind` (i.e. emulated unwinding),
                // as without unwinding, `no_panic` could never detect any panics,
                // and reporting such errors early (as before) is more conservative.
                let import_type = *type_map.get(&import_id).expect("Unexpected op");
                if sess.panic_strategy().unwinds()
                    && fn_parameters.contains_key(&import_id)
                    && !fn_type_has_pointers(import_type)
                {
                    unresolved_fn_imports.push(UnresolvedFnImport {
                        id: import_id,
                        name: name.to_string(),
                    });
                } else {
                    any_err = Some(sess.dcx().err(format!("Unresolved symbol {name:?}")));
                }
                continue;
            }
            Some(&x) => x,
        };
        let import_type = *type_map.get(&import_id).expect("Unexpected op");
        // Make sure the import/export pair has the same type.
        check_tys_equal(
            sess,
            module,
            name,
            import_type,
            export_type,
            &zombie_infected,
        )?;
        rewrite_rules.insert(import_id, export_id);
        if let Some(params) = fn_parameters.get(&import_id) {
            for &param in params {
                killed_parameters.insert(param);
            }
        }
    }

    match any_err {
        Some(err) => Err(err),
        None => Ok((rewrite_rules, killed_parameters, unresolved_fn_imports)),
    }
}

/// Prefix for the `OpName` of the stub definitions given to unresolved function
/// imports (see `define_unresolved_fn_imports_as_stubs`), followed by the symbol.
const UNRESOLVED_SYMBOL_STUB_NAME_PREFIX: &str = "rust-gpu::unresolved_symbol::";

/// Prefix for the `OpName` of "panic site markers", followed by the location
/// (and message) of a `panic!`, that has been redirected to a landing pad (by
/// the inliner, see `CustomInst::CallUnwound`): the marker is called in place of
/// the panic (i.e. before jumping to the landing pad), and as such survives
/// optimizations exactly when the panic does, which makes the markers useful to
/// explain e.g. `no_panic` errors (see `report_calls_to_unresolved_symbol_stubs`).
///
/// They are always removed afterwards, see `remove_panic_site_markers`.
pub(crate) const PANIC_SITE_MARKER_NAME_PREFIX: &str = "rust-gpu::panic_site::";

/// Give each unresolved function import a stub body (just `OpUnreachable`),
/// and name it (see `UNRESOLVED_SYMBOL_STUB_NAME_PREFIX`), so that an error
/// is only reported if any calls to it remain after all optimizations
/// (see `report_calls_to_unresolved_symbol_stubs`, used post-`spirv-opt`).
///
/// This mimics how linkers for other targets only see the symbols referenced
/// by code that survived optimizations (which e.g. `no_panic` relies on).
fn define_unresolved_fn_imports_as_stubs(
    module: &mut Module,
    unresolved_fn_imports: Vec<UnresolvedFnImport>,
) {
    for UnresolvedFnImport { id, name } in unresolved_fn_imports {
        let header = module.header.as_mut().unwrap();
        let label_id = header.bound;
        header.bound += 1;

        let func = module
            .functions
            .iter_mut()
            .find(|f| f.def_id() == Some(id))
            .unwrap();
        assert!(func.blocks.is_empty());

        // Avoid (non-forced) inlining, which would remove all calls to the stub.
        let def = func.def.as_mut().unwrap();
        let control = def.operands[0].unwrap_function_control();
        def.operands[0] = Operand::FunctionControl(
            (control - FunctionControl::INLINE) | FunctionControl::DONT_INLINE,
        );

        func.blocks.push(Block {
            label: Some(Instruction::new(Op::Label, None, Some(label_id), vec![])),
            instructions: vec![Instruction::new(Op::Unreachable, None, None, vec![])],
        });

        module.debug_names.retain(|inst| {
            !(inst.class.opcode == Op::Name && inst.operands[0].unwrap_id_ref() == id)
        });
        module.debug_names.push(Instruction::new(
            Op::Name,
            None,
            None,
            vec![
                Operand::IdRef(id),
                Operand::LiteralString(format!("{UNRESOLVED_SYMBOL_STUB_NAME_PREFIX}{name}")),
            ],
        ));
    }
}

/// Returns `true` if `module` contains any unresolved symbol stubs
/// (see `define_unresolved_fn_imports_as_stubs`).
pub fn has_unresolved_symbol_stubs(module: &Module) -> bool {
    unresolved_symbol_stubs(module).next().is_some()
}

fn unresolved_symbol_stubs(module: &Module) -> impl Iterator<Item = (Word, &str)> {
    functions_with_name_prefix(module, UNRESOLVED_SYMBOL_STUB_NAME_PREFIX)
}

fn panic_site_markers(module: &Module) -> impl Iterator<Item = (Word, &str)> {
    functions_with_name_prefix(module, PANIC_SITE_MARKER_NAME_PREFIX)
}

fn functions_with_name_prefix<'a>(
    module: &'a Module,
    prefix: &'static str,
) -> impl Iterator<Item = (Word, &'a str)> {
    module.debug_names.iter().filter_map(move |inst| {
        if inst.class.opcode != Op::Name {
            return None;
        }
        let name = inst.operands[1]
            .unwrap_literal_string()
            .strip_prefix(prefix)?;
        Some((inst.operands[0].unwrap_id_ref(), name))
    })
}

/// Remove all panic site markers (see `PANIC_SITE_MARKER_NAME_PREFIX`), and
/// calls to them, returning `true` if there were any.
pub fn remove_panic_site_markers(module: &mut Module) -> bool {
    let markers: FxHashSet<Word> = panic_site_markers(module).map(|(id, _)| id).collect();
    if markers.is_empty() {
        return false;
    }
    module
        .functions
        .retain(|f| !markers.contains(&f.def_id().unwrap()));
    for func in &mut module.functions {
        for block in &mut func.blocks {
            block.instructions.retain(|inst| {
                !(inst.class.opcode == Op::FunctionCall
                    && markers.contains(&inst.operands[0].unwrap_id_ref()))
            });
        }
    }
    module.debug_names.retain(|inst| {
        inst.operands
            .first()
            .and_then(|op| op.id_ref_any())
            .is_none_or(|id| !markers.contains(&id))
    });
    true
}

/// Report an "Unresolved symbol" error for every unresolved symbol stub (see
/// `define_unresolved_fn_imports_as_stubs`) that is still called in `module`
/// (which is expected to have been fully optimized, e.g. by `spirv-opt`).
pub fn report_calls_to_unresolved_symbol_stubs(sess: &Session, module: &Module) -> Result<()> {
    let stubs: FxHashMap<Word, &str> = unresolved_symbol_stubs(module).collect();
    if stubs.is_empty() {
        return Ok(());
    }
    let strings: FxHashMap<Word, &str> = module
        .debug_string_source
        .iter()
        .filter(|inst| inst.class.opcode == Op::String)
        .map(|inst| {
            (
                inst.result_id.unwrap(),
                inst.operands[0].unwrap_literal_string(),
            )
        })
        .collect();

    let markers: FxHashMap<Word, &str> = panic_site_markers(module).collect();

    // Stubs (transitively) called by each function (including the stubs
    // themselves), as calls to them may not have been inlined.
    let mut stubs_reachable_from_fn: FxHashMap<Word, FxHashSet<Word>> = stubs
        .keys()
        .map(|&stub| (stub, [stub].into_iter().collect()))
        .collect();
    loop {
        let mut changed = false;
        for func in &module.functions {
            let func_id = func.def_id().unwrap();
            let callees = func
                .blocks
                .iter()
                .flat_map(|block| &block.instructions)
                .filter(|inst| inst.class.opcode == Op::FunctionCall)
                .map(|inst| inst.operands[0].unwrap_id_ref());
            let mut reachable: FxHashSet<Word> = callees
                .filter_map(|callee| stubs_reachable_from_fn.get(&callee))
                .flatten()
                .copied()
                .collect();
            if let Some(existing) = stubs_reachable_from_fn.get(&func_id) {
                reachable.extend(existing.iter().copied());
                if reachable.len() == existing.len() {
                    continue;
                }
            } else if reachable.is_empty() {
                continue;
            }
            stubs_reachable_from_fn.insert(func_id, reachable);
            changed = true;
        }
        if !changed {
            break;
        }
    }

    // Collect all the call sites (and their `OpLine`, if any), for each stub,
    // and also all the panic site markers, still reachable from each function.
    let mut stub_call_sites: rustc_data_structures::fx::FxIndexMap<Word, Vec<String>> =
        Default::default();
    let mut stub_panic_sites: FxHashMap<Word, rustc_data_structures::fx::FxIndexSet<&str>> =
        Default::default();
    for func in &module.functions {
        // Blocks (by index) calling stubs, and blocks calling panic site markers.
        let mut stubs_called_by_block: FxHashMap<usize, Vec<Word>> = FxHashMap::default();
        let mut panic_sites_by_block: Vec<(usize, &str)> = vec![];
        for (block_idx, block) in func.blocks.iter().enumerate() {
            let mut current_line = None;
            for inst in &block.instructions {
                match inst.class.opcode {
                    Op::Line => {
                        current_line = Some(format!(
                            "{}:{}:{}",
                            strings
                                .get(&inst.operands[0].unwrap_id_ref())
                                .copied()
                                .unwrap_or("<unknown file>"),
                            inst.operands[1].unwrap_literal_bit32(),
                            // FIXME see `spirt_passes::controlflow`
                            // for the column numbering convention mismatch.
                            inst.operands[2].unwrap_literal_bit32() + 1,
                        ));
                    }
                    Op::NoLine => current_line = None,
                    Op::FunctionCall => {
                        let callee = inst.operands[0].unwrap_id_ref();
                        if let Some(&panic_site) = markers.get(&callee) {
                            panic_sites_by_block.push((block_idx, panic_site));
                        }
                        if let Some(reachable_stubs) = stubs_reachable_from_fn.get(&callee) {
                            stubs_called_by_block
                                .entry(block_idx)
                                .or_default()
                                .extend(reachable_stubs.iter().copied());
                        }
                        if stubs.contains_key(&callee) {
                            stub_call_sites.entry(callee).or_default().push(
                                current_line
                                    .clone()
                                    .unwrap_or_else(|| "<unknown location>".to_string()),
                            );
                        }
                    }
                    _ => {}
                }
            }
        }

        // Each panic site is attributed to the stubs reachable from it (e.g.
        // through the landing pad, for `no_panic`'s `Drop`-based detection).
        if !panic_sites_by_block.is_empty() && !stubs_called_by_block.is_empty() {
            let label_to_idx: FxHashMap<Word, usize> = func
                .blocks
                .iter()
                .enumerate()
                .map(|(i, b)| (b.label_id().unwrap(), i))
                .collect();
            for (start, panic_site) in panic_sites_by_block {
                let mut visited = FxHashSet::default();
                let mut queue = vec![start];
                while let Some(b) = queue.pop() {
                    if !visited.insert(b) {
                        continue;
                    }
                    for &stub in stubs_called_by_block.get(&b).into_iter().flatten() {
                        stub_panic_sites.entry(stub).or_default().insert(panic_site);
                    }
                    queue.extend(
                        super::simple_passes::outgoing_edges(&func.blocks[b])
                            .filter_map(|target| label_to_idx.get(&target).copied()),
                    );
                }
            }
        }
    }

    let mut any_err = None;
    for (stub, call_sites) in stub_call_sites {
        let mut err = sess
            .dcx()
            .struct_err(format!("Unresolved symbol {:?}", stubs[&stub]));
        err.note("the symbol is still used after all optimizations");
        // HACK avoid showing too many call sites.
        const MAX_CALL_SITES: usize = 8;
        for call_site in call_sites.iter().take(MAX_CALL_SITES) {
            err.note(format!("used at {call_site}"));
        }
        if call_sites.len() > MAX_CALL_SITES {
            err.note(format!(
                "used at {} more locations",
                call_sites.len() - MAX_CALL_SITES
            ));
        }
        // NOTE these are all the panics that were redirected to landing
        // pads (i.e. unwinding), from which a call to the stub is reachable.
        if let Some(panic_sites) = stub_panic_sites.get(&stub) {
            const MAX_PANIC_SITES: usize = 32;
            for panic_site in panic_sites.iter().take(MAX_PANIC_SITES) {
                err.note(format!("possible panic at {panic_site}"));
            }
            if panic_sites.len() > MAX_PANIC_SITES {
                err.note(format!(
                    "{} more possible panics",
                    panic_sites.len() - MAX_PANIC_SITES
                ));
            }
        }
        any_err = Some(err.emit());
    }
    match any_err {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

fn get_linkage_inst(inst: &Instruction) -> Option<(Word, &str, LinkageType)> {
    if inst.class.opcode == Op::Decorate
        && inst.operands[1].unwrap_decoration() == Decoration::LinkageAttributes
    {
        let id = inst.operands[0].unwrap_id_ref();
        let name = inst.operands[2].unwrap_literal_string();
        let linkage_ty = inst.operands[3].unwrap_linkage_type();
        Some((id, name, linkage_ty))
    } else {
        None
    }
}

fn get_type_map(module: &Module) -> FxHashMap<Word, Word> {
    let vars = module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::Variable)
        .map(|i| (i.result_id.unwrap(), i.result_type.unwrap()));
    let fns = module.functions.iter().map(|i| {
        let d = i.def.as_ref().unwrap();
        (d.result_id.unwrap(), d.operands[1].unwrap_id_ref())
    });
    vars.chain(fns).collect()
}

fn fn_parameters(module: &Module) -> FxHashMap<Word, Vec<Word>> {
    module
        .functions
        .iter()
        .map(|f| {
            let params = f.parameters.iter().map(|i| i.result_id.unwrap()).collect();
            (f.def_id().unwrap(), params)
        })
        .collect()
}

fn check_tys_equal(
    sess: &Session,
    module: &Module,
    name: &str,
    import_type: Word,
    export_type: Word,
    zombie_infected: &FxHashSet<Word>,
) -> Result<()> {
    let allowed = import_type == export_type || {
        // HACK(eddyb) zombies can cause types to differ in definition, due to
        // requiring multiple different instances (usually different `Span`s),
        // so we ignore them, as `find_import_export_pairs_and_killed_params`'s
        // own comment explains (zombies will cause errors or be removed, *later*).
        zombie_infected.contains(&import_type) && zombie_infected.contains(&export_type)
    };

    if allowed {
        Ok(())
    } else {
        // We have an error. It's okay to do something really slow now to report the error.
        use std::fmt::Write;
        let ty_defs = module
            .types_global_values
            .iter()
            .filter_map(|inst| Some((inst.result_id?, inst)))
            .collect();
        fn format_ty(ty_defs: &FxHashMap<Word, &Instruction>, ty: Word, buf: &mut String) {
            match ty_defs.get(&ty) {
                Some(def) => {
                    write!(buf, "({}", def.class.opname).unwrap();
                    if let Some(result_type) = def.result_type {
                        write!(buf, " {result_type}").unwrap();
                    }
                    for op in &def.operands {
                        if let Some(id) = op.id_ref_any() {
                            write!(buf, " ").unwrap();
                            format_ty(ty_defs, id, buf);
                        }
                    }
                    write!(buf, ")").unwrap();
                }
                None => write!(buf, "{ty}").unwrap(),
            }
        }
        fn format_ty_(ty_defs: &FxHashMap<Word, &Instruction>, ty: Word) -> String {
            let mut result = String::new();
            format_ty(ty_defs, ty, &mut result);
            result
        }
        Err(sess
            .dcx()
            .struct_err(format!("Types mismatch for {name:?}"))
            .with_note(format!(
                "import type: {}",
                format_ty_(&ty_defs, import_type)
            ))
            .with_note(format!(
                "export type: {}",
                format_ty_(&ty_defs, export_type)
            ))
            .emit())
    }
}

fn replace_all_uses_with(module: &mut Module, rules: &FxHashMap<u32, u32>) {
    module.all_inst_iter_mut().for_each(|inst| {
        if let Some(result_type) = &mut inst.result_type
            && let Some(&rewrite) = rules.get(result_type)
        {
            *result_type = rewrite;
        }

        inst.operands.iter_mut().for_each(|op| {
            if let Some(w) = op.id_ref_any_mut()
                && let Some(&rewrite) = rules.get(w)
            {
                *w = rewrite;
            }
        });
    });
}

fn kill_linkage_instructions(
    opts: &super::Options,
    module: &mut Module,
    rewrite_rules: &FxHashMap<u32, u32>,
) {
    // drop imported functions
    module
        .functions
        .retain(|f| !rewrite_rules.contains_key(&f.def_id().unwrap()));

    // drop imported variables
    module
        .types_global_values
        .retain(|v| v.result_id.is_none_or(|v| !rewrite_rules.contains_key(&v)));

    // NOTE(eddyb) `Options`'s `keep_link_export`s field requests that `Export`s
    // are left in (primarily for unit testing - see also its doc comment).
    let mut kept_any_linkage_decorations = false;
    module.annotations.retain(|inst| {
        !(inst.class.opcode == Op::Decorate
            && inst.operands[1].unwrap_decoration() == Decoration::LinkageAttributes
            && match inst.operands[3].unwrap_linkage_type() {
                LinkageType::Export if opts.keep_link_exports => {
                    kept_any_linkage_decorations = true;
                    false
                }
                _ => true,
            })
    });
    if !kept_any_linkage_decorations {
        // drop OpCapability Linkage
        module.capabilities.retain(|inst| {
            inst.class.opcode != Op::Capability
                || inst.operands[0].unwrap_capability() != Capability::Linkage
        });
    }
}

fn import_kill_annotations_and_debug(
    module: &mut Module,
    rewrite_rules: &FxHashMap<u32, u32>,
    killed_parameters: &FxHashSet<u32>,
) {
    module.annotations.retain(|inst| {
        inst.operands.is_empty()
            || inst.operands[0].id_ref_any().is_none_or(|id| {
                !rewrite_rules.contains_key(&id) && !killed_parameters.contains(&id)
            })
    });
    module.debug_names.retain(|inst| {
        inst.operands.is_empty()
            || inst.operands[0].id_ref_any().is_none_or(|id| {
                !rewrite_rules.contains_key(&id) && !killed_parameters.contains(&id)
            })
    });
    // need to remove OpGroupDecorate members that mention this id
    for inst in &mut module.annotations {
        if inst.class.opcode == Op::GroupDecorate {
            inst.operands.retain(|op| {
                op.id_ref_any().is_none_or(|id| {
                    !rewrite_rules.contains_key(&id) && !killed_parameters.contains(&id)
                })
            });
        }
    }
}
