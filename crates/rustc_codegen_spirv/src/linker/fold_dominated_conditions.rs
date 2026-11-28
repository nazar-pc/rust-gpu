//! Fold `OpBranchConditional` conditions that are implied by dominating branches.
//!
//! This is meant to run *after* `spirv-opt` (which takes care of e.g. constant
//! folding, CSE, and removing dead code, but doesn't propagate facts learned
//! from conditional branches), and be followed by `spirv-opt` again (to clean
//! up the branches that now have constant conditions).
//!
//! Its main purpose is removing redundant bounds checks (and similar checks,
//! leading to panics), such as these (with `i < 8` checked twice in each):
//! ```ignore
//! if i < a.len() { a[i] } else { 0 }
//! while i < 8 { s += a[i]; i += 1; }
//! for i in 0..8 { s += a[i]; }
//! ```
//! (the latter needs upper bounds, see `Bound`, which also allow folding e.g.
//! `a[i & 3]` and `a[i % 4]`, for `a: [T; 4]`, and seeing through the `Option`
//! returned by `Range::next`, see `FuncFolder::resolve` and `Phi` handling).
//!
//! All facts are about SSA values, and learned from the edges of conditional
//! branches (when the edge is the only way to enter a block), so they hold in
//! all blocks dominated by the target of that edge.

use super::simple_passes::outgoing_edges;
use rspirv::dr::{Function, Instruction, Module, Operand};
use rspirv::spirv::{Op, Word};
use rustc_data_structures::fx::FxHashMap;
use smallvec::SmallVec;
use std::cell::RefCell;
use std::rc::Rc;

/// Like `fold_dominated_conditions`, but for unstructured CFGs (i.e. before
/// structurization, where folded branches can be replaced with `OpBranch`es,
/// and any unreachable blocks removed), repeating until no more changes.
///
/// Returns `true` iff any conditions were folded (i.e. `module` was changed).
pub fn fold_dominated_conditions_unstructured(module: &mut Module) -> bool {
    // HACK the number of rounds is arbitrary, but each round can only
    // expose more opportunities if the previous one removed some code.
    const MAX_ROUNDS: usize = 4;

    let mut any_changes = false;
    for _ in 0..MAX_ROUNDS {
        let folded = find_foldable_conditions_in_module(module);
        if folded.is_empty() {
            break;
        }
        any_changes = true;

        for (func_id, block_idx, value) in folded {
            let func = module
                .functions
                .iter_mut()
                .find(|f| f.def_id() == Some(func_id))
                .unwrap();
            let block_label = func.blocks[block_idx].label_id().unwrap();
            let terminator = func.blocks[block_idx].instructions.last_mut().unwrap();
            assert_eq!(terminator.class.opcode, Op::BranchConditional);
            let [taken, not_taken] =
                if value { [1, 2] } else { [2, 1] }.map(|i| terminator.operands[i].unwrap_id_ref());
            *terminator = Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(taken)]);

            // The edge to `not_taken` was removed, so its `OpPhi`s must not
            // refer to this block anymore (unless the edge to `taken` remains).
            if taken != not_taken {
                let not_taken_block = func
                    .blocks
                    .iter_mut()
                    .find(|b| b.label_id() == Some(not_taken))
                    .unwrap();
                for phi in &mut not_taken_block.instructions {
                    if matches!(phi.class.opcode, Op::Line | Op::NoLine) {
                        continue;
                    }
                    if phi.class.opcode != Op::Phi {
                        break;
                    }
                    let old_operands = std::mem::take(&mut phi.operands);
                    for pair in old_operands.chunks(2) {
                        if pair[1].unwrap_id_ref() != block_label {
                            phi.operands.extend(pair.iter().cloned());
                        }
                    }
                }
            }
        }

        for func in &mut module.functions {
            super::simple_passes::block_ordering_pass(func);
            super::inline::remove_phi_operands_from_missing_blocks(func);
        }
    }
    any_changes
}

/// Returns `true` iff any conditions were folded (i.e. `module` was changed).
pub fn fold_dominated_conditions(module: &mut Module) -> bool {
    let mut globals = Globals::collect(module);
    let Some(bool_type) = globals.bool_type else {
        return false;
    };

    let folded = find_foldable_conditions_in_module_with_globals(module, &globals);
    if folded.is_empty() {
        return false;
    }

    // Get (or create) the `bool` constants needed to replace the conditions.
    let mut get_bool_const = |value: bool| {
        if let Some((&id, _)) = globals.bool_consts.iter().find(|&(_, &v)| v == value) {
            return id;
        }
        let header = module.header.as_mut().unwrap();
        let id = header.bound;
        header.bound += 1;
        module.types_global_values.push(Instruction::new(
            if value {
                Op::ConstantTrue
            } else {
                Op::ConstantFalse
            },
            Some(bool_type),
            Some(id),
            vec![],
        ));
        globals.bool_consts.insert(id, value);
        id
    };
    let [false_id, true_id] = [false, true].map(&mut get_bool_const);

    for (func_id, block_idx, value) in folded {
        let func = module
            .functions
            .iter_mut()
            .find(|f| f.def_id() == Some(func_id))
            .unwrap();
        let terminator = func.blocks[block_idx].instructions.last_mut().unwrap();
        assert_eq!(terminator.class.opcode, Op::BranchConditional);
        terminator.operands[0] = Operand::IdRef(if value { true_id } else { false_id });
    }
    true
}

/// Returns `(func_id, block_idx, value)` for every block (in every function)
/// ending in an `OpBranchConditional` with a condition known to be `value`.
fn find_foldable_conditions_in_module(module: &Module) -> Vec<(Word, usize, bool)> {
    find_foldable_conditions_in_module_with_globals(module, &Globals::collect(module))
}

fn find_foldable_conditions_in_module_with_globals(
    module: &Module,
    globals: &Globals,
) -> Vec<(Word, usize, bool)> {
    let mut folded = vec![];
    for func in &module.functions {
        if let Some(folder) = FuncFolder::new(globals, func) {
            folded.extend(
                folder
                    .find_foldable_conditions()
                    .into_iter()
                    .map(|(block_idx, value)| (func.def_id().unwrap(), block_idx, value)),
            );
        }
    }
    folded
}

#[derive(Default)]
struct Globals {
    int_widths: FxHashMap<Word, u32>,
    bool_type: Option<Word>,
    composite_types: FxHashMap<Word, Instruction>,
    consts: FxHashMap<Word, u64>,
    bool_consts: FxHashMap<Word, bool>,
    const_composites: FxHashMap<Word, SmallVec<[Word; 4]>>,
    global_types: FxHashMap<Word, Word>,
}

impl Globals {
    fn collect(module: &Module) -> Self {
        let mut globals = Globals::default();
        for inst in &module.types_global_values {
            let Some(id) = inst.result_id else {
                continue;
            };
            match inst.class.opcode {
                Op::TypeInt => {
                    globals
                        .int_widths
                        .insert(id, inst.operands[0].unwrap_literal_bit32());
                }
                Op::TypeBool => globals.bool_type = globals.bool_type.or(Some(id)),
                Op::TypeStruct | Op::TypeArray | Op::TypeVector => {
                    globals.composite_types.insert(id, inst.clone());
                }
                Op::Constant => {
                    let value = match inst.operands[..] {
                        [Operand::LiteralBit32(x)] => u64::from(x),
                        [Operand::LiteralBit64(x)] => x,
                        _ => continue,
                    };
                    if globals.int_widths.contains_key(&inst.result_type.unwrap()) {
                        globals.consts.insert(id, value);
                    }
                }
                Op::ConstantTrue | Op::ConstantFalse if inst.result_type == globals.bool_type => {
                    globals
                        .bool_consts
                        .insert(id, inst.class.opcode == Op::ConstantTrue);
                }
                Op::ConstantComposite => {
                    globals.const_composites.insert(
                        id,
                        inst.operands.iter().map(|op| op.unwrap_id_ref()).collect(),
                    );
                }
                _ => {}
            }
            if let Some(ty) = inst.result_type {
                globals.global_types.insert(id, ty);
            }
        }
        globals
    }
}

/// A (scalar) value, as a path of (constant) indices into a composite value
/// (for SSA values of composite types, e.g. `Option<usize>` or `Range<usize>`,
/// which `spirv-opt` doesn't scalarize, when they're not in memory).
#[derive(Clone, PartialEq, Eq, Debug)]
struct Value {
    id: Word,
    path: SmallVec<[u32; 2]>,
}

impl Value {
    fn scalar(id: Word) -> Self {
        Self {
            id,
            path: SmallVec::new(),
        }
    }
}

/// An unsigned comparison, in a canonical form: `lhs < rhs` or `lhs <= rhs`.
#[derive(Clone, PartialEq, Eq)]
struct Comparison {
    lhs: Value,
    rhs: Value,
    strict: bool,
}

/// Known facts: values of `bool` SSA values, as a persistent (shared) list,
/// most recent first (so that the facts of each block extend its dominator's).
#[derive(Clone, Default)]
struct Facts(Option<Rc<FactNode>>);

struct FactNode {
    cond: Word,
    value: bool,
    parent: Facts,
}

impl Facts {
    fn push(&mut self, cond: Word, value: bool) {
        *self = Facts(Some(Rc::new(FactNode {
            cond,
            value,
            parent: self.clone(),
        })));
    }

    /// Iterate over all facts, most recent first.
    fn iter(&self) -> impl Iterator<Item = (Word, bool)> + '_ {
        let mut node = self.0.as_deref();
        std::iter::from_fn(move || {
            let n = node?;
            node = n.parent.0.as_deref();
            Some((n.cond, n.value))
        })
    }
}

/// Limit for recursion through definitions (to bound the cost of queries).
const MAX_DEPTH: u32 = 8;

/// Limit on the number of (most recent) facts used to rule out `OpPhi` edges.
const MAX_FACTS_FOR_EDGE_FEASIBILITY: usize = 32;

struct FuncFolder<'a> {
    globals: &'a Globals,
    func: &'a Function,

    label_to_idx: FxHashMap<Word, usize>,
    idom: Vec<usize>,
    reachable_rpo: Vec<usize>,

    /// Fact learned from the (only) edge entering each block, if any.
    edge_fact: Vec<Option<(Word, bool)>>,

    defs: FxHashMap<Word, &'a Instruction>,
    /// Block (index) that each `OpPhi` is in.
    phi_block: FxHashMap<Word, usize>,

    /// Cache for `facts_at`.
    facts_at_cache: RefCell<FxHashMap<usize, Facts>>,
}

enum Bound {
    Known(u64),
    /// The value is (a copy of) a `Value` already being bounded, further up
    /// (i.e. through an `OpPhi` cycle, which is the same as being unconstrained
    /// by this particular `OpPhi` operand, for the purposes of the bound).
    Cycle,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Dir {
    Max,
    Min,
}

impl<'a> FuncFolder<'a> {
    fn new(globals: &'a Globals, func: &'a Function) -> Option<Self> {
        let blocks = &func.blocks;
        if blocks.len() < 2 {
            return None;
        }

        let label_to_idx: FxHashMap<Word, usize> = blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (b.label_id().unwrap(), i))
            .collect();
        let succs: Vec<Vec<usize>> = blocks
            .iter()
            .map(|b| outgoing_edges(b).map(|id| label_to_idx[&id]).collect())
            .collect();
        let mut preds = vec![vec![]; blocks.len()];
        for (i, s) in succs.iter().enumerate() {
            for &j in s {
                if !preds[j].contains(&i) {
                    preds[j].push(i);
                }
            }
        }

        // Reverse post-order (only reachable blocks), via iterative DFS.
        let mut post_order = Vec::with_capacity(blocks.len());
        {
            let mut visited = vec![false; blocks.len()];
            let mut stack = vec![(0, 0)];
            visited[0] = true;
            while let Some((b, next_succ)) = stack.last_mut() {
                if let Some(&s) = succs[*b].get(*next_succ) {
                    *next_succ += 1;
                    if !visited[s] {
                        visited[s] = true;
                        stack.push((s, 0));
                    }
                } else {
                    post_order.push(*b);
                    stack.pop();
                }
            }
        }
        let reachable_rpo: Vec<usize> = post_order.iter().rev().copied().collect();
        let mut rpo_number = vec![usize::MAX; blocks.len()];
        for (i, &b) in reachable_rpo.iter().enumerate() {
            rpo_number[b] = i;
        }

        // Immediate dominators (Cooper, Harvey & Kennedy, "A Simple, Fast
        // Dominance Algorithm"), indexed by block index.
        let mut idom = vec![usize::MAX; blocks.len()];
        idom[0] = 0;
        let mut changed = true;
        while changed {
            changed = false;
            for &b in reachable_rpo.iter().skip(1) {
                let mut new_idom = usize::MAX;
                for &p in &preds[b] {
                    if idom[p] == usize::MAX {
                        continue;
                    }
                    new_idom = if new_idom == usize::MAX {
                        p
                    } else {
                        let (mut x, mut y) = (p, new_idom);
                        while x != y {
                            while rpo_number[x] > rpo_number[y] {
                                x = idom[x];
                            }
                            while rpo_number[y] > rpo_number[x] {
                                y = idom[y];
                            }
                        }
                        x
                    };
                }
                if idom[b] != new_idom {
                    idom[b] = new_idom;
                    changed = true;
                }
            }
        }

        let edge_fact = (0..blocks.len())
            .map(|b| {
                let [p] = preds[b][..] else {
                    return None;
                };
                let terminator = blocks[p].instructions.last().unwrap();
                if terminator.class.opcode != Op::BranchConditional {
                    return None;
                }
                let cond = terminator.operands[0].unwrap_id_ref();
                let [t, f] = [1, 2].map(|i| label_to_idx[&terminator.operands[i].unwrap_id_ref()]);
                (t != f).then_some((cond, b == t))
            })
            .collect();

        let mut defs = FxHashMap::default();
        let mut phi_block = FxHashMap::default();
        for (b, block) in blocks.iter().enumerate() {
            for inst in &block.instructions {
                if let Some(id) = inst.result_id {
                    defs.insert(id, inst);
                    if inst.class.opcode == Op::Phi {
                        phi_block.insert(id, b);
                    }
                }
            }
        }
        for param in &func.parameters {
            defs.insert(param.result_id.unwrap(), param);
        }

        Some(Self {
            globals,
            func,
            label_to_idx,
            idom,
            reachable_rpo,
            edge_fact,
            defs,
            phi_block,
            facts_at_cache: Default::default(),
        })
    }

    /// Returns the blocks (by index) ending in an `OpBranchConditional` with
    /// a condition that is known to have a specific value, and that value.
    fn find_foldable_conditions(&self) -> Vec<(usize, bool)> {
        let mut foldable = vec![];
        for &b in &self.reachable_rpo {
            let terminator = self.func.blocks[b].instructions.last().unwrap();
            if terminator.class.opcode != Op::BranchConditional {
                continue;
            }
            let cond = terminator.operands[0].unwrap_id_ref();
            if self.globals.bool_consts.contains_key(&cond) {
                continue;
            }
            let facts = self.facts_at(b);
            if let Some(value) = self.eval_bool(&facts, cond, 0) {
                foldable.push((b, value));
            }
        }
        foldable
    }

    /// All the facts that hold in block `b` (from all the edges entering the
    /// blocks dominating `b`, and which are the only way to enter them).
    fn facts_at(&self, b: usize) -> Facts {
        // Find the closest dominator with cached facts (or the entry block),
        // then compute (and cache) the facts for all the blocks in between.
        let mut chain = vec![];
        let mut current = b;
        let mut facts = loop {
            if let Some(facts) = self.facts_at_cache.borrow().get(&current) {
                break facts.clone();
            }
            if current == 0 || self.idom[current] == usize::MAX {
                break Facts::default();
            }
            chain.push(current);
            current = self.idom[current];
        };
        for &block in chain.iter().rev() {
            if let Some((cond, value)) = self.edge_fact[block] {
                self.add_fact(&mut facts, cond, value, 0);
            }
            self.facts_at_cache
                .borrow_mut()
                .insert(block, facts.clone());
        }
        facts
    }

    fn add_fact(&self, facts: &mut Facts, cond: Word, value: bool, depth: u32) {
        facts.push(cond, value);
        if depth >= MAX_DEPTH {
            return;
        }
        let Some(inst) = self.def(cond) else {
            return;
        };
        match (inst.class.opcode, &inst.operands[..], value) {
            (Op::LogicalNot, &[Operand::IdRef(a)], _) => {
                self.add_fact(facts, a, !value, depth + 1);
            }
            (Op::LogicalAnd, &[Operand::IdRef(a), Operand::IdRef(b)], true)
            | (Op::LogicalOr, &[Operand::IdRef(a), Operand::IdRef(b)], false) => {
                self.add_fact(facts, a, value, depth + 1);
                self.add_fact(facts, b, value, depth + 1);
            }
            _ => {}
        }
    }

    fn def(&self, id: Word) -> Option<&'a Instruction> {
        self.defs.get(&id).copied()
    }

    /// Normalize `v` by looking through composite construction/insertion and
    /// extraction (i.e. find the definition of the scalar value it refers to).
    fn resolve(&self, mut v: Value) -> Value {
        // HACK arbitrary limit, just in case.
        for _ in 0..64 {
            if let Some(components) = self.globals.const_composites.get(&v.id)
                && let Some((&i, rest)) = v.path.split_first()
                && let Some(&component) = components.get(i as usize)
            {
                v = Value {
                    id: component,
                    path: rest.into(),
                };
                continue;
            }
            let Some(inst) = self.def(v.id) else {
                break;
            };
            match inst.class.opcode {
                Op::CompositeExtract => {
                    let base = inst.operands[0].unwrap_id_ref();
                    let mut path: SmallVec<[u32; 2]> = inst.operands[1..]
                        .iter()
                        .map(|op| op.unwrap_literal_bit32())
                        .collect();
                    path.extend(v.path.iter().copied());
                    v = Value { id: base, path };
                }
                Op::CompositeConstruct if !v.path.is_empty() => {
                    // NOTE vectors can be constructed from smaller
                    // vectors, which would make component indices not match.
                    let ty = inst.result_type.unwrap();
                    let is_struct_or_array = self
                        .globals
                        .composite_types
                        .get(&ty)
                        .is_some_and(|ty| ty.class.opcode != Op::TypeVector);
                    let Some(component) = inst.operands.get(v.path[0] as usize) else {
                        break;
                    };
                    if !is_struct_or_array {
                        break;
                    }
                    v = Value {
                        id: component.unwrap_id_ref(),
                        path: v.path[1..].into(),
                    };
                }
                Op::CompositeInsert if !v.path.is_empty() => {
                    let object = inst.operands[0].unwrap_id_ref();
                    let composite = inst.operands[1].unwrap_id_ref();
                    let insert_path: SmallVec<[u32; 2]> = inst.operands[2..]
                        .iter()
                        .map(|op| op.unwrap_literal_bit32())
                        .collect();
                    if v.path.starts_with(&insert_path) {
                        v = Value {
                            id: object,
                            path: v.path[insert_path.len()..].into(),
                        };
                    } else if insert_path.iter().zip(&v.path).any(|(a, b)| a != b) {
                        v.id = composite;
                    } else {
                        // Partial overlap (the insert is "inside" `v`).
                        break;
                    }
                }
                _ => break,
            }
        }
        v
    }

    /// Get the scalar type of `v` (i.e. the type of the component it refers to).
    fn type_of(&self, v: &Value) -> Option<Word> {
        let mut ty = match self.def(v.id) {
            Some(inst) => inst.result_type?,
            None => *self.globals.global_types.get(&v.id)?,
        };
        for &i in &v.path {
            let ty_inst = self.globals.composite_types.get(&ty)?;
            ty = match ty_inst.class.opcode {
                Op::TypeStruct => ty_inst.operands.get(i as usize)?.unwrap_id_ref(),
                Op::TypeArray | Op::TypeVector => ty_inst.operands[0].unwrap_id_ref(),
                _ => return None,
            };
        }
        Some(ty)
    }

    fn comparison(&self, id: Word) -> Option<Comparison> {
        let inst = self.def(id)?;
        let [a, b] = match inst.operands[..] {
            [Operand::IdRef(a), Operand::IdRef(b)] => [a, b],
            _ => return None,
        };
        let (lhs, rhs, strict) = match inst.class.opcode {
            Op::ULessThan => (a, b, true),
            Op::ULessThanEqual => (a, b, false),
            Op::UGreaterThan => (b, a, true),
            Op::UGreaterThanEqual => (b, a, false),
            _ => return None,
        };
        Some(Comparison {
            lhs: self.resolve(Value::scalar(lhs)),
            rhs: self.resolve(Value::scalar(rhs)),
            strict,
        })
    }

    /// Returns `false` if the edge from `pred` into `phi_block` (and therefore
    /// the corresponding operands of `OpPhi`s in `phi_block`) is impossible,
    /// given `facts` (which should be facts that hold in `phi_block`).
    fn is_edge_feasible(&self, facts: &Facts, phi_block: usize, pred: usize) -> bool {
        facts
            .iter()
            .take(MAX_FACTS_FOR_EDGE_FEASIBILITY)
            .all(|(cond, value)| {
                self.const_eval_on_edge(Value::scalar(cond), phi_block, pred, 0)
                    != Some(u64::from(!value))
            })
    }

    /// Evaluate `v` to a constant, assuming that `phi_block` was entered from
    /// `pred` (i.e. replacing `phi_block`'s `OpPhi`s with their operands).
    fn const_eval_on_edge(
        &self,
        v: Value,
        phi_block: usize,
        pred: usize,
        depth: u32,
    ) -> Option<u64> {
        if depth >= MAX_DEPTH {
            return None;
        }
        let v = self.resolve(v);
        if v.path.is_empty() {
            if let Some(&value) = self.globals.consts.get(&v.id) {
                return Some(value);
            }
            if let Some(&value) = self.globals.bool_consts.get(&v.id) {
                return Some(u64::from(value));
            }
        }
        if self.phi_block.get(&v.id) == Some(&phi_block) {
            let incoming = self.phi_incoming(v.id).find(|&(_, p)| p == pred)?.0;
            return self.const_eval_on_edge(
                Value {
                    id: incoming,
                    path: v.path,
                },
                phi_block,
                pred,
                depth + 1,
            );
        }
        if !v.path.is_empty() {
            return None;
        }
        let inst = self.def(v.id)?;
        let operand = |i: usize| {
            let id = inst.operands.get(i)?.id_ref_any()?;
            self.const_eval_on_edge(Value::scalar(id), phi_block, pred, depth + 1)
        };
        match inst.class.opcode {
            Op::Bitcast | Op::CopyObject => operand(0),
            Op::LogicalNot => Some(u64::from(operand(0)? == 0)),
            Op::IEqual | Op::LogicalEqual => Some(u64::from(operand(0)? == operand(1)?)),
            Op::INotEqual | Op::LogicalNotEqual => Some(u64::from(operand(0)? != operand(1)?)),
            Op::ULessThan => Some(u64::from(operand(0)? < operand(1)?)),
            Op::ULessThanEqual => Some(u64::from(operand(0)? <= operand(1)?)),
            Op::UGreaterThan => Some(u64::from(operand(0)? > operand(1)?)),
            Op::UGreaterThanEqual => Some(u64::from(operand(0)? >= operand(1)?)),
            Op::LogicalAnd => Some(u64::from(operand(0)? != 0 && operand(1)? != 0)),
            Op::LogicalOr => Some(u64::from(operand(0)? != 0 || operand(1)? != 0)),
            Op::Select => {
                if operand(0)? != 0 {
                    operand(1)
                } else {
                    operand(2)
                }
            }
            _ => None,
        }
    }

    /// The `(value, pred_block_idx)` operand pairs of an `OpPhi`.
    fn phi_incoming(&self, phi: Word) -> impl Iterator<Item = (Word, usize)> + '_ {
        self.defs[&phi].operands.chunks(2).filter_map(|pair| {
            Some((
                pair[0].id_ref_any()?,
                *self.label_to_idx.get(&pair[1].id_ref_any()?)?,
            ))
        })
    }

    fn eval_bool(&self, facts: &Facts, cond: Word, depth: u32) -> Option<bool> {
        if let Some(&value) = self.globals.bool_consts.get(&cond) {
            return Some(value);
        }
        if let Some((_, value)) = facts.iter().find(|&(c, _)| c == cond) {
            return Some(value);
        }
        if depth >= MAX_DEPTH {
            return None;
        }
        let inst = self.def(cond)?;
        match (inst.class.opcode, &inst.operands[..]) {
            (Op::LogicalNot, &[Operand::IdRef(a)]) => {
                return self.eval_bool(facts, a, depth + 1).map(|v| !v);
            }
            (Op::Select, &[Operand::IdRef(c), Operand::IdRef(a), Operand::IdRef(b)]) => {
                return match self.eval_bool(facts, c, depth + 1) {
                    Some(true) => self.eval_bool(facts, a, depth + 1),
                    Some(false) => self.eval_bool(facts, b, depth + 1),
                    None => {
                        let a = self.eval_bool(facts, a, depth + 1)?;
                        (self.eval_bool(facts, b, depth + 1)? == a).then_some(a)
                    }
                };
            }
            (Op::Phi, _) => {
                // All feasible incoming values must agree.
                let phi_block = self.phi_block[&cond];
                let mut result = None;
                for (incoming, pred) in self.phi_incoming(cond) {
                    if !self.is_edge_feasible(facts, phi_block, pred) {
                        continue;
                    }
                    let value = self.eval_bool(&self.facts_at(pred), incoming, depth + 1)?;
                    if result.is_some_and(|r| r != value) {
                        return None;
                    }
                    result = Some(value);
                }
                return result;
            }
            _ => {}
        }

        let cmp = self.comparison(cond)?;

        // Same (or opposite) comparison already known (e.g. a repeated check).
        for (fact, value) in facts.iter() {
            if let Some(known) = self.comparison(fact) {
                if known == cmp {
                    return Some(value);
                }
                // `!(a < b)` is `b <= a` (and `!(a <= b)` is `b < a`).
                if known.lhs == cmp.rhs && known.rhs == cmp.lhs && known.strict != cmp.strict {
                    return Some(!value);
                }
            }
        }

        // Comparison decided by the range of values on either side.
        let bound = |v: &Value, dir| self.bound(facts, v.clone(), dir, depth + 1, &mut vec![]);
        let lhs_max = bound(&cmp.lhs, Dir::Max)?;
        let rhs_min = bound(&cmp.rhs, Dir::Min)?;
        if if cmp.strict {
            lhs_max < rhs_min
        } else {
            lhs_max <= rhs_min
        } {
            return Some(true);
        }
        let lhs_min = bound(&cmp.lhs, Dir::Min)?;
        let rhs_max = bound(&cmp.rhs, Dir::Max)?;
        if if cmp.strict {
            lhs_min >= rhs_max
        } else {
            lhs_min > rhs_max
        } {
            return Some(false);
        }
        None
    }

    /// Upper (`Dir::Max`) or lower (`Dir::Min`) bound (inclusive) of the
    /// unsigned integer value `v` (or `None` if `v` isn't an integer).
    fn bound(
        &self,
        facts: &Facts,
        v: Value,
        dir: Dir,
        depth: u32,
        visiting: &mut Vec<Option<Value>>,
    ) -> Option<u64> {
        match self.bound_or_cycle(facts, v, dir, depth, visiting)? {
            Bound::Known(b) => Some(b),
            Bound::Cycle => None,
        }
    }

    fn bound_or_cycle(
        &self,
        facts: &Facts,
        v: Value,
        dir: Dir,
        depth: u32,
        // NOTE `None` entries are "barriers" (e.g. arithmetic), which
        // invalidate `Bound::Cycle` (as the value isn't a copy anymore).
        visiting: &mut Vec<Option<Value>>,
    ) -> Option<Bound> {
        let v = self.resolve(v);
        if v.path.is_empty()
            && let Some(&value) = self.globals.consts.get(&v.id)
        {
            return Some(Bound::Known(value));
        }
        let width = *self.globals.int_widths.get(&self.type_of(&v)?)?;
        let type_max = if width >= 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        let unconstrained = match dir {
            Dir::Max => type_max,
            Dir::Min => 0,
        };

        if let Some(pos) = visiting.iter().rposition(|w| w.as_ref() == Some(&v)) {
            return Some(if visiting[pos..].contains(&None) {
                Bound::Known(unconstrained)
            } else {
                Bound::Cycle
            });
        }
        if depth >= MAX_DEPTH {
            return Some(Bound::Known(unconstrained));
        }
        let tighten = |a: u64, b: u64| match dir {
            Dir::Max => a.min(b),
            Dir::Min => a.max(b),
        };
        let mut result = unconstrained;

        // `OpPhi`s: combine the bounds of all (feasible) incoming values.
        if let Some(&phi_block) = self.phi_block.get(&v.id) {
            visiting.push(Some(v.clone()));
            let mut combined: Option<u64> = None;
            let mut any_unknown = false;
            for (incoming, pred) in self.phi_incoming(v.id) {
                if !self.is_edge_feasible(facts, phi_block, pred) {
                    continue;
                }
                let incoming = Value {
                    id: incoming,
                    path: v.path.clone(),
                };
                match self.bound_or_cycle(&self.facts_at(pred), incoming, dir, depth + 1, visiting)
                {
                    Some(Bound::Known(b)) => {
                        combined = Some(match (combined, dir) {
                            (None, _) => b,
                            (Some(c), Dir::Max) => c.max(b),
                            (Some(c), Dir::Min) => c.min(b),
                        });
                    }
                    Some(Bound::Cycle) => {}
                    None => any_unknown = true,
                }
            }
            visiting.pop();
            if any_unknown {
                combined = None;
            }
            match combined {
                Some(b) => result = tighten(result, b),
                // Only cycles (or no feasible edges at all).
                None if !any_unknown => return Some(Bound::Cycle),
                None => {}
            }
        } else if v.path.is_empty()
            && let Some(inst) = self.def(v.id)
        {
            // Arithmetic (only for `Dir::Max`, as that's what bounds checks need).
            visiting.push(None);
            let operand = |i: usize| inst.operands.get(i).and_then(|op| op.id_ref_any());
            let mut bound_of = |i: usize, dir| {
                let id = operand(i)?;
                self.bound(facts, Value::scalar(id), dir, depth + 1, visiting)
            };
            let const_operand = |i: usize| operand(i).and_then(|id| self.globals.consts.get(&id));
            let from_def = match (inst.class.opcode, dir) {
                // `a & b <= min(a, b)`
                (Op::BitwiseAnd, Dir::Max) => bound_of(0, Dir::Max).min(bound_of(1, Dir::Max)),
                // `a % b < b` (and `<= a`)
                (Op::UMod, Dir::Max) => bound_of(1, Dir::Max)
                    .map(|b| b.saturating_sub(1))
                    .min(bound_of(0, Dir::Max)),
                (Op::ShiftRightLogical, _) => const_operand(1)
                    .copied()
                    .and_then(|s| Some(bound_of(0, dir)? >> s.min(63))),
                (Op::UDiv, _) => const_operand(1)
                    .copied()
                    .filter(|&d| d != 0)
                    .and_then(|d| Some(bound_of(0, dir)? / d)),
                (Op::UConvert | Op::Bitcast | Op::CopyObject, _)
                    if self.type_of(&Value::scalar(operand(0)?)) == self.type_of(&v)
                        || inst.class.opcode == Op::UConvert =>
                {
                    bound_of(0, dir)
                }
                (Op::Select, Dir::Max) => bound_of(1, Dir::Max).max(bound_of(2, Dir::Max)),
                (Op::Select, Dir::Min) => bound_of(1, Dir::Min).min(bound_of(2, Dir::Min)),
                // NOTE integer arithmetic wraps on overflow, so bounds
                // can only be computed when that can't happen.
                (Op::IAdd | Op::IMul, _) => {
                    let checked_op = |a: u64, b: u64| match inst.class.opcode {
                        Op::IAdd => a.checked_add(b).filter(|&r| r <= type_max),
                        _ => a.checked_mul(b).filter(|&r| r <= type_max),
                    };
                    let (a_max, b_max) = (bound_of(0, Dir::Max), bound_of(1, Dir::Max));
                    match (a_max, b_max) {
                        (Some(a_max), Some(b_max)) => checked_op(a_max, b_max).and_then(|max| {
                            Some(match dir {
                                Dir::Max => max,
                                Dir::Min => {
                                    checked_op(bound_of(0, Dir::Min)?, bound_of(1, Dir::Min)?)?
                                }
                            })
                        }),
                        _ => None,
                    }
                }
                (Op::ISub, _) => {
                    let (a_min, b_max) = (bound_of(0, Dir::Min), bound_of(1, Dir::Max));
                    match (a_min, b_max) {
                        // No underflow possible.
                        (Some(a_min), Some(b_max)) if a_min >= b_max => match dir {
                            Dir::Max => Some(bound_of(0, Dir::Max)? - bound_of(1, Dir::Min)?),
                            Dir::Min => Some(a_min - b_max),
                        },
                        _ => None,
                    }
                }
                _ => None,
            };
            visiting.pop();
            if let Some(from_def) = from_def {
                result = tighten(result, from_def.min(type_max));
            }
        }

        // Known comparisons involving `v`.
        visiting.push(None);
        for (fact, value) in facts.iter() {
            let Some(cmp) = self.comparison(fact) else {
                continue;
            };
            let mut bound_of =
                |x: &Value, dir| self.bound(facts, x.clone(), dir, depth + 1, visiting);
            match dir {
                Dir::Max => {
                    // `v < x` or `v <= x` being true.
                    if cmp.lhs == v
                        && value
                        && let Some(x) = bound_of(&cmp.rhs, Dir::Max)
                    {
                        result = result.min(if cmp.strict { x.saturating_sub(1) } else { x });
                    }
                    // `x < v` or `x <= v` being false, i.e. `v <= x` or `v < x`.
                    if cmp.rhs == v
                        && !value
                        && let Some(x) = bound_of(&cmp.lhs, Dir::Max)
                    {
                        result = result.min(if cmp.strict { x } else { x.saturating_sub(1) });
                    }
                }
                Dir::Min => {
                    // `x < v` or `x <= v` being true.
                    if cmp.rhs == v
                        && value
                        && let Some(x) = bound_of(&cmp.lhs, Dir::Min)
                    {
                        result = result.max(if cmp.strict { x.saturating_add(1) } else { x });
                    }
                    // `v < x` or `v <= x` being false, i.e. `x <= v` or `x < v`.
                    if cmp.lhs == v
                        && !value
                        && let Some(x) = bound_of(&cmp.rhs, Dir::Min)
                    {
                        result = result.max(if cmp.strict { x } else { x.saturating_add(1) });
                    }
                }
            }
        }
        visiting.pop();

        Some(Bound::Known(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run `fold_dominated_conditions_unstructured` on a function (taking two `u32`
    /// parameters, `%x` and `%n`) and return whether its only `OpUnreachable`
    /// block (`%panic`, standing in for a panic) was removed as unreachable.
    #[track_caller]
    fn panic_removed(body: &str) -> bool {
        let src = format!(
            "OpCapability Shader
            OpMemoryModel Logical Simple
            %void = OpTypeVoid
            %bool = OpTypeBool
            %u32 = OpTypeInt 32 0
            %fn_ty = OpTypeFunction %void %u32 %u32
            %u32_0 = OpConstant %u32 0
            %u32_1 = OpConstant %u32 1
            %u32_3 = OpConstant %u32 3
            %u32_4 = OpConstant %u32 4
            %u32_8 = OpConstant %u32 8
            %u32_16 = OpConstant %u32 16
            %u32_17 = OpConstant %u32 17
            %u32_32 = OpConstant %u32 32
            %u32_max_minus_15 = OpConstant %u32 4294967280
            %pair = OpTypeStruct %u32 %u32
            %undef_pair = OpUndef %pair
            %f = OpFunction %void None %fn_ty
            %x = OpFunctionParameter %u32
            %n = OpFunctionParameter %u32
            %entry = OpLabel
            {body}
            %panic = OpLabel
            OpUnreachable
            %exit = OpLabel
            OpReturn
            OpFunctionEnd"
        );
        use spirv_tools::assembler::{self, Assembler as _};
        let spv = assembler::create(None)
            .assemble(&src, assembler::AssemblerOptions::default())
            .unwrap();
        let mut module = crate::link::with_rspirv_loader(|loader| {
            rspirv::binary::parse_bytes(AsRef::<[u8]>::as_ref(&spv), loader)
        })
        .unwrap();

        let count_unreachable = |module: &Module| {
            module.functions[0]
                .blocks
                .iter()
                .filter(|b| b.instructions.last().unwrap().class.opcode == Op::Unreachable)
                .count()
        };
        assert_eq!(count_unreachable(&module), 1);
        fold_dominated_conditions_unstructured(&mut module);
        count_unreachable(&module) == 0
    }

    /// `if x < n { if x < n { .. } else { panic } }`, i.e. a check repeated.
    #[test]
    fn repeated_check() {
        assert!(panic_removed(
            "%c = OpULessThan %bool %x %n
            OpBranchConditional %c %body %exit
            %body = OpLabel
            %c2 = OpULessThan %bool %x %n
            OpBranchConditional %c2 %exit %panic"
        ));
    }

    /// `if x >= n { return } if x < n { .. } else { panic }`.
    #[test]
    fn opposite_check() {
        assert!(panic_removed(
            "%c = OpUGreaterThanEqual %bool %x %n
            OpBranchConditional %c %exit %body
            %body = OpLabel
            %c2 = OpULessThan %bool %x %n
            OpBranchConditional %c2 %exit %panic"
        ));
    }

    /// The check isn't on every path to the second check.
    #[test]
    fn non_dominating_check() {
        assert!(!panic_removed(
            "%c = OpULessThan %bool %x %n
            OpBranchConditional %c %body %join
            %body = OpLabel
            OpBranch %join
            %join = OpLabel
            %c2 = OpULessThan %bool %x %n
            OpBranchConditional %c2 %exit %panic"
        ));
    }

    /// `if x < 4 { x < 8 } else { .. }` (upper bound from a known comparison).
    #[test]
    fn smaller_bound() {
        assert!(panic_removed(
            "%c = OpULessThan %bool %x %u32_4
            OpBranchConditional %c %body %exit
            %body = OpLabel
            %c2 = OpULessThan %bool %x %u32_8
            OpBranchConditional %c2 %exit %panic"
        ));
    }

    /// `if x <= 8 { x < 8 }` is off by one.
    #[test]
    fn off_by_one_bound() {
        assert!(!panic_removed(
            "%c = OpULessThanEqual %bool %x %u32_8
            OpBranchConditional %c %body %exit
            %body = OpLabel
            %c2 = OpULessThan %bool %x %u32_8
            OpBranchConditional %c2 %exit %panic"
        ));
    }

    /// `x & 3 < 4` and `x % 4 < 4` (upper bounds from arithmetic).
    #[test]
    fn arithmetic_bounds() {
        assert!(panic_removed(
            "%masked = OpBitwiseAnd %u32 %x %u32_3
            %c = OpULessThan %bool %masked %u32_4
            OpBranchConditional %c %body %panic
            %body = OpLabel
            %rem = OpUMod %u32 %x %u32_4
            %c2 = OpULessThan %bool %rem %u32_4
            OpBranchConditional %c2 %exit %panic"
        ));
        // `x % 8 < 4` doesn't hold.
        assert!(!panic_removed(
            "%rem = OpUMod %u32 %x %u32_8
            %c = OpULessThan %bool %rem %u32_4
            OpBranchConditional %c %exit %panic"
        ));
    }

    /// `(x % 32) + (u32::MAX - 15)` wraps around (to values below 16) sometimes.
    #[test]
    fn wrapping_add() {
        assert!(!panic_removed(
            "%rem = OpUMod %u32 %x %u32_32
            %sum = OpIAdd %u32 %rem %u32_max_minus_15
            %c = OpULessThan %bool %sum %u32_16
            OpBranchConditional %c %exit %panic"
        ));
        // ... but `(x % 16) + (u32::MAX - 15)` never does (and is always `>= 16`).
        assert!(panic_removed(
            "%rem = OpUMod %u32 %x %u32_16
            %sum = OpIAdd %u32 %rem %u32_max_minus_15
            %c = OpUGreaterThanEqual %bool %sum %u32_16
            OpBranchConditional %c %exit %panic"
        ));
    }

    /// The shape of `for i in 0..end { a[i] }` (with `a: [T; 16]`), using the
    /// `Option<usize>` returned by `Range::next` (as a `(discriminant, value)`
    /// pair, merged by `OpPhi`s), and the loop-invariant `end` of the `Range`.
    fn range_loop(end: &str) -> String {
        format!(
            "%init = OpCompositeConstruct %pair %u32_0 {end}
            OpBranch %header
            %header = OpLabel
            %range = OpPhi %pair %init %entry %range_next %latch
            %start = OpCompositeExtract %u32 %range 0
            %range_end = OpCompositeExtract %u32 %range 1
            %has_next = OpULessThan %bool %start %range_end
            OpBranchConditional %has_next %some %none
            %some = OpLabel
            %start_plus_1 = OpIAdd %u32 %start %u32_1
            %range_some = OpCompositeInsert %pair %start_plus_1 %range 0
            %opt_some = OpCompositeConstruct %pair %u32_1 %start
            OpBranch %join
            %none = OpLabel
            %opt_none = OpCompositeInsert %pair %u32_0 %undef_pair 0
            OpBranch %join
            %join = OpLabel
            %range_next = OpPhi %pair %range_some %some %range %none
            %opt = OpPhi %pair %opt_some %some %opt_none %none
            %discriminant = OpCompositeExtract %u32 %opt 0
            %i = OpCompositeExtract %u32 %opt 1
            %is_some = OpINotEqual %bool %discriminant %u32_0
            OpBranchConditional %is_some %check %exit
            %check = OpLabel
            %in_bounds = OpULessThan %bool %i %u32_16
            OpBranchConditional %in_bounds %latch %panic
            %latch = OpLabel
            OpBranch %header"
        )
    }

    #[test]
    fn range_loop_in_bounds() {
        assert!(panic_removed(&range_loop("%u32_16")));
    }

    #[test]
    fn range_loop_out_of_bounds() {
        assert!(!panic_removed(&range_loop("%u32_17")));
        assert!(!panic_removed(&range_loop("%n")));
    }
}
