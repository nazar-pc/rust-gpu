// HACK(eddyb) avoids rewriting all of the imports (see `lib.rs` and `build.rs`).
use crate::maybe_pqp_cg_ssa as rustc_codegen_ssa;

use super::Builder;
use crate::abi::ConvSpirvType;
use crate::builder_spirv::SpirvValue;
use crate::codegen_cx::CodegenCx;
use crate::spirv_type::SpirvType;
use rspirv::dr;
use rspirv::grammar::{LogicalOperand, OperandKind, OperandQuantifier, reflect};
use rspirv::spirv::{
    CooperativeMatrixOperands, FPFastMathMode, FragmentShadingRate, FunctionControl,
    GroupOperation, ImageOperands, KernelProfilingInfo, LoopControl, MemoryAccess, MemorySemantics,
    Op, RayFlags, SelectionControl, StorageClass, Word,
};
use rustc_abi::{BackendRepr, Primitive};
use rustc_ast::ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_codegen_ssa::mir::operand::OperandValue;
use rustc_codegen_ssa::mir::place::PlaceRef;
use rustc_codegen_ssa::traits::{
    AsmBuilderMethods, BackendTypes, BuilderMethods, InlineAsmOperandRef,
};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_middle::{bug, ty::Instance};
use rustc_span::{DUMMY_SP, Span};
use rustc_target::asm::{InlineAsmRegClass, InlineAsmRegOrRegClass, SpirVInlineAsmRegClass};
use smallvec::SmallVec;

pub struct InstructionTable {
    table: FxHashMap<&'static str, &'static rspirv::grammar::Instruction<'static>>,
}

impl InstructionTable {
    pub fn new() -> Self {
        let table = rspirv::grammar::CoreInstructionTable::iter()
            .map(|inst| (inst.opname, inst))
            .collect();
        Self { table }
    }
}

// HACK(eddyb) `InlineAsmOperandRef` lacks `#[derive(Clone)]`
fn inline_asm_operand_ref_clone<'tcx, B: BackendTypes + ?Sized>(
    operand: &InlineAsmOperandRef<'tcx, B>,
) -> InlineAsmOperandRef<'tcx, B> {
    use InlineAsmOperandRef::*;

    match operand {
        &In { reg, value } => In { reg, value },
        &Out { reg, late, place } => Out { reg, late, place },
        &InOut {
            reg,
            late,
            in_value,
            out_place,
        } => InOut {
            reg,
            late,
            in_value,
            out_place,
        },
        Const { string } => Const {
            string: string.clone(),
        },
        &SymFn { instance } => SymFn { instance },
        &SymStatic { def_id } => SymStatic { def_id },
        &Label { label } => Label { label },
    }
}

impl<'a, 'tcx> AsmBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    /* Example asm and the template it compiles to:
    asm!(
        "mov {0}, {1}",
        "add {0}, {number}",
        out(reg) o,
        in(reg) i,
        number = const 5,
    );
    [
    String("mov "),
    Placeholder { operand_idx: 0, modifier: None, span: src/lib.rs:19:18: 19:21 (#0) },
    String(", "),
    Placeholder { operand_idx: 1, modifier: None, span: src/lib.rs:19:23: 19:26 (#0) },
    String("\n"),
    String("add "),
    Placeholder { operand_idx: 0, modifier: None, span: src/lib.rs:20:18: 20:21 (#0) },
    String(", "),
    Placeholder { operand_idx: 2, modifier: None, span: src/lib.rs:20:23: 20:31 (#0) }
    ]
     */

    fn codegen_inline_asm(
        &mut self,
        template: &[InlineAsmTemplatePiece],
        operands: &[InlineAsmOperandRef<'tcx, Self>],
        options: InlineAsmOptions,
        _line_spans: &[Span],
        _instance: Instance<'_>,
        _dest: Option<Self::BasicBlock>,
        _catch_funclet: Option<(Self::BasicBlock, Option<&Self::Funclet>)>,
    ) {
        const SUPPORTED_OPTIONS: InlineAsmOptions = InlineAsmOptions::NORETURN;
        let unsupported_options = options & !SUPPORTED_OPTIONS;
        if !unsupported_options.is_empty() {
            self.err(format!("asm flags not supported: {unsupported_options:?}"));
        }

        // HACK(eddyb) get more accurate pointers types, for pointer operands,
        // from the Rust types available in their respective `OperandRef`s.
        let mut operands: SmallVec<[_; 8]> =
            operands.iter().map(inline_asm_operand_ref_clone).collect();
        for operand in &mut operands {
            let (in_value, out_place) = match operand {
                InlineAsmOperandRef::In { value, .. } => (Some(value), None),
                InlineAsmOperandRef::InOut {
                    in_value,
                    out_place,
                    ..
                } => (Some(in_value), out_place.as_mut()),
                InlineAsmOperandRef::Out { place, .. } => (None, place.as_mut()),

                InlineAsmOperandRef::Const { .. }
                | InlineAsmOperandRef::SymFn { .. }
                | InlineAsmOperandRef::SymStatic { .. }
                | InlineAsmOperandRef::Label { .. } => (None, None),
            };

            if let Some(in_value) = in_value
                && let (BackendRepr::Scalar(scalar), OperandValue::Immediate(in_value_spv)) =
                    (in_value.layout.backend_repr, &mut in_value.val)
                && let Primitive::Pointer(_) = scalar.primitive()
            {
                let in_value_precise_type = in_value.layout.spirv_type(self.span(), self);
                *in_value_spv = self.pointercast(*in_value_spv, in_value_precise_type);
            }
            if let Some(out_place) = out_place {
                let out_place_precise_type = out_place.layout.spirv_type(self.span(), self);
                let out_place_precise_ptr_type = self.type_ptr_to(out_place_precise_type);
                out_place.val.llval =
                    self.pointercast(out_place.val.llval, out_place_precise_ptr_type);
            }
        }

        // vec of lines, and each line is vec of tokens
        let mut tokens = vec![vec![]];
        for piece in template {
            match piece {
                InlineAsmTemplatePiece::String(asm) => {
                    // We cannot use str::lines() here because we don't want the behavior of "the
                    // last newline is optional", we want an empty string for the last line if
                    // there is no newline terminator.
                    // Lambda copied from std LinesAnyMap
                    let lines = asm.split('\n').map(|line| {
                        let l = line.len();
                        if l > 0 && line.as_bytes()[l - 1] == b'\r' {
                            &line[0..l - 1]
                        } else {
                            line
                        }
                    });
                    for (index, line) in lines.enumerate() {
                        if index != 0 {
                            // There was a newline, add a new line.
                            tokens.push(vec![]);
                        }
                        let mut chars = line.chars();
                        while let Some(token) = self.lex_word(&mut chars) {
                            tokens.last_mut().unwrap().push(token);
                        }
                    }
                }
                &InlineAsmTemplatePiece::Placeholder {
                    operand_idx,
                    modifier,
                    span,
                } => {
                    if let Some(modifier) = modifier {
                        self.tcx
                            .dcx()
                            .span_err(span, format!("asm modifiers are not supported: {modifier}"));
                    }
                    let line = tokens.last_mut().unwrap();
                    let typeof_kind = line.last().and_then(|prev| match prev {
                        Token::Word("typeof") => Some(TypeofKind::Plain),
                        Token::Word("typeof*") => Some(TypeofKind::Dereference),
                        _ => None,
                    });
                    match typeof_kind {
                        Some(kind) => {
                            *line.last_mut().unwrap() =
                                Token::Typeof(&operands[operand_idx], span, kind);
                        }
                        None => match &operands[operand_idx] {
                            InlineAsmOperandRef::Const { string } => line.push(Token::Word(string)),
                            item => line.push(Token::Placeholder(item, span)),
                        },
                    }
                }
            }
        }

        let mut id_map = FxHashMap::default();
        let mut defined_ids = FxHashSet::default();
        let mut id_to_type_map = FxHashMap::default();
        for operand in &operands {
            if let InlineAsmOperandRef::In { reg: _, value } = operand {
                let value = value.immediate();
                id_to_type_map.insert(value.def(self), value.ty);
            }
        }

        let mut asm_block = AsmBlock::Open;
        for line in tokens {
            self.codegen_asm(
                &mut id_map,
                &mut defined_ids,
                &mut id_to_type_map,
                &mut asm_block,
                line.into_iter(),
            );
        }

        match (options.contains(InlineAsmOptions::NORETURN), asm_block) {
            (true, AsmBlock::Open) => {
                self.err("`noreturn` requires a terminator at the end");
            }
            (true, AsmBlock::End(_)) => {
                // `noreturn` appends an `OpUnreachable` after the asm block.
                // This requires starting a new block for this.
                let label = self.emit().id();
                self.emit()
                    .insert_into_block(
                        dr::InsertPoint::End,
                        dr::Instruction::new(Op::Label, None, Some(label), vec![]),
                    )
                    .unwrap();
            }
            (false, AsmBlock::Open) => (),
            (false, AsmBlock::End(terminator)) => {
                self.err(format!(
                    "trailing terminator `Op{terminator:?}` requires `options(noreturn)`"
                ));
            }
        }
        for (id, num) in id_map {
            if !defined_ids.contains(&num) {
                self.err(format!("%{id} is used but not defined"));
            }
        }
    }
}

enum TypeofKind {
    Plain,
    Dereference,
}

enum Token<'a, 'cx, 'tcx> {
    Word(&'a str),
    String(String),
    Placeholder(&'a InlineAsmOperandRef<'tcx, Builder<'cx, 'tcx>>, Span),
    Typeof(
        &'a InlineAsmOperandRef<'tcx, Builder<'cx, 'tcx>>,
        Span,
        TypeofKind,
    ),
}

enum OutRegister<'tcx> {
    Regular(Word),
    Place(PlaceRef<'tcx, SpirvValue>),
}

enum AsmBlock {
    Open,
    End(Op),
}

impl<'cx, 'tcx> Builder<'cx, 'tcx> {
    fn lex_word<'a>(&self, line: &mut std::str::Chars<'a>) -> Option<Token<'a, 'cx, 'tcx>> {
        loop {
            let start = line.as_str();
            match line.next()? {
                // skip over leading whitespace
                ch if ch.is_whitespace() => {}
                // lex a string
                '"' => {
                    let mut cooked = String::new();
                    loop {
                        match line.next() {
                            None => {
                                self.err("Unterminated string in instruction");
                                return None;
                            }
                            Some('"') => break,
                            Some('\\') => {
                                let escape = match line.next() {
                                    None => {
                                        self.err("Unterminated string in instruction");
                                        return None;
                                    }
                                    Some('n') => '\n',
                                    Some('r') => '\r',
                                    Some('t') => '\t',
                                    Some('0') => '\0',
                                    Some('\\') => '\\',
                                    Some('\'') => '\'',
                                    Some('"') => '"',
                                    Some(escape) => {
                                        self.err(format!("invalid escape '\\{escape}'"));
                                        return None;
                                    }
                                };
                                cooked.push(escape);
                            }
                            Some(ch) => {
                                cooked.push(ch);
                            }
                        }
                    }
                    break Some(Token::String(cooked));
                }
                // lex a word
                _ => {
                    let end = loop {
                        let end = line.as_str();
                        match line.next() {
                            Some(ch) if !ch.is_whitespace() => {}
                            _ => break end,
                        }
                    };
                    let word = &start[..(start.len() - end.len())];
                    break Some(Token::Word(word));
                }
            }
        }
    }

    fn insert_inst(
        &mut self,
        id_map: &mut FxHashMap<&str, Word>,
        defined_ids: &mut FxHashSet<Word>,
        asm_block: &mut AsmBlock,
        inst: dr::Instruction,
    ) {
        // Types declared must be registered in our type system.
        let new_result_id = match inst.class.opcode {
            Op::TypeVoid => SpirvType::Void.def(self.span(), self),
            Op::TypeBool => SpirvType::Bool.def(self.span(), self),
            Op::TypeInt => SpirvType::Integer(
                inst.operands[0].unwrap_literal_bit32(),
                inst.operands[1].unwrap_literal_bit32() != 0,
            )
            .def(self.span(), self),
            Op::TypeFloat => {
                SpirvType::Float(inst.operands[0].unwrap_literal_bit32()).def(self.span(), self)
            }
            Op::TypeStruct => {
                self.err("OpTypeStruct in asm! is not supported yet");
                return;
            }
            Op::TypeVector => SpirvType::Vector {
                element: inst.operands[0].unwrap_id_ref(),
                count: inst.operands[1].unwrap_literal_bit32(),
            }
            .def(self.span(), self),
            Op::TypeMatrix => SpirvType::Matrix {
                element: inst.operands[0].unwrap_id_ref(),
                count: inst.operands[1].unwrap_literal_bit32(),
            }
            .def(self.span(), self),
            Op::TypeArray => {
                self.err("OpTypeArray in asm! is not supported yet");
                return;
            }
            Op::TypeRuntimeArray => SpirvType::RuntimeArray {
                element: inst.operands[0].unwrap_id_ref(),
            }
            .def(self.span(), self),
            Op::TypePointer => {
                let storage_class = inst.operands[0].unwrap_storage_class();
                if storage_class != StorageClass::Generic {
                    self.struct_err("TypePointer in asm! requires `Generic` storage class")
                        .with_note(format!(
                            "`{storage_class:?}` storage class was specified"
                        ))
                        .with_help(format!(
                            "the storage class will be inferred automatically (e.g. to `{storage_class:?}`)"
                        ))
                        .emit();
                }
                SpirvType::Pointer {
                    pointee: inst.operands[1].unwrap_id_ref(),
                }
                .def(self.span(), self)
            }
            Op::TypeImage => SpirvType::Image {
                sampled_type: inst.operands[0].unwrap_id_ref(),
                dim: inst.operands[1].unwrap_dim(),
                depth: inst.operands[2].unwrap_literal_bit32(),
                arrayed: inst.operands[3].unwrap_literal_bit32(),
                multisampled: inst.operands[4].unwrap_literal_bit32(),
                sampled: inst.operands[5].unwrap_literal_bit32(),
                image_format: inst.operands[6].unwrap_image_format(),
            }
            .def(self.span(), self),
            Op::TypeSampledImage => SpirvType::SampledImage {
                image_type: inst.operands[0].unwrap_id_ref(),
            }
            .def(self.span(), self),
            Op::TypeSampler => SpirvType::Sampler.def(self.span(), self),
            Op::TypeAccelerationStructureKHR => {
                SpirvType::AccelerationStructureKhr.def(self.span(), self)
            }
            Op::TypeRayQueryKHR => SpirvType::RayQueryKhr.def(self.span(), self),
            Op::Variable => {
                // OpVariable with Function storage class should be emitted inside the function,
                // however, all other OpVariables should appear in the global scope instead.
                if inst.operands[0].unwrap_storage_class() == StorageClass::Function {
                    let mut builder = self.emit();
                    builder.select_block(Some(0)).unwrap();
                    builder
                        .insert_into_block(dr::InsertPoint::Begin, inst)
                        .unwrap();
                } else {
                    self.emit_global()
                        .insert_types_global_values(dr::InsertPoint::End, inst);
                }
                return;
            }

            op => {
                // NOTE(eddyb) allowing the instruction to be added below avoids
                // spurious "`noreturn` requires a terminator at the end" errors.
                if let Op::Return | Op::ReturnValue = op {
                    self.struct_err(format!(
                        "using `Op{op:?}` to return from within `asm!` is disallowed"
                    ))
                    .with_note(
                        "resuming execution, without falling through the end \
                        of the `asm!` block, is always undefined behavior",
                    )
                    .emit();
                }

                self.emit()
                    .insert_into_block(dr::InsertPoint::End, inst)
                    .unwrap();

                *asm_block = match *asm_block {
                    AsmBlock::Open => {
                        if reflect::is_block_terminator(op) {
                            AsmBlock::End(op)
                        } else {
                            AsmBlock::Open
                        }
                    }
                    AsmBlock::End(terminator) => {
                        if op != Op::Label {
                            self.err(format!(
                                "expected `OpLabel` after terminator `Op{terminator:?}`"
                            ));
                        }

                        AsmBlock::Open
                    }
                };

                return;
            }
        };
        for value in id_map.values_mut() {
            if *value == inst.result_id.unwrap() {
                *value = new_result_id;
            }
        }
        if defined_ids.remove(&inst.result_id.unwrap()) {
            // Note this may be a duplicate insert, if the type was deduplicated.
            defined_ids.insert(new_result_id);
        }
    }

    fn codegen_asm<'a>(
        &mut self,
        id_map: &mut FxHashMap<&'a str, Word>,
        defined_ids: &mut FxHashSet<Word>,
        id_to_type_map: &mut FxHashMap<Word, Word>,
        asm_block: &mut AsmBlock,
        mut tokens: impl Iterator<Item = Token<'a, 'cx, 'tcx>>,
    ) where
        'cx: 'a,
        'tcx: 'a,
    {
        let mut first_token = match tokens.next() {
            Some(tok) => tok,
            None => return,
        };
        // Parse result_id in front of instruction:
        // %z = OpAdd %ty %x %y
        let out_register = if match first_token {
            Token::Placeholder(_, _) => true,
            Token::Word(id_str) if id_str.starts_with('%') => true,
            Token::Word(_) | Token::String(_) | Token::Typeof(_, _, _) => false,
        } {
            let result_id = match self.parse_id_out(id_map, defined_ids, first_token) {
                Some(result_id) => result_id,
                None => return,
            };
            if let Some(Token::Word("=")) = tokens.next() {
            } else {
                self.err("expected equals after result id specifier");
                return;
            }
            first_token = if let Some(tok) = tokens.next() {
                tok
            } else {
                self.err("expected instruction after equals");
                return;
            };
            Some(result_id)
        } else {
            None
        };
        let inst_name = match first_token {
            Token::Word(inst_name) => inst_name,
            Token::String(_) => {
                self.err("cannot use a string as an instruction");
                return;
            }
            Token::Placeholder(_, span) | Token::Typeof(_, span, _) => {
                self.tcx
                    .dcx()
                    .span_err(span, "cannot use a dynamic value as an instruction type");
                return;
            }
        };
        let inst_class = inst_name
            .strip_prefix("Op")
            .and_then(|n| self.cx.instruction_table.table.get(n));
        let inst_class = if let Some(inst) = inst_class {
            inst
        } else {
            self.err(format!("unknown spirv instruction {inst_name}"));
            return;
        };
        let result_id = match out_register {
            Some(OutRegister::Regular(reg)) => Some(reg),
            Some(OutRegister::Place(_)) => Some(self.emit().id()),
            None => None,
        };
        let mut instruction = dr::Instruction {
            class: inst_class,
            result_type: None,
            result_id,
            operands: vec![],
        };
        self.parse_operands(id_map, id_to_type_map, tokens, &mut instruction);
        if let Some(result_type) = instruction.result_type {
            id_to_type_map.insert(instruction.result_id.unwrap(), result_type);
        }
        self.insert_inst(id_map, defined_ids, asm_block, instruction);
        if let Some(OutRegister::Place(place)) = out_register {
            self.emit()
                .store(
                    place.val.llval.def(self),
                    result_id.unwrap(),
                    None,
                    std::iter::empty(),
                )
                .unwrap();
        }
    }

    fn parse_operands<'a>(
        &mut self,
        id_map: &mut FxHashMap<&'a str, Word>,
        id_to_type_map: &FxHashMap<Word, Word>,
        mut tokens: impl Iterator<Item = Token<'a, 'cx, 'tcx>>,
        instruction: &mut dr::Instruction,
    ) where
        'cx: 'a,
        'tcx: 'a,
    {
        let mut saw_id_result = false;
        let mut need_result_type_infer = false;

        let mut logical_operand_stack = instruction
            .class
            .operands
            .iter()
            .cloned()
            .collect::<std::collections::VecDeque<_>>();

        while let Some(LogicalOperand { kind, quantifier }) = logical_operand_stack.pop_front() {
            if kind == OperandKind::IdResult {
                assert_eq!(quantifier, OperandQuantifier::One);
                if instruction.result_id.is_none() {
                    self.err(format!(
                        "instruction {} expects a result id",
                        instruction.class.opname
                    ));
                }
                saw_id_result = true;
                continue;
            }

            if kind == OperandKind::IdResultType {
                assert_eq!(quantifier, OperandQuantifier::One);
                if let Some(token) = tokens.next() {
                    if let Token::Word("_") = token {
                        need_result_type_infer = true;
                    } else if let Some(id) = self.parse_id_in(id_map, token) {
                        instruction.result_type = Some(id);
                    }
                } else {
                    self.err(format!(
                        "instruction {} expects a result type",
                        instruction.class.opname
                    ));
                }
                continue;
            }

            let operands_start = instruction.operands.len();

            match quantifier {
                OperandQuantifier::One => {
                    if !self.parse_one_operand(id_map, instruction, kind, &mut tokens) {
                        self.err(format!(
                            "expected operand after instruction: {}",
                            instruction.class.opname
                        ));
                        return;
                    }
                }
                OperandQuantifier::ZeroOrOne => {
                    let _ = self.parse_one_operand(id_map, instruction, kind, &mut tokens);
                    // If this return false, well, it's optional, do nothing
                }
                OperandQuantifier::ZeroOrMore => {
                    while self.parse_one_operand(id_map, instruction, kind, &mut tokens) {}
                }
            }

            // Parsed operands can add more optional operands that need to be parsed
            // to an instruction - so push then on the stack here, after parsing
            for op in instruction.operands[operands_start..].iter() {
                logical_operand_stack.extend(op.additional_operands());
            }
        }

        if !saw_id_result && instruction.result_id.is_some() {
            self.err(format!(
                "instruction {} does not expect a result id",
                instruction.class.opname
            ));
        }
        if tokens.next().is_some() {
            self.tcx.dcx().err(format!(
                "too many operands to instruction: {}",
                instruction.class.opname
            ));
        }

        if need_result_type_infer {
            assert!(instruction.result_type.is_none());

            match self.infer_result_type(id_to_type_map, instruction) {
                Some(result_type) => instruction.result_type = Some(result_type),
                None => self.err(format!(
                    "instruction {} cannot have its result type inferred",
                    instruction.class.opname
                )),
            }
        }
    }

    fn infer_result_type(
        &self,
        id_to_type_map: &FxHashMap<Word, Word>,
        instruction: &dr::Instruction,
    ) -> Option<Word> {
        use crate::spirv_type_constraints::{InstSig, TyListPat, TyPat, instruction_signatures};

        #[derive(Debug)]
        struct Unapplicable;

        /// Recursively match `ty` against `pat`, returning one of:
        /// * `Ok([None])`: `pat` matched but contained no type variables
        /// * `Ok([Some(var)])`: `pat` matched and `var` is the type variable
        /// * `Err(Mismatch)`: `pat` didn't match or isn't supported right now
        fn match_ty_pat(
            cx: &CodegenCx<'_>,
            pat: &TyPat<'_>,
            ty: Word,
        ) -> Result<[Option<Word>; 1], Unapplicable> {
            match pat {
                TyPat::Any => Ok([None]),
                &TyPat::T => Ok([Some(ty)]),
                TyPat::Either(a, b) => {
                    match_ty_pat(cx, a, ty).or_else(|Unapplicable| match_ty_pat(cx, b, ty))
                }
                _ => match (pat, cx.lookup_type(ty)) {
                    (TyPat::Any | &TyPat::T | TyPat::Either(..), _) => unreachable!(),

                    (TyPat::Void, SpirvType::Void) => Ok([None]),
                    (TyPat::Pointer(_, pat), SpirvType::Pointer { pointee: ty, .. })
                    | (TyPat::Vector(pat), SpirvType::Vector { element: ty, .. })
                    | (
                        TyPat::Vector4(pat),
                        SpirvType::Vector {
                            element: ty,
                            count: 4,
                        },
                    )
                    | (
                        TyPat::Image(pat),
                        SpirvType::Image {
                            sampled_type: ty, ..
                        },
                    )
                    | (TyPat::SampledImage(pat), SpirvType::SampledImage { image_type: ty }) => {
                        match_ty_pat(cx, pat, ty)
                    }
                    _ => Err(Unapplicable),
                },
            }
        }

        #[derive(Debug)]
        struct Ambiguous;

        /// Construct a type from `pat`, replacing `TyPat::Var(i)` with `ty_vars[i]`.
        /// `leftover_operands` is used for `IndexComposite` patterns, if any exist.
        /// If the pattern isn't constraining enough to determine an unique type,
        /// `Err(Ambiguous)` is returned instead.
        fn subst_ty_pat(
            cx: &CodegenCx<'_>,
            pat: &TyPat<'_>,
            ty_vars: &[Option<Word>],
            leftover_operands: &[dr::Operand],
        ) -> Result<Word, Ambiguous> {
            Ok(match pat {
                &TyPat::Var(i) => match ty_vars.get(i) {
                    Some(&Some(ty)) => ty,
                    _ => return Err(Ambiguous),
                },

                TyPat::Pointer(_, pat) => SpirvType::Pointer {
                    pointee: subst_ty_pat(cx, pat, ty_vars, leftover_operands)?,
                }
                .def(DUMMY_SP, cx),

                TyPat::Vector4(pat) => SpirvType::Vector {
                    element: subst_ty_pat(cx, pat, ty_vars, leftover_operands)?,
                    count: 4,
                }
                .def(DUMMY_SP, cx),

                TyPat::SampledImage(pat) => SpirvType::SampledImage {
                    image_type: subst_ty_pat(cx, pat, ty_vars, leftover_operands)?,
                }
                .def(DUMMY_SP, cx),

                TyPat::IndexComposite(pat) => {
                    let mut ty = subst_ty_pat(cx, pat, ty_vars, leftover_operands)?;
                    for index in leftover_operands {
                        let index_to_usize = || match *index {
                            // FIXME(eddyb) support more than just literals,
                            // by looking up `IdRef`s as constant integers.
                            dr::Operand::LiteralBit32(i) => usize::try_from(i).ok(),

                            _ => None,
                        };
                        ty = match cx.lookup_type(ty) {
                            SpirvType::Array { element, .. }
                            | SpirvType::RuntimeArray { element }
                            // HACK(eddyb) this is pretty bad because it's not
                            // checking that the index is an `OpConstant 0`, but
                            // there's no other valid choice anyway.
                            | SpirvType::InterfaceBlock { inner_type: element } => element,

                            SpirvType::Adt { field_types, .. } => *index_to_usize()
                                .and_then(|i| field_types.get(i))
                                .ok_or(Ambiguous)?,

                            // FIXME(eddyb) support more than just arrays and structs.
                            _ => return Err(Ambiguous),
                        };
                    }
                    ty
                }

                _ => return Err(Ambiguous),
            })
        }

        // FIXME(eddyb) try multiple signatures until one fits.
        let mut sig = match instruction_signatures(instruction.class.opcode)? {
            [
                sig @ InstSig {
                    output_type: Some(_),
                    ..
                },
            ] => *sig,
            _ => return None,
        };

        let mut combined_ty_vars = [None];

        let mut operands = instruction.operands.iter();
        let mut next_id_operand = || operands.find_map(|o| o.id_ref_any());
        while let TyListPat::Cons { first: pat, suffix } = *sig.input_types {
            sig.input_types = suffix;

            let match_result = match id_to_type_map.get(&next_id_operand()?) {
                Some(&ty) => match_ty_pat(self, pat, ty),

                // Non-value ID operand (or value operand of unknown type),
                // only `TyPat::Any` is valid.
                None => match pat {
                    TyPat::Any => Ok([None]),
                    _ => Err(Unapplicable),
                },
            };

            let ty_vars = match match_result {
                Ok(ty_vars) => ty_vars,
                Err(Unapplicable) => return None,
            };

            for (&var, combined_var) in ty_vars.iter().zip(&mut combined_ty_vars) {
                if let Some(var) = var {
                    match *combined_var {
                        Some(combined_var) => {
                            // FIXME(eddyb) this could use some error reporting
                            // (it's a type mismatch), although we could also
                            // just use the first type and let validation take
                            // care of the mismatch
                            if var != combined_var {
                                return None;
                            }
                        }
                        None => *combined_var = Some(var),
                    }
                }
            }
        }
        match sig.input_types {
            TyListPat::Cons { .. } => unreachable!(),

            TyListPat::Any => {}
            TyListPat::Nil => {
                if next_id_operand().is_some() {
                    return None;
                }
            }
            _ => return None,
        }

        // HACK(eddyb) clippy false positive, `.ok()` loses information.
        #[allow(clippy::manual_ok_err)]
        match subst_ty_pat(
            self,
            sig.output_type.unwrap(),
            &combined_ty_vars,
            operands.as_slice(),
        ) {
            Ok(ty) => Some(ty),
            Err(Ambiguous) => None,
        }
    }

    fn check_reg(&mut self, span: Span, reg: &InlineAsmRegOrRegClass) {
        match reg {
            InlineAsmRegOrRegClass::RegClass(InlineAsmRegClass::SpirV(
                SpirVInlineAsmRegClass::reg,
            )) => {}
            _ => {
                self.tcx
                    .dcx()
                    .span_err(span, format!("invalid register: {reg}"));
            }
        }
    }

    fn parse_id_out<'a>(
        &mut self,
        id_map: &mut FxHashMap<&'a str, Word>,
        defined_ids: &mut FxHashSet<Word>,
        token: Token<'a, 'cx, 'tcx>,
    ) -> Option<OutRegister<'tcx>> {
        match token {
            Token::Word(word) => {
                if let Some(id) = word.strip_prefix('%') {
                    Some(OutRegister::Regular({
                        let num = *id_map.entry(id).or_insert_with(|| self.emit().id());
                        if !defined_ids.insert(num) {
                            self.err(format!("%{id} is defined more than once"));
                        }
                        num
                    }))
                } else {
                    self.err("expected ID");
                    None
                }
            }
            Token::String(_) => {
                self.err("expected ID, not string");
                None
            }
            Token::Typeof(_, span, _) => {
                self.tcx
                    .dcx()
                    .span_err(span, "cannot assign to a typeof expression");
                None
            }
            Token::Placeholder(hole, span) => match hole {
                InlineAsmOperandRef::In { reg, value: _ } => {
                    self.check_reg(span, reg);
                    self.tcx
                        .dcx()
                        .span_err(span, "in register cannot be assigned to");
                    None
                }
                InlineAsmOperandRef::Out {
                    reg,
                    late: _,
                    place,
                } => {
                    self.check_reg(span, reg);
                    if let Some(place) = place {
                        Some(OutRegister::Place(*place))
                    } else {
                        self.tcx.dcx().span_err(span, "missing place for register");
                        None
                    }
                }
                InlineAsmOperandRef::InOut {
                    reg,
                    late: _,
                    in_value: _,
                    out_place,
                } => {
                    self.check_reg(span, reg);
                    if let Some(out_place) = out_place {
                        Some(OutRegister::Place(*out_place))
                    } else {
                        self.tcx.dcx().span_err(span, "missing place for register");
                        None
                    }
                }
                InlineAsmOperandRef::Const { string: _ } => {
                    self.tcx
                        .dcx()
                        .span_err(span, "cannot write to const asm argument");
                    None
                }
                InlineAsmOperandRef::SymFn { instance: _ } => {
                    self.tcx
                        .dcx()
                        .span_err(span, "cannot write to function asm argument");
                    None
                }
                InlineAsmOperandRef::SymStatic { def_id: _ } => {
                    self.tcx
                        .dcx()
                        .span_err(span, "cannot write to static variable asm argument");
                    None
                }
                InlineAsmOperandRef::Label { label: _ } => {
                    self.tcx
                        .dcx()
                        .span_err(span, "cannot write to label asm argument");
                    None
                }
            },
        }
    }

    fn parse_id_in<'a>(
        &mut self,
        id_map: &mut FxHashMap<&'a str, Word>,
        token: Token<'a, 'cx, 'tcx>,
    ) -> Option<Word> {
        match token {
            Token::Word(word) => {
                if let Some(id) = word.strip_prefix('%') {
                    Some(*id_map.entry(id).or_insert_with(|| self.emit().id()))
                } else {
                    self.err("expected ID");
                    None
                }
            }
            Token::String(_) => {
                self.err("expected ID, not string");
                None
            }
            Token::Typeof(hole, span, kind) => match hole {
                InlineAsmOperandRef::In { reg, value } => {
                    self.check_reg(span, reg);
                    let ty = value.immediate().ty;
                    Some(match kind {
                        TypeofKind::Plain => ty,
                        TypeofKind::Dereference => match self.lookup_type(ty) {
                            SpirvType::Pointer { pointee } => pointee,
                            other => {
                                self.tcx.dcx().span_err(
                                    span,
                                    format!(
                                        "cannot use typeof* on non-pointer type: {}",
                                        other.debug(ty, self)
                                    ),
                                );
                                ty
                            }
                        },
                    })
                }
                InlineAsmOperandRef::Out {
                    reg,
                    late: _,
                    place,
                } => {
                    self.check_reg(span, reg);
                    if let Some(place) = place {
                        match self.lookup_type(place.val.llval.ty) {
                            SpirvType::Pointer { pointee } => Some(pointee),
                            other => {
                                self.tcx.dcx().span_err(
                                    span,
                                    format!(
                                        "out register type not pointer: {}",
                                        other.debug(place.val.llval.ty, self)
                                    ),
                                );
                                None
                            }
                        }
                    } else {
                        self.tcx
                            .dcx()
                            .span_err(span, "missing place for out register typeof");
                        None
                    }
                }
                InlineAsmOperandRef::InOut {
                    reg,
                    late: _,
                    in_value,
                    out_place: _,
                } => {
                    self.check_reg(span, reg);
                    Some(in_value.immediate().ty)
                }
                InlineAsmOperandRef::Const { string: _ } => {
                    self.tcx
                        .dcx()
                        .span_err(span, "cannot take the type of a const asm argument");
                    None
                }
                InlineAsmOperandRef::SymFn { instance: _ } => {
                    self.tcx
                        .dcx()
                        .span_err(span, "cannot take the type of a function asm argument");
                    None
                }
                InlineAsmOperandRef::SymStatic { def_id: _ } => {
                    self.tcx.dcx().span_err(
                        span,
                        "cannot take the type of a static variable asm argument",
                    );
                    None
                }
                InlineAsmOperandRef::Label { label: _ } => {
                    self.tcx
                        .dcx()
                        .span_err(span, "cannot take the type of a label asm argument");
                    None
                }
            },
            Token::Placeholder(hole, span) => match hole {
                InlineAsmOperandRef::In { reg, value } => {
                    self.check_reg(span, reg);
                    Some(value.immediate().def(self))
                }
                InlineAsmOperandRef::Out {
                    reg,
                    late: _,
                    place: _,
                } => {
                    self.check_reg(span, reg);
                    self.tcx
                        .dcx()
                        .span_err(span, "out register cannot be used as a value");
                    None
                }
                InlineAsmOperandRef::InOut {
                    reg,
                    late: _,
                    in_value,
                    out_place: _,
                } => {
                    self.check_reg(span, reg);
                    Some(in_value.immediate().def(self))
                }
                InlineAsmOperandRef::Const { string: _ } => {
                    self.tcx
                        .dcx()
                        .span_err(span, "const asm argument not supported yet");
                    None
                }
                InlineAsmOperandRef::SymFn { instance: _ } => {
                    self.tcx
                        .dcx()
                        .span_err(span, "function asm argument not supported yet");
                    None
                }
                InlineAsmOperandRef::SymStatic { def_id: _ } => {
                    self.tcx
                        .dcx()
                        .span_err(span, "static variable asm argument not supported yet");
                    None
                }
                InlineAsmOperandRef::Label { label: _ } => {
                    self.tcx
                        .dcx()
                        .span_err(span, "label asm argument not supported yet");
                    None
                }
            },
        }
    }

    fn parse_one_operand<'a>(
        &mut self,
        id_map: &mut FxHashMap<&'a str, Word>,
        inst: &mut dr::Instruction,
        kind: OperandKind,
        tokens: &mut impl Iterator<Item = Token<'a, 'cx, 'tcx>>,
    ) -> bool
    where
        'cx: 'a,
        'tcx: 'a,
    {
        let token = match tokens.next() {
            Some(tok) => tok,
            None => return false,
        };
        let word = match token {
            Token::Word(word) => Some(word),
            Token::String(_) | Token::Placeholder(_, _) | Token::Typeof(_, _, _) => None,
        };
        match (kind, word) {
            (OperandKind::IdResultType | OperandKind::IdResult, _) => {
                bug!("should be handled by parse_operands");
            }
            (OperandKind::IdMemorySemantics, _) => {
                if let Some(id) = self.parse_id_in(id_map, token) {
                    inst.operands.push(dr::Operand::IdMemorySemantics(id));
                }
            }
            (OperandKind::IdScope, _) => {
                if let Some(id) = self.parse_id_in(id_map, token) {
                    inst.operands.push(dr::Operand::IdScope(id));
                }
            }
            (OperandKind::IdRef, _) => {
                if let Some(id) = self.parse_id_in(id_map, token) {
                    inst.operands.push(dr::Operand::IdRef(id));
                }
            }

            (OperandKind::LiteralInteger, Some(word)) => match word.parse() {
                Ok(v) => inst.operands.push(dr::Operand::LiteralBit32(v)),
                Err(e) => self.err(format!("invalid integer: {e}")),
            },
            (OperandKind::LiteralFloat, Some(word)) => match word.parse() {
                Ok(v) => inst
                    .operands
                    .push(dr::Operand::LiteralBit32(f32::to_bits(v))),
                Err(e) => self.err(format!("invalid float: {e}")),
            },
            (OperandKind::LiteralString, _) => {
                if let Token::String(value) = token {
                    inst.operands.push(dr::Operand::LiteralString(value));
                }
            }
            (OperandKind::LiteralContextDependentNumber, Some(word)) => {
                assert!(matches!(inst.class.opcode, Op::Constant | Op::SpecConstant));
                let ty = inst.result_type.unwrap();
                fn parse(ty: SpirvType<'_>, w: &str) -> Result<dr::Operand, String> {
                    fn fmt(x: impl ToString) -> String {
                        x.to_string()
                    }
                    Ok(match ty {
                        SpirvType::Integer(8, false) => {
                            dr::Operand::LiteralBit32(w.parse::<u8>().map_err(fmt)? as u32)
                        }
                        SpirvType::Integer(16, false) => {
                            dr::Operand::LiteralBit32(w.parse::<u16>().map_err(fmt)? as u32)
                        }
                        SpirvType::Integer(32, false) => {
                            dr::Operand::LiteralBit32(w.parse::<u32>().map_err(fmt)?)
                        }
                        SpirvType::Integer(64, false) => {
                            dr::Operand::LiteralBit64(w.parse::<u64>().map_err(fmt)?)
                        }
                        SpirvType::Integer(8, true) => {
                            dr::Operand::LiteralBit32(w.parse::<i8>().map_err(fmt)? as i32 as u32)
                        }
                        SpirvType::Integer(16, true) => {
                            dr::Operand::LiteralBit32(w.parse::<i16>().map_err(fmt)? as i32 as u32)
                        }
                        SpirvType::Integer(32, true) => {
                            dr::Operand::LiteralBit32(w.parse::<i32>().map_err(fmt)? as u32)
                        }
                        SpirvType::Integer(64, true) => {
                            dr::Operand::LiteralBit64(w.parse::<i64>().map_err(fmt)? as u64)
                        }
                        SpirvType::Float(32) => {
                            dr::Operand::LiteralBit32(w.parse::<f32>().map_err(fmt)?.to_bits())
                        }
                        SpirvType::Float(64) => {
                            dr::Operand::LiteralBit64(w.parse::<f64>().map_err(fmt)?.to_bits())
                        }
                        _ => return Err("expected number literal in OpConstant".to_string()),
                    })
                }
                match parse(self.lookup_type(ty), word) {
                    Ok(op) => inst.operands.push(op),
                    Err(err) => self.err(err),
                }
            }
            (OperandKind::LiteralExtInstInteger, Some(word)) => match word.parse() {
                Ok(v) => inst.operands.push(dr::Operand::LiteralExtInstInteger(v)),
                Err(e) => self.err(format!("invalid integer: {e}")),
            },
            (OperandKind::LiteralSpecConstantOpInteger, Some(word)) => {
                match self.instruction_table.table.get(word) {
                    Some(v) => {
                        inst.operands
                            .push(dr::Operand::LiteralSpecConstantOpInteger(v.opcode));
                    }
                    None => self.err("invalid instruction in OpSpecConstantOp"),
                }
            }
            (OperandKind::PairLiteralIntegerIdRef, _) => {
                self.err("PairLiteralIntegerIdRef not supported yet");
            }
            (OperandKind::PairIdRefLiteralInteger, _) => {
                if let Some(id) = self.parse_id_in(id_map, token) {
                    inst.operands.push(dr::Operand::IdRef(id));
                    match tokens.next() {
                        Some(Token::Word(word)) => match word.parse() {
                            Ok(v) => inst.operands.push(dr::Operand::LiteralBit32(v)),
                            Err(e) => {
                                self.err(format!("invalid integer: {e}"));
                            }
                        },
                        Some(Token::String(_)) => {
                            self.err(format!("expected a literal, not a string for a {kind:?}"));
                        }
                        Some(Token::Placeholder(_, span)) => {
                            self.tcx.dcx().span_err(
                                span,
                                format!("expected a literal, not a dynamic value for a {kind:?}"),
                            );
                        }
                        Some(Token::Typeof(_, span, _)) => {
                            self.tcx.dcx().span_err(
                                span,
                                format!("expected a literal, not a type for a {kind:?}"),
                            );
                        }
                        None => {
                            self.err("expected operand after instruction");
                        }
                    }
                }
            }
            (OperandKind::PairIdRefIdRef, _) => {
                if let Some(id) = self.parse_id_in(id_map, token) {
                    inst.operands.push(dr::Operand::IdRef(id));
                    match tokens.next() {
                        Some(token) => {
                            if let Some(id) = self.parse_id_in(id_map, token) {
                                inst.operands.push(dr::Operand::IdRef(id));
                            }
                        }
                        None => self.err("expected operand after instruction"),
                    }
                }
            }

            (OperandKind::ImageOperands, Some(word)) => {
                match parse_bitflags_operand(IMAGE_OPERANDS, word) {
                    Some(x) => inst.operands.push(dr::Operand::ImageOperands(x)),
                    None => self.err(format!("Unknown ImageOperands {word}")),
                }
            }
            (OperandKind::FPFastMathMode, Some(word)) => {
                match parse_bitflags_operand(FP_FAST_MATH_MODE, word) {
                    Some(x) => inst.operands.push(dr::Operand::FPFastMathMode(x)),
                    None => self.err(format!("Unknown FPFastMathMode {word}")),
                }
            }
            (OperandKind::SelectionControl, Some(word)) => {
                match parse_bitflags_operand(SELECTION_CONTROL, word) {
                    Some(x) => inst.operands.push(dr::Operand::SelectionControl(x)),
                    None => self.err(format!("Unknown SelectionControl {word}")),
                }
            }
            (OperandKind::LoopControl, Some(word)) => {
                match parse_bitflags_operand(LOOP_CONTROL, word) {
                    Some(x) => inst.operands.push(dr::Operand::LoopControl(x)),
                    None => self.err(format!("Unknown LoopControl {word}")),
                }
            }
            (OperandKind::FunctionControl, Some(word)) => {
                match parse_bitflags_operand(FUNCTION_CONTROL, word) {
                    Some(x) => inst.operands.push(dr::Operand::FunctionControl(x)),
                    None => self.err(format!("Unknown FunctionControl {word}")),
                }
            }
            (OperandKind::MemorySemantics, Some(word)) => {
                match parse_bitflags_operand(MEMORY_SEMANTICS, word) {
                    Some(x) => inst.operands.push(dr::Operand::MemorySemantics(x)),
                    None => self.err(format!("Unknown MemorySemantics {word}")),
                }
            }
            (OperandKind::MemoryAccess, Some(word)) => {
                match parse_bitflags_operand(MEMORY_ACCESS, word) {
                    Some(x) => inst.operands.push(dr::Operand::MemoryAccess(x)),
                    None => self.err(format!("Unknown MemoryAccess {word}")),
                }
            }
            (OperandKind::KernelProfilingInfo, Some(word)) => {
                match parse_bitflags_operand(KERNEL_PROFILING_INFO, word) {
                    Some(x) => inst.operands.push(dr::Operand::KernelProfilingInfo(x)),
                    None => self.err(format!("Unknown KernelProfilingInfo {word}")),
                }
            }
            (OperandKind::RayFlags, Some(word)) => match parse_bitflags_operand(RAY_FLAGS, word) {
                Some(x) => inst.operands.push(dr::Operand::RayFlags(x)),
                None => self.err(format!("Unknown RayFlags {word}")),
            },
            (OperandKind::FragmentShadingRate, Some(word)) => {
                match parse_bitflags_operand(FRAGMENT_SHADING_RATE, word) {
                    Some(x) => inst.operands.push(dr::Operand::FragmentShadingRate(x)),
                    None => self.err(format!("Unknown FragmentShadingRate {word}")),
                }
            }

            (OperandKind::SourceLanguage, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::SourceLanguage(x)),
                Err(()) => self.err(format!("Unknown SourceLanguage {word}")),
            },
            (OperandKind::ExecutionModel, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::ExecutionModel(x)),
                Err(()) => self.err(format!("unknown ExecutionModel {word}")),
            },
            (OperandKind::AddressingModel, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::AddressingModel(x)),
                Err(()) => self.err(format!("unknown AddressingModel {word}")),
            },
            (OperandKind::MemoryModel, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::MemoryModel(x)),
                Err(()) => self.err(format!("unknown MemoryModel {word}")),
            },
            (OperandKind::ExecutionMode, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::ExecutionMode(x)),
                Err(()) => self.err(format!("unknown ExecutionMode {word}")),
            },
            (OperandKind::StorageClass, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::StorageClass(x)),
                Err(()) => self.err(format!("unknown StorageClass {word}")),
            },
            (OperandKind::Dim, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::Dim(x)),
                Err(()) => self.err(format!("unknown Dim {word}")),
            },
            (OperandKind::SamplerAddressingMode, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::SamplerAddressingMode(x)),
                Err(()) => self.err(format!("unknown SamplerAddressingMode {word}")),
            },
            (OperandKind::SamplerFilterMode, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::SamplerFilterMode(x)),
                Err(()) => self.err(format!("unknown SamplerFilterMode {word}")),
            },
            (OperandKind::ImageFormat, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::ImageFormat(x)),
                Err(()) => self.err(format!("unknown ImageFormat {word}")),
            },
            (OperandKind::ImageChannelOrder, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::ImageChannelOrder(x)),
                Err(()) => self.err(format!("unknown ImageChannelOrder {word}")),
            },
            (OperandKind::ImageChannelDataType, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::ImageChannelDataType(x)),
                Err(()) => self.err(format!("unknown ImageChannelDataType {word}")),
            },
            (OperandKind::FPRoundingMode, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::FPRoundingMode(x)),
                Err(()) => self.err(format!("unknown FPRoundingMode {word}")),
            },
            (OperandKind::LinkageType, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::LinkageType(x)),
                Err(()) => self.err(format!("unknown LinkageType {word}")),
            },
            (OperandKind::AccessQualifier, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::AccessQualifier(x)),
                Err(()) => self.err(format!("unknown AccessQualifier {word}")),
            },
            (OperandKind::FunctionParameterAttribute, Some(word)) => match word.parse() {
                Ok(x) => inst
                    .operands
                    .push(dr::Operand::FunctionParameterAttribute(x)),
                Err(()) => self.err(format!("unknown FunctionParameterAttribute {word}")),
            },
            (OperandKind::Decoration, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::Decoration(x)),
                Err(()) => self.err(format!("unknown Decoration {word}")),
            },
            (OperandKind::BuiltIn, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::BuiltIn(x)),
                Err(()) => self.err(format!("unknown BuiltIn {word}")),
            },
            (OperandKind::Scope, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::Scope(x)),
                Err(()) => self.err(format!("unknown Scope {word}")),
            },
            (OperandKind::GroupOperation, Some(word)) => {
                match word.parse::<u32>().ok().and_then(GroupOperation::from_u32) {
                    Some(id) => inst.operands.push(dr::Operand::GroupOperation(id)),
                    None => match word.parse() {
                        Ok(x) => inst.operands.push(dr::Operand::GroupOperation(x)),
                        Err(()) => self.err(format!("unknown GroupOperation {word}")),
                    },
                }
            }
            (OperandKind::KernelEnqueueFlags, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::KernelEnqueueFlags(x)),
                Err(()) => self.err(format!("unknown KernelEnqueueFlags {word}")),
            },
            (OperandKind::Capability, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::Capability(x)),
                Err(()) => self.err(format!("unknown Capability {word}")),
            },
            (OperandKind::RayQueryIntersection, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::RayQueryIntersection(x)),
                Err(()) => self.err(format!("unknown RayQueryIntersection {word}")),
            },
            (OperandKind::RayQueryCommittedIntersectionType, Some(word)) => match word.parse() {
                Ok(x) => inst
                    .operands
                    .push(dr::Operand::RayQueryCommittedIntersectionType(x)),
                Err(()) => self.err(format!("unknown RayQueryCommittedIntersectionType {word}")),
            },
            (OperandKind::RayQueryCandidateIntersectionType, Some(word)) => match word.parse() {
                Ok(x) => inst
                    .operands
                    .push(dr::Operand::RayQueryCandidateIntersectionType(x)),
                Err(()) => self.err(format!("unknown RayQueryCandidateIntersectionType {word}")),
            },
            (OperandKind::FPDenormMode, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::FPDenormMode(x)),
                Err(()) => self.err(format!("unknown FPDenormMode {word}")),
            },
            (OperandKind::QuantizationModes, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::QuantizationModes(x)),
                Err(()) => self.err(format!("unknown QuantizationModes {word}")),
            },
            (OperandKind::FPOperationMode, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::FPOperationMode(x)),
                Err(()) => self.err(format!("unknown FPOperationMode {word}")),
            },
            (OperandKind::OverflowModes, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::OverflowModes(x)),
                Err(()) => self.err(format!("unknown OverflowModes {word}")),
            },
            (OperandKind::PackedVectorFormat, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::PackedVectorFormat(x)),
                Err(()) => self.err(format!("unknown PackedVectorFormat {word}")),
            },
            (OperandKind::HostAccessQualifier, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::HostAccessQualifier(x)),
                Err(()) => self.err(format!("unknown HostAccessQualifier {word}")),
            },
            (OperandKind::CooperativeMatrixOperands, Some(word)) => {
                match parse_bitflags_operand(COOPERATIVE_MATRIX_OPERANDS, word) {
                    Some(x) => inst
                        .operands
                        .push(dr::Operand::CooperativeMatrixOperands(x)),
                    None => self.err(format!("Unknown CooperativeMatrixOperands {word}")),
                }
            }
            (OperandKind::CooperativeMatrixLayout, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::CooperativeMatrixLayout(x)),
                Err(()) => self.err(format!("unknown CooperativeMatrixLayout {word}")),
            },
            (OperandKind::CooperativeMatrixUse, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::CooperativeMatrixUse(x)),
                Err(()) => self.err(format!("unknown CooperativeMatrixUse {word}")),
            },
            (OperandKind::InitializationModeQualifier, Some(word)) => match word.parse() {
                Ok(x) => inst
                    .operands
                    .push(dr::Operand::InitializationModeQualifier(x)),
                Err(()) => self.err(format!("unknown InitializationModeQualifier {word}")),
            },
            (OperandKind::LoadCacheControl, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::LoadCacheControl(x)),
                Err(()) => self.err(format!("unknown LoadCacheControl {word}")),
            },
            (OperandKind::StoreCacheControl, Some(word)) => match word.parse() {
                Ok(x) => inst.operands.push(dr::Operand::StoreCacheControl(x)),
                Err(()) => self.err(format!("unknown StoreCacheControl {word}")),
            },
            (kind, None) => match token {
                Token::Word(_) => bug!(),
                Token::String(_) => {
                    self.err(format!("expected a literal, not a string for a {kind:?}"));
                }
                Token::Placeholder(_, span) => {
                    self.tcx.dcx().span_err(
                        span,
                        format!("expected a literal, not a dynamic value for a {kind:?}"),
                    );
                }
                Token::Typeof(_, span, _) => {
                    self.tcx.dcx().span_err(
                        span,
                        format!("expected a literal, not a type for a {kind:?}"),
                    );
                }
            },
        }
        true
    }
}

pub const IMAGE_OPERANDS: &[(&str, ImageOperands)] = &[
    ("None", ImageOperands::NONE),
    ("Bias", ImageOperands::BIAS),
    ("Lod", ImageOperands::LOD),
    ("Grad", ImageOperands::GRAD),
    ("ConstOffset", ImageOperands::CONST_OFFSET),
    ("Offset", ImageOperands::OFFSET),
    ("ConstOffsets", ImageOperands::CONST_OFFSETS),
    ("Sample", ImageOperands::SAMPLE),
    ("MinLod", ImageOperands::MIN_LOD),
    ("MakeTexelAvailable", ImageOperands::MAKE_TEXEL_AVAILABLE),
    (
        "MakeTexelAvailableKHR",
        ImageOperands::MAKE_TEXEL_AVAILABLE_KHR,
    ),
    ("MakeTexelVisible", ImageOperands::MAKE_TEXEL_VISIBLE),
    ("MakeTexelVisibleKHR", ImageOperands::MAKE_TEXEL_VISIBLE_KHR),
    ("NonPrivateTexel", ImageOperands::NON_PRIVATE_TEXEL),
    ("NonPrivateTexelKHR", ImageOperands::NON_PRIVATE_TEXEL_KHR),
    ("VolatileTexel", ImageOperands::VOLATILE_TEXEL),
    ("VolatileTexelKHR", ImageOperands::VOLATILE_TEXEL_KHR),
    ("SignExtend", ImageOperands::SIGN_EXTEND),
    ("ZeroExtend", ImageOperands::ZERO_EXTEND),
];
pub const FP_FAST_MATH_MODE: &[(&str, FPFastMathMode)] = &[
    ("None", FPFastMathMode::NONE),
    ("NotNan", FPFastMathMode::NOT_NAN),
    ("NotInf", FPFastMathMode::NOT_INF),
    ("Nsz", FPFastMathMode::NSZ),
    ("AllowRecip", FPFastMathMode::ALLOW_RECIP),
    ("Fast", FPFastMathMode::FAST),
];
pub const SELECTION_CONTROL: &[(&str, SelectionControl)] = &[
    ("None", SelectionControl::NONE),
    ("Flatten", SelectionControl::FLATTEN),
    ("DontFlatten", SelectionControl::DONT_FLATTEN),
];
pub const LOOP_CONTROL: &[(&str, LoopControl)] = &[
    ("None", LoopControl::NONE),
    ("Unroll", LoopControl::UNROLL),
    ("DontUnroll", LoopControl::DONT_UNROLL),
    ("DependencyInfinite", LoopControl::DEPENDENCY_INFINITE),
    ("DependencyLength", LoopControl::DEPENDENCY_LENGTH),
    ("MinIterations", LoopControl::MIN_ITERATIONS),
    ("MaxIterations", LoopControl::MAX_ITERATIONS),
    ("IterationMultiple", LoopControl::ITERATION_MULTIPLE),
    ("PeelCount", LoopControl::PEEL_COUNT),
    ("PartialCount", LoopControl::PARTIAL_COUNT),
];
pub const FUNCTION_CONTROL: &[(&str, FunctionControl)] = &[
    ("None", FunctionControl::NONE),
    ("Inline", FunctionControl::INLINE),
    ("DontInline", FunctionControl::DONT_INLINE),
    ("Pure", FunctionControl::PURE),
    ("Const", FunctionControl::CONST),
];
pub const MEMORY_SEMANTICS: &[(&str, MemorySemantics)] = &[
    ("Relaxed", MemorySemantics::RELAXED),
    ("None", MemorySemantics::NONE),
    ("Acquire", MemorySemantics::ACQUIRE),
    ("Release", MemorySemantics::RELEASE),
    ("AcquireRelease", MemorySemantics::ACQUIRE_RELEASE),
    (
        "SequentiallyConsistent",
        MemorySemantics::SEQUENTIALLY_CONSISTENT,
    ),
    ("UniformMemory", MemorySemantics::UNIFORM_MEMORY),
    ("SubgroupMemory", MemorySemantics::SUBGROUP_MEMORY),
    ("WorkgroupMemory", MemorySemantics::WORKGROUP_MEMORY),
    (
        "CrossWorkgroupMemory",
        MemorySemantics::CROSS_WORKGROUP_MEMORY,
    ),
    (
        "AtomicCounterMemory",
        MemorySemantics::ATOMIC_COUNTER_MEMORY,
    ),
    ("ImageMemory", MemorySemantics::IMAGE_MEMORY),
    ("OutputMemory", MemorySemantics::OUTPUT_MEMORY),
    ("OutputMemoryKHR", MemorySemantics::OUTPUT_MEMORY_KHR),
    ("MakeAvailable", MemorySemantics::MAKE_AVAILABLE),
    ("MakeAvailableKHR", MemorySemantics::MAKE_AVAILABLE_KHR),
    ("MakeVisible", MemorySemantics::MAKE_VISIBLE),
    ("MakeVisibleKHR", MemorySemantics::MAKE_VISIBLE_KHR),
    ("Volatile", MemorySemantics::VOLATILE),
];
pub const MEMORY_ACCESS: &[(&str, MemoryAccess)] = &[
    ("None", MemoryAccess::NONE),
    ("Volatile", MemoryAccess::VOLATILE),
    ("Aligned", MemoryAccess::ALIGNED),
    ("Nontemporal", MemoryAccess::NONTEMPORAL),
    ("MakePointerAvailable", MemoryAccess::MAKE_POINTER_AVAILABLE),
    (
        "MakePointerAvailableKHR",
        MemoryAccess::MAKE_POINTER_AVAILABLE_KHR,
    ),
    ("MakePointerVisible", MemoryAccess::MAKE_POINTER_VISIBLE),
    (
        "MakePointerVisibleKHR",
        MemoryAccess::MAKE_POINTER_VISIBLE_KHR,
    ),
    ("NonPrivatePointer", MemoryAccess::NON_PRIVATE_POINTER),
    (
        "NonPrivatePointerKHR",
        MemoryAccess::NON_PRIVATE_POINTER_KHR,
    ),
];
pub const KERNEL_PROFILING_INFO: &[(&str, KernelProfilingInfo)] = &[
    ("None", KernelProfilingInfo::NONE),
    ("CmdExecTime", KernelProfilingInfo::CMD_EXEC_TIME),
];
pub const RAY_FLAGS: &[(&str, RayFlags)] = &[
    ("NoneKHR", RayFlags::NONE_KHR),
    ("OpaqueKHR", RayFlags::OPAQUE_KHR),
    ("NoOpaqueKHR", RayFlags::NO_OPAQUE_KHR),
    (
        "TerminateOnFirstHitKHR",
        RayFlags::TERMINATE_ON_FIRST_HIT_KHR,
    ),
    (
        "SkipClosestHitShaderKHR",
        RayFlags::SKIP_CLOSEST_HIT_SHADER_KHR,
    ),
    (
        "CullBackFacingTrianglesKHR",
        RayFlags::CULL_BACK_FACING_TRIANGLES_KHR,
    ),
    (
        "CullFrontFacingTrianglesKHR",
        RayFlags::CULL_FRONT_FACING_TRIANGLES_KHR,
    ),
    ("CullOpaqueKHR", RayFlags::CULL_OPAQUE_KHR),
    ("CullNoOpaqueKHR", RayFlags::CULL_NO_OPAQUE_KHR),
    ("SkipTrianglesKHR", RayFlags::SKIP_TRIANGLES_KHR),
    ("SkipAabBsKHR", RayFlags::SKIP_AAB_BS_KHR),
];
pub const FRAGMENT_SHADING_RATE: &[(&str, FragmentShadingRate)] = &[
    ("VERTICAL2_PIXELS", FragmentShadingRate::VERTICAL2_PIXELS),
    ("VERTICAL4_PIXELS", FragmentShadingRate::VERTICAL4_PIXELS),
    (
        "HORIZONTAL2_PIXELS",
        FragmentShadingRate::HORIZONTAL2_PIXELS,
    ),
    (
        "HORIZONTAL4_PIXELS",
        FragmentShadingRate::HORIZONTAL4_PIXELS,
    ),
];
pub const COOPERATIVE_MATRIX_OPERANDS: &[(&str, CooperativeMatrixOperands)] = &[
    ("NONE_KHR", CooperativeMatrixOperands::NONE_KHR),
    (
        "MATRIX_A_SIGNED_COMPONENTS_KHR",
        CooperativeMatrixOperands::MATRIX_A_SIGNED_COMPONENTS_KHR,
    ),
    (
        "MATRIX_B_SIGNED_COMPONENTS_KHR",
        CooperativeMatrixOperands::MATRIX_B_SIGNED_COMPONENTS_KHR,
    ),
    (
        "MATRIX_C_SIGNED_COMPONENTS_KHR",
        CooperativeMatrixOperands::MATRIX_C_SIGNED_COMPONENTS_KHR,
    ),
    (
        "MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
        CooperativeMatrixOperands::MATRIX_RESULT_SIGNED_COMPONENTS_KHR,
    ),
    (
        "SATURATING_ACCUMULATION_KHR",
        CooperativeMatrixOperands::SATURATING_ACCUMULATION_KHR,
    ),
];

fn parse_bitflags_operand<T: std::ops::BitOr<Output = T> + Copy>(
    values: &'static [(&'static str, T)],
    word: &str,
) -> Option<T> {
    let mut result = None;
    'outer: for item in word.split('|') {
        for &(key, value) in values {
            if item == key {
                result = Some(result.map_or(value, |x| x | value));
                continue 'outer;
            }
        }
        return None;
    }
    result
}
