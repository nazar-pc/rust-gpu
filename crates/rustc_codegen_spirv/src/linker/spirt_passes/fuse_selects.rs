use itertools::Itertools as _;
use rustc_data_structures::fx::FxHashMap;
use smallvec::SmallVec;
use spirt::func_at::FuncAt;
use spirt::transform::InnerInPlaceTransform;
use spirt::visit::InnerVisit;
use spirt::{Context, FuncDefBody, NodeKind, Region, SelectionKind, Value};
use std::mem;

use super::{ReplaceValueWith, VisitAllRegionsAndNodes};

/// Combine consecutive `Select`s in `func_def_body`.
pub(crate) fn fuse_selects_in_func(_cx: &Context, func_def_body: &mut FuncDefBody) {
    // HACK(eddyb) this kind of random-access is easier than using `spirt::transform`.
    let mut all_regions = vec![];

    func_def_body.inner_visit_with(&mut VisitAllRegionsAndNodes {
        state: (),
        visit_region: |_: &mut (), func_at_region: FuncAt<'_, Region>| {
            all_regions.push(func_at_region.position);
        },
        visit_node: |_: &mut (), _| {},
    });

    // HACK(eddyb) reusing allocations for `Var` -> `Value` mappings.
    let mut reusable_var_replacements = FxHashMap::default();

    for region in all_regions {
        // HACK(eddyb) buffering the `Node`s to remove from this region,
        // as iterating and modifying a list at the same time isn't supported.
        let mut nodes_to_remove = SmallVec::<[_; 8]>::new();

        let mut func_at_children_iter = func_def_body.at_mut(region).at_children().into_iter();
        while let Some(func_at_child) = func_at_children_iter.next() {
            let base_node = func_at_child.position;
            let base_node_def = func_at_child.def();
            if let NodeKind::Select(SelectionKind::BoolCond) = &base_node_def.kind {
                let base_cond = base_node_def.inputs[0];
                let base_cases = base_node_def.child_regions.clone();

                // Scan ahead for candidate `Select`s (with the same condition).
                let mut fusion_candidate_iter = func_at_children_iter.reborrow();
                while let Some(func_at_fusion_candidate) = fusion_candidate_iter.next() {
                    let fusion_candidate = func_at_fusion_candidate.position;
                    let mut func = func_at_fusion_candidate.at(());
                    let fusion_candidate_def = func.reborrow().at(fusion_candidate).def();
                    match &fusion_candidate_def.kind {
                        NodeKind::Select(SelectionKind::BoolCond)
                            if fusion_candidate_def.inputs[0] == base_cond =>
                        {
                            // FIXME(eddyb) handle outputs from the second `Select`.
                            if !fusion_candidate_def.outputs.is_empty() {
                                break;
                            }

                            let cases_to_fuse = fusion_candidate_def.child_regions.clone();

                            // Concatenate the `Select`s' respective cases
                            // ("then" with "then", "else" with "else", etc.).
                            for (&base_case, &case_to_fuse) in base_cases.iter().zip(&cases_to_fuse)
                            {
                                let children_of_case_to_fuse =
                                    mem::take(&mut func.reborrow().at(case_to_fuse).def().children);

                                // Replace uses of the outputs of the first `Select`,
                                // in the second one's case, with the specific values
                                // (e.g. `let y = if c { x } ...; if c { f(y) }`
                                // has to become `let y = if c { f(x); x } ...`).
                                let outputs_of_base_case = &func.regions[base_case].outputs;

                                // HACK(eddyb) due to no `func.vars` access through
                                // `ReplaceValueWith`, this is the next best thing.
                                let mut var_replacements = reusable_var_replacements;
                                assert_eq!(var_replacements.len(), 0);
                                var_replacements.extend(
                                    func.nodes[base_node]
                                        .outputs
                                        .iter()
                                        .copied()
                                        .zip_eq(outputs_of_base_case.iter().copied()),
                                );
                                func.reborrow()
                                    .at(children_of_case_to_fuse)
                                    .into_iter()
                                    .inner_in_place_transform_with(&mut ReplaceValueWith(
                                        |v| match v {
                                            Value::Const(_) => None,
                                            Value::Var(v) => var_replacements.get(&v).copied(),
                                        },
                                    ));
                                var_replacements.clear();
                                reusable_var_replacements = var_replacements;

                                func.regions[base_case]
                                    .children
                                    .append(children_of_case_to_fuse, func.nodes);
                            }

                            nodes_to_remove.push(fusion_candidate);
                        }

                        _ => break,
                    }
                }
            }
        }

        let region_children = &mut func_def_body.regions[region].children;
        for node in nodes_to_remove {
            region_children.remove(node, &mut func_def_body.nodes);
        }
    }
}
