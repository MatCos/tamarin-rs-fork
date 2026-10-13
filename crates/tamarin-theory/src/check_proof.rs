// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! Replay a proof given as a tree of proof methods — HS `checkProof`
//! (Theory/Proof.hs) with the prover for unhandled cases fixed to
//! `sorry (Just "Unhandled Case") Nothing`, as the TamRL fork's
//! `checkproof` handler calls it.
//!
//! [`crate::replay`] does the same for proofs parsed from a `.spthy` file,
//! where each step has to be matched against the system by its text.
//! Here every step is already a [`ProofMethod`] (decoded from the RL
//! client's tokens), so it is run directly.

use std::collections::BTreeMap;

use crate::constraint::solver::context::ProofContext;
use crate::constraint::solver::proof_method::{check_and_exec_proof_method, ProofMethod};
use crate::constraint::solver::search::{node_status_of, NodeStatus, ProofNode};
use crate::constraint::system::System;

/// A proof as a client sends it: the method applied at this node and the
/// proof of each case, keyed by case name.
#[derive(Debug, Clone, PartialEq)]
pub struct MethodTree {
    pub method: ProofMethod,
    pub cases: BTreeMap<String, MethodTree>,
}

/// Replay `proof` from `sys` (HS `checkProof ctxt checkProver 0 sys proof`):
///
/// - a `Sorry` step is kept, its cases are kept without systems;
/// - a step that applies (`checkAndExecProofMethod`) keeps its method, and
///   its cases are merged with the cases the method produced: both →
///   replayed recursively; produced only → `sorry "Unhandled Case"`
///   without a system; given only → kept without systems;
/// - a step that does not apply becomes `sorry "invalid proof step
///   encountered"` with the given subtree, without systems, as its single
///   unnamed case.
///
/// "Without a system" is `annotated == false` (HS `Nothing`), printed as
/// `/* unannotated */` and counted as `Undetermined` by
/// [`proof_status`](crate::constraint::solver::search::proof_status).
pub fn check_proof(ctx: &ProofContext, sys: System, proof: &MethodTree) -> ProofNode {
    if let ProofMethod::Sorry(reason) = &proof.method {
        let children = without_systems(&proof.cases, &sys);
        return node(ProofMethod::Sorry(reason.clone()), sys, children, true);
    }
    match check_and_exec_proof_method(ctx, &proof.method, &sys) {
        Some(produced) => {
            let mut children = BTreeMap::new();
            for (name, case_sys) in produced {
                let child = match proof.cases.get(&name) {
                    Some(sub) => check_proof(ctx, case_sys, sub),
                    None => node(
                        ProofMethod::Sorry(Some("Unhandled Case".into())),
                        case_sys,
                        BTreeMap::new(),
                        false,
                    ),
                };
                children.insert(name, child);
            }
            for (name, sub) in &proof.cases {
                if !children.contains_key(name) {
                    children.insert(name.clone(), without_system(sub, &sys));
                }
            }
            node(proof.method.clone(), sys, children, true)
        }
        None => {
            let mut children = BTreeMap::new();
            children.insert(String::new(), without_system(proof, &sys));
            node(
                ProofMethod::Sorry(Some("invalid proof step encountered".into())),
                sys,
                children,
                true,
            )
        }
    }
}

/// HS `noSystemPrf`: the subtree as given, every node without a system.
/// `ProofNode` needs a system anyway; `placeholder` fills it and is never
/// read for unannotated nodes.
fn without_system(proof: &MethodTree, placeholder: &System) -> ProofNode {
    node(
        proof.method.clone(),
        placeholder.clone(),
        without_systems(&proof.cases, placeholder),
        false,
    )
}

fn without_systems(
    cases: &BTreeMap<String, MethodTree>,
    placeholder: &System,
) -> BTreeMap<String, ProofNode> {
    cases
        .iter()
        .map(|(name, sub)| (name.clone(), without_system(sub, placeholder)))
        .collect()
}

fn node(
    method: ProofMethod,
    sys: System,
    children: BTreeMap<String, ProofNode>,
    annotated: bool,
) -> ProofNode {
    let status = match &method {
        ProofMethod::Finished(r) => node_status_of(r),
        ProofMethod::Sorry(_) => NodeStatus::Sorry,
        _ => NodeStatus::Open,
    };
    ProofNode {
        method,
        sys,
        children,
        status,
        annotated,
    }
}
