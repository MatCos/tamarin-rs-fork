//! (No HS analog.) The RL system codec (`system_codec`) on real proof states.
//!
//! Walks the proof tree of a few lemmas (one per builtin family) and, at
//! every system reached, checks that
//!
//! - `decode(encode(sys)) == sys` under `System: PartialEq`, the `LessAtom`
//!   reasons (which that equality ignores) agree, and encoding the decoded
//!   system gives the same token again (this covers every written field,
//!   including `Fact.annotations`, which `Fact: PartialEq` also ignores);
//! - the decoded system behaves the same: the same ranked methods (the RL
//!   server's list: `candidate_methods` filtered by
//!   `is_applicable_for_display`), and the same cases from every method;
//! - every ranked method round-trips.
//!
//! Only built with the `serde` feature.  Skips without maude only when
//! `TAM_ALLOW_NO_MAUDE=1` (see `common::maude_path`).

#![cfg(feature = "serde")]

use std::path::PathBuf;

use tamarin_theory::constraint::solver::context::{CutStrategy, ProofContext};
use tamarin_theory::constraint::solver::proof_method::{
    exec_proof_method, is_applicable_for_display, ProofMethod,
};
use tamarin_theory::constraint::solver::search::candidate_methods;
use tamarin_theory::constraint::system::System;
use tamarin_theory::prove::{build_lemma_proof_context, CliHeuristic};
use tamarin_theory::system_codec::{
    decode_proof_method, decode_system, encode_proof_method, encode_system,
};

mod common;

/// Systems checked per lemma, maximum depth, and how many methods per
/// system have their cases pushed onto the walk.
const MAX_SYSTEMS: usize = 40;
const MAX_DEPTH: usize = 4;
const BRANCHES: usize = 2;

fn ranked(ctx: &ProofContext, sys: &System) -> Vec<ProofMethod> {
    candidate_methods(sys, ctx, 0)
        .into_iter()
        .filter(|m| is_applicable_for_display(ctx, m, sys))
        .collect()
}

fn sorted_cases(
    ctx: &ProofContext,
    m: &ProofMethod,
    sys: &System,
) -> Option<Vec<(String, System)>> {
    exec_proof_method(ctx, m, sys).map(|mut cs| {
        cs.sort_by(|a, b| a.0.cmp(&b.0));
        cs
    })
}

fn round_trip(sys: &System, at: &str) -> System {
    let token = encode_system(sys);
    let back = decode_system(&token).unwrap_or_else(|e| panic!("{at}: decode failed: {e}"));
    assert!(back == *sys, "{at}: decoded system differs");
    let reasons = |s: &System| s.less_atoms.iter().map(|l| l.reason).collect::<Vec<_>>();
    assert_eq!(
        reasons(&back),
        reasons(sys),
        "{at}: less-atom reasons differ"
    );
    assert!(
        encode_system(&back) == token,
        "{at}: re-encoding gives a different token"
    );
    back
}

fn walk(rel_path: &str, lemma: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tamarin-prover/examples")
        .join(rel_path);
    let Some((parsed, _, maude)) = common::load_theory_with_maude(&path) else {
        return;
    };
    let (ctx, initial, _, _user_funs) = build_lemma_proof_context(
        &parsed,
        lemma,
        maude,
        None,
        path.to_str().unwrap(),
        &CliHeuristic::default(),
        CutStrategy::Dfs,
        None,
    )
    .unwrap_or_else(|e| panic!("{rel_path} {lemma}: {e:?}"));

    let mut stack = vec![(initial, 0usize, String::new())];
    let mut checked = 0;
    while let Some((sys, depth, case_path)) = stack.pop() {
        if checked == MAX_SYSTEMS {
            break;
        }
        checked += 1;
        let at = format!("{rel_path} {lemma} [{case_path}]");
        let back = round_trip(&sys, &at);

        let methods = ranked(&ctx, &sys);
        assert!(
            ranked(&ctx, &back) == methods,
            "{at}: ranked methods differ after decode"
        );
        for m in &methods {
            let token = encode_proof_method(m);
            let m2 = decode_proof_method(&token).unwrap_or_else(|e| panic!("{at}: {m:?}: {e}"));
            assert!(m2 == *m, "{at}: method {m:?} differs after decode");
            assert!(
                encode_proof_method(&m2) == token,
                "{at}: method {m:?} re-encodes differently"
            );
        }

        if depth == MAX_DEPTH {
            continue;
        }
        let mut expanded = 0;
        for m in &methods {
            let cases = sorted_cases(&ctx, m, &sys);
            let cases_back = sorted_cases(&ctx, m, &back);
            let names = |cs: &Option<Vec<(String, System)>>| {
                cs.as_ref()
                    .map(|v| v.iter().map(|c| c.0.clone()).collect::<Vec<_>>())
            };
            assert_eq!(
                names(&cases_back),
                names(&cases),
                "{at}: {m:?}: case names differ"
            );
            assert!(cases_back == cases, "{at}: {m:?}: case systems differ");
            match cases {
                Some(cs) if !cs.is_empty() && expanded < BRANCHES => {
                    expanded += 1;
                    stack.extend(
                        cs.into_iter()
                            .map(|(name, s)| (s, depth + 1, format!("{case_path}/{name}"))),
                    );
                }
                _ => {}
            }
        }
    }
    assert!(
        checked >= 10,
        "{rel_path} {lemma}: the walk reached only {checked} systems"
    );
}

#[test]
fn tutorial_client_auth() {
    walk("Tutorial.spthy", "Client_auth");
}

#[test]
fn diffie_hellman() {
    walk(
        "thesis-benedikt/running-examples/DH_Message_Deduction.spthy",
        "key_deducible",
    );
}

#[test]
fn xor() {
    walk("features/xor/basicfunctionality/xor0.spthy", "crworks");
}

#[test]
fn multiset() {
    walk("csf20-disputeResolution/PR6_ShHm.spthy", "functional");
}

#[test]
fn induction() {
    walk(
        "features/injectivity/injectivity.spthy",
        "injectivity_check",
    );
}
