//! (No HS analog.) The pieces the RL server builds its routes from, on
//! Tutorial.spthy: a lemma's start system and context from a
//! `ProverSession`, the ranked method list, and replaying a client's proof
//! with `check_proof`.
//!
//! Skips without maude only when `TAM_ALLOW_NO_MAUDE=1` (see
//! `tamarin_test_support::require_maude_path`).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use tamarin_term::maude_proc::MaudeHandle;
use tamarin_theory::check_proof::{check_proof, MethodTree};
use tamarin_theory::constraint::solver::proof_method::{ProofMethod, Result as MethodResult};
use tamarin_theory::constraint::solver::search::{
    proof_size, proof_status, ranked_methods, ProofNode, ProofStatus, SysRetention,
};
use tamarin_theory::pretty_theory::pretty_proof_body;
use tamarin_theory::prove::{prove_lemma_in_session, ProverSession, ProverSessionOptions};

fn tutorial_session() -> Option<ProverSession> {
    let maude_path = tamarin_test_support::require_maude_path()?;
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tamarin-prover/examples/Tutorial.spthy");
    let source = std::fs::read_to_string(path).expect("read theory");
    let parsed = tamarin_parser::parse_theory(&source, &[]).expect("parse theory");
    let theory = tamarin_theory::elaborate::elaborate(&parsed).expect("elaborate");
    let maude = MaudeHandle::start(&maude_path, theory.signature.clone())
        .unwrap_or_else(|e| panic!("maude at {maude_path} failed to start: {e:?}"));
    // Keep the autoprover's systems, so its root can be compared with the
    // start system (the server keeps them too).
    let options = ProverSessionOptions {
        sys_retention: SysRetention::KeepAll,
        ..ProverSessionOptions::default()
    };
    Some(ProverSession::build(Arc::new(theory), maude, options).expect("session"))
}

/// The methods of a proof tree, as a client would send them back.
fn methods_of(node: &ProofNode) -> MethodTree {
    MethodTree {
        method: node.method.clone(),
        cases: node
            .children
            .iter()
            .map(|(name, child)| (name.clone(), methods_of(child)))
            .collect(),
    }
}

fn auto_proof(session: &ProverSession, lemma: &str) -> ProofNode {
    prove_lemma_in_session(session, lemma, usize::MAX).expect("prove")
}

#[test]
fn lemma_start_is_the_system_the_prover_starts_from() {
    let Some(session) = tutorial_session() else {
        return;
    };
    for lemma in ["Client_auth", "Client_session_key_honest_setup"] {
        let (ctx, start) = session.context_and_system_for_lemma(lemma).expect("start");
        assert!(start == auto_proof(&session, lemma).sys, "{lemma}");
        let methods = ranked_methods(&start, &ctx, 0).expect("ranked methods");
        assert!(
            methods.contains(&ProofMethod::Simplify),
            "{lemma}: {methods:?}"
        );
        assert!(
            methods.contains(&ProofMethod::Induction),
            "{lemma}: {methods:?}"
        );
    }
    assert!(session
        .context_and_system_for_lemma("no_such_lemma")
        .is_err());
}

fn replay_auto_proof(session: &ProverSession, lemma: &str) -> (ProofNode, ProofNode) {
    let auto = auto_proof(session, lemma);
    let (ctx, start) = session.context_and_system_for_lemma(lemma).expect("start");
    let checked = check_proof(&ctx, start, &methods_of(&auto)).expect("check proof");
    (auto, checked)
}

#[test]
fn replaying_the_autoprover_proof_gives_the_same_proof() {
    let Some(session) = tutorial_session() else {
        return;
    };
    for lemma in [
        "Client_session_key_secrecy",
        "Client_auth",
        "Client_auth_injective",
    ] {
        let (auto, checked) = replay_auto_proof(&session, lemma);
        assert_eq!(
            pretty_proof_body(&checked),
            pretty_proof_body(&auto),
            "{lemma}"
        );
        assert_eq!(proof_status(&checked), ProofStatus::Complete, "{lemma}");
        assert_eq!(proof_size(&checked), proof_size(&auto), "{lemma}");
    }
}

/// For an exists-trace lemma the autoprover keeps only the path to the
/// trace, so the replay marks the other cases `sorry "Unhandled Case"`
/// (with their systems); the trace wins, the result still says TraceFound.
#[test]
fn replaying_a_trace_reports_trace_found() {
    let Some(session) = tutorial_session() else {
        return;
    };
    let (auto, checked) = replay_auto_proof(&session, "Client_session_key_honest_setup");
    assert_eq!(proof_status(&auto), ProofStatus::TraceFound);
    assert_eq!(proof_status(&checked), ProofStatus::TraceFound);
    let body = pretty_proof_body(&checked);
    assert!(body.contains("SOLVED // trace found"), "{body}");
    assert!(body.contains("by sorry /* Unhandled Case */"), "{body}");
    assert!(!body.contains("/* unannotated */"), "{body}");
}

#[test]
fn missing_extra_and_invalid_steps_follow_check_proof() {
    let Some(session) = tutorial_session() else {
        return;
    };
    let lemma = "Client_auth";
    let auto = auto_proof(&session, lemma);
    let (ctx, start_sys) = session.context_and_system_for_lemma(lemma).expect("start");
    let start = || start_sys.clone();
    let full = methods_of(&auto);

    // A node with several cases, as a path of case names from the root.
    fn find_split(t: &MethodTree, path: &mut Vec<String>) -> bool {
        if t.cases.len() > 1 {
            return true;
        }
        for (name, sub) in &t.cases {
            path.push(name.clone());
            if find_split(sub, path) {
                return true;
            }
            path.pop();
        }
        false
    }
    let mut path = Vec::new();
    assert!(
        find_split(&full, &mut path),
        "Client_auth's proof has a case split"
    );
    fn at<'a>(t: &'a mut MethodTree, path: &[String]) -> &'a mut MethodTree {
        path.iter().fold(t, |t, p| t.cases.get_mut(p).unwrap())
    }
    fn node_at<'a>(n: &'a ProofNode, path: &[String]) -> &'a ProofNode {
        path.iter().fold(n, |n, p| &n.children[p])
    }

    // A case left out: it comes back as a `sorry` with the case's system,
    // so the proof is Incomplete. (The fork's HS handler left it without a
    // system; Complete absorbed the resulting Undetermined and a proof with
    // missing cases was reported Complete.)
    let mut missing = full.clone();
    let dropped = {
        let split = at(&mut missing, &path);
        let name = split.cases.keys().next().unwrap().clone();
        split.cases.remove(&name);
        name
    };
    let checked = check_proof(&ctx, start(), &missing).expect("check proof");
    let mut p = path.clone();
    p.push(dropped);
    let n = node_at(&checked, &p);
    assert_eq!(n.method, ProofMethod::Sorry(Some("Unhandled Case".into())));
    assert!(n.annotated && n.children.is_empty());
    assert_eq!(proof_status(&checked), ProofStatus::Incomplete);

    // A case the method does not produce: kept as given, without systems.
    let mut extra = full.clone();
    let leaf = MethodTree {
        method: ProofMethod::Finished(MethodResult::Contradictory(None)),
        cases: BTreeMap::new(),
    };
    at(&mut extra, &path)
        .cases
        .insert("no_such_case".into(), leaf.clone());
    let checked = check_proof(&ctx, start(), &extra).expect("check proof");
    p = path.clone();
    p.push("no_such_case".into());
    let n = node_at(&checked, &p);
    assert_eq!(n.method, leaf.method);
    assert!(!n.annotated);
    assert_eq!(proof_size(&checked), proof_size(&auto) + 1);

    // A step that does not apply (the start system is not finished): a
    // `sorry` whose single unnamed case is the given subtree, unannotated.
    let invalid = MethodTree {
        method: ProofMethod::Finished(MethodResult::Solved),
        cases: full.cases.clone(),
    };
    let checked = check_proof(&ctx, start(), &invalid).expect("check proof");
    assert_eq!(
        checked.method,
        ProofMethod::Sorry(Some("invalid proof step encountered".into()))
    );
    assert!(checked.annotated);
    let inner = &checked.children[""];
    assert_eq!(inner.method, invalid.method);
    assert!(!inner.annotated && inner.children.values().all(|c| !c.annotated));
    assert_eq!(proof_status(&checked), ProofStatus::Incomplete);

    // A `sorry` keeps its cases without systems and is not run.
    let sorry = MethodTree {
        method: ProofMethod::Sorry(None),
        cases: full.cases.clone(),
    };
    let checked = check_proof(&ctx, start(), &sorry).expect("check proof");
    assert!(checked.annotated);
    assert!(checked.children.values().all(|c| !c.annotated));
    assert_eq!(proof_status(&checked), ProofStatus::Incomplete);
    assert_eq!(proof_size(&checked), proof_size(&auto));
}
