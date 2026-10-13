//! (No HS analog.) Round trips of the RL system codec on small hand-built
//! values, one per hand-written serde impl.  Real proof states are covered by
//! `tests/system_codec_walk.rs`.

use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde::Serialize;
use tamarin_parser::ast as p;
use tamarin_term::function_symbols::{AcFctSym, Constructability, NdcState, NoEqSym, Privacy};
use tamarin_term::intern::{intern_bytes, intern_str};
use tamarin_term::lterm::{LNTerm, LSort, LVar, NameId};
use tamarin_term::vterm::var_term;

use super::*;
use crate::constraint::constraints::{Disj, Goal, LessAtom, Reason};
use crate::constraint::solver::proof_method::Result as ProofResult;
use crate::constraint::system::GoalStatus;
use crate::fact::{proto_fact_ann, FactAnnotation, FactTag, LNFact, Multiplicity};
use crate::guarded::Guarded;
use crate::guarded_types::{BVar, GAtom, GTerm};
use crate::rule::ProtoRuleName;
use crate::tools::equation_store::SplitId;
use crate::tools::EquationStore;

fn round_trip<T: Serialize + DeserializeOwned>(v: &T) -> T {
    decode(&encode(v)).unwrap()
}

fn node(idx: u64) -> LVar {
    LVar::new("vr", LSort::Node, idx)
}

fn msg(name: &str, idx: u64) -> LNTerm {
    var_term(LVar::new(name, LSort::Msg, idx))
}

fn annotated_fact() -> LNFact {
    proto_fact_ann(
        Multiplicity::Linear,
        "St_codec",
        [FactAnnotation::SolveFirst, FactAnnotation::NoSources]
            .into_iter()
            .collect(),
        vec![msg("x", 3), msg("y", 7)],
    )
}

fn gvar(name: &str) -> GTerm {
    GTerm::Var(BVar::Free(p::VarSpec {
        name: name.to_string(),
        idx: 0,
        sort: p::SortHint::Msg,
        typ: None,
    }))
}

/// One `x = y op z` formula per `BinOp` variant, the AC one included.
fn binop_formulas() -> Vec<Arc<Guarded>> {
    [
        p::BinOp::Exp,
        p::BinOp::Mult,
        p::BinOp::Union,
        p::BinOp::Xor,
        p::BinOp::NatPlus,
        p::BinOp::AcFct("codec_ac"),
    ]
    .into_iter()
    .map(|op| {
        let rhs = GTerm::BinOp(op, Arc::new(gvar("y")), Arc::new(gvar("z")));
        Arc::new(Guarded::Atom(GAtom::Eq(gvar("x"), rhs)))
    })
    .collect()
}

/// A system touching each hand-written impl: interned names in variables and
/// fact tags, an annotated fact, every `LessAtom` reason, `BinOp`s in
/// formulas, a non-default equation store, and the public non-content fields.
fn small_system() -> System {
    let mut sys = System::empty();
    let c = sys.content_mut();
    c.less_atoms = [
        Reason::Formula,
        Reason::InjectiveFacts,
        Reason::Fresh,
        Reason::Adversary,
        Reason::NormalForm,
    ]
    .into_iter()
    .enumerate()
    .map(|(i, r)| LessAtom::new(node(i as u64), node(i as u64 + 10), r))
    .collect();
    c.goals = Arc::new(vec![
        (
            Goal::Action(node(1), annotated_fact()),
            GoalStatus {
                looping: true,
                solved: false,
                nr: 4,
            },
        ),
        (
            Goal::Disj(Disj(vec![(*binop_formulas()[5]).clone()])),
            GoalStatus::default(),
        ),
    ]);
    c.formulas = binop_formulas();
    c.last_atom = Some(node(2));
    let mut eq = EquationStore::empty();
    eq.next_split = SplitId(3);
    sys.set_eq_store(Arc::new(eq));
    sys.next_goal_nr = 5;
    sys.used_sources = vec!["source_a".to_string()];
    sys
}

#[test]
fn interned_names_are_reinterned_on_read() {
    let v = round_trip(&LVar::new("codec_var", LSort::Fresh, 9));
    assert_eq!(v, LVar::new("codec_var", LSort::Fresh, 9));
    assert!(std::ptr::eq(v.name, intern_str("codec_var")));

    let n = round_trip(&NameId::new("codec_name"));
    assert!(std::ptr::eq(n.0, intern_str("codec_name")));

    let s = NoEqSym::new("codec_f", 2, Privacy::Private, Constructability::Destructor);
    let s2 = round_trip(&s);
    assert_eq!(s2, s);
    assert!(std::ptr::eq(s2.name, intern_bytes(b"codec_f")));

    let a = AcFctSym::new(
        "codec_ac",
        Privacy::Public,
        Constructability::Constructor,
        NdcState::IsNdc,
    );
    let a2 = round_trip(&a);
    assert_eq!(a2, a);
    assert!(std::ptr::eq(a2.name, intern_bytes(b"codec_ac")));
}

#[test]
fn every_variant_of_the_mirrored_enums_round_trips() {
    let tags = [
        FactTag::Proto(Multiplicity::Persistent, intern_str("codec_P"), 3),
        FactTag::Fresh,
        FactTag::Out,
        FactTag::In,
        FactTag::Ku,
        FactTag::Kd,
        FactTag::Ded,
        FactTag::Term,
    ];
    for t in tags {
        assert_eq!(round_trip(&t), t);
    }
    for n in [ProtoRuleName::Fresh, ProtoRuleName::Stand("codec_rule")] {
        assert_eq!(round_trip(&n), n);
    }
    let formulas = binop_formulas();
    assert_eq!(round_trip(&formulas), formulas);
}

/// `Fact`'s `PartialEq` ignores `annotations` and the two caches, so check
/// them directly: annotations survive, caches match a freshly built fact.
#[test]
fn fact_keeps_annotations_and_recomputes_caches() {
    let f = annotated_fact();
    let g: LNFact = round_trip(&f);
    assert_eq!(g, f);
    assert_eq!(g.annotations, f.annotations);
    assert_eq!(g.bloom(), f.bloom());
    assert_eq!(g.max_var_cached(), Some(7));
}

#[test]
fn system_round_trips_with_reasons_and_fresh_stamps() {
    let sys = small_system();
    let token = encode_system(&sys);
    let back = decode_system(&token).unwrap();
    assert_eq!(back, sys);
    // `LessAtom`'s `PartialEq` ignores `reason`.
    let reasons = |s: &System| s.less_atoms.iter().map(|l| l.reason).collect::<Vec<_>>();
    assert_eq!(reasons(&back), reasons(&sys));
    // Every written field survives exactly: writing the decoded system
    // again gives the same token.
    assert_eq!(encode_system(&back), token);
    // Stamps are minted afresh, as for `System::empty`.
    assert_ne!(back.subst_stamp(), 0);
    assert_ne!(back.subst_stamp(), sys.subst_stamp());
}

#[test]
fn empty_system_round_trips() {
    let sys = System::empty();
    assert_eq!(decode_system(&encode_system(&sys)).unwrap(), sys);
}

#[test]
fn proof_methods_round_trip() {
    let methods = [
        ProofMethod::Simplify,
        ProofMethod::Induction,
        ProofMethod::Sorry(Some("codec".to_string())),
        ProofMethod::SolveGoal(Goal::Action(node(1), annotated_fact())),
        ProofMethod::SolveGoal(Goal::Split(SplitId(2))),
        ProofMethod::Finished(ProofResult::Contradictory(None)),
    ];
    for m in &methods {
        let token = encode_proof_method(m);
        assert_eq!(&decode_proof_method(&token).unwrap(), m);
        assert_eq!(
            encode_proof_method(&decode_proof_method(&token).unwrap()),
            token
        );
    }
}

#[test]
fn base64url_is_written_padded_and_read_with_or_without_padding() {
    assert_eq!(B64.encode([0xfb, 0xff]), "-_8=");
    assert_eq!(B64.decode("-_8=").unwrap(), [0xfb, 0xff]);
    assert_eq!(B64.decode("-_8").unwrap(), [0xfb, 0xff]);
}

#[test]
fn garbage_is_an_error_not_a_panic() {
    assert!(matches!(
        decode_system("not base64!"),
        Err(CodecError::Base64(_))
    ));
    assert!(matches!(decode_system("AAAA"), Err(CodecError::Gzip(_))));
    // Valid gzip, but a method's bytes are too short for a system.
    let method = encode_proof_method(&ProofMethod::Simplify);
    assert!(matches!(
        decode_system(&method),
        Err(CodecError::Postcard(_))
    ));
    // A system's first bytes can read as a method; the rest is not ignored.
    let system = encode_system(&small_system());
    assert!(matches!(
        decode_proof_method(&system),
        Err(CodecError::TrailingBytes(_))
    ));
}
