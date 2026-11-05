//! (No HS analog.) `goal_type_and_args` against strings the TamRL Haskell
//! fork's server produced. Each expected string is copied from TamRL's
//! decision files (`project/harvest_data/decisions/*.jsonl`), except the
//! `<k.0:fresh, h('c')>` Subterm goal, which comes from a live run of the
//! fork's server.

use std::sync::Arc;

use tamarin_parser::parser::parse_formula_str;
use tamarin_term::builtin::{dh_neutral, exp, fst, mult, pair, snd};
use tamarin_term::function_symbols::{nat_one_sym, AcSym, Constructability, NoEqSym, Privacy};
use tamarin_term::lterm::{pub_term, LNTerm, LSort, LVar};
use tamarin_term::maude_sig::nat_maude_sig;
use tamarin_term::subst_vfresh::SubstVFresh;
use tamarin_term::term::{f_app_ac, f_app_no_eq};
use tamarin_term::vterm::var_term;

use super::*;
use crate::constraint::constraints::Disj;
use crate::elaborate::formula_to_guarded_parsed;
use crate::fact::{ku_fact, proto_fact, Fact, FactAnnotation, FactTag, Multiplicity};
use crate::rule::{ConcIdx, PremIdx};
use crate::tools::equation_store::EqDisj;
use crate::tools::EquationStore;

fn lv(name: &str, sort: LSort, idx: u64) -> LVar {
    LVar::new(name, sort, idx)
}

fn v(name: &str, sort: LSort, idx: u64) -> LNTerm {
    var_term(lv(name, sort, idx))
}

fn fun(name: &str, args: Vec<LNTerm>) -> LNTerm {
    let sym = NoEqSym::new(
        name,
        args.len(),
        Privacy::Public,
        Constructability::Constructor,
    );
    f_app_no_eq(sym, args)
}

fn nat_one() -> LNTerm {
    f_app_no_eq(nat_one_sym(), vec![])
}

fn args_of(goal: Goal, sys: &System) -> (&'static str, Vec<String>) {
    goal_type_and_args(&goal, sys)
}

#[test]
fn action_goals_use_ascii_operators_and_sorted_vars() {
    let sys = System::empty();
    // `x1 ++ x2 ++ x3 ++ x4` under XOR, and `*` inside an exponent.
    let union = f_app_ac(
        AcSym::Union,
        (0..4).map(|i| v("x", LSort::Msg, i)).collect(),
    );
    let f5 = fun(
        "f5",
        vec![v("k", LSort::Fresh, 0), v("RAND", LSort::Msg, 0)],
    );
    assert_eq!(
        args_of(
            Goal::Action(
                lv("vk", LSort::Node, 4),
                ku_fact(f_app_ac(AcSym::Xor, vec![f5, union]))
            ),
            &sys
        ),
        (
            "Action",
            vec![
                "!KU((f5(k.0:fresh, RAND.0:msg) XOR (x.0:msg ++ x.1:msg ++ x.2:msg ++ x.3:msg)))"
                    .to_string(),
                "vk.4:node".to_string(),
            ]
        )
    );
    let dh = exp(
        pub_term("g"),
        mult(v("lkI", LSort::Fresh, 0), v("x", LSort::Fresh, 0)),
    );
    let fa = ku_fact(fun("h", vec![pair(v("z", LSort::Msg, 0), dh)]));
    assert_eq!(
        args_of(Goal::Action(lv("vk", LSort::Node, 5), fa), &sys).1[0],
        "!KU(h(<z.0:msg, 'g'^(lkI.0:fresh * x.0:fresh)>))"
    );
    // Annotations follow the `)` directly.
    let annotated = Fact::fresh_annotated(
        FactTag::Ku,
        [FactAnnotation::SolveFirst].into_iter().collect(),
        vec![fun("GTK", vec![v("x", LSort::Fresh, 0)])],
    );
    assert_eq!(
        args_of(Goal::Action(lv("vk", LSort::Node, 0), annotated), &sys).1[0],
        "!KU(GTK(x.0:fresh))[+]"
    );
}

#[test]
fn premise_and_chain_goals_print_node_and_index() {
    let sys = System::empty();
    let st = proto_fact(
        Multiplicity::Linear,
        "St_1_UE",
        ["tid", "supi", "idHN", "k", "sqn_root"]
            .iter()
            .map(|n| v(n, LSort::Fresh, 0))
            .collect(),
    );
    assert_eq!(
        args_of(
            Goal::Premise((lv("i", LSort::Node, 0), PremIdx(0)), st),
            &sys
        ),
        (
            "Premise",
            vec![
                "St_1_UE(tid.0:fresh, supi.0:fresh, idHN.0:fresh, k.0:fresh, sqn_root.0:fresh)"
                    .to_string(),
                "(i.0:node, 0)".to_string(),
            ]
        )
    );
    assert_eq!(
        args_of(
            Goal::Chain(
                (lv("vl", LSort::Node, 1), ConcIdx(0)),
                (lv("vk", LSort::Node, 0), PremIdx(0))
            ),
            &sys
        ),
        (
            "Chain",
            vec!["(vl.1:node, 0)".to_string(), "(vk.0:node, 0)".to_string()]
        )
    );
}

/// A fact name starting with `F_` or `L_` is the old way of marking a
/// solve-first or solve-last fact (the fork's `isSolveFirstFact` /
/// `isSolveLastFact`). The fork's `showFactTag` drops that prefix
/// (upstream's does not). `!AgentKey(ltkR.0:fresh)` is the fork's
/// output for `!F_AgentKey(~ltk)` in csf17/counter-protocol.spthy; the other
/// cases follow that code.
#[test]
fn f_and_l_name_prefixes_are_dropped() {
    let sys = System::empty();
    let premise = |mult, name: &str| {
        let fa = proto_fact(mult, name, vec![v("ltkR", LSort::Fresh, 0)]);
        let goal = Goal::Premise((lv("i", LSort::Node, 0), PremIdx(0)), fa);
        args_of(goal, &sys).1.remove(0)
    };
    assert_eq!(
        premise(Multiplicity::Persistent, "F_AgentKey"),
        "!AgentKey(ltkR.0:fresh)"
    );
    assert_eq!(
        premise(Multiplicity::Linear, "L_State"),
        "State(ltkR.0:fresh)"
    );
    // One prefix only, `F_` tried first, and only at the start.
    assert_eq!(premise(Multiplicity::Linear, "F_L_X"), "L_X(ltkR.0:fresh)");
    assert_eq!(premise(Multiplicity::Linear, "XF_Y"), "XF_Y(ltkR.0:fresh)");
    // Facts inside formulas go through the same `showFactTag`.
    assert_eq!(
        disj_args(&["All #j. F_Inv(id) @ #j ==> F"]),
        ["All j.0:node. (Inv(id.0:msg) @ j.0:node) ==> F"]
    );
}

#[test]
fn subterm_goals() {
    let sys = System::empty();
    let n_plus_one = f_app_ac(AcSym::NatPlus, vec![v("n", LSort::Nat, 0), nat_one()]);
    assert_eq!(
        args_of(Goal::Subterm((v("n", LSort::Nat, 1), n_plus_one)), &sys),
        (
            "Subterm",
            vec!["n.1:nat".to_string(), "(n.0:nat %+ %1)".to_string()]
        )
    );
    let k_hc = pair(v("k", LSort::Fresh, 0), fun("h", vec![pub_term("c")]));
    assert_eq!(
        args_of(Goal::Subterm((v("x", LSort::Msg, 0), k_hc)), &sys).1,
        ["x.0:msg", "<k.0:fresh, h('c')>"]
    );
}

#[test]
fn split_goals_print_each_case_of_the_split() {
    let x30 = || v("x", LSort::Msg, 30);
    let with_var = SubstVFresh::from_list([
        (lv("z", LSort::Msg, 0), fst(x30())),
        (lv("x", LSort::Msg, 1), x30()),
        (lv("z", LSort::Msg, 2), snd(x30())),
    ]);
    let without_vars = SubstVFresh::from_list([
        (lv("hkI", LSort::Msg, 0), dh_neutral()),
        (lv("z", LSort::Msg, 0), dh_neutral()),
    ]);
    let mut eq = EquationStore::empty();
    eq.conj = vec![
        EqDisj {
            split_id: SplitId(0),
            substs: vec![with_var],
        },
        EqDisj {
            split_id: SplitId(1),
            substs: vec![without_vars],
        },
    ];
    let mut sys = System::empty();
    sys.set_eq_store(Arc::new(eq));
    assert_eq!(
        args_of(Goal::Split(SplitId(0)), &sys),
        (
            "Split",
            vec![
                "Ex x.30:msg; z.0:msg = fst(x.30:msg) & x.1:msg = x.30:msg & z.2:msg = snd(x.30:msg)"
                    .to_string()
            ]
        )
    );
    assert_eq!(
        args_of(Goal::Split(SplitId(1)), &sys).1,
        ["Ex ; hkI.0:msg = DH_neutral & z.0:msg = DH_neutral"]
    );
    // A split the store doesn't have (the fork errors here).
    assert!(args_of(Goal::Split(SplitId(7)), &sys).1.is_empty());
}

/// The Disj goal over `formulas`, each parsed and converted to a guarded
/// formula.
fn disj_args(formulas: &[&str]) -> Vec<String> {
    let sig = nat_maude_sig();
    let gs = formulas
        .iter()
        .map(|s| formula_to_guarded_parsed(&parse_formula_str(s, &sig).unwrap(), &sig).unwrap())
        .collect();
    args_of(Goal::Disj(Disj(gs)), &System::empty()).1
}

#[test]
fn disj_goals_print_each_disjunct() {
    assert_eq!(
        disj_args(&[
            "Ex #j. Running(b, a, <'UE', 'eNB', 'K_eNB_star', r>) @ #j & not(last(#j)) \
             & (All a2 b2 #i2. Commit(a2, b2, <'UE', 'eNB', 'K_eNB_star', r>) @ #i2 \
             ==> (last(#i2) | #i2 = #i))",
            "All #k. K(GTK)[+] @ #k ==> F",
            "#i < #j",
            // Both sides free.
            "(%n %+ %1) << n.1",
        ]),
        [
            "Ex j.0:node. (Running(b.0:msg, a.0:msg, <'UE', 'eNB', 'K_eNB_star', r.0:msg>) @ j.0:node) \
             & (not(last(j.0:node))) & (All a2.0:msg b2.0:msg i2.0:node. \
             (Commit(a2.0:msg, b2.0:msg, <'UE', 'eNB', 'K_eNB_star', r.0:msg>) @ i2.0:node) \
             ==> ((last(i2.0:node)) | (i2.0:node = i.0:node)))",
            "All k.0:node. (K(GTK.0:msg)[+] @ k.0:node) ==> F",
            "i.0:node < j.0:node",
            "(n.0:nat %+ %1) << n.1:msg",
        ]
    );
}
