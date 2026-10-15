use super::*;

use std::sync::Arc;

use tamarin_parser::ast as p;
use tamarin_parser::parser::parse_formula_str;
use tamarin_term::builtin::{pair, pk, union, xor};
use tamarin_term::lterm::pub_term;

use crate::bliss_proc::bliss_available;
use crate::constraint::constraints::{Edge, NodeId};
use crate::constraint::system::GoalStatus;
use crate::fact::{ku_fact, proto_fact, proto_fact_ann, FactAnnotation, Multiplicity};
use crate::guarded::formula_to_guarded;
use crate::rule::{
    ConcIdx, PremIdx, ProtoRuleACInstInfo, ProtoRuleName, Rule, RuleAttributes, RuleInfo,
};
use crate::tools::equation_store::{EqDisj, LNSubst, LNSubstVFresh, SplitId};

fn msg(name: &str, idx: u64) -> LVar {
    LVar::new(name, LSort::Msg, idx)
}

fn fresh(name: &str, idx: u64) -> LVar {
    LVar::new(name, LSort::Fresh, idx)
}

fn node(idx: u64) -> NodeId {
    LVar::new("i", LSort::Node, idx)
}

fn v(x: LVar) -> LNTerm {
    var_term(x)
}

fn fact(name: &str, terms: Vec<LNTerm>) -> LNFact {
    proto_fact(Multiplicity::Linear, name, terms)
}

fn rule(
    name: &'static str,
    prems: Vec<LNFact>,
    acts: Vec<LNFact>,
    concs: Vec<LNFact>,
) -> RuleACInst {
    Rule::new(
        RuleInfo::Proto(ProtoRuleACInstInfo {
            name: ProtoRuleName::Stand(name),
            attributes: RuleAttributes::empty(),
            loop_breakers: Vec::new(),
        }),
        prems,
        concs,
        acts,
    )
}

fn g(s: &str) -> Guarded {
    let f = parse_formula_str(s).unwrap_or_else(|e| panic!("parse {s:?}: {e}"));
    formula_to_guarded(&f).unwrap_or_else(|e| panic!("formula_to_guarded {s:?}: {e}"))
}

fn alt(entries: Vec<(LVar, LNTerm)>) -> LNSubstVFresh {
    LNSubstVFresh::from_list(entries)
}

fn set_conj(sys: &mut System, conj: Vec<Vec<LNSubstVFresh>>) {
    let store = sys.eq_store_mut();
    store.conj = conj
        .into_iter()
        .enumerate()
        .map(|(i, substs)| EqDisj {
            split_id: SplitId(i as i64),
            substs,
        })
        .collect();
}

fn goals(sys: &mut System, goals: Vec<(Goal, bool)>) {
    sys.content_mut().goals = Arc::new(
        goals
            .into_iter()
            .map(|(goal, solved)| {
                (
                    goal,
                    GoalStatus {
                        solved,
                        ..Default::default()
                    },
                )
            })
            .collect(),
    );
}

const ALL: TermGraphOptions = TermGraphOptions {
    check_generators: true,
    certificate: true,
};

/// The term-graph form, with the generator check and the certificate.
fn canon(sys: &System) -> (CanonicalSystem, CanonLabelling, TermGraphStats) {
    let (form, labelling, stats) =
        canonicalize_constraint_system_via_term_graph_with_labelling(sys, ALL)
            .expect("canonicalize");
    assert_eq!(
        stats.non_genuine_generators, 0,
        "every automorphism of the term graph must be a symmetry of the system"
    );
    (form, labelling, stats)
}

fn form(sys: &System) -> CanonicalSystem {
    canon(sys).0
}

// -- The worked example (PLAN.md) ---------------------------------------------

/// Two rule instances joined by an edge, a formula, and an equation-store
/// disjunction `{x ↦ pk(u1)} ∨ {x ↦ u2 ⊕ v2}`, over the given variables.
fn worked_example(x: LVar, k: LVar, i: NodeId, j: NodeId, locals: [LVar; 3]) -> System {
    let st = || fact("St", vec![v(x), v(k)]);
    let mut sys = System::empty();
    sys.add_node(i, rule("R1", vec![], vec![], vec![st()]));
    sys.add_node(
        j,
        rule("R2", vec![st()], vec![fact("Fin", vec![v(x)])], vec![]),
    );
    sys.content_mut().edges.push(Edge {
        src: (i, ConcIdx(0)),
        tgt: (j, PremIdx(0)),
    });
    let formula = format!("All #t. K({}) @ #t ==> F", pretty_var(x));
    sys.formulas_mut().push(Arc::new(g(&formula)));
    let [u1, u2, v2] = locals;
    set_conj(
        &mut sys,
        vec![vec![
            alt(vec![(x, pk(v(u1)))]),
            alt(vec![(x, xor(v(u2), v(v2)))]),
        ]],
    );
    sys
}

/// `x` as the formula parser reads it back to the same `LVar`.
fn pretty_var(x: LVar) -> String {
    format!("{}.{}", x.name, x.idx)
}

fn worked() -> System {
    worked_example(
        msg("x", 0),
        fresh("k", 0),
        node(0),
        node(1),
        [msg("u", 0), msg("u", 1), msg("v", 1)],
    )
}

#[test]
fn the_worked_example_has_one_automorphism_and_it_fixes_every_free_variable() {
    if !bliss_available() {
        return;
    }
    let sys = worked();
    let term = system_to_term(&sys);
    let shown = term.to_string();
    for part in [
        "∃_Root",
        "Nodes{RuleInst[R1](",
        "Edges{Edge[0,0](",
        "LessAtoms{}, ⟨NoLast⟩",
        "EqConj{EqDisj{∃_EqAlternative",
    ] {
        assert!(shown.contains(part), "{part:?} not in {shown}");
    }

    let (_, labelling, stats) = canon(&sys);
    assert_eq!(stats.free_vars, 4, "x, ~k, #i, #j");
    assert_eq!(
        stats.aut_size.as_deref(),
        Some("2"),
        "only u2 <-> v2 under the xor of the second alternative"
    );
    assert_eq!(stats.generators_moving_free_vars, 0);
    assert_eq!(
        stats.generator_moves.keys().collect::<Vec<_>>(),
        [
            "Binder(EqAlternative) > BoundVar(EqAlternative,Msg)",
            "Fun(Ac(Xor)) > Occ(Msg)"
        ]
    );
    let theta = labelling.theta();
    assert_eq!(theta[&msg("x", 0)], LVar::new("mv", LSort::Msg, 0));
    assert_eq!(theta[&fresh("k", 0)], LVar::new("fv", LSort::Fresh, 0));
    assert_ne!(theta[&node(0)], theta[&node(1)]);
}

#[test]
fn a_renamed_copy_has_the_same_form_and_certificate() {
    if !bliss_available() {
        return;
    }
    let a = worked();
    let b = worked_example(
        msg("y", 7),
        fresh("q", 3),
        node(9),
        node(2),
        [msg("w", 5), msg("w", 4), msg("w", 3)],
    );
    let (form_a, theta_a, stats_a) = canon(&a);
    let (form_b, theta_b, stats_b) = canon(&b);
    assert_eq!(form_a, form_b);
    assert_eq!(stats_a.certificate, stats_b.certificate);
    assert_eq!(theta_a.theta()[&node(0)], theta_b.theta()[&node(9)]);
    assert_eq!(theta_a.theta()[&msg("x", 0)], theta_b.theta()[&msg("y", 7)]);
}

#[test]
fn moving_the_edge_changes_the_form() {
    if !bliss_available() {
        return;
    }
    let a = worked();
    let mut b = worked();
    b.content_mut().edges[0] = Edge {
        src: (node(1), ConcIdx(0)),
        tgt: (node(0), PremIdx(0)),
    };
    assert_ne!(form(&a), form(&b));
}

// -- Faithfulness of the encoding ---------------------------------------------

fn ku_goal_system(terms: Vec<LNTerm>) -> System {
    let mut sys = System::empty();
    goals(
        &mut sys,
        terms
            .into_iter()
            .enumerate()
            .map(|(n, t)| (Goal::Action(node(n as u64), ku_fact(t)), false))
            .collect(),
    );
    sys
}

/// `x ∪ x ∪ y` next to `<'a', x>`: which variable occurs twice under the AC
/// symbol is pinned by the second goal. One occurrence vertex per
/// occurrence: with one vertex per variable, `x ∪ x ∪ y` and `x ∪ y ∪ y`
/// would have the same edges.
fn doubled_union(doubled: LNTerm, single: LNTerm, anchored: LNTerm) -> System {
    ku_goal_system(vec![
        union(union(doubled.clone(), doubled), single),
        pair(pub_term("a"), anchored),
    ])
}

#[test]
fn ac_multiplicity_of_a_variable_counts() {
    if !bliss_available() {
        return;
    }
    let (x, y) = (v(msg("x", 0)), v(msg("y", 0)));
    let xxy = doubled_union(x.clone(), y.clone(), x.clone());
    let xyy = doubled_union(y.clone(), x.clone(), x.clone());
    let (z, w) = (v(msg("z", 3)), v(msg("w", 1)));
    let zzw = doubled_union(z.clone(), w, z);
    assert_ne!(form(&xxy), form(&xyy));
    assert_eq!(form(&xxy), form(&zzw));
}

#[test]
fn a_duplicated_formula_is_not_a_single_one() {
    if !bliss_available() {
        return;
    }
    let with = |n: usize| {
        let mut sys = System::empty();
        for _ in 0..n {
            sys.formulas_mut()
                .push(Arc::new(g("All #t. P(x.0) @ #t ==> F")));
        }
        sys
    };
    assert_ne!(form(&with(1)), form(&with(2)));
}

#[test]
fn names_are_constants() {
    if !bliss_available() {
        return;
    }
    let ku = |tag: &str, var: &str| ku_goal_system(vec![pair(pub_term(tag), v(msg(var, 0)))]);
    assert_ne!(form(&ku("1", "x")), form(&ku("2", "x")));
    assert_eq!(form(&ku("1", "x")), form(&ku("1", "y")));
}

#[test]
fn which_node_is_last_counts() {
    if !bliss_available() {
        return;
    }
    let with_last = |last: u64| {
        let mut sys = System::empty();
        sys.add_node(node(0), rule("A", vec![], vec![], vec![]));
        sys.add_node(node(1), rule("A", vec![], vec![], vec![fact("M", vec![])]));
        sys.content_mut().last_atom = Some(node(last));
        sys
    };
    assert_ne!(form(&with_last(0)), form(&with_last(1)));
}

/// Two rules with the same facts but different names give different forms;
/// renaming the variable still merges.
#[test]
fn which_rule_a_node_instantiates_counts() {
    if !bliss_available() {
        return;
    }
    let with_rule = |name: &'static str, var: LVar| {
        let mut sys = System::empty();
        sys.add_node(
            node(0),
            rule(
                name,
                vec![fact("In", vec![v(var)])],
                vec![],
                vec![fact("M", vec![v(var)])],
            ),
        );
        form(&sys)
    };
    assert_ne!(with_rule("A1", msg("x", 0)), with_rule("A2", msg("x", 0)));
    assert_eq!(with_rule("A1", msg("x", 0)), with_rule("A1", msg("y", 3)));
}

/// Annotations are solver hints: rule facts and formula facts that differ
/// only in them are equal.
#[test]
fn fact_annotations_do_not_count() {
    if !bliss_available() {
        return;
    }
    let annotated = |ann: bool| {
        let anns = if ann {
            [FactAnnotation::SolveFirst].into_iter().collect()
        } else {
            Default::default()
        };
        let mut sys = System::empty();
        let st = proto_fact_ann(Multiplicity::Linear, "St", anns, vec![v(msg("x", 0))]);
        sys.add_node(node(0), rule("R", vec![st], vec![], vec![]));
        let formula_fact = GFact {
            persistent: false,
            name: "P".into(),
            args: Arc::from([GTerm::Var(BVar::Free(p::VarSpec {
                name: "x".into(),
                idx: 0,
                sort: p::SortHint::Msg,
                typ: None,
            }))]),
            annotations: if ann {
                vec![p::FactAnnotation::SolveFirst]
            } else {
                Vec::new()
            },
        };
        sys.formulas_mut().push(Arc::new(Guarded::Disj(Arc::from([
            Guarded::Atom(GAtom::Pred(formula_fact)),
            crate::guarded::gfalse(),
        ]))));
        sys
    };
    assert_eq!(form(&annotated(false)), form(&annotated(true)));
}

// -- The equation store -------------------------------------------------------

/// Each alternative is its own scope: renaming its local variables and
/// reordering the alternatives changes nothing.
#[test]
fn alpha_equivalent_alternatives_in_either_order_are_equal() {
    if !bliss_available() {
        return;
    }
    let (x, y) = (msg("x", 0), msg("y", 0));
    let system = |alts: Vec<LNSubstVFresh>| {
        let mut sys = ku_goal_system(vec![pair(v(x), v(y))]);
        set_conj(&mut sys, vec![alts]);
        sys
    };
    let (u, w) = (msg("u", 1), msg("w", 2));
    let a = system(vec![
        alt(vec![(x, pk(v(u))), (y, v(u))]),
        alt(vec![(x, v(w)), (y, pk(v(w)))]),
    ]);
    let b = system(vec![
        alt(vec![(x, v(u)), (y, pk(v(u)))]),
        alt(vec![(x, pk(v(w))), (y, v(w))]),
    ]);
    let swapped_keys = system(vec![
        alt(vec![(x, pk(v(u))), (y, v(u))]),
        alt(vec![(x, pk(v(w))), (y, v(w))]),
    ]);
    assert_eq!(form(&a), form(&b));
    assert_ne!(form(&a), form(&swapped_keys));
}

/// A range variable is local even when its raw identity is a key's: it
/// cannot be captured by the free variable of the same name.
#[test]
fn a_range_variable_named_like_a_key_is_still_local() {
    if !bliss_available() {
        return;
    }
    let x = msg("x", 0);
    let system = |range: LVar| {
        let mut sys = ku_goal_system(vec![v(x)]);
        set_conj(&mut sys, vec![vec![alt(vec![(x, pk(v(range)))])]]);
        sys
    };
    assert_eq!(form(&system(x)), form(&system(msg("u", 7))));
}

/// The store-only variables `a`, `b` link two disjunctions (the shape of
/// idbased/BP_IBS_1): renaming them and reordering everything changes
/// nothing, and cutting the link changes the form.
#[test]
fn store_only_variables_link_disjunctions() {
    if !bliss_available() {
        return;
    }
    let (x, x1) = (msg("x", 0), msg("x", 1));
    let linking = |a: LVar, b: LVar, w: u64| {
        let w0 = v(msg("w", w));
        let (w1, w2) = (v(msg("w", w + 1)), v(msg("w", w + 2)));
        vec![
            alt(vec![
                (x, w0.clone()),
                (x1, w0.clone()),
                (a, w0.clone()),
                (b, w0),
            ]),
            alt(vec![
                (x, w1.clone()),
                (x1, w2.clone()),
                (a, w1.clone()),
                (b, w2.clone()),
            ]),
            alt(vec![(x, w1.clone()), (x1, w2.clone()), (a, w2), (b, w1)]),
        ]
    };
    let variant = |a: LVar, b: LVar, w: u64| {
        let p = pub_term("P");
        vec![
            alt(vec![(a, v(msg("w", w))), (b, p.clone())]),
            alt(vec![(a, p), (b, v(msg("w", w + 1)))]),
        ]
    };
    let system = |conj: Vec<Vec<LNSubstVFresh>>| {
        let mut sys = ku_goal_system(vec![pair(v(x), v(x1))]);
        set_conj(&mut sys, conj);
        sys
    };
    let (a, b) = (msg("x", 4), msg("x", 5));
    let linked = system(vec![linking(a, b, 20), variant(a, b, 30)]);
    let (a2, b2) = (msg("y", 41), msg("y", 40));
    let mut reversed = linking(a2, b2, 70);
    reversed.reverse();
    let renamed = system(vec![variant(a2, b2, 50), reversed]);
    let (c, d) = (msg("x", 6), msg("x", 7));
    let unlinked = system(vec![linking(a, b, 20), variant(c, d, 30)]);
    assert_eq!(form(&linked), form(&renamed));
    assert_ne!(form(&linked), form(&unlinked));
}

/// The store-only variables `a`, `b` look alike (each occurs once, with a
/// variable range), but swapping them gives a different store that is
/// equivalent only via the renaming `a ↔ b`.
#[test]
fn symmetric_store_only_variables_get_the_same_form() {
    if !bliss_available() {
        return;
    }
    let (x, a, b) = (msg("x", 0), msg("x", 4), msg("x", 5));
    let system = |first: LVar, second: LVar| {
        let w = v(msg("w", 20));
        let mut sys = ku_goal_system(vec![v(x)]);
        set_conj(
            &mut sys,
            vec![vec![
                alt(vec![(x, w.clone()), (first, w.clone())]),
                alt(vec![(x, pub_term("P")), (second, w)]),
            ]],
        );
        sys
    };
    assert_eq!(form(&system(a, b)), form(&system(b, a)));
}

/// `subst`'s domain variables no longer occur anywhere else once
/// `substSystem` has applied it; they are free variables all the same.
#[test]
fn a_substituted_variable_outside_the_graph_is_named() {
    if !bliss_available() {
        return;
    }
    let x = msg("x", 0);
    let system = |gone: LVar| {
        let mut sys = ku_goal_system(vec![v(x)]);
        sys.eq_store_mut().subst = LNSubst::from_list(vec![(gone, v(x))]);
        sys
    };
    let (canon, labelling, _) = canon(&system(msg("x", 9)));
    assert!(labelling.theta().contains_key(&msg("x", 9)));
    assert_eq!(canon, form(&system(msg("y", 3))));
    assert_ne!(canon, form(&ku_goal_system(vec![v(x)])));
}

/// The induction hypothesis puts a free `Last(#v)` into a formula; the term
/// graph names `#v` like any other free variable.
#[test]
fn a_free_node_variable_only_in_a_formula_is_named() {
    if !bliss_available() {
        return;
    }
    let vs = p::VarSpec {
        name: "v".into(),
        idx: 3,
        sort: p::SortHint::Node,
        typ: None,
    };
    let mut sys = System::empty();
    sys.formulas_mut()
        .push(Arc::new(Guarded::Atom(GAtom::Last(GTerm::Var(
            BVar::Free(vs.clone()),
        )))));
    let (_, labelling, _) = canon(&sys);
    assert_eq!(
        labelling.theta()[&varspec_to_lvar(&vs)],
        LVar::new("tv", LSort::Node, 0)
    );
}

// -- Ties that only part of the system breaks --------------------------------

/// `!KU(x ⊕ y)` alone cannot tell `x` from `y`; a subterm goal on `x` can.
/// Swapping the raw indices must not change the form.
#[test]
fn a_renaming_tie_is_broken_by_the_rest_of_the_system() {
    if !bliss_available() {
        return;
    }
    let with_raw_indices = |x_idx: u64, y_idx: u64| {
        let (x, y) = (v(msg("x", x_idx)), v(msg("y", y_idx)));
        let mut sys = System::empty();
        goals(
            &mut sys,
            vec![
                (Goal::Action(node(0), ku_fact(xor(x.clone(), y))), false),
                (Goal::Subterm((x, pub_term("c"))), false),
            ],
        );
        sys
    };
    assert_eq!(form(&with_raw_indices(1, 2)), form(&with_raw_indices(2, 1)));
}

fn ku_pair_goals(goals_spec: &[(u64, &str, &str, bool)]) -> System {
    let mut sys = System::empty();
    goals(
        &mut sys,
        goals_spec
            .iter()
            .map(|&(n, tag, var, solved)| {
                let t = pair(pub_term(tag), v(msg(var, 0)));
                (Goal::Action(node(n), ku_fact(t)), solved)
            })
            .collect(),
    );
    sys
}

#[test]
fn a_tie_break_is_independent_of_node_numbering_and_variable_names() {
    if !bliss_available() {
        return;
    }
    let original = ku_pair_goals(&[
        (0, "a", "x", true),
        (1, "a", "y", false),
        (2, "b", "x", false),
        (3, "b", "y", false),
    ]);
    let renumbered_and_renamed = ku_pair_goals(&[
        (0, "a", "w", false),
        (1, "b", "u", false),
        (2, "b", "w", false),
        (3, "a", "u", true),
    ]);
    let solved_moved = ku_pair_goals(&[
        (0, "a", "x", false),
        (1, "a", "y", false),
        (2, "b", "x", true),
        (3, "b", "y", false),
    ]);
    assert_eq!(form(&original), form(&renumbered_and_renamed));
    assert_ne!(form(&original), form(&solved_moved));
}

/// Two interchangeable sessions: bliss reports their swap, it moves free
/// variables, and it is a symmetry of the system.
#[test]
fn a_symmetry_of_the_system_is_a_genuine_generator() {
    if !bliss_available() {
        return;
    }
    let sys = ku_pair_goals(&[(0, "a", "x", false), (1, "a", "y", false)]);
    let (_, _, stats) = canon(&sys);
    assert_eq!(stats.aut_size.as_deref(), Some("2"));
    assert_eq!(stats.generators_moving_free_vars, 1);
}

/// The shape of `xor0`'s split: `~a ⊕ ~b` and `~a.1 ⊕ ~b.1`, and the
/// disjunction `{a, a.1 ↦ u; b, b.1 ↦ v} ∨ {a, b.1 ↦ v; b, a.1 ↦ u}`.
/// Swapping `~a` and `~b` swaps the two alternatives: a symmetry, which the
/// check must confirm. More node variables than fresh ones make a labelling
/// whose cursors are rebuilt from θ's image (`CanonLabelling::from_theta`)
/// name the alternatives' local variables differently, so a check built that
/// way reports the symmetry as not genuine.
#[test]
fn a_symmetry_swapping_equation_store_alternatives_is_genuine() {
    if !bliss_available() {
        return;
    }
    let (a, b, a1, b1) = (fresh("a", 0), fresh("b", 0), fresh("a", 1), fresh("b", 1));
    let (u, w) = (fresh("x", 6), fresh("x", 7));
    let mut sys = System::empty();
    let mut gs = vec![
        (Goal::Action(node(0), ku_fact(xor(v(a), v(b)))), false),
        (Goal::Action(node(1), ku_fact(xor(v(a1), v(b1)))), false),
    ];
    gs.extend((2..8).map(|n| (Goal::Action(node(n), ku_fact(pub_term("c"))), false)));
    goals(&mut sys, gs);
    set_conj(
        &mut sys,
        vec![vec![
            alt(vec![(a, v(u)), (b, v(w)), (a1, v(u)), (b1, v(w))]),
            alt(vec![(a, v(w)), (b, v(u)), (a1, v(u)), (b1, v(w))]),
        ]],
    );
    let (_, _, stats) = canon(&sys);
    assert!(stats.generators_moving_free_vars >= 1);
}

#[test]
fn a_system_without_free_variables_skips_bliss() {
    let mut sys = System::empty();
    sys.formulas_mut().push(Arc::new(crate::guarded::gtrue()));
    let (form_true, labelling, stats) =
        canonicalize_constraint_system_via_term_graph_with_labelling(
            &sys,
            TermGraphOptions::default(),
        )
        .expect("canonicalize");
    assert!(!stats.bliss_ran);
    assert!(labelling.theta().is_empty());

    let mut other = System::empty();
    other
        .formulas_mut()
        .push(Arc::new(crate::guarded::gfalse()));
    let (form_false, _, _) = canonicalize_constraint_system_via_term_graph_with_labelling(
        &other,
        TermGraphOptions::default(),
    )
    .expect("canonicalize");
    assert_ne!(form_true, form_false);
}

// -- Which systems are equal -------------------------------------------------

/// Over a set of systems, exactly the renamed copies get the same form.
#[test]
fn exactly_the_renamed_copies_are_equal() {
    if !bliss_available() {
        return;
    }
    let (x, y, z) = (v(msg("x", 0)), v(msg("y", 0)), v(msg("z", 2)));
    let mut systems = vec![
        worked(),
        worked_example(
            msg("y", 7),
            fresh("q", 3),
            node(9),
            node(2),
            [msg("w", 5), msg("w", 4), msg("w", 3)],
        ),
        doubled_union(x.clone(), y.clone(), x.clone()),
        doubled_union(y.clone(), x.clone(), x.clone()),
        doubled_union(z.clone(), x.clone(), z.clone()),
        ku_pair_goals(&[
            (0, "a", "x", true),
            (1, "a", "y", false),
            (2, "b", "x", false),
        ]),
        ku_pair_goals(&[
            (0, "a", "w", true),
            (1, "b", "w", false),
            (2, "a", "u", false),
        ]),
        ku_pair_goals(&[
            (0, "a", "x", false),
            (1, "a", "y", true),
            (2, "b", "x", false),
        ]),
    ];
    let mut moved = worked();
    moved.content_mut().edges[0] = Edge {
        src: (node(1), ConcIdx(0)),
        tgt: (node(0), PremIdx(0)),
    };
    systems.push(moved);

    let forms: Vec<CanonicalSystem> = systems.iter().map(form).collect();
    let renamed_copies = [(0, 1), (2, 4), (5, 6)];
    for i in 0..systems.len() {
        for j in i + 1..systems.len() {
            assert_eq!(
                forms[i] == forms[j],
                renamed_copies.contains(&(i, j)),
                "systems {i} and {j}"
            );
        }
    }
}
