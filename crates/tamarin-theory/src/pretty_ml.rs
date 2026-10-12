//! (No HS analog.) The goal strings the RL server shows TamRL's model
//! (`goal.type`, `goal.args`), printed as the TamRL Haskell fork's
//! `toJSONGoal` (Constraints.hs) prints them.
//!
//! The fork changed Tamarin's printers through their parameters and operator
//! constants: plain-ASCII operators, variables as `name.idx:sort`, fact names
//! without an `F_`/`L_` prefix, everything on one line. Here the same changes
//! are the styles [`ML_TERM`] and [`ML_FORMULA`], passed to the existing
//! one-line printers.

use tamarin_term::lterm::{LNTerm, LVar};
use tamarin_term::pretty::{pp_var_sorted, pretty_lnterm_styled, TermStyle};

use crate::constraint::constraints::{Goal, NodeId};
use crate::constraint::system::System;
use crate::fact::LNFact;
use crate::pretty_formula::{pretty_guarded_styled, FormulaStyle};
use crate::pretty_system::pretty_fact_styled;
use crate::tools::equation_store::{LNSubstVFresh, SplitId};

/// The fork's `prettyTerm` operators and `prettyLVar`.
pub const ML_TERM: TermStyle = TermStyle {
    mult: " * ",
    xor: " XOR ",
    union: " ++ ",
    nat_plus: " %+ ",
    sorted_vars: true,
};

/// The fork's `prettyGuarded`, `Theory.Text.Pretty` operators and
/// `prettyFact`.
pub const ML_FORMULA: FormulaStyle = FormulaStyle {
    term: ML_TERM,
    falsum: "F",
    verum: "T",
    or: " | ",
    and: " & ",
    implies: " ==> ",
    subterm: " << ",
    not: "not",
    forall: "All",
    exists: "Ex",
    flat_facts: true,
};

/// `goal.type` and `goal.args` of `goal`, which belongs to `sys` (a Split
/// goal's arguments come from `sys`'s equation store).
pub fn goal_type_and_args(goal: &Goal, sys: &System) -> (&'static str, Vec<String>) {
    match goal {
        Goal::Action(v, fa) => ("Action", vec![fact(fa), var(v)]),
        Goal::Chain((c, ci), (p, pi)) => ("Chain", vec![node_idx(c, ci.0), node_idx(p, pi.0)]),
        Goal::Premise((p, pi), fa) => ("Premise", vec![fact(fa), node_idx(p, pi.0)]),
        Goal::Split(id) => ("Split", split_args(sys, *id)),
        Goal::Disj(d) => (
            "Disj",
            d.0.iter()
                .map(|g| pretty_guarded_styled(g, &ML_FORMULA))
                .collect(),
        ),
        Goal::Subterm((l, r)) => ("Subterm", vec![term(l), term(r)]),
    }
}

fn term(t: &LNTerm) -> String {
    pretty_lnterm_styled(t, &ML_TERM)
}

fn fact(fa: &LNFact) -> String {
    pretty_fact_styled(fa, &ML_FORMULA)
}

fn var(v: &LVar) -> String {
    let mut s = String::new();
    pp_var_sorted(v.name, v.idx, v.sort, &mut s);
    s
}

/// `prettyNodeConc` / `prettyNodePrem`: `(i.0:node, 0)`.
fn node_idx(v: &NodeId, idx: usize) -> String {
    format!("({}, {idx})", var(v))
}

/// The fork's `getSubstAtSplitId`: the split's substitutions in set order
/// (`S.toList`), not in the case order a split emits. Empty if the store has
/// no such split (the fork errors there).
fn split_args(sys: &System, id: SplitId) -> Vec<String> {
    let Some(d) = sys.eq_store.conj.iter().find(|d| d.split_id == id) else {
        return Vec::new();
    };
    let mut substs: Vec<&LNSubstVFresh> = d.substs.iter().collect();
    substs.sort();
    substs.into_iter().map(split_subst).collect()
}

/// The fork's `ppSubst`: `Ex x.0:msg y.0:msg; a.0:msg = t & b.0:msg = u`
/// (with no fresh variables, `Ex ; ...`).
fn split_subst(s: &LNSubstVFresh) -> String {
    let vars: Vec<String> = s.vars_range().iter().map(var).collect();
    let eqs: Vec<String> = s
        .to_list()
        .iter()
        .map(|(v, t)| format!("{} = {}", var(v), term(t)))
        .collect();
    format!("Ex {}; {}", vars.join(" "), eqs.join(" & "))
}

#[cfg(test)]
#[path = "pretty_ml_tests.rs"]
mod tests;
