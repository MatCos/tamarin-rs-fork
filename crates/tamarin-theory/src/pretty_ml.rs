//! (No HS analog.) The goal strings the RL server shows TamRL's model
//! (`goal.type`, `goal.args`), printed as the TamRL Haskell fork's
//! `toJSONGoal` prints them.
//!
//! The fork changed Tamarin's printers through their parameters and operator
//! constants: plain-ASCII operators, variables as `name.idx:sort`, fact names
//! without an `F_`/`L_` prefix, everything on one line. Tamarin's own printers
//! here are `Doc` builders without those parameters, so this module has its
//! own one-line printers for terms, facts and guarded formulas.

use std::fmt::Write as _;

use tamarin_term::function_symbols::{
    diff_sym, exp_sym, nat_one_sym, pair_sym, AcSym, CSym, FunSym, EMAP_SYM_STRING,
};
use tamarin_term::lterm::{frees, sort_suffix, LNTerm, LVar};
use tamarin_term::pretty::{ac_op_symbol, pp_name};
use tamarin_term::term::Term;
use tamarin_term::vterm::Lit;
use tamarin_utils::fresh::PreciseFreshState;

use crate::atom::{Atom, ProtoAtom};
use crate::constraint::constraints::{Goal, NodeId};
use crate::constraint::system::System;
use crate::fact::{fact_tag_multiplicity, fact_tag_name, FactAnnotation, LNFact, Multiplicity};
use crate::formula::Quantifier;
use crate::guarded::{bvar_to_lvar, open_guarded, Guarded};
use crate::tools::equation_store::{LNSubstVFresh, SplitId};

/// `goal.type` and `goal.args` of `goal`, which belongs to `sys` (a Split
/// goal's arguments come from `sys`'s equation store).
pub fn goal_type_and_args(goal: &Goal, sys: &System) -> (&'static str, Vec<String>) {
    match goal {
        Goal::Action(v, fa) => ("Action", vec![fact(fa), var(v)]),
        Goal::Chain((c, ci), (p, pi)) => ("Chain", vec![node_idx(c, ci.0), node_idx(p, pi.0)]),
        Goal::Premise((p, pi), fa) => ("Premise", vec![fact(fa), node_idx(p, pi.0)]),
        Goal::Split(id) => ("Split", split_args(sys, *id)),
        Goal::Disj(d) => ("Disj", d.0.iter().map(guarded).collect()),
        Goal::Subterm((l, r)) => ("Subterm", vec![term(l), term(r)]),
    }
}

/// A term as a goal argument prints it.
pub fn term(t: &LNTerm) -> String {
    let mut s = String::new();
    pp_term(t, &mut s);
    s
}

fn var(v: &LVar) -> String {
    let mut s = String::new();
    pp_var(v, &mut s);
    s
}

/// The fork's `prettyLVar`: `name.idx:sort`, the index always printed.
fn pp_var(v: &LVar, out: &mut String) {
    let _ = write!(out, "{}.{}:{}", v.name, v.idx, sort_suffix(v.sort));
}

/// `prettyNodeConc` / `prettyNodePrem`: `(i.0:node, 0)`.
fn node_idx(v: &NodeId, idx: usize) -> String {
    format!("({}, {idx})", var(v))
}

/// The fork's `getSubstAtSplitId`: the split's substitutions in set order
/// (`S.toList`), not in the case order a split emits. Empty if the store has
/// no such split (the fork errors there).
fn split_args(sys: &System, id: SplitId) -> Vec<String> {
    let Some(d) = sys.eq_store().conj.iter().find(|d| d.split_id == id) else {
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

// =============================================================================
// Terms: the fork's `prettyTerm`
// =============================================================================

/// The fork's `prettyTerm` on one line: Tamarin's term printer with the
/// builtin AC operators spelled ` * `, ` XOR `, ` ++ `, ` %+ ` and the
/// variables as [`pp_var`] prints them.
fn pp_term(t: &LNTerm, out: &mut String) {
    match t {
        Term::Lit(Lit::Var(v)) => pp_var(v, out),
        Term::Lit(Lit::Con(n)) => pp_name(n, out),
        // A nullary user-defined AC symbol prints as its bare name.
        Term::App(FunSym::Ac(AcSym::AcFct(sym)), ts) if ts.is_empty() => {
            out.push_str(&String::from_utf8_lossy(sym.name));
        }
        Term::App(FunSym::Ac(o), ts) => {
            out.push('(');
            pp_args(ts, ac_separator(*o), out);
            out.push(')');
        }
        Term::App(FunSym::NoEq(sym), ts) if ts.len() == 2 && *sym == exp_sym() => {
            pp_term(&ts[0], out);
            out.push('^');
            pp_term(&ts[1], out);
        }
        Term::App(FunSym::NoEq(sym), ts) if ts.len() == 2 && *sym == diff_sym() => {
            out.push_str("diff(");
            pp_args(ts, ", ", out);
            out.push(')');
        }
        Term::App(FunSym::NoEq(sym), ts) if ts.is_empty() && *sym == nat_one_sym() => {
            out.push_str("%1");
        }
        Term::App(FunSym::NoEq(sym), _) if *sym == pair_sym() => {
            let mut flat: Vec<&LNTerm> = Vec::new();
            pair_spine(t, &mut flat);
            out.push('<');
            for (i, c) in flat.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                pp_term(c, out);
            }
            out.push('>');
        }
        Term::App(FunSym::NoEq(sym), ts) => {
            out.push_str(&String::from_utf8_lossy(sym.name));
            if !ts.is_empty() {
                out.push('(');
                pp_args(ts, ", ", out);
                out.push(')');
            }
        }
        Term::App(FunSym::C(CSym::EMap), ts) => {
            out.push_str(&String::from_utf8_lossy(EMAP_SYM_STRING));
            out.push('(');
            pp_args(ts, ", ", out);
            out.push(')');
        }
        Term::App(FunSym::List, ts) => {
            out.push_str("LIST(");
            pp_args(ts, ", ", out);
            out.push(')');
        }
    }
}

fn pp_args(ts: &[LNTerm], sep: &str, out: &mut String) {
    for (i, c) in ts.iter().enumerate() {
        if i > 0 {
            out.push_str(sep);
        }
        pp_term(c, out);
    }
}

/// The fork's AC separators. A user-defined AC symbol `f` keeps Tamarin's
/// ` f `.
fn ac_separator(op: AcSym) -> &'static str {
    match op {
        AcSym::Mult => " * ",
        AcSym::Xor => " XOR ",
        AcSym::Union => " ++ ",
        AcSym::NatPlus => " %+ ",
        AcSym::AcFct(_) => ac_op_symbol(op),
    }
}

/// `pair(a, pair(b, c))` as `[a, b, c]`. Only the right child continues the
/// list, so `pair(pair(a, b), c)` keeps its left child as one element.
fn pair_spine<'a>(t: &'a LNTerm, out: &mut Vec<&'a LNTerm>) {
    match t {
        Term::App(FunSym::NoEq(sym), ts) if ts.len() == 2 && *sym == pair_sym() => {
            out.push(&ts[0]);
            pair_spine(&ts[1], out);
        }
        _ => out.push(t),
    }
}

// =============================================================================
// Facts and atoms
// =============================================================================

/// The fork's `prettyFact`: `!Name(a, b)` with the annotations right after
/// the `)` (`!KU(x.0:fresh)[+]`), and an `F_`/`L_` name prefix dropped (the
/// fork's `showFactTag`).
fn fact(fa: &LNFact) -> String {
    let mut s = String::new();
    if fact_tag_multiplicity(&fa.tag) == Multiplicity::Persistent {
        s.push('!');
    }
    s.push_str(strip_fact_name_prefix(&fact_tag_name(&fa.tag)));
    s.push('(');
    pp_args(&fa.terms, ", ", &mut s);
    s.push(')');
    if !fa.annotations.is_empty() {
        let anns: Vec<&str> = fa
            .annotations
            .iter()
            .map(|a| match a {
                FactAnnotation::SolveFirst => "+",
                FactAnnotation::SolveLast => "-",
                FactAnnotation::NoSources => "no_precomp",
            })
            .collect();
        let _ = write!(s, "[{}]", anns.join(", "));
    }
    s
}

/// The TamRL Haskell fork's `showFactTag` drops an `F_` or `L_` prefix of a
/// fact name.
fn strip_fact_name_prefix(name: &str) -> &str {
    name.strip_prefix("F_")
        .or_else(|| name.strip_prefix("L_"))
        .unwrap_or(name)
}

fn atom(a: &Atom<LNTerm>, out: &mut String) {
    match a {
        ProtoAtom::Action(t, fa) => {
            out.push_str(&fact(fa));
            out.push_str(" @ ");
            pp_term(t, out);
        }
        ProtoAtom::EqE(l, r) => pp_infix(l, " = ", r, out),
        ProtoAtom::Less(l, r) => pp_infix(l, " < ", r, out),
        ProtoAtom::Subterm(l, r) => pp_infix(l, " << ", r, out),
        ProtoAtom::Last(t) => {
            out.push_str("last(");
            pp_term(t, out);
            out.push(')');
        }
        // An `Atom` carries no syntactic sugar.
        ProtoAtom::Syntactic(_) => {}
    }
}

fn pp_infix(l: &LNTerm, op: &str, r: &LNTerm, out: &mut String) {
    pp_term(l, out);
    out.push_str(op);
    pp_term(r, out);
}

// =============================================================================
// Guarded formulas: the fork's `prettyGuarded` on one line
// =============================================================================

/// A disjunct of a Disj goal. Bound variables get their display index the
/// way Tamarin's `prettyGuarded` draws it: a name that is also a free
/// variable of `g` starts above that variable's index.
fn guarded(g: &Guarded) -> String {
    let mut state =
        PreciseFreshState::avoid_precise(frees(g).into_iter().map(|v| (v.name.to_string(), v.idx)));
    let mut s = String::new();
    pp_guarded(g, false, &mut state, &mut s);
    s
}

/// `paren` is the caller's `opParens`, which wraps every shape, so a
/// disjunction under a conjunction prints as `((a | b))`.
fn pp_guarded(g: &Guarded, paren: bool, state: &mut PreciseFreshState, out: &mut String) {
    if paren {
        out.push('(');
    }
    match g {
        Guarded::Atom(a) => atom(&bvar_to_lvar(a), out),
        Guarded::Disj(xs) if xs.is_empty() => out.push('F'),
        Guarded::Conj(xs) if xs.is_empty() => out.push('T'),
        Guarded::Disj(xs) => {
            out.push('(');
            pp_junction(xs, " | ", state, out);
            out.push(')');
        }
        Guarded::Conj(xs) => pp_junction(xs, " & ", state, out),
        Guarded::GGuarded { .. } => state.scope_freshness(|state| pp_gguarded(g, state, out)),
    }
    if paren {
        out.push(')');
    }
}

fn pp_junction(xs: &[Guarded], sep: &str, state: &mut PreciseFreshState, out: &mut String) {
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            out.push_str(sep);
        }
        pp_guarded(x, true, state, out);
    }
}

/// `All x.0:msg. (A) & (B) ==> body`, `Ex x.0:msg. (A) & body`, and
/// `not(A)` for a universal with no variables and a false body.
fn pp_gguarded(g: &Guarded, state: &mut PreciseFreshState, out: &mut String) {
    let (qua, vs, guards, body) = open_guarded(g, state).expect("pp_gguarded: not a GGuarded");
    let mut dante = String::new();
    if guards.is_empty() {
        dante.push('T');
    }
    for (i, a) in guards.iter().enumerate() {
        if i > 0 {
            dante.push_str(" & ");
        }
        dante.push('(');
        atom(a, &mut dante);
        dante.push(')');
    }
    let body_false = matches!(&body, Guarded::Disj(xs) if xs.is_empty());
    let body_true = matches!(&body, Guarded::Conj(xs) if xs.is_empty());
    if matches!(qua, Quantifier::All) && vs.is_empty() && body_false {
        out.push_str("not");
        out.push_str(&dante);
        return;
    }
    out.push_str(match qua {
        Quantifier::All => "All ",
        Quantifier::Ex => "Ex ",
    });
    for (i, v) in vs.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        pp_var(v, out);
    }
    out.push_str(". ");
    out.push_str(&dante);
    if matches!(qua, Quantifier::Ex) && body_true {
        return;
    }
    out.push_str(match qua {
        Quantifier::All => " ==> ",
        Quantifier::Ex => " & ",
    });
    pp_guarded(&body, false, state, out);
}

#[cfg(test)]
#[path = "pretty_ml_tests.rs"]
mod tests;
