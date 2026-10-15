// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! (No HS analog.) Canonicalizing a whole constraint [`System`] through ONE
//! colored digraph: the system is written as one term ([`system_to_term`]),
//! the term becomes a colored digraph by the standard reduction from
//! AC-equality with commutative binders to graph isomorphism (Basin, *Equality
//! of terms containing associative-commutative functions and commutative
//! binding operators is isomorphism complete*, Cornell TR 89-1020, Lemma 2;
//! [`term_to_graph`]), and bliss's canonical labeling of that graph names the
//! system's variables.
//!
//! Every free variable is a vertex with an edge to each of its occurrences,
//! so bliss sees the whole system, and its canonical labeling is already
//! canonical: an automorphism of a faithful encoding is a symmetry of the
//! system, so any labeling bliss picks gives the same result, and nothing
//! has to be minimized over the automorphism group.
//!
//! **Soundness does not depend on the encoding.** bliss only decides the
//! variable names (θ). The canonical form is then built from the system
//! itself, renamed by θ ([`canonicalize_constraint_system_via_term_graph_with_labelling`]),
//! so two systems with equal forms are equivalent whatever the graph looked
//! like. A lossy encoding can only cost merges, never cause a false one; the
//! generator check ([`TermGraphOptions::check_generators`]) detects it.
//!
//! # Step 1: the system as one term
//!
//! [`STerm`] is a term over a signature of [`Op`]s, each free (ordered
//! arguments), AC (a multiset of arguments, nested applications flattened)
//! or C (two unordered arguments), plus a commutative binder
//! ([`STerm::Exists`]). `LNTerm` cannot hold it: it has no binders, and its
//! AC symbols belong to the theory.
//!
//! ```text
//! ∃_Root x̄. System[source_kind, side, contradictory](
//!     Nodes{…}, Edges{…}, LessAtoms{…}, Last(#i) | ⟨NoLast⟩,
//!     Formulas{…}, SolvedFormulas{…}, Lemmas{…},
//!     EqSubst{…}, EqConj{…}, Subterms{…}, SolvedSubterms{…}, NegSubterms{…}, Goals{…} )
//! ```
//!
//! Each `Name{…}` is an AC section operator with one argument per entry of
//! its field. The root binder binds every free variable of the system.
//!
//! **Collections.** A field that is a `Vec` in `System` is a multiset: an AC
//! operator keeps duplicates as separate arguments, because the solver
//! keeps alpha-equivalent entries apart and their count is proof-search
//! state (CONVENTIONS rule 2). That covers `nodes`, `edges`, `less_atoms`,
//! `formulas`, `solved_formulas`, `lemmas`, `goals`, a disjunction goal's
//! alternatives, `eq_store.conj`, each `EqDisj`'s alternatives,
//! `subterm_store.subterms`/`solved_subterms`, and a quantifier's guards.
//! Sets are deduplicated modulo AC first: `subterm_store.neg_subterms` (a
//! set in `System`), and `Conj`/`Disj` inside a formula (which
//! `ac_normalize_guarded` deduplicates). Deduplicating modulo AC before
//! renaming equals deduplicating after it, since the renaming is a bijection.
//! Sequences stay ordered: a rule's premises/actions/conclusions (edges and
//! goals refer to their indices), arguments of free symbols, and a
//! quantifier's binders (de Bruijn indices refer to their positions).
//!
//! **Field audit** (CONVENTIONS rule 3; the same verdicts as the
//! `CanonicalSystem` built from the system):
//!
//! | Field | Term |
//! |---|---|
//! | `nodes` | `RuleInst[rule](#i, Prems(…), Acts(…), Concs(…))`; `rule` is the rule's identity ([`RuleId`]); `new_vars` excluded (derived from the facts) |
//! | facts | `Fact[tag](t1 … tn)`; annotations excluded: solver hints without semantics |
//! | `LNTerm` | `Fun(f)` free / AC (flattened) / C by the symbol; a name is a constant; a variable an occurrence |
//! | `edges` | `Edge[c, p](#src, #tgt)` |
//! | `less_atoms` | `Less(#a, #b)`; `reason` excluded |
//! | `last_atom` | `Last(#i)` or the constant `NoLast` |
//! | `formulas`, `solved_formulas`, `lemmas` | the `ac_normalize_guarded` form, translated as below; three separate sections |
//! | `eq_store.subst` | `EqBind(#x, t)` |
//! | `eq_store.conj` | `EqDisj{ ∃_Alt ȳ. EqAlt{ EqBind(#x, t) … } … }`: every variable in an alternative's range is local to that alternative ([`VarRef::Local`]), so each alternative is its own commutative existential scope; `split_id` and `next_split` excluded |
//! | `subterm_store` | `Sub(s, t)` in three sections; `contradictory` in `System`'s label; `old_neg_subterms`, `propagated` excluded |
//! | `goals` | `Goal[solved](payload)`; `Split` goals (they only name an `EqDisj`), `looping` and `nr` excluded |
//! | `source_kind`, `side` | `System`'s label |
//! | `next_goal_nr`, `used_sources`, `sources_lemma_universals`, caches | excluded (PLAN.md's field audit) |
//!
//! Formulas keep their de Bruijn indices as constants, so a quantifier
//! block's binder order stays significant, as in the `CanonicalSystem`.
//! `Conj`/`Disj` and a quantifier's guards are AC, `=` is C, `em` is C, the
//! AC binary operators of formula terms (`is_ac_binop`) are AC.
//!
//! **Closed subterms** (an `LNTerm`, fact, formula, atom or formula term
//! without a free variable) become one constant keyed by the fingerprint of
//! their AC normal form: two of them are equal modulo AC iff their normal
//! forms are, and no renaming touches them. This keeps lemmas and ground
//! terms down to one vertex each.
//!
//! # Step 2: the term as a colored digraph
//!
//! [`term_to_graph`] follows the reduction case by case. Every vertex gets a
//! key `(label, pos)`, where `pos` is its argument position under a free
//! parent and `None` under an AC or C parent or a binder:
//!
//! - a constant is one vertex;
//! - an application is one vertex with an edge to each argument (an AC
//!   operator's nested applications of itself are flattened into it);
//! - a variable occurrence is a NEW vertex `Occ(sort)`, with an edge from the
//!   vertex of the variable, `BoundVar(scope, sort)`;
//! - a binder is a vertex with an edge to each of its variables' vertices and
//!   one to its body.
//!
//! Departures from the paper, each keeping the encoding faithful:
//! - Edge labels (argument positions) are folded into the child's key: bliss
//!   colors vertices only, and every vertex but a variable vertex has exactly
//!   one parent, so "my i-th child" and "my child at position i" coincide.
//! - Every occurrence gets its own vertex. Pointing the parents at one vertex
//!   per variable would make the AC terms `f(x, x, y)` and `f(x, y, y)` the
//!   same edge set, since bliss drops duplicate edges.
//! - Sorts are part of the variable and occurrence labels, so a renaming that
//!   an isomorphism induces is sort-respecting.
//!
//! A vertex's color is the RANK of its key among the keys of the graph: an
//! isomorphism between two systems' graphs maps keys to equal keys, so both
//! have the same key set and the same ranks.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::time::{Duration, Instant};

use tamarin_term::alpha_eq_ac::{apply_renaming, rename_var, CanonLabelling};
use tamarin_term::fingerprint::fingerprint_term;
use tamarin_term::function_symbols::FunSym;
use tamarin_term::lterm::{HasFrees, LNTerm, LSort, LVar, Name};
use tamarin_term::term::Term;
use tamarin_term::vterm::{var_term, Lit};
use tamarin_utils::fingerprint::{Fingerprint, FingerprintHasher};

use crate::bliss_proc::{run_bliss_with_stats, to_dimacs, BlissError, Color, Permutation};
use crate::canon::{
    canonicalize_system_content, fact_to_term, fingerprint_atom, fingerprint_gterm, graph_marker,
    graph_tag, index_marker, rule_id, rule_identity_marker, varspec_to_lvar, CanonicalSystem,
    RuleId,
};
use crate::constraint::constraints::Goal;
use crate::constraint::system::{Side, SourceKind, System};
use crate::fact::{FactTag, LNFact};
use crate::guarded::{self, BVar, GAtom, GFact, GTerm, Guarded, Quant};
use crate::rule::RuleACInst;

// =============================================================================
// The term language
// =============================================================================

/// A term of the system's signature (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub enum STerm {
    App(Op, Vec<STerm>),
    Var(VarRef),
    Const(ConstKey),
    /// A commutative binder `∃ x1 … xn. body`.
    Exists(Scope, Vec<LVar>, Box<STerm>),
}

/// A variable occurrence. Free variables are bound by the root binder; a
/// local one by the innermost enclosing [`STerm::Exists`] that lists it.
/// Keeping the two apart, rather than resolving by name, means an
/// equation-store alternative whose range happens to reuse a free
/// variable's raw identity cannot capture it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VarRef {
    Free(LVar),
    Local(LVar),
}

/// Which binder an [`STerm::Exists`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scope {
    /// Binds every free variable of the system.
    Root,
    /// Binds the range variables of one equation-store alternative.
    EqAlternative,
}

/// How an operator treats its arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    Free,
    Ac,
    C,
}

/// The fields of the system that are collections, in `System`'s argument
/// order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Section {
    Nodes,
    Edges,
    LessAtoms,
    Formulas,
    SolvedFormulas,
    Lemmas,
    EqSubst,
    EqConj,
    Subterms,
    SolvedSubterms,
    NegSubterms,
    Goals,
}

/// Which group of a rule's facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FactGroup {
    Prems,
    Acts,
    Concs,
}

/// The operators. Structural ones are their own variants, so they can never
/// collide with a theory symbol (which only ever appears wrapped in `Fun`,
/// `Fact`, `GFact` or a `G…` formula-term variant).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Op {
    System {
        source_kind: Option<SourceKind>,
        side: Option<Side>,
        contradictory: bool,
    },
    Section(Section),
    RuleInst(RuleId),
    FactGroup(FactGroup),
    Fact(FactTag),
    Fun(FunSym),
    Edge(usize, usize),
    Less,
    Last,
    EqBind,
    EqDisj,
    EqAlt,
    Sub,
    Goal {
        solved: bool,
    },
    GoalAction,
    GoalChain(usize, usize),
    GoalPremise(usize),
    GoalDisj,
    GoalSubterm,
    // Formulas (`Guarded`) ...
    GConj,
    GDisj,
    /// A quantifier block: existential or not, and the sort hints of its
    /// binders in order (spelled as written, as the `CanonicalSystem` keeps
    /// them).
    GQuant {
        ex: bool,
        sorts: Vec<String>,
    },
    GGuards,
    GEq,
    GLess,
    GLessMset,
    GSubterm,
    GAction,
    GLast,
    GPred,
    GFact {
        name: String,
        persistent: bool,
    },
    // ... and their terms (`GTerm`).
    GApp(String),
    /// The commutative application `em(a, b)`.
    GAppC(String),
    GAlgApp(String),
    GPair,
    GDiff,
    GPatMatch,
    GBinOp {
        op: String,
        ac: bool,
    },
}

impl Op {
    pub fn kind(&self) -> OpKind {
        match self {
            Op::Section(_) | Op::EqDisj | Op::EqAlt | Op::GoalDisj => OpKind::Ac,
            Op::GConj | Op::GDisj | Op::GGuards => OpKind::Ac,
            Op::GEq | Op::GAppC(_) => OpKind::C,
            Op::GBinOp { ac: true, .. } => OpKind::Ac,
            Op::Fun(sym) if sym.is_ac() => OpKind::Ac,
            Op::Fun(sym) if sym.is_c() => OpKind::C,
            _ => OpKind::Free,
        }
    }
}

/// What a closed subterm was, so fingerprints of different kinds of object
/// never compare equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ClosedKind {
    Term,
    Fact,
    Formula,
    Atom,
    FormulaTerm,
}

/// A leaf that no renaming touches.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstKey {
    /// A name of an `LNTerm` (`'a'`, `~'n'`, `%'n'`): names are constants.
    Name(Name),
    /// A name in a formula term: `PubLit`/`FreshLit`/`NatLit` (0/1/2).
    FormulaName(u8, String),
    /// A formula literal other than a name (`Number`, `1`, `%1`, `DH_neutral`).
    FormulaLiteral(String),
    /// A de Bruijn index: a bound occurrence in a formula.
    DeBruijn(u32),
    NoLast,
    /// A subterm without free variables, by the fingerprint of its AC normal
    /// form (see the module docs).
    Closed(ClosedKind, Fingerprint),
}

// =============================================================================
// Step 1: System -> STerm
// =============================================================================

/// The whole of `sys` as one closed term (see the module docs).
pub fn system_to_term(sys: &System) -> STerm {
    let mut free: BTreeSet<LVar> = BTreeSet::new();
    let mut tr = Translator { free: &mut free };

    let nodes = sys
        .nodes
        .iter()
        .map(|(nid, ru)| tr.rule_instance(*nid, ru))
        .collect();
    let edges = sys
        .edges
        .iter()
        .map(|e| {
            STerm::App(
                Op::Edge(e.src.1 .0, e.tgt.1 .0),
                vec![tr.free_var(e.src.0), tr.free_var(e.tgt.0)],
            )
        })
        .collect();
    let less_atoms = sys
        .less_atoms
        .iter()
        .map(|la| {
            STerm::App(
                Op::Less,
                vec![tr.free_var(la.smaller), tr.free_var(la.larger)],
            )
        })
        .collect();
    let last = match sys.last_atom {
        Some(nid) => STerm::App(Op::Last, vec![tr.free_var(nid)]),
        None => STerm::Const(ConstKey::NoLast),
    };
    let formulas = sys.formulas.iter().map(|f| tr.formula(f)).collect();
    let solved_formulas = sys.solved_formulas.iter().map(|f| tr.formula(f)).collect();
    let lemmas = sys.lemmas.iter().map(|f| tr.formula(f)).collect();

    let eq_subst = sys
        .eq_store
        .subst
        .iter()
        .map(|(x, t)| STerm::App(Op::EqBind, vec![tr.free_var(*x), tr.term(t, false)]))
        .collect();
    let eq_conj = sys
        .eq_store
        .conj
        .iter()
        .map(|disj| {
            let alternatives = disj
                .substs
                .iter()
                .map(|alt| {
                    let mut locals: BTreeSet<LVar> = BTreeSet::new();
                    for (_, t) in alt.iter() {
                        t.for_each_free(&mut |v| {
                            locals.insert(*v);
                        });
                    }
                    let binds = alt
                        .iter()
                        .map(|(x, t)| {
                            STerm::App(Op::EqBind, vec![tr.free_var(*x), tr.term(t, true)])
                        })
                        .collect();
                    STerm::Exists(
                        Scope::EqAlternative,
                        locals.into_iter().collect(),
                        Box::new(STerm::App(Op::EqAlt, binds)),
                    )
                })
                .collect();
            STerm::App(Op::EqDisj, alternatives)
        })
        .collect();

    let store = &sys.subterm_store;
    let pair = |tr: &mut Translator, s: &LNTerm, t: &LNTerm| {
        STerm::App(Op::Sub, vec![tr.term(s, false), tr.term(t, false)])
    };
    let subterms = store
        .subterms
        .iter()
        .map(|c| pair(&mut tr, &c.small, &c.big))
        .collect();
    let solved_subterms = store
        .solved_subterms
        .iter()
        .map(|c| pair(&mut tr, &c.small, &c.big))
        .collect();
    let neg_subterms: BTreeSet<(LNTerm, LNTerm)> = store
        .neg_subterms
        .iter()
        .map(|(s, t)| (ac_normal(s), ac_normal(t)))
        .collect();
    let neg_subterms = neg_subterms
        .iter()
        .map(|(s, t)| pair(&mut tr, s, t))
        .collect();

    let goals = sys
        .goals
        .iter()
        .filter_map(|(goal, status)| {
            tr.goal(goal).map(|payload| {
                STerm::App(
                    Op::Goal {
                        solved: status.solved,
                    },
                    vec![payload],
                )
            })
        })
        .collect();

    let section = |s: Section, items: Vec<STerm>| STerm::App(Op::Section(s), items);
    let body = STerm::App(
        Op::System {
            source_kind: sys.source_kind,
            side: sys.side,
            contradictory: store.contradictory,
        },
        vec![
            section(Section::Nodes, nodes),
            section(Section::Edges, edges),
            section(Section::LessAtoms, less_atoms),
            last,
            section(Section::Formulas, formulas),
            section(Section::SolvedFormulas, solved_formulas),
            section(Section::Lemmas, lemmas),
            section(Section::EqSubst, eq_subst),
            section(Section::EqConj, eq_conj),
            section(Section::Subterms, subterms),
            section(Section::SolvedSubterms, solved_subterms),
            section(Section::NegSubterms, neg_subterms),
            section(Section::Goals, goals),
        ],
    );
    STerm::Exists(Scope::Root, free.into_iter().collect(), Box::new(body))
}

/// The AC normal form of `t` (the smart constructors re-sort AC arguments
/// on rebuild).
fn ac_normal(t: &LNTerm) -> LNTerm {
    t.clone().map_free(&mut |v| v)
}

/// Translates the parts of one system, collecting its free variables.
struct Translator<'a> {
    free: &'a mut BTreeSet<LVar>,
}

impl Translator<'_> {
    fn free_var(&mut self, v: LVar) -> STerm {
        self.free.insert(v);
        STerm::Var(VarRef::Free(v))
    }

    fn rule_instance(&mut self, nid: LVar, ru: &RuleACInst) -> STerm {
        let group = |tr: &mut Self, g: FactGroup, facts: &[LNFact]| {
            STerm::App(Op::FactGroup(g), facts.iter().map(|f| tr.fact(f)).collect())
        };
        let node = self.free_var(nid);
        let prems = group(self, FactGroup::Prems, &ru.premises);
        let acts = group(self, FactGroup::Acts, &ru.actions);
        let concs = group(self, FactGroup::Concs, &ru.conclusions);
        STerm::App(Op::RuleInst(rule_id(ru)), vec![node, prems, acts, concs])
    }

    fn fact(&mut self, fact: &LNFact) -> STerm {
        if fact.terms.iter().all(is_ground) {
            let fp = fingerprint_term(&ac_normal(&fact_to_term(fact)));
            return STerm::Const(ConstKey::Closed(ClosedKind::Fact, fp));
        }
        let args = fact.terms.iter().map(|t| self.term(t, false)).collect();
        STerm::App(Op::Fact(fact.tag), args)
    }

    /// `t`, with every variable local (`local`: a range term of an
    /// equation-store alternative) or free.
    fn term(&mut self, t: &LNTerm, local: bool) -> STerm {
        if is_ground(t) {
            return STerm::Const(ConstKey::Closed(
                ClosedKind::Term,
                fingerprint_term(&ac_normal(t)),
            ));
        }
        self.term_rec(t, local)
    }

    fn term_rec(&mut self, t: &LNTerm, local: bool) -> STerm {
        match t {
            Term::Lit(Lit::Var(v)) if local => STerm::Var(VarRef::Local(*v)),
            Term::Lit(Lit::Var(v)) => self.free_var(*v),
            Term::Lit(Lit::Con(n)) => STerm::Const(ConstKey::Name(*n)),
            Term::App(sym, args) => STerm::App(
                Op::Fun(*sym),
                args.iter().map(|a| self.term(a, local)).collect(),
            ),
        }
    }

    /// One formula, translated in the form `ac_normalize_guarded` gives it --
    /// the form the content stage canonicalizes.
    fn formula(&mut self, g: &Guarded) -> STerm {
        self.guarded(&crate::canon::ac_normalize_guarded(g))
    }

    fn guarded(&mut self, g: &Guarded) -> STerm {
        if !guarded_has_free(g) {
            let fp = crate::canon::fingerprint_guarded(g);
            return STerm::Const(ConstKey::Closed(ClosedKind::Formula, fp));
        }
        match g {
            Guarded::Atom(a) => self.atom(a),
            Guarded::Conj(items) => {
                STerm::App(Op::GConj, items.iter().map(|g| self.guarded(g)).collect())
            }
            Guarded::Disj(items) => {
                STerm::App(Op::GDisj, items.iter().map(|g| self.guarded(g)).collect())
            }
            Guarded::GGuarded {
                qua,
                vars,
                guards,
                body,
            } => {
                let op = Op::GQuant {
                    ex: matches!(qua, Quant::Ex),
                    sorts: vars.iter().map(|b| format!("{:?}", b.sort)).collect(),
                };
                let guards = STerm::App(Op::GGuards, guards.iter().map(|a| self.atom(a)).collect());
                STerm::App(op, vec![guards, self.guarded(body)])
            }
        }
    }

    fn atom(&mut self, a: &GAtom) -> STerm {
        if !atom_has_free(a) {
            return STerm::Const(ConstKey::Closed(ClosedKind::Atom, fingerprint_atom(a)));
        }
        let two = |tr: &mut Self, op: Op, x: &GTerm, y: &GTerm| {
            let args = vec![tr.gterm(x), tr.gterm(y)];
            STerm::App(op, args)
        };
        match a {
            GAtom::Eq(x, y) => two(self, Op::GEq, x, y),
            GAtom::Less(x, y) => two(self, Op::GLess, x, y),
            GAtom::LessMset(x, y) => two(self, Op::GLessMset, x, y),
            GAtom::Subterm(x, y) => two(self, Op::GSubterm, x, y),
            GAtom::Action(fact, t) => {
                let args = vec![self.gfact(fact), self.gterm(t)];
                STerm::App(Op::GAction, args)
            }
            GAtom::Last(t) => STerm::App(Op::GLast, vec![self.gterm(t)]),
            GAtom::Pred(fact) => STerm::App(Op::GPred, vec![self.gfact(fact)]),
        }
    }

    /// A formula fact: annotations are left out (see the module docs).
    fn gfact(&mut self, fact: &GFact) -> STerm {
        let op = Op::GFact {
            name: fact.name.clone(),
            persistent: fact.persistent,
        };
        STerm::App(op, fact.args.iter().map(|t| self.gterm(t)).collect())
    }

    fn gterm(&mut self, t: &GTerm) -> STerm {
        if !gterm_has_free(t) {
            return STerm::Const(ConstKey::Closed(
                ClosedKind::FormulaTerm,
                fingerprint_gterm(t),
            ));
        }
        match t {
            GTerm::Var(BVar::Free(vs)) => self.free_var(varspec_to_lvar(vs)),
            GTerm::Var(BVar::Bound(i)) => STerm::Const(ConstKey::DeBruijn(*i)),
            GTerm::PubLit(s) => STerm::Const(ConstKey::FormulaName(0, s.clone())),
            GTerm::FreshLit(s) => STerm::Const(ConstKey::FormulaName(1, s.clone())),
            GTerm::NatLit(s) => STerm::Const(ConstKey::FormulaName(2, s.clone())),
            GTerm::Number(n) => STerm::Const(ConstKey::FormulaLiteral(format!("Number({n})"))),
            GTerm::NumberOne => STerm::Const(ConstKey::FormulaLiteral("NumberOne".into())),
            GTerm::NatOne => STerm::Const(ConstKey::FormulaLiteral("NatOne".into())),
            GTerm::DhNeutral => STerm::Const(ConstKey::FormulaLiteral("DhNeutral".into())),
            GTerm::App(n, args) if &**n == "em" && args.len() == 2 => STerm::App(
                Op::GAppC(n.to_string()),
                args.iter().map(|a| self.gterm(a)).collect(),
            ),
            GTerm::App(n, args) => STerm::App(
                Op::GApp(n.to_string()),
                args.iter().map(|a| self.gterm(a)).collect(),
            ),
            GTerm::AlgApp(n, a, b) => {
                let args = vec![self.gterm(a), self.gterm(b)];
                STerm::App(Op::GAlgApp(n.to_string()), args)
            }
            GTerm::Pair(items) => {
                STerm::App(Op::GPair, items.iter().map(|a| self.gterm(a)).collect())
            }
            GTerm::Diff(a, b) => {
                let args = vec![self.gterm(a), self.gterm(b)];
                STerm::App(Op::GDiff, args)
            }
            GTerm::PatMatch(x) => STerm::App(Op::GPatMatch, vec![self.gterm(x)]),
            GTerm::BinOp(op, a, b) => {
                let op_key = Op::GBinOp {
                    op: format!("{op:?}"),
                    ac: guarded::is_ac_binop(op),
                };
                let args = vec![self.gterm(a), self.gterm(b)];
                STerm::App(op_key, args)
            }
        }
    }

    /// One goal's payload; `None` for a `Split` goal (see the module docs).
    fn goal(&mut self, goal: &Goal) -> Option<STerm> {
        Some(match goal {
            Goal::Split(_) => return None,
            Goal::Action(nid, fact) => {
                let args = vec![self.free_var(*nid), self.fact(fact)];
                STerm::App(Op::GoalAction, args)
            }
            Goal::Chain((src, conc), (tgt, prem)) => {
                let args = vec![self.free_var(*src), self.free_var(*tgt)];
                STerm::App(Op::GoalChain(conc.0, prem.0), args)
            }
            Goal::Premise((nid, prem), fact) => {
                let args = vec![self.free_var(*nid), self.fact(fact)];
                STerm::App(Op::GoalPremise(prem.0), args)
            }
            Goal::Disj(disj) => STerm::App(
                Op::GoalDisj,
                disj.0.iter().map(|g| self.formula(g)).collect(),
            ),
            Goal::Subterm((small, big)) => {
                let args = vec![self.term(small, false), self.term(big, false)];
                STerm::App(Op::GoalSubterm, args)
            }
        })
    }
}

fn is_ground(t: &LNTerm) -> bool {
    match t {
        Term::Lit(Lit::Var(_)) => false,
        Term::Lit(Lit::Con(_)) => true,
        Term::App(_, args) => args.iter().all(is_ground),
    }
}

fn gterm_has_free(t: &GTerm) -> bool {
    match t {
        GTerm::Var(BVar::Free(_)) => true,
        GTerm::Var(BVar::Bound(_))
        | GTerm::PubLit(_)
        | GTerm::FreshLit(_)
        | GTerm::NatLit(_)
        | GTerm::Number(_)
        | GTerm::NumberOne
        | GTerm::NatOne
        | GTerm::DhNeutral => false,
        GTerm::App(_, args) | GTerm::Pair(args) => args.iter().any(gterm_has_free),
        GTerm::AlgApp(_, a, b) | GTerm::Diff(a, b) | GTerm::BinOp(_, a, b) => {
            gterm_has_free(a) || gterm_has_free(b)
        }
        GTerm::PatMatch(x) => gterm_has_free(x),
    }
}

fn atom_has_free(a: &GAtom) -> bool {
    match a {
        GAtom::Eq(x, y) | GAtom::Less(x, y) | GAtom::LessMset(x, y) | GAtom::Subterm(x, y) => {
            gterm_has_free(x) || gterm_has_free(y)
        }
        GAtom::Action(fact, t) => fact.args.iter().any(gterm_has_free) || gterm_has_free(t),
        GAtom::Last(t) => gterm_has_free(t),
        GAtom::Pred(fact) => fact.args.iter().any(gterm_has_free),
    }
}

fn guarded_has_free(g: &Guarded) -> bool {
    match g {
        Guarded::Atom(a) => atom_has_free(a),
        Guarded::Conj(items) | Guarded::Disj(items) => items.iter().any(guarded_has_free),
        Guarded::GGuarded { guards, body, .. } => {
            guards.iter().any(atom_has_free) || guarded_has_free(body)
        }
    }
}

// =============================================================================
// Step 2: STerm -> colored digraph
// =============================================================================

/// What a vertex stands for (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Label {
    Op(Op),
    Const(ConstKey),
    Binder(Scope),
    BoundVar(Scope, LSort),
    Occ(LSort),
}

/// A vertex's key: its label and its argument position under a free parent.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VKey {
    pub label: Label,
    pub pos: Option<usize>,
}

/// The colored digraph of a system term.
#[derive(Debug, Clone, Default)]
pub struct TermGraph {
    pub keys: Vec<VKey>,
    /// `colors[v]`: the rank of `keys[v]` among the distinct keys.
    pub colors: Vec<Color>,
    pub edges: Vec<(usize, usize)>,
    /// The root binder's variable vertices: the system's free variables.
    pub free_vars: Vec<(usize, LVar)>,
    /// Each vertex's parent in the term (a variable vertex's is its binder,
    /// an occurrence's the application it is an argument of); `None` for the
    /// root.
    pub parents: Vec<Option<usize>>,
}

/// The colored digraph of `t` (see the module docs). Panics on a variable
/// occurrence no enclosing binder binds: a translation bug, never a case to
/// pass over.
pub fn term_to_graph(t: &STerm) -> TermGraph {
    let mut b = GraphBuilder {
        graph: TermGraph::default(),
        scopes: Vec::new(),
    };
    b.build(t, None);
    let mut graph = b.graph;
    let ranks: BTreeMap<&VKey, Color> = graph
        .keys
        .iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .zip(0..)
        .collect();
    graph.colors = graph.keys.iter().map(|k| ranks[k]).collect();
    graph
}

struct GraphBuilder {
    graph: TermGraph,
    /// The enclosing binders, innermost last: their scope and variable
    /// vertices.
    scopes: Vec<(Scope, BTreeMap<LVar, usize>)>,
}

impl GraphBuilder {
    fn vertex(&mut self, label: Label, pos: Option<usize>) -> usize {
        self.graph.keys.push(VKey { label, pos });
        self.graph.parents.push(None);
        self.graph.keys.len() - 1
    }

    fn edge(&mut self, src: usize, tgt: usize) {
        self.graph.edges.push((src, tgt));
    }

    /// The edge from `parent` to its term child `child`.
    fn child_edge(&mut self, parent: usize, child: usize) {
        self.graph.parents[child] = Some(parent);
        self.edge(parent, child);
    }

    fn build(&mut self, t: &STerm, pos: Option<usize>) -> usize {
        match t {
            STerm::Const(c) => self.vertex(Label::Const(c.clone()), pos),
            STerm::Var(r) => {
                let (v, w) = match r {
                    VarRef::Free(v) => {
                        let root = self.scopes.first().filter(|(s, _)| *s == Scope::Root);
                        (v, root.and_then(|(_, vars)| vars.get(v)).copied())
                    }
                    VarRef::Local(v) => (
                        v,
                        self.scopes
                            .iter()
                            .rev()
                            .filter(|(s, _)| *s != Scope::Root)
                            .find_map(|(_, vars)| vars.get(v))
                            .copied(),
                    ),
                };
                let w = w.unwrap_or_else(|| {
                    panic!("term_to_graph: no binder binds {r:?} -- system_to_term missed {v}")
                });
                let occ = self.vertex(Label::Occ(v.sort), pos);
                self.edge(w, occ);
                occ
            }
            STerm::App(op, args) => {
                let me = self.vertex(Label::Op(op.clone()), pos);
                match op.kind() {
                    OpKind::Free => {
                        for (i, a) in args.iter().enumerate() {
                            let child = self.build(a, Some(i));
                            self.child_edge(me, child);
                        }
                    }
                    OpKind::C => {
                        for a in args {
                            let child = self.build(a, None);
                            self.child_edge(me, child);
                        }
                    }
                    OpKind::Ac => {
                        let mut frontier = Vec::new();
                        flatten(op, args, &mut frontier);
                        for a in frontier {
                            let child = self.build(a, None);
                            self.child_edge(me, child);
                        }
                    }
                }
                me
            }
            STerm::Exists(scope, vars, body) => {
                let me = self.vertex(Label::Binder(*scope), pos);
                let mut bound = BTreeMap::new();
                for v in vars {
                    let w = self.vertex(Label::BoundVar(*scope, v.sort), None);
                    self.child_edge(me, w);
                    bound.insert(*v, w);
                    if *scope == Scope::Root {
                        self.graph.free_vars.push((w, *v));
                    }
                }
                self.scopes.push((*scope, bound));
                let child = self.build(body, None);
                self.scopes.pop();
                self.child_edge(me, child);
                me
            }
        }
    }
}

/// The arguments of an AC application of `op`, with nested applications of
/// `op` replaced by their own arguments (the reduction's frontier).
fn flatten<'t>(op: &Op, args: &'t [STerm], out: &mut Vec<&'t STerm>) {
    for a in args {
        match a {
            STerm::App(inner, inner_args) if inner == op => flatten(op, inner_args, out),
            _ => out.push(a),
        }
    }
}

// =============================================================================
// The driver
// =============================================================================

/// What the driver does beyond computing the canonical form.
#[derive(Debug, Clone, Copy, Default)]
pub struct TermGraphOptions {
    /// Check that every automorphism generator bliss reports is a symmetry of
    /// the system ([`TermGraphStats::non_genuine_generators`]). Costs one
    /// content pass per generator that moves a free variable.
    pub check_generators: bool,
    /// Compute [`TermGraphStats::certificate`]; runs bliss even when the
    /// system has no free variables.
    pub certificate: bool,
}

/// Measurements of one canonicalization.
#[derive(Debug, Clone, Default)]
pub struct TermGraphStats {
    pub vertices: usize,
    pub edges: usize,
    pub free_vars: usize,
    /// Whether bliss ran: it is skipped when the system has no free variable
    /// and no certificate was asked for.
    pub bliss_ran: bool,
    /// `|Aut|` of the term graph, as bliss prints it.
    pub aut_size: Option<String>,
    pub bliss_search_nodes: u64,
    pub generators: usize,
    /// Generators that move at least one free-variable vertex. The others
    /// only permute the inside of the term (identical AC arguments, an
    /// alternative's local variables) and cannot change θ.
    pub generators_moving_free_vars: usize,
    /// Generators under which the canonical form changes: each one is a
    /// symmetry of the graph that is not one of the system, i.e. the
    /// encoding lost information. Counted only with
    /// [`TermGraphOptions::check_generators`].
    pub non_genuine_generators: usize,
    /// What the generators permute: per generator, each distinct
    /// `parent > subtree` pair of labels of a moved subtree whose parent stays
    /// fixed (see [`generator_moves`]).
    pub generator_moves: BTreeMap<String, usize>,
    /// [`Self::generator_moves`] of the non-genuine generators only.
    pub non_genuine_moves: BTreeMap<String, usize>,
    /// The `CanonicalSystem` fields a non-genuine generator changes.
    pub non_genuine_fields: BTreeMap<String, usize>,
    /// A hash of bliss's canonical graph: the key table, the colors in
    /// canonical order and the canonical edges. Equal for isomorphic graphs;
    /// whether it is also only equal for equivalent systems is what comparing
    /// it with the form measures.
    pub certificate: Option<Fingerprint>,
    pub time_term: Duration,
    pub time_graph: Duration,
    pub time_bliss: Duration,
    pub time_content: Duration,
    pub time_check: Duration,
}

/// Canonicalizes `sys` through its term graph (see the module docs),
/// returning the canonical form, the labelling θ it was built with, and the
/// measurements.
///
/// The steps: the term and its graph; bliss's canonical labeling of the
/// graph; θ names the free variables, per sort, in the order of their
/// vertices' canonical positions; the form is `sys` renamed by θ, field by
/// field. The equation store's alternatives are still canonicalized by the
/// content stage's own per-alternative canonizer.
pub fn canonicalize_constraint_system_via_term_graph_with_labelling(
    sys: &System,
    options: TermGraphOptions,
) -> Result<(CanonicalSystem, CanonLabelling, TermGraphStats), BlissError> {
    let mut stats = TermGraphStats::default();

    let start = Instant::now();
    let term = system_to_term(sys);
    stats.time_term = start.elapsed();

    let start = Instant::now();
    let graph = term_to_graph(&term);
    stats.vertices = graph.keys.len();
    stats.edges = graph.edges.len();
    stats.free_vars = graph.free_vars.len();
    stats.time_graph = start.elapsed();

    let mut labeling: Option<Permutation> = None;
    let mut generators: Vec<Permutation> = Vec::new();
    if !graph.free_vars.is_empty() || options.certificate {
        let start = Instant::now();
        let (result, bliss_stats) = run_bliss_with_stats(&to_dimacs(&graph.colors, &graph.edges)?)?;
        stats.time_bliss = start.elapsed();
        stats.bliss_ran = true;
        stats.aut_size = Some(bliss_stats.aut_size);
        stats.bliss_search_nodes = bliss_stats.search_nodes;
        stats.generators = result.generators.len();
        if options.certificate {
            stats.certificate = Some(certificate(&graph, &result.canonical_labeling));
        }
        labeling = Some(result.canonical_labeling);
        generators = result.generators;
    }

    let mut order: Vec<(usize, LVar)> = graph
        .free_vars
        .iter()
        .map(|&(w, v)| {
            let position = labeling
                .as_ref()
                .expect("bliss runs whenever there is a free variable")
                .image_of(w);
            (position, v)
        })
        .collect();

    // Sort according to bliss' canonical labelling. The tuple (usize, LVar)
    // sorts on the first entry, which is the Lvar's image, i.e., the canonical
    // order on its vertex, under the canonical labelling.
    order.sort();
    let mut labelling = CanonLabelling::empty();
    for (_, v) in &order {
        labelling.bind_next(*v);
    }

    let start = Instant::now();
    let canon = content_under(sys, &labelling);
    stats.time_content = start.elapsed();

    let var_of: BTreeMap<usize, LVar> = graph.free_vars.iter().copied().collect();
    let start = Instant::now();
    for g in &generators {
        let moves = generator_moves(&graph, g);
        for moved in &moves {
            *stats.generator_moves.entry(moved.clone()).or_insert(0) += 1;
        }
        let sigma: BTreeMap<LVar, LVar> = graph
            .free_vars
            .iter()
            .map(|&(w, v)| {
                let image = var_of.get(&g.image_of(w)).unwrap_or_else(|| {
                    panic!(
                        "an automorphism maps the variable vertex of {v} to a non-variable vertex"
                    )
                });
                (v, *image)
            })
            .collect();
        if sigma.iter().all(|(v, image)| v == image) {
            continue;
        }
        stats.generators_moving_free_vars += 1;
        if options.check_generators {
            // θ∘σ, built like θ itself: `CanonLabelling::from_theta` would
            // move the per-sort cursors, and with them the names the
            // content stage gives an alternative's local variables.
            let preimage: BTreeMap<LVar, LVar> =
                sigma.iter().map(|(v, image)| (*image, *v)).collect();
            let mut moved = CanonLabelling::empty();
            for (_, v) in &order {
                moved.bind_next(preimage[v]);
            }
            let other = content_under(sys, &moved);
            if other != canon {
                stats.non_genuine_generators += 1;
                for moved in &moves {
                    *stats.non_genuine_moves.entry(moved.clone()).or_insert(0) += 1;
                }
                let report = crate::canon_fingerprint::compare_fingerprints(
                    &crate::canon_fingerprint::fingerprint_constraint_system(&canon),
                    &crate::canon_fingerprint::fingerprint_constraint_system(&other),
                );
                for field in report.mismatched_fields() {
                    *stats
                        .non_genuine_fields
                        .entry(field.to_string())
                        .or_insert(0) += 1;
                }
            }
        }
    }
    stats.time_check = start.elapsed();

    Ok((canon, labelling, stats))
}

/// The distinct `parent > subtree` label pairs of the subtrees `g` moves
/// whose parent it fixes: what a symmetry of the term graph swaps, e.g.
/// `EqAlt > Binder` (two alpha-equivalent alternatives) or
/// `Binder(EqAlternative) > BoundVar(EqAlternative, Msg)` (two local variables
/// of one alternative).
pub fn generator_moves(graph: &TermGraph, g: &Permutation) -> BTreeSet<String> {
    (0..graph.keys.len())
        .filter(|&v| g.image_of(v) != v)
        .filter_map(|v| {
            let parent = graph.parents[v]?;
            (g.image_of(parent) == parent).then(|| {
                format!(
                    "{} > {}",
                    label_name(&graph.keys[parent]),
                    label_name(&graph.keys[v])
                )
            })
        })
        .collect()
}

/// A short name for a vertex's label, for [`generator_moves`].
fn label_name(key: &VKey) -> String {
    match &key.label {
        Label::Op(op) => match op {
            Op::Fun(sym) => format!("Fun({})", fun_sym_name(sym)),
            Op::Fact(tag) => format!("Fact({})", crate::fact::fact_tag_name(tag)),
            Op::GFact { name, .. } => format!("GFact({name})"),
            Op::RuleInst(_) => "RuleInst".into(),
            other => {
                let shown = format!("{other}");
                shown.split('[').next().unwrap_or(&shown).to_string()
            }
        },
        Label::Const(c) => match c {
            ConstKey::Closed(kind, _) => format!("Closed({kind:?})"),
            _ => "Const".into(),
        },
        Label::Binder(scope) => format!("Binder({scope:?})"),
        Label::BoundVar(scope, sort) => format!("BoundVar({scope:?},{sort:?})"),
        Label::Occ(sort) => format!("Occ({sort:?})"),
    }
}

/// `sys` renamed by `labelling`, which names every free variable of it: a
/// variable `system_to_term` missed has no entry, and renaming it panics.
fn content_under(sys: &System, labelling: &CanonLabelling) -> CanonicalSystem {
    let graph_part = renamed_graph_part_term(sys, labelling.theta());
    canonicalize_system_content(sys, labelling, graph_part)
}

/// The graph part of `sys` renamed by `theta` and sorted: rule instances
/// (with their rule's identity), edges, less-than atoms and `last_atom`.
/// Formula and goal actions are not repeated here: the form keeps them in its
/// `formulas` and `goals`.
pub fn renamed_graph_part_term(sys: &System, theta: &BTreeMap<LVar, LVar>) -> LNTerm {
    let node = |nid: LVar| var_term(rename_var(theta, nid));
    let mut nodes: Vec<LNTerm> = sys
        .nodes
        .iter()
        .map(|(nid, ru)| {
            graph_tag(
                "RuleInstanceVertexTerm",
                vec![
                    node(*nid),
                    rule_identity_marker(ru),
                    apply_renaming(theta, crate::canon::rule_to_term(ru)),
                ],
            )
        })
        .collect();
    nodes.sort();
    let mut edges: Vec<LNTerm> = sys
        .edges
        .iter()
        .map(|e| {
            graph_tag(
                "EdgeTerm",
                vec![
                    node(e.src.0),
                    index_marker(e.src.1 .0),
                    node(e.tgt.0),
                    index_marker(e.tgt.1 .0),
                ],
            )
        })
        .collect();
    edges.sort();
    let mut less: Vec<LNTerm> = sys
        .less_atoms
        .iter()
        .map(|la| graph_tag("LessTerm", vec![node(la.smaller), node(la.larger)]))
        .collect();
    less.sort();
    let last = match sys.last_atom {
        Some(nid) => graph_tag("LastAtomTerm", vec![node(nid)]),
        None => graph_marker("NoLastAtom"),
    };
    graph_tag(
        "TermGraphPartTerm",
        vec![
            graph_tag("NodesTerm", nodes),
            graph_tag("EdgesTerm", edges),
            graph_tag("LessAtomsTerm", less),
            last,
        ],
    )
}

/// A hash of `graph` relabeled by `labeling` (see
/// [`TermGraphStats::certificate`]). Colors are ranks among this graph's
/// keys, so the key table itself is hashed too: two graphs with different
/// keys can share every rank.
fn certificate(graph: &TermGraph, labeling: &Permutation) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    let table: BTreeSet<&VKey> = graph.keys.iter().collect();
    h.u64(table.len() as u64);
    for key in table {
        h.bytes(format!("{key:?}").as_bytes());
    }
    let mut colors = vec![0; graph.colors.len()];
    for (v, c) in graph.colors.iter().enumerate() {
        colors[labeling.image_of(v)] = *c;
    }
    h.u64(colors.len() as u64);
    for c in colors {
        h.u64(u64::from(c));
    }
    let edges: BTreeSet<(usize, usize)> = graph
        .edges
        .iter()
        .map(|&(s, t)| (labeling.image_of(s), labeling.image_of(t)))
        .collect();
    h.u64(edges.len() as u64);
    for (s, t) in edges {
        h.u64(s as u64).u64(t as u64);
    }
    h.finish()
}

// =============================================================================
// Display (for dumps and tests)
// =============================================================================

impl fmt::Display for STerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            STerm::Var(VarRef::Free(v)) | STerm::Var(VarRef::Local(v)) => write!(f, "{v}"),
            STerm::Const(c) => write!(f, "⟨{c}⟩"),
            STerm::Exists(scope, vars, body) => {
                write!(f, "∃_{scope:?}")?;
                for v in vars {
                    write!(f, " {v}")?;
                }
                write!(f, ". {body}")
            }
            STerm::App(op, args) => {
                let (open, close) = match op.kind() {
                    OpKind::Ac => ('{', '}'),
                    OpKind::Free | OpKind::C => ('(', ')'),
                };
                write!(f, "{op}{open}")?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{a}")?;
                }
                write!(f, "{close}")
            }
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::System {
                source_kind,
                side,
                contradictory,
            } => write!(f, "System[{source_kind:?},{side:?},{contradictory}]"),
            Op::Section(s) => write!(f, "{s:?}"),
            Op::RuleInst(id) => match id {
                RuleId::Fresh => write!(f, "RuleInst[Fresh]"),
                RuleId::Proto(name) => write!(f, "RuleInst[{name}]"),
                RuleId::Intr(info) => write!(f, "RuleInst[{info:?}]"),
            },
            Op::FactGroup(g) => write!(f, "{g:?}"),
            Op::Fact(tag) => write!(f, "{}", crate::fact::fact_tag_name(tag)),
            Op::Fun(sym) => write!(f, "{}", fun_sym_name(sym)),
            Op::Edge(c, p) => write!(f, "Edge[{c},{p}]"),
            Op::GoalChain(c, p) => write!(f, "ChainG[{c},{p}]"),
            Op::GoalPremise(p) => write!(f, "PremiseG[{p}]"),
            Op::Goal { solved } => write!(f, "Goal[{}]", if *solved { "solved" } else { "open" }),
            Op::GQuant { ex, sorts } => {
                write!(f, "{}[{}]", if *ex { "Ex" } else { "All" }, sorts.join(","))
            }
            Op::GFact { name, persistent } => {
                write!(f, "{}{name}", if *persistent { "!" } else { "" })
            }
            Op::GApp(n) | Op::GAppC(n) | Op::GAlgApp(n) => write!(f, "{n}"),
            Op::GBinOp { op, .. } => write!(f, "{op}"),
            other => write!(f, "{other:?}"),
        }
    }
}

impl fmt::Display for ConstKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstKey::Name(n) => write!(f, "{}", LNTerm::Lit(Lit::Con(*n))),
            ConstKey::FormulaName(_, s) | ConstKey::FormulaLiteral(s) => write!(f, "{s}"),
            ConstKey::DeBruijn(i) => write!(f, "db{i}"),
            ConstKey::NoLast => write!(f, "NoLast"),
            ConstKey::Closed(kind, fp) => {
                write!(f, "{kind:?}:")?;
                for b in &fp[..4] {
                    write!(f, "{b:02x}")?;
                }
                Ok(())
            }
        }
    }
}

/// A function symbol's name, for display.
fn fun_sym_name(sym: &FunSym) -> String {
    match sym {
        FunSym::NoEq(s) => String::from_utf8_lossy(s.name).into_owned(),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
#[path = "canon_term_graph_tests.rs"]
mod tests;
