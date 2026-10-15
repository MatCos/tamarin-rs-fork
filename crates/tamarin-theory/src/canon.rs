//! Lifting `CAN_alphaeqac` (canonization of terms for alpha-equivalence
//! modulo AC, `tamarin_term::alpha_eq_ac`) to facts and rules, per
//! `work.tex`'s "Canonization of Facts and Rule Instances": facts and rule
//! instances carry no equational theory of their own, so canonizing them
//! reduces to canonizing a single term that faithfully encodes their
//! structure.
//!
//! Unlike `work.tex`'s formula, which uses one fixed-arity symbol `f` for
//! every fact (`f(t_1, ..., t_n)`) and rule (`f(i, p_1, ..., p_k, a_1, ...,
//! a_l, c_1, ..., c_m)`), a fact's/rule's arity varies per instance here, so
//! we build the lifted term with [`FunSym::List`] — a free n-ary combinator
//! with no equational theory of its own — instead of committing to one
//! symbol per shape.
//!
//! - A fact `F(t_1, ..., t_n)` becomes `List(marker(F), t_1, ..., t_n)`,
//!   where `marker(F)` is a distinguishing 0-ary constructor symbol built
//!   from `F`'s tag (see [`fact_tag_marker`]). It is a function symbol, not
//!   a literal, so `CAN_alphaeqac` — which only ever renames literals —
//!   leaves it untouched: two facts with different tags can therefore never
//!   canonize equal, no matter their arguments.
//! - A rule is lifted the same way, one level up: its premises, actions,
//!   and conclusions each become their own `List` of lifted fact terms, and
//!   the three groups are combined into one outer `List` (`rule_to_term`).
//!   Grouping — rather than one flat concatenation of the `k + l + m`
//!   facts, as `work.tex`'s formula reads literally — keeps a rule's
//!   premise/action/conclusion boundaries structurally explicit: nothing
//!   else here plays the role `work.tex`'s `i` (an external timepoint, not
//!   part of the rule itself) does, so without grouping, e.g. a
//!   1-premise/1-action/1-conclusion rule could canonize equal to a
//!   0-premise/2-action/1-conclusion rule whenever their facts happen to
//!   line up.
//!
//! `new_vars` is intentionally NOT part of the lifted term. `Rule<I>::
//! new_vars` is a deterministic function of the premises, actions, and
//! conclusions alone — computed once as `(conclusion vars ∪ action vars) −
//! premise vars` (Haskell `newVariables`) and always kept in lockstep with
//! the rest of the rule by the same machinery that already walks
//! premises/actions/conclusions (`HasFrees for Rule<I>` and
//! `apply_subst_rule` in `rule.rs` fold/substitute over all four fields
//! identically); it never diverges from what the other three fields imply.
//! Two rules with alpha-eq-ac premises/actions/conclusions therefore always
//! have alpha-eq-ac `new_vars` too, so including `new_vars` in the lifted
//! term could only ever be redundant, never distinguishing. This mirrors an
//! existing precedent in this codebase: `write_rule_to_key_excl_new_vars`
//! (`constraint/solver/sources.rs`) already excludes `new_vars` when
//! building the canonical key used to detect redundant/duplicate
//! constraint systems, with the comment "Crucial: rule.new_vars EXCLUDED
//! per `compareRulesUpToNewVars`."
//!
//! `info: I` is not part of the lifted rule term either, mirroring
//! `Rule<I>`'s own `HasFrees` impl (`rule.rs`): the generic bound is
//! `Clone`, not a to-term conversion. A rule INSTANCE of a constraint system
//! is lifted with its rule's identity next to it ([`rule_identity_marker`]):
//! two instances of differently named rules with the same facts are
//! different proof-search states, since a proof names the rule it uses and
//! rule names can enter the ranking (oracles, tactics). SAPIC produces such
//! pairs (`sapic/fast/basic/channels1`'s `incx_1_121` and `incx_3_121`).

use std::sync::Arc;

use tamarin_term::alpha_eq_ac::{
    canonicalize_alpha_eq_ac, canonicalize_alpha_eq_ac_seeded, CanonLabelling,
};
use tamarin_term::function_symbols::{Constructability, FunSym, NoEqSym, Privacy};
use tamarin_term::lterm::LNTerm;
use tamarin_term::term::f_app_list;

use crate::fact::{FactTag, LNFact};
use crate::rule::{ConcIdx, IntrRuleACInfo, PremIdx, ProtoRuleName, Rule, RuleACInst, RuleInfo};

/// A NUL-prefixed marker name for `tag`, distinct for every distinct
/// `FactTag` (the variant name is baked into the string, so e.g. a
/// user-defined `Proto` fact literally named `"Fresh"` can never collide
/// with `FactTag::Fresh`'s marker). The NUL prefix additionally guarantees
/// this can never collide with a real function symbol drawn from a Tamarin
/// model's signature, since no valid Tamarin identifier contains one.
fn fact_tag_marker_name(tag: &FactTag) -> Vec<u8> {
    let suffix = match tag {
        FactTag::Proto(mult, name, arity) => format!("Proto:{mult:?}:{name}:{arity}"),
        FactTag::Fresh => "Fresh".to_string(),
        FactTag::Out => "Out".to_string(),
        FactTag::In => "In".to_string(),
        FactTag::Ku => "Ku".to_string(),
        FactTag::Kd => "Kd".to_string(),
        FactTag::Ded => "Ded".to_string(),
        FactTag::Term => "Term".to_string(),
    };
    let mut name = vec![0u8];
    name.extend_from_slice(suffix.as_bytes());
    name
}

/// The 0-ary marker term for a fact's tag (see the module docs).
fn fact_tag_marker(tag: &FactTag) -> LNTerm {
    let sym = NoEqSym::new(
        fact_tag_marker_name(tag),
        0,
        Privacy::Public,
        Constructability::Constructor,
    );
    LNTerm::App(FunSym::NoEq(sym), Arc::from([]))
}

/// Lifts a fact `F(t_1, ..., t_n)` to the term `List(marker(F), t_1, ...,
/// t_n)` (see the module docs).
pub fn fact_to_term(fact: &LNFact) -> LNTerm {
    let mut args = Vec::with_capacity(fact.terms.len() + 1);
    args.push(fact_tag_marker(&fact.tag));
    args.extend(fact.terms.iter().cloned());
    f_app_list(args)
}

/// Canonizes a fact w.r.t. $\alphaeqac$: two facts are $\alphaeqac$ iff
/// their [`fact_to_term`] lifts canonize syntactically equal.
pub fn canonicalize_fact(fact: &LNFact) -> LNTerm {
    canonicalize_alpha_eq_ac(&fact_to_term(fact))
}

/// Lifts a group of facts (a rule's premises, actions, or conclusions) to
/// `List(fact_to_term(f_1), ..., fact_to_term(f_n))`.
fn facts_to_term(facts: &[LNFact]) -> LNTerm {
    f_app_list(facts.iter().map(fact_to_term).collect())
}

/// Lifts a rule to a term nesting its premises, actions, and conclusions
/// (see the module docs for why `info` and `new_vars` are excluded, and why
/// the groups are nested rather than flattened).
pub fn rule_to_term<I>(rule: &Rule<I>) -> LNTerm {
    f_app_list(vec![
        facts_to_term(&rule.premises),
        facts_to_term(&rule.actions),
        facts_to_term(&rule.conclusions),
    ])
}

/// A rule's identity: what tells two rule instances with the same facts
/// apart. Intruder rules carry their full info, so a protocol rule named
/// like an intruder rule never collides with it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuleId {
    Fresh,
    Proto(&'static str),
    Intr(IntrRuleACInfo),
}

/// The identity of `ru`'s rule.
pub fn rule_id(ru: &RuleACInst) -> RuleId {
    match &ru.info {
        RuleInfo::Proto(p) => match p.name {
            ProtoRuleName::Fresh => RuleId::Fresh,
            ProtoRuleName::Stand(s) => RuleId::Proto(s),
        },
        RuleInfo::Intr(info) => RuleId::Intr(info.clone()),
    }
}

/// A 0-ary marker naming `ru`'s rule ([`rule_id`]), lifted next to a rule
/// instance's content (see the module docs).
pub fn rule_identity_marker(ru: &RuleACInst) -> LNTerm {
    graph_marker(&format!("rule:{:?}", rule_id(ru)))
}

/// Canonizes a rule w.r.t. $\alphaeqac$ (see [`rule_to_term`]).
pub fn canonicalize_rule<I>(rule: &Rule<I>) -> LNTerm {
    canonicalize_alpha_eq_ac(&rule_to_term(rule))
}

// =============================================================================
// Graph-part markers
// =============================================================================
//
// The graph part of a canonical form (rule instances, edges, `<` atoms,
// `last_atom`; `canon_term_graph::renamed_graph_part_term`) is a term built
// from type-tagged NoEq functions, so that a pairing of two sub-terms can
// never coincide with another pairing of the same shape, nor with a function
// symbol of the theory.

/// A NUL,NUL-prefixed marker name for `suffix`, distinct from both a real
/// function symbol (no valid Tamarin identifier contains a NUL byte) and
/// from [`fact_tag_marker_name`]'s own NUL-prefixed markers (which use a
/// SINGLE leading NUL followed immediately by ASCII text — a second NUL
/// byte here guarantees the two marker families can never collide).
fn graph_marker_name(suffix: &str) -> Vec<u8> {
    let mut name = vec![0u8, 0u8];
    name.extend_from_slice(suffix.as_bytes());
    name
}

/// An `args.len()`-ary NoEq function named `tag` (via [`graph_marker_name`],
/// so it cannot collide with a theory's function symbols or with
/// [`fact_tag_marker`]'s markers): tags a graph-part sub-term with what kind
/// of thing it is, so two sub-terms of the same shape but different meaning
/// never coincide, and a dumped term says what each part is.
pub(crate) fn graph_tag(tag: &str, args: Vec<LNTerm>) -> LNTerm {
    let sym = NoEqSym::new(
        graph_marker_name(tag),
        args.len(),
        Privacy::Public,
        Constructability::Constructor,
    );
    LNTerm::App(FunSym::NoEq(sym), args.into())
}

/// The 0-ary marker term for `suffix` (see [`graph_marker_name`]) — a
/// function symbol, so it is never renamed. A 0-ary [`graph_tag`].
pub(crate) fn graph_marker(suffix: &str) -> LNTerm {
    graph_tag(suffix, Vec::new())
}

/// A marker term for the number `i` (a premise or conclusion index): a
/// function symbol, not a variable, since a number is fixed content that no
/// renaming may touch.
pub(crate) fn index_marker(i: usize) -> LNTerm {
    graph_marker(&format!("idx:{i}"))
}

// =============================================================================
// Guarded formulas
// =============================================================================
//
// Per `work.tex`'s "Canonization of Formula Constraints": a guarded
// formula's free variables are always bound/introduced elsewhere in the
// constraint system (by an action formula, a rule constraint, or a
// rule/fact argument — see the paragraph building up to
// `ex:canon_guarded`), so by the time formulas are canonized a canonical
// labelling `theta` for them already exists (the term graph names every
// free variable of the system, formulas included; see
// `canon_term_graph`). `theta` is assumed EXHAUSTIVE: every
// free variable `g` mentions must have an entry, or canonization panics
// (see [`tamarin_term::alpha_eq_ac::rename_var`]) — a miss means `theta` was
// built incorrectly, not a case to paper over with a silent fallback. Given
// that labelling, canonizing a `Guarded` is two steps:
//
// 1. Substitute every free variable via `theta` ([`subst_via_theta_guarded`]).
//    Name constants (`PubLit`/`FreshLit`/`NatLit`) are left as they are:
//    names are never renamed (see `tamarin_term::alpha_eq_ac`'s module doc),
//    so they need no `theta` entry and may occur only in a formula.
//    Bound variables need no substitution: a `Guarded`'s bound occurrences
//    are already De-Bruijn indices (position-determined), which is already
//    a canonical representation — the only non-canonical thing about them is
//    each binder's cosmetic display NAME (`GBinding::name`), which carries
//    no semantic content and is erased (see [`ac_normalize_guarded`]).
// 2. Bring $\land$/$\lor$ (`Guarded::Conj`/`Disj`) and $=$ (`GAtom::Eq`) into
//    AC normal form ([`ac_normalize_guarded`]), reusing this crate's
//    existing `guarded.rs` machinery: `gconj`/`gdisj` already flatten nested
//    conjunctions/disjunctions and drop duplicates (mirroring `CAN_AC`'s
//    flattening of terms); this file adds the sorting half — by the
//    already-HS-faithful [`crate::guarded::cmp_guarded`] /
//    [`crate::guarded::cmp_atom`] total orders, which are exactly the "order
//    on formulas" / "atoms according to string representation" `work.tex`
//    only gestures at ("we compare atoms according to string representation
//    and fix some order on the function symbols" — `\todo{Better
//    explanation?}"). `work.tex`'s own worked example additionally assumes
//    an illustrative `'∃' < '='` operator order that is NOT what
//    `cmp_guarded` (Atom < Disj < Conj < GGuarded) gives; this module uses
//    the codebase's real, already-tested order rather than inventing a
//    second one just to match the paper's illustration — see
//    [`tests::canonicalize_guarded_matches_work_tex_example`] for the
//    concrete difference this makes.

use crate::guarded::{
    self, cmp_atom, cmp_guarded, cmp_term, ga, gall, gex, map_lvars_in_guarded, BVar, GAtom,
    GBinding, GFact, GTerm, Guarded, Quant,
};
use tamarin_parser::ast as p;
use tamarin_term::alpha_eq_ac::{apply_renaming, rename_var};
use tamarin_term::lterm::{LSort, LVar};
use tamarin_utils::fingerprint::{Fingerprint, FingerprintHasher};

/// The canonical labelling's renaming: variables only, names are never
/// renamed (see `tamarin_term::alpha_eq_ac`'s module doc).
type Theta = std::collections::BTreeMap<LVar, LVar>;

/// `p::SortHint` to `LSort`. `Untagged` (a bare, sigil-less name) resolves
/// to `Msg`, matching the parser's own default for that position; every
/// other hint (including the `:msg|:pub|...` suffix spelling) maps to its
/// named sort. Small, self-contained duplicate of the same conversion this
/// crate already carries at several call sites (e.g.
/// `constraint::solver::sources::varspec_sort_to_lsort`,
/// `elaborate::lnterm_to_term`) — kept local rather than imported across a
/// module boundary that has no other reason to depend on this one.
fn sort_hint_to_lsort(s: p::SortHint) -> LSort {
    use p::{SortHint as S, SuffixSort as SS};
    match s {
        S::Msg | S::Untagged | S::Suffix(SS::Msg) => LSort::Msg,
        S::Pub | S::Suffix(SS::Pub) => LSort::Pub,
        S::Fresh | S::Suffix(SS::Fresh) => LSort::Fresh,
        S::Node | S::Suffix(SS::Node) => LSort::Node,
        S::Nat | S::Suffix(SS::Nat) => LSort::Nat,
    }
}

/// `LSort` to `p::SortHint` (the inverse of [`sort_hint_to_lsort`], modulo
/// `Untagged`/`Suffix`, which `LSort` has no counterpart for).
fn lsort_to_sort_hint(s: LSort) -> p::SortHint {
    match s {
        LSort::Msg => p::SortHint::Msg,
        LSort::Pub => p::SortHint::Pub,
        LSort::Fresh => p::SortHint::Fresh,
        LSort::Node => p::SortHint::Node,
        LSort::Nat => p::SortHint::Nat,
    }
}

/// A parser-AST variable spec as the `LVar` it denotes.
pub(crate) fn varspec_to_lvar(v: &p::VarSpec) -> tamarin_term::lterm::LVar {
    tamarin_term::lterm::LVar::new(v.name.as_str(), sort_hint_to_lsort(v.sort), v.idx)
}

/// An `LVar` as the parser-AST variable spec that denotes it (no SAPIC type
/// annotation — canonical variables never carry one).
fn lvar_to_varspec(v: &tamarin_term::lterm::LVar) -> p::VarSpec {
    p::VarSpec {
        name: v.name.to_string(),
        idx: v.idx,
        sort: lsort_to_sort_hint(v.sort),
        typ: None,
    }
}

/// Substitutes every free variable of `g` via `theta`, leaving Bound
/// variables (already canonical by De-Bruijn position) and name constants
/// (never renamed) untouched. Panics if some free variable has no entry in
/// `theta` — see [`rename_var`]: a guarded formula's free variables are
/// ALWAYS assumed to be already covered by an exhaustively-computed
/// canonical labelling (per `work.tex`'s argument that every free variable
/// in a constraint system's formula is bound by some action formula or rule
/// constraint elsewhere in the same system), so a miss means `theta` was
/// built incorrectly or this formula wasn't actually closed the way
/// `work.tex` assumes — either way a caller bug.
pub fn subst_via_theta_guarded(g: &Guarded, theta: &Theta) -> Guarded {
    map_lvars_in_guarded(g, |v| lvar_to_varspec(&rename_var(theta, varspec_to_lvar(v))))
}

/// Canonically orders an atom's operands where doing so is sound: `=` is
/// symmetric (`s = t` and `t = s` are the same atom), so its two sides are
/// reordered smaller-first by [`cmp_term`]. Every other `GAtom` variant is
/// directional (`Less`/`Subterm`, or simply not binary) and is returned
/// unchanged.
///
/// A fact's annotations are dropped: they are hints to the solver's goal
/// ranking with no meaning of their own, and [`fact_to_term`] leaves them out
/// of rule and goal facts too.
fn normalize_atom(a: &GAtom) -> GAtom {
    let t = ac_normalize_term;
    let f = |fact: &GFact| GFact {
        persistent: fact.persistent,
        name: fact.name.clone(),
        args: fact.args.iter().map(ac_normalize_term).collect(),
        annotations: Vec::new(),
    };
    match a {
        GAtom::Eq(x, y) => {
            let (x, y) = (t(x), t(y));
            // Compare the AC-NORMALIZED sides: swapping has to be decided
            // after flattening/sorting any AC operator each side carries,
            // not before — e.g. `xor(b,a) = c` and `c = xor(a,b)` must
            // settle on the same orientation, which only holds once both
            // `xor`s are already in their sorted form.
            if cmp_term(&x, &y) == std::cmp::Ordering::Greater {
                GAtom::Eq(y, x)
            } else {
                GAtom::Eq(x, y)
            }
        }
        GAtom::Less(x, y) => GAtom::Less(t(x), t(y)),
        GAtom::LessMset(x, y) => GAtom::LessMset(t(x), t(y)),
        GAtom::Subterm(x, y) => GAtom::Subterm(t(x), t(y)),
        GAtom::Action(fact, time) => GAtom::Action(f(fact), t(time)),
        GAtom::Last(x) => GAtom::Last(t(x)),
        GAtom::Pred(fact) => GAtom::Pred(f(fact)),
    }
}

/// Brings a term reachable from a guarded formula's atoms into `CAN_AC`
/// normal form: recursively normalizes every subterm bottom-up, then at
/// each AC operator (`xor`/`union`/`mult`/`tplus`/a user `[AC]` symbol,
/// i.e. [`crate::guarded::is_ac_binop`]) flattens the (now-normalized)
/// chain and sorts the flattened operands by [`cmp_term`] before rebuilding
/// — the same flatten-then-sort `CAN_AC` already applies to `LNTerm` via
/// `f_app_ac` (`tamarin_term::term`), just rebuilt here as a right-nested
/// `GTerm::BinOp` chain since `GTerm` (unlike `Term`) has no n-ary AC
/// application node to flatten INTO.
///
/// The bilinear-pairing C symbol `em` — commutative but not associative,
/// parsed as `GTerm::App("em", [_, _])` (see
/// [`crate::guarded::funsym_key`]'s special case) — gets the same
/// treatment minus the flattening: its two (normalized) arguments are
/// sorted, never merged with a nested `em`.
fn ac_normalize_term(t: &GTerm) -> GTerm {
    match t {
        GTerm::BinOp(op, a, b) if guarded::is_ac_binop(op) => {
            let combined = GTerm::BinOp(*op, ga(ac_normalize_term(a)), ga(ac_normalize_term(b)));
            let mut leaves = Vec::new();
            guarded::flatten_ac_binop(op, &combined, &mut leaves);
            let mut sorted: Vec<GTerm> = leaves.into_iter().cloned().collect();
            sorted.sort_by(cmp_term);
            rebuild_ac_chain(*op, sorted)
        }
        GTerm::BinOp(op, a, b) => {
            GTerm::BinOp(*op, ga(ac_normalize_term(a)), ga(ac_normalize_term(b)))
        }
        GTerm::App(n, args) if &**n == "em" && args.len() == 2 => {
            let mut sorted: Vec<GTerm> = args.iter().map(ac_normalize_term).collect();
            sorted.sort_by(cmp_term);
            GTerm::App(n.clone(), sorted.into())
        }
        GTerm::App(n, args) => GTerm::App(n.clone(), args.iter().map(ac_normalize_term).collect()),
        GTerm::AlgApp(n, a, b) => GTerm::AlgApp(
            n.clone(),
            ga(ac_normalize_term(a)),
            ga(ac_normalize_term(b)),
        ),
        GTerm::Pair(items) => GTerm::Pair(items.iter().map(ac_normalize_term).collect()),
        GTerm::Diff(a, b) => GTerm::Diff(ga(ac_normalize_term(a)), ga(ac_normalize_term(b))),
        GTerm::PatMatch(x) => GTerm::PatMatch(ga(ac_normalize_term(x))),
        GTerm::Var(_)
        | GTerm::PubLit(_)
        | GTerm::FreshLit(_)
        | GTerm::NatLit(_)
        | GTerm::Number(_)
        | GTerm::NumberOne
        | GTerm::NatOne
        | GTerm::DhNeutral => t.clone(),
    }
}

/// Rebuilds a flattened, sorted (>= 2 elements — an actual `BinOp` node
/// always flattens to at least its own two leaves) AC operand list as a
/// right-nested `BinOp` chain, preserving the sorted left-to-right order:
/// `[s0, s1, s2]` becomes `BinOp(op, s0, BinOp(op, s1, s2))`.
fn rebuild_ac_chain(op: p::BinOp, sorted: Vec<GTerm>) -> GTerm {
    let mut rev = sorted.into_iter().rev();
    let mut acc = rev
        .next()
        .expect("an AC BinOp always flattens to at least two leaves");
    for x in rev {
        acc = GTerm::BinOp(op, ga(x), ga(acc));
    }
    acc
}

/// Brings `g` into `CAN_AC` normal form: recursively normalizes every
/// subformula bottom-up, then at each level — symmetric `=` operands
/// reordered, a quantifier's guard atoms sorted, and `Conj`/`Disj` children
/// flattened, deduplicated (via [`crate::guarded::gconj`] /
/// [`crate::guarded::gdisj`]), and sorted — all by the total orders
/// [`cmp_term`]/[`cmp_atom`]/[`cmp_guarded`] already established in
/// `guarded.rs`. Every quantifier's binder list also has its (purely
/// cosmetic) display names erased, since only each binder's De-Bruijn
/// POSITION is semantically meaningful — see the module docs.
pub fn ac_normalize_guarded(g: &Guarded) -> Guarded {
    match g {
        Guarded::Atom(a) => Guarded::Atom(normalize_atom(a)),
        Guarded::Disj(items) => {
            let mut normalized: Vec<Guarded> = items.iter().map(ac_normalize_guarded).collect();
            normalized.sort_by(cmp_guarded);
            guarded::gdisj(normalized)
        }
        Guarded::Conj(items) => {
            let mut normalized: Vec<Guarded> = items.iter().map(ac_normalize_guarded).collect();
            normalized.sort_by(cmp_guarded);
            guarded::gconj(normalized)
        }
        Guarded::GGuarded {
            qua,
            vars,
            guards,
            body,
        } => {
            let anonymous_vars: Vec<GBinding> = vars
                .iter()
                .map(|b| GBinding {
                    name: String::new(),
                    sort: b.sort,
                })
                .collect();
            let mut sorted_guards: Vec<GAtom> = guards.iter().map(normalize_atom).collect();
            sorted_guards.sort_by(cmp_atom);
            let normalized_body = ac_normalize_guarded(body);
            match qua {
                Quant::All => gall(anonymous_vars, sorted_guards, normalized_body),
                Quant::Ex => gex(anonymous_vars, sorted_guards, normalized_body),
            }
        }
    }
}

/// Canonizes a guarded formula w.r.t. $\alphaeqac$: substitutes its free
/// variables via the already-canonical `theta`, then brings the result into
/// `CAN_AC` normal form (see the module docs, [`subst_via_theta_guarded`],
/// and [`ac_normalize_guarded`]).
pub fn canonicalize_guarded(
    g: &Guarded,
    theta: &Theta,
) -> Guarded {
    ac_normalize_guarded(&subst_via_theta_guarded(g, theta))
}

/// Fingerprints an ALREADY-CANONICAL guarded formula — the output of
/// [`canonicalize_guarded`]. Shares its digest type with
/// [`tamarin_term::fingerprint::fingerprint_term`] (which also covers
/// [`canonicalize_fact`]/[`canonicalize_rule`]'s outputs, both plain
/// canonical `LNTerm`s): one uniform fingerprint type across every
/// canonized object kind this crate produces.
///
/// A Merkle hash, exactly like `fingerprint_term`: a compound node's
/// fingerprint is built from the fingerprints of its already-fingerprinted
/// children (the "subhash" composition), not their raw content, so a
/// `Guarded`'s `Conj`/`Disj`/`GGuarded` fingerprint is a function of its
/// subformulas' fingerprints, and (once a constraint-system-level
/// canonizer exists) a system's fingerprint can be built the same way from
/// its rules'/formulas' fingerprints in turn. Only a genuine leaf (a
/// literal, a name, a bare De-Bruijn index) ever hashes its own fields
/// directly — there's nothing further to recurse into.
pub fn fingerprint_guarded(g: &Guarded) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    match g {
        Guarded::Atom(a) => {
            h.tag("Atom");
            h.digest(&fingerprint_atom(a));
        }
        Guarded::Disj(items) => {
            h.tag("Disj");
            h.u64(items.len() as u64);
            for it in items.iter() {
                h.digest(&fingerprint_guarded(it));
            }
        }
        Guarded::Conj(items) => {
            h.tag("Conj");
            h.u64(items.len() as u64);
            for it in items.iter() {
                h.digest(&fingerprint_guarded(it));
            }
        }
        Guarded::GGuarded {
            qua,
            vars,
            guards,
            body,
        } => {
            h.tag("GGuarded");
            hash_quant(&mut h, *qua);
            h.u64(vars.len() as u64);
            for v in vars.iter() {
                hash_binding(&mut h, v);
            }
            h.u64(guards.len() as u64);
            for a in guards.iter() {
                h.digest(&fingerprint_atom(a));
            }
            h.digest(&fingerprint_guarded(body));
        }
    }
    h.finish()
}

fn hash_quant(h: &mut FingerprintHasher, q: Quant) {
    h.u8(match q {
        Quant::All => 0,
        Quant::Ex => 1,
    });
}

fn hash_binding(h: &mut FingerprintHasher, b: &GBinding) {
    // `name` is purely cosmetic (erased to `""` by `ac_normalize_guarded`)
    // but still hashed for totality — a value that skipped normalization
    // would then also produce a different fingerprint, rather than
    // silently colliding with a properly-normalized one.
    h.bytes(b.name.as_bytes());
    hash_sort_hint(h, b.sort);
}

pub(crate) fn fingerprint_atom(a: &GAtom) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    match a {
        GAtom::Eq(x, y) => {
            h.tag("Eq");
            h.digest(&fingerprint_gterm(x));
            h.digest(&fingerprint_gterm(y));
        }
        GAtom::Less(x, y) => {
            h.tag("Less");
            h.digest(&fingerprint_gterm(x));
            h.digest(&fingerprint_gterm(y));
        }
        GAtom::LessMset(x, y) => {
            h.tag("LessMset");
            h.digest(&fingerprint_gterm(x));
            h.digest(&fingerprint_gterm(y));
        }
        GAtom::Subterm(x, y) => {
            h.tag("Subterm");
            h.digest(&fingerprint_gterm(x));
            h.digest(&fingerprint_gterm(y));
        }
        GAtom::Action(fact, time) => {
            h.tag("Action");
            h.digest(&fingerprint_fact(fact));
            h.digest(&fingerprint_gterm(time));
        }
        GAtom::Last(x) => {
            h.tag("Last");
            h.digest(&fingerprint_gterm(x));
        }
        GAtom::Pred(fact) => {
            h.tag("Pred");
            h.digest(&fingerprint_fact(fact));
        }
    }
    h.finish()
}

fn fingerprint_fact(f: &GFact) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    h.tag("GFact");
    h.u8(u8::from(f.persistent));
    h.bytes(f.name.as_bytes());
    h.u64(f.args.len() as u64);
    for a in f.args.iter() {
        h.digest(&fingerprint_gterm(a));
    }
    h.u64(f.annotations.len() as u64);
    for ann in f.annotations.iter() {
        h.u8(fact_annotation_byte(ann));
    }
    h.finish()
}

fn fact_annotation_byte(a: &p::FactAnnotation) -> u8 {
    match a {
        p::FactAnnotation::SolveFirst => 0,
        p::FactAnnotation::SolveLast => 1,
        p::FactAnnotation::NoSources => 2,
    }
}

pub(crate) fn fingerprint_gterm(t: &GTerm) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    match t {
        GTerm::Var(v) => {
            h.tag("Var");
            hash_bvar(&mut h, v);
        }
        GTerm::PubLit(s) => {
            h.tag("PubLit");
            h.bytes(s.as_bytes());
        }
        GTerm::FreshLit(s) => {
            h.tag("FreshLit");
            h.bytes(s.as_bytes());
        }
        GTerm::NatLit(s) => {
            h.tag("NatLit");
            h.bytes(s.as_bytes());
        }
        GTerm::Number(n) => {
            h.tag("Number");
            h.u64(*n);
        }
        GTerm::NumberOne => {
            h.tag("NumberOne");
        }
        GTerm::NatOne => {
            h.tag("NatOne");
        }
        GTerm::DhNeutral => {
            h.tag("DhNeutral");
        }
        GTerm::App(n, args) => {
            h.tag("App");
            h.bytes(n.as_bytes());
            h.u64(args.len() as u64);
            for a in args.iter() {
                h.digest(&fingerprint_gterm(a));
            }
        }
        GTerm::AlgApp(n, a, b) => {
            h.tag("AlgApp");
            h.bytes(n.as_bytes());
            h.digest(&fingerprint_gterm(a));
            h.digest(&fingerprint_gterm(b));
        }
        GTerm::Pair(items) => {
            h.tag("Pair");
            h.u64(items.len() as u64);
            for it in items.iter() {
                h.digest(&fingerprint_gterm(it));
            }
        }
        GTerm::Diff(a, b) => {
            h.tag("Diff");
            h.digest(&fingerprint_gterm(a));
            h.digest(&fingerprint_gterm(b));
        }
        GTerm::BinOp(op, a, b) => {
            h.tag("BinOp");
            hash_binop(&mut h, op);
            h.digest(&fingerprint_gterm(a));
            h.digest(&fingerprint_gterm(b));
        }
        GTerm::PatMatch(x) => {
            h.tag("PatMatch");
            h.digest(&fingerprint_gterm(x));
        }
    }
    h.finish()
}

fn hash_binop(h: &mut FingerprintHasher, op: &p::BinOp) {
    match op {
        p::BinOp::Exp => {
            h.tag("Exp");
        }
        p::BinOp::Mult => {
            h.tag("Mult");
        }
        p::BinOp::Union => {
            h.tag("Union");
        }
        p::BinOp::Xor => {
            h.tag("Xor");
        }
        p::BinOp::NatPlus => {
            h.tag("NatPlus");
        }
        p::BinOp::AcFct(name) => {
            h.tag("AcFct");
            h.bytes(name.as_bytes());
        }
    }
}

fn hash_bvar(h: &mut FingerprintHasher, v: &BVar) {
    match v {
        BVar::Bound(n) => {
            h.tag("Bound");
            h.u64(*n as u64);
        }
        BVar::Free(vs) => {
            h.tag("Free");
            hash_varspec(h, vs);
        }
    }
}

fn hash_varspec(h: &mut FingerprintHasher, v: &p::VarSpec) {
    h.bytes(v.name.as_bytes());
    h.u64(v.idx);
    hash_sort_hint(h, v.sort);
    match &v.typ {
        Some(t) => {
            h.tag("Some");
            h.bytes(t.as_bytes());
        }
        None => {
            h.tag("None");
        }
    }
}

fn hash_sort_hint(h: &mut FingerprintHasher, s: p::SortHint) {
    use p::{SortHint as S, SuffixSort as SS};
    let b = match s {
        S::Msg => 0,
        S::Pub => 1,
        S::Fresh => 2,
        S::Node => 3,
        S::Nat => 4,
        S::Suffix(SS::Msg) => 5,
        S::Suffix(SS::Pub) => 6,
        S::Suffix(SS::Fresh) => 7,
        S::Suffix(SS::Node) => 8,
        S::Suffix(SS::Nat) => 9,
        // Should never occur in an ALREADY-CANONICAL formula (`subst_via_
        // theta_guarded`'s successful-lookup path always produces a
        // resolved `SortHint` via `lvar_to_varspec`), but still given a
        // distinct tag rather than panicking or aliasing a resolved sort.
        S::Untagged => 10,
    };
    h.u8(b);
}

// =============================================================================
// Whole-system assembly -- field-by-field canonicalization of a whole
// constraint `System`
// =============================================================================
//
// `canonicalize_constraint_system` gets the labelling from the term graph
// (`canon_term_graph`) and renames every field through it here: the graph
// part, formulas/solved_formulas/lemmas, eq_store, subterm_store and goals.

use crate::bliss_proc::BlissError;
use crate::constraint::constraints::Goal;
use crate::constraint::solver::proof_method::{ProofMethod, Result as FinishedResult};
use crate::constraint::solver::Contradiction;
use crate::constraint::system::{GoalStatus, Side, SourceKind, System};
use crate::tools::equation_store::{EqDisj, EquationStore, LNSubst, LNSubstVFresh};
use crate::tools::subterm_store::{SubtermConstraint, SubtermStore};
use tamarin_term::vterm::var_term;

/// The complete canonical form of a constraint `System`.
///
/// `PartialEq` only, no `Eq`/`Ord` derive: `Guarded` itself derives only
/// `PartialEq` (no `Eq`, no `Ord` -- only the free [`cmp_guarded`]), so
/// neither derives here for `CanonicalSystem` either.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalSystem {
    /// Rule instances (with their rule's identity), `System::edges`,
    /// `less_atoms` and `last_atom`, renamed and sorted
    /// (`canon_term_graph::renamed_graph_part_term`).
    pub graph_part: LNTerm,
    /// `formulas`: canonicalized via the system's labelling, sorted via
    /// `cmp_guarded` -- **NOT deduplicated**
    /// (REVISED 2026-09-23, per direct user instruction: `stores_contains`
    /// only prevents RAW-syntactic duplicates at insertion, not
    /// alpha-equivalent ones, so two solver-tracked-as-distinct formulas
    /// could canonicalize identically; see [`sort_guarded`]'s own doc
    /// comment for the full reasoning, which mirrors `eq_store.conj`'s
    /// pre-existing multiset-preservation policy). KEPT SEPARATE
    /// from `solved_formulas` -- membership in one store vs. the other is
    /// live proof-search state (e.g. `ProofMethod::Induction`'s
    /// applicability, `isInitialSystem`), not pure memoization, so
    /// merging them would conflate two non-interchangeable systems.
    pub formulas: Vec<Guarded>,
    /// `solved_formulas`, canonicalized the same way as `formulas`
    /// (sorted, not deduplicated), as its own independent list.
    pub solved_formulas: Vec<Guarded>,
    /// `lemmas`, canonicalized the same way.
    pub lemmas: Vec<Guarded>,
    pub eq_store: CanonicalEqStore,
    pub subterm_store: CanonicalSubtermStore,
    /// `sys.goals`, minus `Goal::Split` (dropped -- it only NAMES an
    /// `EqDisj` already present in `eq_store.conj`; nothing else ever
    /// reads its raw id, the exact same reasoning that already dropped
    /// `EqDisj::split_id` itself). See [`canonicalize_goals`] for each
    /// variant's canonicalization and why `GoalStatus::solved` is kept
    /// (content-relevant: gates whether `candidate_methods` offers the goal
    /// again) while `::looping`/`::nr` are not (pure heuristic-ranking /
    /// path-dependent bookkeeping). Sorted, NOT deduplicated -- see
    /// [`canonicalize_goals`]'s own doc comment.
    pub goals: Vec<CanonicalGoal>,
    pub source_kind: Option<SourceKind>,
    pub side: Option<Side>,
}

/// Canonicalized form of one non-`Split` [`Goal`], paired
/// with whether it's currently solved -- see [`CanonicalSystem::goals`]'s
/// own doc comment.
///
/// `PartialEq` only, no `Eq`/`Ord`: `Disj` embeds `Vec<Guarded>`, and
/// `Guarded` itself has no `Ord`/`Eq` -- same reason [`CanonicalSystem`]
/// itself only derives `PartialEq`. [`cmp_canonical_goal`] is the total
/// order.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalGoal {
    pub kind: CanonicalGoalKind,
    pub solved: bool,
}

/// The canonicalized payload of one [`CanonicalGoal`] -- see
/// [`canonicalize_goals`] for exactly how each variant is derived from
/// the matching [`Goal`] variant.
#[derive(Debug, Clone, PartialEq)]
pub enum CanonicalGoalKind {
    /// `Goal::Action(LVar, LNFact)`: the `NodeId` and the fact canonicalized
    /// via `theta`, and (in [`CanonicalGoal::solved`]) whether the goal is
    /// solved, which decides what `candidate_methods` offers.
    Action(LVar, LNTerm),
    /// `Goal::Chain(NodeConc, NodePrem)`: both endpoints' `NodeId`s
    /// canonicalized via `theta`; the port indices need no renaming.
    Chain(LVar, ConcIdx, LVar, PremIdx),
    /// `Goal::Premise(NodePrem, LNFact)`: the `NodeId` canonicalized the
    /// same way as `Chain`'s endpoints, plus the fact itself
    /// canonicalized via `theta`.
    Premise(LVar, PremIdx, LNTerm),
    /// `Goal::Disj(Disj<Guarded>)`: each alternative canonicalized via
    /// `canonicalize_guarded` against `theta`, then sorted -- NOT
    /// deduplicated, same treatment as `formulas`/`solved_formulas`/`lemmas`.
    Disj(Vec<Guarded>),
    /// `Goal::Subterm((LNTerm, LNTerm))`: both terms canonicalized via
    /// `apply_renaming` -- same treatment as `subterm_store`'s
    /// pairs.
    Subterm(LNTerm, LNTerm),
}

/// All four fields are `Ord` on their own, so this derives it directly
/// (unlike [`CanonicalSystem`], which needs a custom comparison only
/// because of its `Vec<Guarded>` fields).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CanonicalEqStore {
    /// `eq_store.subst`, canonical-key sorted.
    pub subst: Vec<(LVar, LNTerm)>,
    /// `eq_store.conj` -- `split_id` DROPPED: its only consumer is
    /// `Goal::Split`, which `CanonicalSystem::goals` drops too (it only
    /// NAMES an `EqDisj` already present here; nothing else ever reads a
    /// `split_id` value -- see `CanonicalSystem::goals`'s own doc
    /// comment). Each alternative is stored
    /// fully canonicalized, not as a raw `LNSubstVFresh`: domain keys
    /// renamed via the shared `theta`, each range term canonicalized via
    /// a per-alternative-forked `CanonLabelling` pass (see
    /// [`canonicalize_eq_disj_alternative`]). The middle `Vec` (one
    /// `EqDisj`'s alternatives) and the
    /// outer `Vec` (the list of `EqDisj`s) are both sorted by their own
    /// canonicalized content directly, as MULTISETS: duplicates from
    /// alpha-equivalent alternatives/disjunctions are preserved, never
    /// deduplicated. Tamarin's own solver keeps them distinct on purpose
    /// (see `tools/equation_store.rs`'s own comment on its
    /// `applyBound`/`S.fromList` handling for the proven real regression:
    /// alpha-canonically deduping a disjunction's alternatives there
    /// collapses a real 6-substitution case to 5, dropping a split case
    /// and changing the proof tree), so collapsing them here would
    /// likewise conflate two non-interchangeable systems.
    pub conj: Vec<Vec<Vec<(LVar, LNTerm)>>>,
}

/// All four fields are `Ord` on their own, so this derives it directly --
/// see [`CanonicalEqStore`]'s own doc comment.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CanonicalSubtermStore {
    pub subterms: Vec<(LNTerm, LNTerm)>,
    pub solved_subterms: Vec<(LNTerm, LNTerm)>,
    pub contradictory: bool,
    pub neg_subterms: Vec<(LNTerm, LNTerm)>,
}

/// Total order over [`CanonicalGoal`] -- constructor rank
/// (`Action < Chain < Premise < Disj < Subterm`, this enum's declaration order,
/// mirroring [`crate::constraint::constraints::cmp_goal`]'s own style)
/// then payload, `solved` last (so two goals with the same canonicalized
/// content but different solved-status sort adjacently, not scattered).
fn cmp_canonical_goal(a: &CanonicalGoal, b: &CanonicalGoal) -> std::cmp::Ordering {
    cmp_canonical_goal_kind(&a.kind, &b.kind).then_with(|| a.solved.cmp(&b.solved))
}

fn cmp_canonical_goal_kind(a: &CanonicalGoalKind, b: &CanonicalGoalKind) -> std::cmp::Ordering {
    fn rank(k: &CanonicalGoalKind) -> u8 {
        match k {
            CanonicalGoalKind::Action(..) => 0,
            CanonicalGoalKind::Chain(..) => 1,
            CanonicalGoalKind::Premise(..) => 2,
            CanonicalGoalKind::Disj(_) => 3,
            CanonicalGoalKind::Subterm(..) => 4,
        }
    }
    let (ra, rb) = (rank(a), rank(b));
    if ra != rb {
        return ra.cmp(&rb);
    }
    // Rank equality above guarantees the same variant, so each `let …
    // else` binding of `b` is infallible.
    match a {
        CanonicalGoalKind::Action(n1, f1) => {
            let CanonicalGoalKind::Action(n2, f2) = b else {
                unreachable!("rank matched Action")
            };
            n1.cmp(n2).then_with(|| f1.cmp(f2))
        }
        CanonicalGoalKind::Chain(cl1, ci1, pl1, pi1) => {
            let CanonicalGoalKind::Chain(cl2, ci2, pl2, pi2) = b else {
                unreachable!("rank matched Chain")
            };
            cl1.cmp(cl2)
                .then_with(|| ci1.cmp(ci2))
                .then_with(|| pl1.cmp(pl2))
                .then_with(|| pi1.cmp(pi2))
        }
        CanonicalGoalKind::Premise(pl1, pi1, f1) => {
            let CanonicalGoalKind::Premise(pl2, pi2, f2) = b else {
                unreachable!("rank matched Premise")
            };
            pl1.cmp(pl2).then_with(|| pi1.cmp(pi2)).then_with(|| f1.cmp(f2))
        }
        CanonicalGoalKind::Disj(d1) => {
            let CanonicalGoalKind::Disj(d2) = b else {
                unreachable!("rank matched Disj")
            };
            cmp_guarded_slice(d1, d2)
        }
        CanonicalGoalKind::Subterm(a1, b1) => {
            let CanonicalGoalKind::Subterm(a2, b2) = b else {
                unreachable!("rank matched Subterm")
            };
            a1.cmp(a2).then_with(|| b1.cmp(b2))
        }
    }
}

/// Lexicographic comparison of two already-canonical `Guarded` slices via
/// `cmp_guarded` (there is no `Ord for Guarded` to fall back on), tie-broken
/// by length when one
/// is a prefix of the other.
fn cmp_guarded_slice(a: &[Guarded], b: &[Guarded]) -> std::cmp::Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        let c = cmp_guarded(x, y);
        if c != std::cmp::Ordering::Equal {
            return c;
        }
    }
    a.len().cmp(&b.len())
}

/// Sorts `v` via `cmp_guarded` (no `Ord for Guarded`, see above) into a
/// content-driven, cross-system-stable order -- **deliberately does NOT
/// dedup**, unlike the similarly-named `sort_dedup_guarded` local closure
/// in `constraint/solver/rename_precise.rs` (that one mirrors HS's `S.Set
/// LNGuarded` rebuild, which sorts AND collision-dedups after every
/// rename -- a real solver-internal behavior, not this function's
/// concern). REVISED 2026-09-23, per direct user instruction: every
/// `CanonicalSystem` field built from a `System` `Vec` (not an
/// already-deduplicated `Set`-typed field, e.g. `neg_subterms`) must
/// preserve its exact multiplicity, not just its distinct content --
/// exactly the same principle `eq_store.conj`'s multiset preservation was
/// already built on (see `CanonicalEqStore::conj`'s own doc comment: a
/// solver-level dedup can silently drop a real split case, confirmed on a
/// real regression). `formulas`/`solved_formulas`/`lemmas` (`Vec<Arc<Guarded>>`
/// in `System`) and a `Goal::Disj`'s own alternatives (`Vec<Guarded>`) are
/// exactly this shape -- two syntactically-identical entries are not
/// assumed to be redundant, since nothing here has proven the solver never
/// produces a real duplicate.
fn sort_guarded(mut v: Vec<Guarded>) -> Vec<Guarded> {
    v.sort_by(cmp_guarded);
    v
}

/// Canonicalizes a whole constraint `System`: two systems get the same form
/// iff they are equal up to a sort-respecting renaming of their variables and
/// AC. bliss's canonical labeling of the system's term graph names the
/// variables ([`crate::canon_term_graph`]), and the form is the system
/// renamed by that labelling, field by field
/// ([`canonicalize_system_content`]).
///
/// Fallible only through the bliss subprocess.
pub fn canonicalize_constraint_system(sys: &System) -> Result<CanonicalSystem, BlissError> {
    canonicalize_constraint_system_with_labelling(sys).map(|(canon, _)| canon)
}

/// [`canonicalize_constraint_system`], also returning the labelling that
/// maps `sys` onto its canonical form, for canonicalizing anything else
/// derived from `sys` consistently with it (e.g. its proof methods,
/// [`canonicalize_proof_method`]). When `sys` has a symmetry, another
/// labelling gives the same form, so anything canonicalized through it
/// should be compared as a multiset, which a symmetry maps onto itself.
pub fn canonicalize_constraint_system_with_labelling(
    sys: &System,
) -> Result<(CanonicalSystem, CanonLabelling), BlissError> {
    crate::canon_term_graph::canonicalize_constraint_system_via_term_graph_with_labelling(
        sys,
        crate::canon_term_graph::TermGraphOptions::default(),
    )
    .map(|(canon, labelling, _)| (canon, labelling))
}

/// `sys` canonicalized field by field under `labelling`, which must name
/// every free variable of `sys` (a miss panics, see [`rename_var`]), with
/// `graph_part` the system's graph part already renamed by it.
pub(crate) fn canonicalize_system_content(
    sys: &System,
    labelling: &CanonLabelling,
    graph_part: LNTerm,
) -> CanonicalSystem {
    let theta = labelling.theta();
    // Three separate fields, not one: `solved_formulas` membership is live
    // proof-search state, not memoization -- `ProofMethod::Induction`'s
    // applicability and `isInitialSystem` both key off it.
    let canonicalize_all = |fs: &[Arc<Guarded>]| {
        sort_guarded(fs.iter().map(|f| canonicalize_guarded(f, theta)).collect())
    };
    let formulas = canonicalize_all(&sys.formulas);
    let solved_formulas = canonicalize_all(&sys.solved_formulas);
    let lemmas = canonicalize_all(&sys.lemmas);
    CanonicalSystem {
        graph_part,
        formulas,
        solved_formulas,
        lemmas,
        eq_store: canonicalize_eq_store(&sys.eq_store, labelling),
        subterm_store: canonicalize_subterm_store(&sys.subterm_store, theta),
        goals: canonicalize_goals(&sys.goals, theta),
        source_kind: sys.source_kind,
        side: sys.side,
    }
}

/// Canonicalizes `store` under `labelling`, which must name every free
/// variable of the store.
///
/// **Which variables are free.** Per EquationStore.hs's `EqStore` docs, the
/// variables of `subst` and the DOMAIN keys of every alternative `sigma_ij`
/// in `conj` are free (globally existentially quantified), while the
/// variables in an alternative's RANGE are fresh, local to that one
/// alternative. A free variable can occur ONLY in the store: merging two rule
/// instances unifies their variant-carrying terms and adds the unifiers as a
/// disjunction over both instances' variables, and once the merged instance
/// is gone its variables survive only as keys of that disjunction and of its
/// rule-variant disjunction (idbased/BP_IBS_1). Such a variable links the
/// disjunctions it occurs in, so it is neither dropped nor local; the term
/// graph names it like every other free variable.
fn canonicalize_eq_store(store: &EquationStore, labelling: &CanonLabelling) -> CanonicalEqStore {
    // eq_store.subst: domain AND range variables are free.
    let mut subst: Vec<(LVar, LNTerm)> = store
        .subst
        .iter()
        .map(|(v, t)| {
            let canon_key = rename_var(labelling.theta(), *v);
            let canon_term = apply_renaming(labelling.theta(), t.clone());
            (canon_key, canon_term)
        })
        .collect();
    subst.sort();

    // eq_store.conj: `split_id` DROPPED (see this file's struct docs
    // above). Each `EqDisj`'s alternatives are a MULTISET (see
    // `CanonicalEqStore::conj`'s own doc comment for why duplicates must
    // survive), sorted once canonical; likewise the outer list of
    // `EqDisj`s.
    let mut conj: Vec<Vec<Vec<(LVar, LNTerm)>>> = store
        .conj
        .iter()
        .map(|disj| canonicalize_eq_disj(disj, labelling))
        .collect();
    conj.sort();

    CanonicalEqStore { subst, conj }
}

/// Canonicalizes one `EqDisj`'s alternatives, preserving multiplicity (no
/// alpha-dedup -- see `CanonicalEqStore::conj`'s own doc comment) and
/// sorting by the now-canonical content.
fn canonicalize_eq_disj(disj: &EqDisj, labelling: &CanonLabelling) -> Vec<Vec<(LVar, LNTerm)>> {
    let mut alternatives: Vec<Vec<(LVar, LNTerm)>> = disj
        .substs
        .iter()
        .map(|alt| canonicalize_eq_disj_alternative(alt, labelling))
        .collect();
    alternatives.sort();
    alternatives
}

/// Canonicalizes ONE alternative substitution (one `sigma_ij` in
/// EquationStore.hs's `sigma_i1 ∨ … ∨ sigma_ik_i` notation) of an
/// `EqDisj`.
///
/// **Domain keys** are free variables (the `x_i` in EquationStore.hs's own
/// semantics) -- canonicalized via the shared `theta`, which names every
/// one of them (a miss panics, see [`rename_var`]) -- and sorted by that CANONICAL
/// identity, not raw `LVar` Ord, so the range terms below get visited
/// (and their local witnesses numbered) in a content-driven,
/// cross-system-stable order rather than one that depends on incidental
/// proof-search allocation order.
///
/// **Range terms** are entirely, uniformly LOCAL/fresh: per `SubstVFresh`'s
/// own contract, every variable in a range term is existentially bound to
/// THIS ONE alternative, regardless of whether its raw identity happens
/// to look like a real system variable's -- a routine case, not a corner
/// case: Maude unifies actual terms containing actual system variables,
/// and `freshen_witness_range` leaves a variable that legitimately
/// belongs to the original input equations untouched. So:
///
/// 1. **Freshen the WHOLE alternative's range in one call**
///    ([`LNSubstVFresh::fresh_to_free_avoiding`] -- shares one rename
///    cache across every entry, so a variable repeated across two range
///    terms gets the SAME fresh identity both times, and leaves domain
///    keys untouched), with an allocator that avoids every raw variable
///    index already claimed by `labelling.theta()`
///    ([`raw_var_idx_avoid_floor`]).
/// 2. **Canonize all freshened range terms together, as ONE list term in
///    canonical key order, in a single call against a FRESH FORK of
///    `labelling`** (`labelling.clone()`), then split the result back into
///    per-key range terms and DISCARD the fork. Never mutate the real,
///    continuing `labelling` itself. One call ties a witness shared
///    between two range terms together (see
///    `eq_disj_alternative_shares_a_witness_across_two_of_its_own_range_terms`),
///    and it lets every range term take part in naming every witness: the
///    `Canonizer` settles a tie between renamings that produce the same
///    term by keeping the first candidate, i.e. by raw variable order, so
///    canonizing the range terms one after another would settle a tie
///    inside one of them that way and hand the choice on to the later ones
///    (see `eq_disj_alternative_resolves_a_tie_across_its_range_terms_by_content`).
///
/// Step 2's fork is NOT optional, unlike an earlier revision of this
/// function assumed. `EqDisj`'s alternatives (`sigma_i1 ∨ … ∨ sigma_ik_i`)
/// are each other's INDEPENDENT existential scopes -- both within one
/// `EqDisj` and across different ones -- so two alternatives with the
/// EXACT SAME shape, differing only in raw witness identity (mirroring
/// Maude's arbitrary per-call numbering), must canonize to the IDENTICAL
/// result regardless of which one happens to be processed first. Canonizing
/// against a SHARED, advancing `labelling` breaks that: whichever
/// alternative runs first claims the lower canonical indices, and an
/// otherwise-identical sibling processed after it is forced to discover
/// its own local witnesses starting from wherever the first one left the
/// counters, landing on DIFFERENT canonical names for the same shape --
/// caught directly by
/// `eq_disj_preserves_multiplicity_of_alpha_equivalent_alternatives`
/// (which failed under that earlier revision). Forking wholesale and
/// discarding afterward sidesteps this entirely: every alternative always
/// starts from the exact same baseline `labelling`, so processing order
/// can never leak into the result. This is safe to discard because
/// nothing downstream of `eq_store.conj` ever discovers a new literal
/// (`subterm_store` is the only field processed after it, and it's
/// read-only against `theta`, same as every other field before
/// `eq_store.conj`), so there is no legitimate consumer for an advanced counter
/// to be preserved for in the first place.
fn canonicalize_eq_disj_alternative(
    alt: &LNSubstVFresh,
    labelling: &CanonLabelling,
) -> Vec<(LVar, LNTerm)> {
    let mut entries: Vec<(LVar, LVar)> = alt
        .iter()
        .map(|(v, _)| (rename_var(labelling.theta(), *v), *v))
        .collect();
    // `entries` is created from the alternative's domain keys; i.e., a BTreeMap's keys.
    // Thus, each key is unique. This is important for the injectivity assertion below.
    entries.sort_by_key(|(canon, _)| *canon);

    // INVARIANT: `theta` is injective -- a core `Canonizer` guarantee, not
    // an incidental implementation detail. Every newly-discovered literal
    // gets a BRAND NEW canonical index (`Canonizer::canonize_literals`
    // zips a batch's distinct original literals 1:1 against indices drawn
    // from a monotonic, never-reset, never-reused counter; a literal
    // already in `theta` is excluded from "uncanonicalized" before that
    // even runs, so it's never reprocessed), so two DISTINCT domain keys
    // of one alternative can never canonicalize to the SAME `LVar`. If
    // they somehow did (e.g. a `theta` hand-built via `CanonLabelling::
    // from_theta` with a bug, bypassing the Canonizer's own accumulation
    // entirely), the stable sort above would fall back to `alt`'s raw,
    // incidental `BTreeMap` order to break the tie -- silently making
    // which of their two (possibly different) range terms is associated
    // with which output position depend on raw `LVar` identity rather
    // than content, exactly the class of bug this whole design exists to
    // rule out. Fail loudly instead of ever risking that silently.
    for pair in entries.windows(2) {
        assert_ne!(
            pair[0].0, pair[1].0,
            "canonicalize_eq_disj_alternative: two distinct domain keys \
             canonicalized to the SAME literal ({:?}) -- theta is no \
             longer injective, which should be impossible; the ordering \
             between their (potentially different) range terms would be \
             undefined",
            pair[0].0
        );
    }

    if entries.is_empty() {
        // No domain keys, so no range terms either -- nothing to freshen
        // or canonize.
        return Vec::new();
    }

    let mut next_idx = raw_var_idx_avoid_floor(labelling.theta());
    let freshened: LNSubst = alt.fresh_to_free_avoiding(|n| {
        let start = next_idx;
        next_idx += n;
        start
    });

    let range_terms: Vec<LNTerm> = entries
        .iter()
        .map(|(_, v)| {
            // `fresh_to_free_avoiding`'s output drops a trivial `v -> v`
            // mapping (`Subst::from_list`'s own trivial-drop) -- which
            // can only happen if `v` had no occurrences anywhere in the
            // alternative's range terms at all (freshening never
            // reassigns a variable to its own original identity, since
            // the allocator only ever hands out indices strictly above
            // everything currently in scope). Either way, `v` itself
            // (unchanged) is the correct range term to canonize.
            freshened
                .image_of(v)
                .cloned()
                .unwrap_or_else(|| var_term(*v))
        })
        .collect();
    let LNTerm::App(_, canon_ranges) =
        canonicalize_alpha_eq_ac_seeded(&f_app_list(range_terms), &mut labelling.clone())
    else {
        unreachable!("canonizing a list application yields a list application")
    };
    entries
        .into_iter()
        .map(|(canon_key, _)| canon_key)
        .zip(canon_ranges.iter().cloned())
        .collect()
}

/// The smallest raw variable index guaranteed not to collide with any
/// variable key already in `theta` -- the floor
/// [`canonicalize_eq_disj_alternative`]'s range-freshening allocator must
/// start from so a freshly-minted witness identity can never accidentally
/// match an existing (and therefore already-resolved) literal. Only
/// `theta`'s KEYS matter here (what a lookup is keyed by, and the raw
/// literals originally seen -- proof-search-allocated indices, typically
/// large and monotonically increasing over a run), not its VALUES (the
/// canonical output literals, a completely separate, small,
/// sequentially-counted namespace that a lookup is never keyed by).
fn raw_var_idx_avoid_floor(theta: &Theta) -> u64 {
    theta.keys().map(|v| v.idx).max().map_or(0, |m| m + 1)
}

/// Canonicalizes `store` (the `subterm_store`): drops
/// `propagated`/`old_neg_subterms` (Rust-only bookkeeping, not
/// semantically meaningful content), renames every term pair via `theta`
/// (`apply_renaming`, same as `eq_store.subst`'s range), and
/// re-sorts everything -- `neg_subterms` is already a sorted, deduplicated
/// set pre-canonicalization, but renaming can change relative order and
/// even collapse two distinct pairs into one, so the existing sort/dedup
/// can't be trusted to survive it unchanged.
fn canonicalize_subterm_store(
    store: &SubtermStore,
    theta: &Theta,
) -> CanonicalSubtermStore {
    let canon_pair =
        |c: &SubtermConstraint| apply_renaming(theta, (c.small.clone(), c.big.clone()));

    let mut subterms: Vec<(LNTerm, LNTerm)> = store.subterms.iter().map(canon_pair).collect();
    subterms.sort();
    let mut solved_subterms: Vec<(LNTerm, LNTerm)> =
        store.solved_subterms.iter().map(canon_pair).collect();
    solved_subterms.sort();
    let mut neg_subterms: Vec<(LNTerm, LNTerm)> = store
        .neg_subterms
        .iter()
        .map(|(a, b)| apply_renaming(theta, (a.clone(), b.clone())))
        .collect();
    neg_subterms.sort();
    neg_subterms.dedup();

    CanonicalSubtermStore {
        subterms,
        solved_subterms,
        contradictory: store.contradictory,
        neg_subterms,
    }
}

/// Canonicalizes `sys.goals` into [`CanonicalGoal`]s -- read-only against
/// `theta`, like every other field (a missing entry panics via
/// [`rename_var`], not a silent fallback).
///
/// `Goal::Split` is skipped: it only NAMES an `EqDisj` already in
/// `eq_store.conj` (see [`CanonicalSystem::goals`]'s own doc comment).
///
/// Every kept variant keeps [`GoalStatus::solved`] (content-relevant
/// -- see [`CanonicalGoal`]'s own doc comment) but drop `::looping`/`::nr`
/// (pure heuristic-ranking / path-dependent bookkeeping, matching
/// `next_goal_nr`'s own exclusion from the canonical form). Sorted, but
/// **NOT deduped** (REVISED 2026-09-23, per direct user instruction --
/// same reasoning as [`sort_guarded`]'s own doc comment): although
/// `add_goal_with_loop_flag`'s own insertion-time dedup (`canonical_goal_for_dedup`,
/// `constraint/system.rs`) makes `sys.goals` map-like under RAW `Goal`
/// equality, that's raw identity, not alpha-equivalence -- two goals that
/// are raw-distinct (e.g. a `Disj` over different free-variable
/// identities) but canonicalize to the SAME `CanonicalGoal` are exactly
/// the kind of real, solver-tracked-as-distinct entries a post-rename
/// dedup here would silently collapse.
fn canonicalize_goals(
    goals: &[(Goal, GoalStatus)],
    theta: &Theta,
) -> Vec<CanonicalGoal> {
    let mut out: Vec<CanonicalGoal> = goals
        .iter()
        .filter_map(|(goal, status)| {
            canonicalize_goal_kind(goal, theta).map(|kind| CanonicalGoal {
                kind,
                solved: status.solved,
            })
        })
        .collect();
    out.sort_by(cmp_canonical_goal);
    out
}

/// One goal's canonical payload via `theta` (panicking on an uncovered
/// variable, like every other field) -- `None` for `Goal::Split`,
/// whose raw id the canonical form drops (see [`canonicalize_goals`]).
pub fn canonicalize_goal_kind(
    goal: &Goal,
    theta: &Theta,
) -> Option<CanonicalGoalKind> {
    Some(match goal {
        Goal::Split(_) => return None,
        Goal::Action(nid, fact) => CanonicalGoalKind::Action(
            rename_var(theta, *nid),
            fact_to_term(&apply_renaming(theta, fact.clone())),
        ),
        Goal::Chain((conc_nid, conc_idx), (prem_nid, prem_idx)) => CanonicalGoalKind::Chain(
            rename_var(theta, *conc_nid),
            *conc_idx,
            rename_var(theta, *prem_nid),
            *prem_idx,
        ),
        Goal::Premise((prem_nid, prem_idx), fact) => CanonicalGoalKind::Premise(
            rename_var(theta, *prem_nid),
            *prem_idx,
            fact_to_term(&apply_renaming(theta, fact.clone())),
        ),
        Goal::Disj(disj) => CanonicalGoalKind::Disj(sort_guarded(
            disj.0.iter().map(|g| canonicalize_guarded(g, theta)).collect(),
        )),
        Goal::Subterm((small, big)) => {
            let (small, big) = apply_renaming(theta, (small.clone(), big.clone()));
            CanonicalGoalKind::Subterm(small, big)
        }
    })
}

/// A [`ProofMethod`] renamed through a system's canonical labelling (see
/// [`canonicalize_proof_method`]): two canonically equal systems must offer
/// the same MULTISET of these. Only a multiset -- `candidate_methods` ranks
/// goals using `depth`, `GoalStatus::looping` and `GoalStatus::nr`, none of
/// which are part of the canonical form, so the order may differ.
#[derive(Debug, Clone, PartialEq)]
pub enum CanonicalProofMethod {
    Simplify,
    Induction,
    Sorry(Option<String>),
    /// Result and contradiction KIND only: the payload (`NodeId`s, a term)
    /// belongs to whichever contradiction `is_finished` happens to find
    /// first, which depends on raw iteration order.
    Finished {
        result: &'static str,
        contradiction: Option<&'static str>,
    },
    Solve(CanonicalGoalKind),
    /// `solve` of a case split, identified by the canonicalized alternatives
    /// of the `EqDisj` it names -- the canonical form drops split ids.
    SolveSplit(Vec<Vec<(LVar, LNTerm)>>),
}

/// Renames `method` (offered for `sys`) through `labelling`, `sys`'s
/// canonical labelling from [`canonicalize_constraint_system_with_labelling`].
/// Panics on a literal `labelling` doesn't cover, on a split goal naming no
/// `EqDisj` in `sys.eq_store.conj`, and on the display-only
/// `Invalidated`/`RawSolve`, which `candidate_methods` never produces.
pub fn canonicalize_proof_method(
    method: &ProofMethod,
    sys: &System,
    labelling: &CanonLabelling,
) -> CanonicalProofMethod {
    match method {
        ProofMethod::Simplify => CanonicalProofMethod::Simplify,
        ProofMethod::Induction => CanonicalProofMethod::Induction,
        ProofMethod::Sorry(reason) => CanonicalProofMethod::Sorry(reason.clone()),
        ProofMethod::Finished(result) => {
            let (result, contradiction) = match result {
                FinishedResult::Solved => ("solved", None),
                FinishedResult::Contradictory(c) => {
                    ("contradictory", c.as_ref().map(contradiction_kind))
                }
                FinishedResult::Unfinishable => ("unfinishable", None),
            };
            CanonicalProofMethod::Finished {
                result,
                contradiction,
            }
        }
        ProofMethod::SolveGoal(Goal::Split(id)) => {
            let disj = sys
                .eq_store
                .conj
                .iter()
                .find(|d| d.split_id == *id)
                .unwrap_or_else(|| panic!("split goal {id:?} names no EqDisj in eq_store.conj"));
            CanonicalProofMethod::SolveSplit(canonicalize_eq_disj(disj, labelling))
        }
        ProofMethod::SolveGoal(goal) => CanonicalProofMethod::Solve(
            canonicalize_goal_kind(goal, labelling.theta())
                .expect("only Goal::Split has no goal kind, handled above"),
        ),
        ProofMethod::Invalidated | ProofMethod::RawSolve(_) => {
            panic!("{method:?} is display-only and never a candidate proof method")
        }
    }
}

fn contradiction_kind(c: &Contradiction) -> &'static str {
    match c {
        Contradiction::Cyclic => "cyclic",
        Contradiction::SubtermCyclic => "subterm_cyclic",
        Contradiction::NonNormalTerms => "non_normal_terms",
        Contradiction::ForbiddenExp => "forbidden_exp",
        Contradiction::ForbiddenBP => "forbidden_bp",
        Contradiction::ForbiddenKD => "forbidden_kd",
        Contradiction::ImpossibleChain => "impossible_chain",
        Contradiction::ForbiddenChain => "forbidden_chain",
        Contradiction::ForbiddenACConstrChain => "forbidden_ac_constr_chain",
        Contradiction::NonInjectiveFactInstance(..) => "non_injective_fact_instance",
        Contradiction::IncompatibleEqs => "incompatible_eqs",
        Contradiction::FormulasFalse => "formulas_false",
        Contradiction::SuperfluousLearn(..) => "superfluous_learn",
        Contradiction::NodeAfterLast(..) => "node_after_last",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tamarin_term::alpha_eq_ac::fingerprint_term;
    use tamarin_term::function_symbols::pair_sym;
    use tamarin_term::lterm::{LSort, LVar};
    use tamarin_term::term::{f_app_no_eq, Term};
    use tamarin_term::vterm::var_term;

    use crate::fact::{fresh_fact, in_fact, out_fact, proto_fact, Multiplicity};

    fn v(name: &str, sort: LSort) -> LNTerm {
        var_term(LVar::new(name, sort, 0))
    }

    fn rule(premises: Vec<LNFact>, actions: Vec<LNFact>, conclusions: Vec<LNFact>) -> Rule<()> {
        Rule::new((), premises, conclusions, actions)
    }

    // -- Facts ----------------------------------------------------------

    /// Two `Proto` facts of the same tag, differing only by a renamed
    /// variable, canonize equal — `canonicalize_fact` correctly delegates to
    /// `CAN_alphaeqac` on the lifted term.
    #[test]
    fn same_tag_facts_renamed_vars_are_alpha_eq() {
        let f1 = proto_fact(Multiplicity::Linear, "Login", vec![v("x", LSort::Msg)]);
        let f2 = proto_fact(Multiplicity::Linear, "Login", vec![v("y", LSort::Msg)]);
        assert_eq!(canonicalize_fact(&f1), canonicalize_fact(&f2));
        assert_eq!(
            fingerprint_term(&canonicalize_fact(&f1)),
            fingerprint_term(&canonicalize_fact(&f2))
        );
    }

    /// Two structurally-identical-looking facts with DIFFERENT tags must
    /// never canonize equal — this is what the marker (a function symbol,
    /// never renamed) is for.
    #[test]
    fn different_tag_facts_are_not_alpha_eq() {
        let f1 = proto_fact(Multiplicity::Linear, "Login", vec![v("x", LSort::Msg)]);
        let f2 = proto_fact(Multiplicity::Linear, "Logout", vec![v("x", LSort::Msg)]);
        assert_ne!(canonicalize_fact(&f1), canonicalize_fact(&f2));
    }

    /// Persistent vs. linear facts of the same name/arity are different
    /// tags and so must not canonize equal either.
    #[test]
    fn same_name_different_multiplicity_facts_are_not_alpha_eq() {
        let f1 = proto_fact(Multiplicity::Linear, "St", vec![v("x", LSort::Msg)]);
        let f2 = proto_fact(Multiplicity::Persistent, "St", vec![v("x", LSort::Msg)]);
        assert_ne!(canonicalize_fact(&f1), canonicalize_fact(&f2));
    }

    /// A built-in tag (`Fresh`) must never collide with a user-defined
    /// `Proto` fact whose display name happens to read the same way
    /// (`show_fact_tag` would print both as `"Fr"`/`"!Fr"`-shaped text) —
    /// the marker's variant-prefixed encoding keeps them apart.
    #[test]
    fn builtin_tag_does_not_collide_with_same_named_proto_fact() {
        let builtin = fresh_fact(v("n", LSort::Fresh));
        let user_defined = proto_fact(Multiplicity::Linear, "Fr", vec![v("n", LSort::Fresh)]);
        assert_ne!(
            canonicalize_fact(&builtin),
            canonicalize_fact(&user_defined)
        );
    }

    // -- Rules ------------------------------------------------------------

    /// Two rules with alpha-eq premises/actions/conclusions (renamed
    /// variables throughout) canonize equal.
    #[test]
    fn alpha_eq_rules_canonize_equal() {
        let r1 = rule(
            vec![in_fact(v("x", LSort::Msg))],
            vec![proto_fact(
                Multiplicity::Linear,
                "Recv",
                vec![v("x", LSort::Msg)],
            )],
            vec![out_fact(v("x", LSort::Msg))],
        );
        let r2 = rule(
            vec![in_fact(v("y", LSort::Msg))],
            vec![proto_fact(
                Multiplicity::Linear,
                "Recv",
                vec![v("y", LSort::Msg)],
            )],
            vec![out_fact(v("y", LSort::Msg))],
        );
        assert_eq!(canonicalize_rule(&r1), canonicalize_rule(&r2));
        assert_eq!(
            fingerprint_term(&canonicalize_rule(&r1)),
            fingerprint_term(&canonicalize_rule(&r2))
        );
    }

    /// The whole rule is lifted as ONE term, so a variable shared between a
    /// premise and a conclusion must canonize to the SAME literal in both
    /// places — the propagation regression from `alpha_eq_ac.rs`'s
    /// `wrong_ac_canon_example_propagates_shared_literal`, now at the rule
    /// level: `r1`'s `x` is shared between its `In` premise and first `Out`
    /// conclusion, `r2`'s `y` plays the same shared role at the same
    /// conclusion INDEX (`List` is not AC — see
    /// `same_facts_different_group_assignment_are_not_alpha_eq` — so the
    /// index has to line up, only the local names differ), so `{x -> y}`
    /// witnesses $\alphaeqac$ even though `r1`/`r2` each also mention a
    /// second, unshared variable (`z`/`w`) at the second conclusion index.
    #[test]
    fn shared_variable_across_premise_and_conclusion_propagates() {
        let r1 = rule(
            vec![in_fact(v("x", LSort::Msg))],
            vec![],
            vec![out_fact(v("x", LSort::Msg)), out_fact(v("z", LSort::Msg))],
        );
        let r2 = rule(
            vec![in_fact(v("y", LSort::Msg))],
            vec![],
            vec![out_fact(v("y", LSort::Msg)), out_fact(v("w", LSort::Msg))],
        );
        assert_eq!(canonicalize_rule(&r1), canonicalize_rule(&r2));
        assert_eq!(
            fingerprint_term(&canonicalize_rule(&r1)),
            fingerprint_term(&canonicalize_rule(&r2))
        );
    }

    /// A variable shared between a premise and a conclusion is NOT
    /// $\alphaeqac$ to the same rule with that sharing broken (two distinct
    /// variables instead) — the negative counterpart of the previous test,
    /// exactly `alpha_eq_ac.rs::ac_canon`'s point at the rule level.
    #[test]
    fn breaking_shared_variable_is_not_alpha_eq() {
        let shared = rule(
            vec![in_fact(v("x", LSort::Msg))],
            vec![],
            vec![out_fact(v("x", LSort::Msg))],
        );
        let not_shared = rule(
            vec![in_fact(v("x", LSort::Msg))],
            vec![],
            vec![out_fact(v("y", LSort::Msg))],
        );
        assert_ne!(canonicalize_rule(&shared), canonicalize_rule(&not_shared));
    }

    /// Moving a fact from one group to another (same total facts, same
    /// arguments) must not canonize equal — this is what nesting the three
    /// groups in `rule_to_term`, rather than flattening them, is for.
    #[test]
    fn same_facts_different_group_assignment_are_not_alpha_eq() {
        let f = proto_fact(Multiplicity::Linear, "P", vec![v("x", LSort::Msg)]);
        let as_premise = rule(vec![f.clone()], vec![], vec![]);
        let as_conclusion = rule(vec![], vec![], vec![f]);
        assert_ne!(
            canonicalize_rule(&as_premise),
            canonicalize_rule(&as_conclusion)
        );
    }

    /// `new_vars` does NOT participate in the lifted term: two otherwise-
    /// identical rules that differ only in `new_vars` canonize equal.
    /// `new_vars` is a deterministic function of premises/actions/conclusions
    /// (see the module docs), so once those agree, `new_vars` carries no
    /// distinguishing information; dropping it from `rule_to_term` mirrors
    /// `write_rule_to_key_excl_new_vars` in `constraint/solver/sources.rs`.
    #[test]
    fn new_vars_do_not_affect_canonization() {
        let base = rule(vec![in_fact(v("x", LSort::Msg))], vec![], vec![]);
        let with_new_var = base.clone().with_new_vars(vec![v("n", LSort::Fresh)]);
        assert_eq!(canonicalize_rule(&base), canonicalize_rule(&with_new_var));
        assert_eq!(
            fingerprint_term(&canonicalize_rule(&base)),
            fingerprint_term(&canonicalize_rule(&with_new_var))
        );
    }

    /// Idempotence carries over from `CAN_alphaeqac` (`can1`): canonizing an
    /// already-canonized rule term is a fixed point.
    #[test]
    fn canonicalize_rule_is_idempotent() {
        let r = rule(
            vec![in_fact(v("x", LSort::Msg))],
            vec![proto_fact(
                Multiplicity::Linear,
                "Recv",
                vec![v("x", LSort::Msg)],
            )],
            vec![out_fact(v("x", LSort::Msg))],
        );
        let once = canonicalize_rule(&r);
        let twice = canonicalize_alpha_eq_ac(&once);
        assert_eq!(once, twice);
        assert_eq!(fingerprint_term(&once), fingerprint_term(&twice));
    }

    /// End-to-end sanity check against the real `RuleACInst` type the
    /// constraint system actually stores (`SystemContent::nodes`,
    /// `constraint/system.rs`), not just the `Rule<()>` placeholder used
    /// elsewhere in this module: two instances of "the same" rule — same
    /// `ProtoRuleACInstInfo.name`, alpha-renamed variables, and DIFFERENT
    /// `new_vars` — canonize equal.
    #[test]
    fn rule_ac_inst_renamed_vars_and_new_vars_are_alpha_eq() {
        use crate::rule::{
            ProtoRuleACInstInfo, ProtoRuleName, RuleACInst, RuleAttributes, RuleInfo,
        };

        let info = || {
            RuleInfo::Proto(ProtoRuleACInstInfo {
                name: ProtoRuleName::Stand("Login"),
                attributes: RuleAttributes::empty(),
                loop_breakers: Vec::new(),
            })
        };

        let r1: RuleACInst = Rule::new(
            info(),
            vec![in_fact(v("x", LSort::Msg))],
            vec![out_fact(v("x", LSort::Msg))],
            vec![proto_fact(
                Multiplicity::Linear,
                "Recv",
                vec![v("x", LSort::Msg)],
            )],
        );
        let r2: RuleACInst = Rule::new(
            info(),
            vec![in_fact(v("y", LSort::Msg))],
            vec![out_fact(v("y", LSort::Msg))],
            vec![proto_fact(
                Multiplicity::Linear,
                "Recv",
                vec![v("y", LSort::Msg)],
            )],
        )
        .with_new_vars(vec![v("n", LSort::Fresh)]);

        assert_eq!(canonicalize_rule(&r1), canonicalize_rule(&r2));
        assert_eq!(
            fingerprint_term(&canonicalize_rule(&r1)),
            fingerprint_term(&canonicalize_rule(&r2))
        );
    }

    // -- Guarded formulas ---------------------------------------------------

    use crate::guarded::formula_to_guarded;
    use tamarin_parser::parser::parse_formula_str;

    /// Parses a surface formula string straight to its guarded form —
    /// mirrors `guarded_tests.rs`'s own `g()` helper, panicking (rather
    /// than returning a `Result`) since every call site here is on the
    /// happy path.
    fn g(s: &str) -> Guarded {
        let f = parse_formula_str(s).unwrap_or_else(|e| panic!("parse {s:?}: {e}"));
        formula_to_guarded(&f).unwrap_or_else(|e| panic!("formula_to_guarded {s:?}: {e}"))
    }

    /// A canonical msg variable literal — same naming scheme as the
    /// term-level tests in `tamarin_term::alpha_eq_ac`'s own test suite
    /// (`mv0`, `mv1`, ...): one fixed name `"mv"`, index in `LVar::idx`.
    fn mv(idx: u64) -> LVar {
        LVar::new("mv", LSort::Msg, idx)
    }

    /// A bare free `VarSpec` (`Untagged` sort, idx 0) — matches what the
    /// parser assigns to a plain identifier like `y` in a formula string.
    fn vs(name: &str) -> p::VarSpec {
        p::VarSpec {
            name: name.to_string(),
            idx: 0,
            sort: p::SortHint::Untagged,
            typ: None,
        }
    }

    /// A free node (timepoint) `VarSpec` — matches what the parser assigns
    /// to a bare `#i` occurrence outside any binder.
    fn vs_node(name: &str) -> p::VarSpec {
        p::VarSpec {
            name: name.to_string(),
            idx: 0,
            sort: p::SortHint::Node,
            typ: None,
        }
    }

    /// The `LVar` key/value `theta` uses for `v`.
    fn var_lit(v: &p::VarSpec) -> LVar {
        varspec_to_lvar(v)
    }

    /// An exhaustive IDENTITY `theta` over `vars`: each variable maps to
    /// itself. Point 2 requires `theta` to cover every free variable a
    /// formula mentions (a miss panics), so tests whose free variables don't
    /// need to actually CHANGE still have to give `theta` an entry for each
    /// of them.
    fn identity_theta(vars: &[p::VarSpec]) -> Theta {
        vars.iter().map(|v| (var_lit(v), var_lit(v))).collect()
    }

    // -- 1) work.tex's `ex:canon_guarded`: $z = y \land \exists\ x\ i.\
    //    g(x)@i$, free `y`,`z`, canonical labelling $\theta = \{y \mapsto
    //    mv_1, z \mapsto mv_2\}$. The independent check: build the "expected"
    //    side by parsing the SAME formula with the canonical names spelled
    //    directly (`mv.1 = mv.2 & ...`) and canonizing via an IDENTITY
    //    `theta` (no value changes, but point 2 still requires an entry for
    //    every free variable) — two different code paths (substitute-then-
    //    normalize vs. already-canonical-then-normalize) that must converge.
    //
    //    The result's operand order differs from work.tex's own — the paper
    //    illustrates with an assumed `'∃' < '='` operator order, but
    //    `cmp_guarded`'s real (Atom < GGuarded) order sorts the equality
    //    BEFORE the existential, the opposite way around. See the module
    //    docs for why this module uses the codebase's real order rather
    //    than the paper's illustrative one.
    #[test]
    fn canonicalize_guarded_matches_work_tex_example() {
        let original = g("z = y & Ex x #i. G(x)@i");
        let theta = std::collections::BTreeMap::from([
            (var_lit(&vs("y")), mv(1)),
            (var_lit(&vs("z")), mv(2)),
        ]);
        let got = canonicalize_guarded(&original, &theta);

        // `mv.1`/`mv.2` — dot-suffix index syntax — spells the SAME `LVar`
        // as `mv(1)`/`mv(2)`: one name `"mv"`, index in the `idx` field,
        // matching `alpha_eq_ac.rs`'s actual canonical-naming scheme; a bare
        // `mv1`/`mv2` would instead parse as two unrelated names both at
        // `idx` 0.
        let identity = std::collections::BTreeMap::from([(mv(1), mv(1)), (mv(2), mv(2))]);
        let expected = canonicalize_guarded(&g("mv.1 = mv.2 & Ex x #i. G(x)@i"), &identity);
        assert_eq!(got, expected);
        assert_eq!(fingerprint_guarded(&got), fingerprint_guarded(&expected));

        // Pin the shape explicitly too: the equality atom sorts BEFORE the
        // existential (per `cmp_guarded`'s Atom < GGuarded order), and its
        // operands come out `mv1 = mv2` (smaller-first), matching the
        // string work.tex itself settles on for the equality's own operand
        // order (`mv_1 = mv_2`), just not the top-level conjunct order.
        match &got {
            Guarded::Conj(items) => {
                assert_eq!(items.len(), 2);
                assert!(matches!(&items[0], Guarded::Atom(GAtom::Eq(_, _))));
                assert!(matches!(
                    &items[1],
                    Guarded::GGuarded { qua: Quant::Ex, .. }
                ));
            }
            other => panic!("expected a top-level Conj, got {other:?}"),
        }
    }

    // -- 2) `=` is symmetric: `x = y` and `y = x` canonize equal. ----------
    #[test]
    fn swapped_eq_operands_are_alpha_eq() {
        let theta = identity_theta(&[vs("x"), vs("y")]);
        let f1 = canonicalize_guarded(&g("x = y"), &theta);
        let f2 = canonicalize_guarded(&g("y = x"), &theta);
        assert_eq!(f1, f2);
        assert_eq!(fingerprint_guarded(&f1), fingerprint_guarded(&f2));
    }

    // -- 3) `&` is commutative: swapping a top-level conjunction's operands
    //    canonizes equal. ----------------------------------------------------
    #[test]
    fn reordered_conjunction_operands_are_alpha_eq() {
        let theta = identity_theta(&[vs("x"), vs("y"), vs_node("i"), vs_node("j")]);
        let f1 = canonicalize_guarded(&g("P(x) @ #i & Q(y) @ #j"), &theta);
        let f2 = canonicalize_guarded(&g("Q(y) @ #j & P(x) @ #i"), &theta);
        assert_eq!(f1, f2);
        assert_eq!(fingerprint_guarded(&f1), fingerprint_guarded(&f2));
    }

    // -- 4) Nested conjunctions flatten (via `gconj`, reused unchanged from
    //    `guarded.rs`) before sorting, so `(A & B) & C` and `C & (B & A)`
    //    — same three conjuncts, different nesting AND order — canonize
    //    equal, exactly `CAN_AC`'s flatten-then-sort for AC terms. ----------
    #[test]
    fn nested_conjunction_flattens_and_sorts() {
        let theta = identity_theta(&[
            vs("x"),
            vs("y"),
            vs("z"),
            vs_node("i"),
            vs_node("j"),
            vs_node("k"),
        ]);
        let f1 = canonicalize_guarded(&g("(P(x) @ #i & Q(y) @ #j) & R(z) @ #k"), &theta);
        let f2 = canonicalize_guarded(&g("R(z) @ #k & (Q(y) @ #j & P(x) @ #i)"), &theta);
        assert_eq!(f1, f2);
        assert_eq!(fingerprint_guarded(&f1), fingerprint_guarded(&f2));
    }

    // -- 5) A quantifier's bound-variable display names are purely cosmetic
    //    — only their De-Bruijn POSITION matters — so `Ex x #i. G(x)@i` and
    //    `Ex a #b. G(a)@b` (same structure, differently-named binders)
    //    canonize equal. Without erasing `GBinding::name` in
    //    `ac_normalize_guarded`, this would fail: `Guarded`'s derived
    //    `PartialEq` compares binder names structurally. Both formulas are
    //    fully closed (x/i, a/b are bound), so the empty `theta` is already
    //    exhaustive. -----------------------------------------------------
    #[test]
    fn bound_variable_display_names_do_not_affect_canonization() {
        let empty = std::collections::BTreeMap::new();
        let f1 = canonicalize_guarded(&g("Ex x #i. G(x)@i"), &empty);
        let f2 = canonicalize_guarded(&g("Ex a #b. G(a)@b"), &empty);
        assert_eq!(f1, f2);
        assert_eq!(fingerprint_guarded(&f1), fingerprint_guarded(&f2));
    }

    // -- 5b) Same point, but with TWO co-bound variables in one binder list
    //    (rather than one variable per quantifier): `All x y #i. P(x,y) @
    //    #i` and `All a b #i. P(a,b) @ #i` must canonize equal too — both
    //    binder names change, not just one, and the two variables keep
    //    their relative POSITIONS (`x`/`a` first, `y`/`b` second). --------
    #[test]
    fn multi_variable_binder_display_names_do_not_affect_canonization() {
        let empty = std::collections::BTreeMap::new();
        let f1 = canonicalize_guarded(&g("All x y #i. P(x,y) @ #i ==> F"), &empty);
        let f2 = canonicalize_guarded(&g("All a b #i. P(a,b) @ #i ==> F"), &empty);
        assert_eq!(f1, f2);
        assert_eq!(fingerprint_guarded(&f1), fingerprint_guarded(&f2));
    }

    // -- 5c) Same point again, with NESTED quantifiers (a binder inside
    //    another binder's body), each level renamed independently. --------
    #[test]
    fn nested_quantifier_display_names_do_not_affect_canonization() {
        let empty = std::collections::BTreeMap::new();
        let f1 = canonicalize_guarded(
            &g("All k #i. Setup(k) @ #i ==> Ex j #t. Foo(j) @ #t"),
            &empty,
        );
        let f2 = canonicalize_guarded(
            &g("All p #q. Setup(p) @ #q ==> Ex w #z. Foo(w) @ #z"),
            &empty,
        );
        assert_eq!(f1, f2);
        assert_eq!(fingerprint_guarded(&f1), fingerprint_guarded(&f2));
    }

    // -- 6) Two formulas built from differently-named free variables, each
    //    with a `theta` renaming them to the SAME canonical labelling,
    //    canonize equal — the general form of work.tex's example. ----------
    #[test]
    fn free_variables_are_substituted_via_theta() {
        let theta1 = std::collections::BTreeMap::from([
            (var_lit(&vs("y1")), mv(1)),
            (var_lit(&vs("z1")), mv(2)),
        ]);
        let theta2 = std::collections::BTreeMap::from([
            (var_lit(&vs("y2")), mv(1)),
            (var_lit(&vs("z2")), mv(2)),
        ]);
        let f1 = canonicalize_guarded(&g("z1 = y1 & Ex x #i. G(x)@i"), &theta1);
        let f2 = canonicalize_guarded(&g("z2 = y2 & Ex x #i. G(x)@i"), &theta2);
        assert_eq!(f1, f2);
        assert_eq!(fingerprint_guarded(&f1), fingerprint_guarded(&f2));
    }

    // -- 7) Different fact names are never alpha-eq, substitution or not. --
    #[test]
    fn different_fact_names_are_not_alpha_eq() {
        let theta = identity_theta(&[vs("x"), vs_node("i")]);
        let f1 = canonicalize_guarded(&g("P(x) @ #i"), &theta);
        let f2 = canonicalize_guarded(&g("Q(x) @ #i"), &theta);
        assert_ne!(f1, f2);
    }

    // -- 8) Point 1: name CONSTANTS (`PubLit`/`FreshLit`/`NatLit`) are
    //    never renamed (see `tamarin_term::alpha_eq_ac`'s module doc): they
    //    need no `theta` entry, are left as they are, and two formulas that
    //    differ only in a name stay different. -----------------------------
    #[test]
    fn name_literals_are_kept_fixed() {
        let empty = Theta::new();
        let got = canonicalize_guarded(&g("~'m' = ~'n'"), &empty);
        let expected = canonicalize_guarded(&g("~'n' = ~'m'"), &empty);
        assert_eq!(got, expected, "`=` is still brought into AC normal form");
        match &got {
            Guarded::Atom(GAtom::Eq(GTerm::FreshLit(a), GTerm::FreshLit(b))) => {
                let mut names = [a.as_str(), b.as_str()];
                names.sort();
                assert_eq!(names, ["m", "n"]);
            }
            other => panic!("expected Eq(FreshLit, FreshLit), got {other:?}"),
        }

        let i = vs_node("i");
        let theta = identity_theta(std::slice::from_ref(&i));
        let got = canonicalize_guarded(&g("P('a', %'b') @ #i"), &theta);
        match &got {
            Guarded::Atom(GAtom::Action(f, _)) => {
                assert!(matches!(&f.args[0], GTerm::PubLit(s) if s == "a"));
                assert!(matches!(&f.args[1], GTerm::NatLit(s) if s == "b"));
            }
            other => panic!("expected an Action atom, got {other:?}"),
        }
    }

    // -- 9) Formulas that differ only in a public constant (a message tag)
    //    must NOT canonicalize equal, even though renaming '1' to '2' would
    //    map one onto the other: the theory refers to '1' and '2' by
    //    identity. --------------------------------------------------------
    #[test]
    fn formulas_differing_only_in_a_name_are_not_alpha_eq() {
        let theta = identity_theta(&[vs("x"), vs_node("i")]);
        let f1 = canonicalize_guarded(&g("P(<'1', x>) @ #i"), &theta);
        let f2 = canonicalize_guarded(&g("P(<'2', x>) @ #i"), &theta);
        assert_ne!(f1, f2);
        assert_ne!(fingerprint_guarded(&f1), fingerprint_guarded(&f2));
    }

    // -- 9b) A name that occurs ONLY in a formula (here: under a
    //    quantifier) needs no `theta`
    //    entry. ------------------------------------------------------------
    #[test]
    fn name_only_under_a_quantifier_needs_no_theta_entry() {
        let empty = Theta::new();
        let got = canonicalize_guarded(&g("not (Ex #j. Reveal('c') @ #j)"), &empty);
        let other = canonicalize_guarded(&g("not (Ex #j. Reveal('d') @ #j)"), &empty);
        assert_ne!(got, other);
    }

    // -- A term inside an atom is ALSO brought into `CAN_AC` normal form,
    //    not just the formula-level `&`/`|`/`=`: an AC operator's operands
    //    (`xor` here) are sorted, so `a XOR b = c` and `b XOR a = c`
    //    canonize equal even though `=`'s two SIDES (`xor(..)` vs `c`)
    //    never need swapping (`xor(..)` already sorts after a bare `c` per
    //    `cmp_term`'s Lit-before-FApp class ordering, so this test isolates
    //    the fix from `normalize_atom`'s Eq-side-swap). ---------------------
    #[test]
    fn xor_equality_operands_are_ac_normalized() {
        let theta = identity_theta(&[vs("a"), vs("b"), vs("c")]);
        let f1 = canonicalize_guarded(&g("a XOR b = c"), &theta);
        let f2 = canonicalize_guarded(&g("b XOR a = c"), &theta);
        assert_eq!(f1, f2);
        match &f1 {
            // `cmp_term`'s Lit-class-before-FApp-class ordering
            // (`term_class`) puts the bare variable `c` before the `xor`
            // application, so `=`'s sides settle here, not swapped: this is
            // `normalize_atom`'s (correct) Eq-side ordering, not the AC
            // fix itself — see `xor(a,b)`'s own operand order below for
            // that.
            Guarded::Atom(GAtom::Eq(
                GTerm::Var(BVar::Free(c)),
                GTerm::BinOp(p::BinOp::Xor, x, y),
            )) => {
                assert_eq!(c.name, "c");
                // Sorted by `cmp_term`: both `a`/`b` are `Msg` vars with the
                // same `idx` (0), so the tie-break is the name string.
                assert!(matches!(&**x, GTerm::Var(BVar::Free(v)) if v.name == "a"));
                assert!(matches!(&**y, GTerm::Var(BVar::Free(v)) if v.name == "b"));
            }
            other => panic!("expected Eq(Var(c), BinOp(Xor, _, _)), got {other:?}"),
        }
    }

    // -- The same point, but exercising FLATTENING across different
    //    (re)association, not just a 2-operand swap: `a XOR b XOR c` (left-
    //    associated by the parser: `(a xor b) xor c`) and `c XOR (a XOR b)`
    //    — three leaves, differently nested AND ordered — must flatten to
    //    the same sorted operand list, exactly `CAN_AC`'s flatten-then-sort
    //    for `LNTerm`'s `f_app_ac`. --------------------------------------
    #[test]
    fn xor_chain_flattens_across_reassociation() {
        let theta = identity_theta(&[vs("a"), vs("b"), vs("c"), vs("z")]);
        let f1 = canonicalize_guarded(&g("a XOR b XOR c = z"), &theta);
        let f2 = canonicalize_guarded(&g("c XOR (a XOR b) = z"), &theta);
        assert_eq!(f1, f2);
        match &f1 {
            // Same Lit-before-FApp ordering as the previous test puts the
            // bare `z` first.
            Guarded::Atom(GAtom::Eq(
                GTerm::Var(BVar::Free(z)),
                GTerm::BinOp(p::BinOp::Xor, x, y),
            )) => {
                assert_eq!(z.name, "z");
                // Right-nested, sorted left-to-right: `xor(a, xor(b, c))`.
                assert!(matches!(&**x, GTerm::Var(BVar::Free(v)) if v.name == "a"));
                match &**y {
                    GTerm::BinOp(p::BinOp::Xor, y1, y2) => {
                        assert!(matches!(&**y1, GTerm::Var(BVar::Free(v)) if v.name == "b"));
                        assert!(matches!(&**y2, GTerm::Var(BVar::Free(v)) if v.name == "c"));
                    }
                    other => panic!("expected a nested Xor, got {other:?}"),
                }
            }
            other => panic!("expected Eq(Var(z), BinOp(Xor, _, _)), got {other:?}"),
        }
    }

    // -- The bilinear-pairing C symbol `em` (commutative, not associative,
    //    parsed as a plain `GTerm::App("em", [_, _])`) gets the same
    //    argument-sorting treatment, minus the flattening. ------------------
    #[test]
    fn em_arguments_are_sorted() {
        let theta = identity_theta(&[vs("a"), vs("b"), vs_node("i")]);
        let f1 = canonicalize_guarded(&g("P(em(a,b)) @ #i"), &theta);
        let f2 = canonicalize_guarded(&g("P(em(b,a)) @ #i"), &theta);
        assert_eq!(f1, f2);
        match &f1 {
            Guarded::Atom(GAtom::Action(fact, _)) => match &fact.args[..] {
                [GTerm::App(n, args)] if &**n == "em" => {
                    assert!(
                        matches!(&args[0], GTerm::Var(v) if matches!(v, BVar::Free(v) if v.name == "a"))
                    );
                    assert!(
                        matches!(&args[1], GTerm::Var(v) if matches!(v, BVar::Free(v) if v.name == "b"))
                    );
                }
                other => panic!("expected [App(\"em\", [_, _])], got {other:?}"),
            },
            other => panic!("expected an Action atom, got {other:?}"),
        }
    }

    // -- 10) Point 2: a free variable with NO entry in `theta` is a caller
    //    bug, not a case to fall back on silently — canonization panics. --
    #[test]
    #[should_panic(expected = "not covered by the canonical renaming")]
    fn missing_free_variable_in_theta_panics() {
        let empty = std::collections::BTreeMap::new();
        let _ = canonicalize_guarded(&g("x = y"), &empty);
    }

    // -- 11) Point 2, continued: same for a missing free NODE variable. ----
    #[test]
    #[should_panic(expected = "not covered by the canonical renaming")]
    fn missing_node_variable_in_theta_panics() {
        let empty = Theta::new();
        let _ = canonicalize_guarded(&g("P('a') @ #i"), &empty);
    }

    // -- 12) Idempotence: canonizing an already-canonical formula (with an
    //    identity `theta` — no further substitution to actually do, but
    //    point 2 still requires `mv(1)`/`mv(2)` to be covered) is a fixed
    //    point. -----------------------------------------------------------
    #[test]
    fn canonicalize_guarded_is_idempotent() {
        let theta = std::collections::BTreeMap::from([
            (var_lit(&vs("y")), mv(1)),
            (var_lit(&vs("z")), mv(2)),
        ]);
        let once = canonicalize_guarded(&g("z = y & Ex x #i. G(x)@i"), &theta);
        let identity = std::collections::BTreeMap::from([(mv(1), mv(1)), (mv(2), mv(2))]);
        let twice = canonicalize_guarded(&once, &identity);
        assert_eq!(once, twice);
        assert_eq!(fingerprint_guarded(&once), fingerprint_guarded(&twice));
    }

    // -- Rule instances --

    fn rule_ac_inst(
        name: &'static str,
        premises: Vec<LNFact>,
        actions: Vec<LNFact>,
        conclusions: Vec<LNFact>,
    ) -> crate::rule::RuleACInst {
        use crate::rule::{ProtoRuleACInstInfo, ProtoRuleName, RuleAttributes, RuleInfo};
        Rule::new(
            RuleInfo::Proto(ProtoRuleACInstInfo {
                name: ProtoRuleName::Stand(name),
                attributes: RuleAttributes::empty(),
                loop_breakers: Vec::new(),
            }),
            premises,
            conclusions,
            actions,
        )
    }

    fn node(idx: u64) -> crate::constraint::constraints::NodeId {
        LVar::new("i", LSort::Node, idx)
    }

    // -- Test helpers --

    use crate::bliss_proc::bliss_available;

    /// A ground 0-ary NoEq function symbol term, e.g. for `name = "aaa"`
    /// a term that is never renamed by `CAN_alphaeqac` (unlike a name/var
    /// literal) and orders by NAME.
    fn zero_ary_fun_term(name: &'static str) -> LNTerm {
        let sym = NoEqSym::new(
            name.as_bytes().to_vec(),
            0,
            Privacy::Public,
            Constructability::Constructor,
        );
        LNTerm::App(FunSym::NoEq(sym), Arc::from([]))
    }

    // -- eq_store.conj -----------------------------------------------------

    /// A `CanonLabelling` whose theta already covers `domain_vars`, each
    /// mapped to its own distinct canonical var of the same sort -- enough
    /// to satisfy `lookup_theta`'s panic-on-miss for a domain-key lookup,
    /// without canonicalizing a whole system first.
    /// Uses the SAME per-sort naming as the real scheme (`canonical_var` in
    /// `tamarin-term/src/alpha_eq_ac.rs`: `mv`/`fv`/`pv`/`tv`/`nv` for
    /// msg/fresh/pub/node/nat) rather than one sort-blind `cv` family, so a
    /// reader of these tests sees the same convention `CanonLabelling`
    /// itself produces -- NOT the real `CanonLabelling` scheme otherwise
    /// (this hands out indices purely by `domain_vars`' own order, one
    /// counter per sort, rather than by any real Canonizer discovery
    /// order). Canonical keys are handed out in `domain_vars`' own order
    /// within each sort (index 0, 1, ...), so a test can predict which
    /// output entry belongs to which input domain var.
    fn labelling_covering(domain_vars: &[LVar]) -> CanonLabelling {
        let mut theta: Theta = std::collections::BTreeMap::new();
        let mut next_idx: std::collections::BTreeMap<LSort, u64> = std::collections::BTreeMap::new();
        for dv in domain_vars {
            let idx = next_idx.entry(dv.sort).or_insert(0);
            let name = match dv.sort {
                LSort::Msg => "mv",
                LSort::Fresh => "fv",
                LSort::Pub => "pv",
                LSort::Node => "tv",
                LSort::Nat => "nv",
            };
            theta.insert(*dv, LVar::new(name, dv.sort, *idx));
            *idx += 1;
        }
        CanonLabelling::from_theta(theta)
    }

    #[test]
    fn eq_disj_alternative_with_no_domain_keys_is_empty() {
        let alt: LNSubstVFresh = LNSubstVFresh::from_list(Vec::<(LVar, LNTerm)>::new());
        let labelling = CanonLabelling::empty();
        assert_eq!(
            canonicalize_eq_disj_alternative(&alt, &labelling),
            Vec::new()
        );
    }

    #[test]
    fn eq_disj_alternative_domain_via_theta_range_discovered_fresh() {
        let x = LVar::new("x", LSort::Msg, 7);
        let w = LVar::new("w", LSort::Msg, 900); // an undiscovered witness
        let alt: LNSubstVFresh = LNSubstVFresh::from_list(vec![(x, var_term(w))]);
        let labelling = labelling_covering(&[x]);

        let out = canonicalize_eq_disj_alternative(&alt, &labelling);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, LVar::new("mv", LSort::Msg, 0));
        // The witness got SOME fresh canonical literal of its own -- not
        // `w` itself (which was never a canonical form to begin with),
        // and not x's own canonical key.
        assert_ne!(out[0].1, var_term(w));
        assert_ne!(out[0].1, var_term(out[0].0));
    }

    /// The core correctness property: two alternatives with the SAME
    /// logical shape but DIFFERENT raw witness identities (mirroring
    /// Maude's arbitrary per-call witness numbering) canonize to the
    /// IDENTICAL result.
    #[test]
    fn eq_disj_alternative_is_witness_numbering_invariant() {
        let x = LVar::new("x", LSort::Msg, 7);
        let alt_a: LNSubstVFresh = LNSubstVFresh::from_list(vec![(
            x,
            f_app_no_eq(
                pair_sym(),
                vec![
                    var_term(LVar::new("w", LSort::Msg, 100)),
                    var_term(LVar::new("w", LSort::Msg, 101)),
                ],
            ),
        )]);
        let alt_b: LNSubstVFresh = LNSubstVFresh::from_list(vec![(
            x,
            f_app_no_eq(
                pair_sym(),
                vec![
                    var_term(LVar::new("z", LSort::Msg, 500)),
                    var_term(LVar::new("z", LSort::Msg, 501)),
                ],
            ),
        )]);

        let labelling_a = labelling_covering(&[x]);
        let labelling_b = labelling_covering(&[x]);

        let out_a = canonicalize_eq_disj_alternative(&alt_a, &labelling_a);
        let out_b = canonicalize_eq_disj_alternative(&alt_b, &labelling_b);

        assert_eq!(out_a, out_b);
    }

    /// A witness appearing in TWO of an alternative's own range terms (not
    /// just twice within one) must canonize to the SAME canonical literal
    /// both times -- confirming that canonizing all range terms in one
    /// `Canonizer` call shares identity across them.
    #[test]
    fn eq_disj_alternative_shares_a_witness_across_two_of_its_own_range_terms() {
        let p = LVar::new("p", LSort::Msg, 7);
        let q = LVar::new("q", LSort::Msg, 8);
        let w = LVar::new("w", LSort::Msg, 900);
        let alt: LNSubstVFresh = LNSubstVFresh::from_list(vec![
            (p, f_app_no_eq(pair_sym(), vec![var_term(w), var_term(w)])),
            (q, var_term(w)),
        ]);
        let labelling = labelling_covering(&[p, q]);

        let out = canonicalize_eq_disj_alternative(&alt, &labelling);

        assert_eq!(out.len(), 2);
        let p_key = LVar::new("mv", LSort::Msg, 0);
        let q_key = LVar::new("mv", LSort::Msg, 1);
        let p_range = &out.iter().find(|(k, _)| *k == p_key).expect("p entry").1;
        let q_range = &out.iter().find(|(k, _)| *k == q_key).expect("q entry").1;
        match p_range {
            Term::App(_, args) => {
                assert_eq!(args.len(), 2);
                // Both occurrences of `w` WITHIN p's own pair(..) term
                // share one canonical literal.
                assert_eq!(args[0], args[1]);
                // AND q's lone range term (just `w`) matches that SAME
                // literal -- the shared identity crosses entries, not
                // just occurrences within one.
                assert_eq!(&args[0], q_range);
            }
            other => panic!("expected a pair(..) term, got {other:?}"),
        }
    }

    /// `w1 ⊕ w2` alone cannot tell its two witnesses apart; the second range
    /// term, `pair(w, w)` for one of them, can. The four alternatives below
    /// are equal up to renaming their local witnesses, and their keys get the
    /// same canonical names, so they must canonize identically whichever
    /// witness `q`'s range uses and whatever the keys' raw indices -- the tie
    /// in `w1 ⊕ w2` must be settled by the other range term, not by raw
    /// variable order.
    #[test]
    fn eq_disj_alternative_resolves_a_tie_across_its_range_terms_by_content() {
        use tamarin_term::builtin::xor;
        let w1 = LVar::new("w", LSort::Msg, 900);
        let w2 = LVar::new("w", LSort::Msg, 901);
        let mut outputs = Vec::new();
        for (p_idx, q_idx) in [(7, 8), (8, 7)] {
            let p = LVar::new("p", LSort::Msg, p_idx);
            let q = LVar::new("q", LSort::Msg, q_idx);
            let labelling = labelling_covering(&[p, q]);
            for w in [w1, w2] {
                let alt = LNSubstVFresh::from_list(vec![
                    (p, xor(var_term(w1), var_term(w2))),
                    (q, f_app_no_eq(pair_sym(), vec![var_term(w), var_term(w)])),
                ]);
                outputs.push(canonicalize_eq_disj_alternative(&alt, &labelling));
            }
        }
        assert!(
            outputs.windows(2).all(|pair| pair[0] == pair[1]),
            "equivalent alternatives canonized differently: {outputs:#?}"
        );
    }

    /// Regression test for the exact collision risk `canonicalize_eq_disj_alternative`'s
    /// own doc comment describes: a range variable whose raw `LVar`
    /// happens to coincide with an UNRELATED real variable already
    /// covered by `theta`. `SubstVFresh`'s "range vars are fresh" is only
    /// an interpretive convention (not a data-level guarantee), so this
    /// raw-identity collision must not cause the witness to be silently
    /// treated as the real variable it happens to share an `LVar` with.
    #[test]
    fn eq_disj_alternative_range_witness_does_not_alias_an_unrelated_real_variable() {
        let x = LVar::new("x", LSort::Msg, 7);
        let y = LVar::new("y", LSort::Msg, 42);
        // `w` deliberately reuses `y`'s exact raw identity.
        let w = y;
        let alt: LNSubstVFresh = LNSubstVFresh::from_list(vec![(x, var_term(w))]);
        let labelling = labelling_covering(&[x, y]);
        let y_canonical = *labelling
            .theta()
            .get(&y)
            .expect("labelling_covering covers y");

        let out = canonicalize_eq_disj_alternative(&alt, &labelling);

        assert_eq!(out.len(), 1);
        assert_ne!(
            out[0].1,
            var_term(y_canonical),
            "the witness (raw-identical to `y`) must get its OWN canonical \
             identity, not `y`'s -- aliasing them would silently fuse an \
             existentially-local witness with an unrelated real variable"
        );
    }

    /// Reviewed corner case, found IMPOSSIBLE under normal operation: two
    /// distinct domain keys of one alternative canonicalizing to the SAME
    /// literal, with DIFFERENT range terms -- which would make the
    /// domain-key sort's tie-break (and thus which range term ends up
    /// associated with that canonical position) depend on incidental raw
    /// `LVar` order rather than content. `theta` is injective by
    /// construction (every `Canonizer`-discovered literal gets a
    /// brand-new canonical index -- see `canonicalize_eq_disj_alternative`'s
    /// own doc comment), so this can't happen via the real accumulation
    /// path; every OTHER test in this file builds its labelling that way.
    /// This test instead hand-constructs a `theta` that violates
    /// injectivity directly (via `CanonLabelling::from_theta`, bypassing
    /// the `Canonizer` entirely -- something production code never does),
    /// to confirm the function fails LOUDLY rather than silently picking
    /// an arbitrary, non-deterministic order if that invariant is ever
    /// broken some other way in the future.
    #[test]
    #[should_panic(expected = "theta is no longer injective")]
    fn eq_disj_alternative_panics_if_theta_is_not_injective() {
        let v1 = LVar::new("v1", LSort::Msg, 1);
        let v2 = LVar::new("v2", LSort::Msg, 2);
        let shared_canonical = LVar::new("mv", LSort::Msg, 0);
        let mut theta: Theta = std::collections::BTreeMap::new();
        // Deliberately broken: two DIFFERENT raw domain vars mapped to the
        // SAME canonical literal -- impossible via real accumulation.
        theta.insert(v1, shared_canonical);
        theta.insert(v2, shared_canonical);
        let labelling = CanonLabelling::from_theta(theta);

        let alt: LNSubstVFresh = LNSubstVFresh::from_list(vec![
            (v1, var_term(LVar::new("w", LSort::Msg, 100))),
            (v2, var_term(LVar::new("w", LSort::Msg, 200))),
        ]);

        canonicalize_eq_disj_alternative(&alt, &labelling);
    }

    #[test]
    fn eq_disj_preserves_multiplicity_of_alpha_equivalent_alternatives() {
        let x = LVar::new("x", LSort::Msg, 7);
        let alt_a: LNSubstVFresh =
            LNSubstVFresh::from_list(vec![(x, var_term(LVar::new("w", LSort::Msg, 100)))]);
        let alt_b: LNSubstVFresh =
            LNSubstVFresh::from_list(vec![(x, var_term(LVar::new("w", LSort::Msg, 200)))]);
        let disj = EqDisj {
            split_id: crate::tools::equation_store::SplitId(0),
            substs: vec![alt_a, alt_b],
        };
        let labelling = labelling_covering(&[x]);

        let out = canonicalize_eq_disj(&disj, &labelling);

        assert_eq!(
            out.len(),
            2,
            "two alpha-equivalent alternatives must NOT be deduplicated -- \
             Tamarin's own solver keeps them distinct on purpose (see \
             tools/equation_store.rs's own comment on its \
             applyBound/S.fromList handling)"
        );
        assert_eq!(
            out[0], out[1],
            "and their canonical forms must actually be equal"
        );
    }

    // -- Goals (`canonicalize_goals`) ------------------------------------

    #[test]
    fn canonicalize_goals_drops_split_goals() {
        let goals = vec![(
            Goal::Split(crate::tools::equation_store::SplitId(0)),
            GoalStatus::default(),
        )];
        let out = canonicalize_goals(&goals, &std::collections::BTreeMap::new());
        assert!(
            out.is_empty(),
            "Split only names an EqDisj already in eq_store.conj and must not \
             appear in the canonical goal list: {out:?}"
        );
    }

    #[test]
    fn canonicalize_goals_keeps_action_goals_with_their_solved_status() {
        let nid = node(0);
        let x = LVar::new("x", LSort::Msg, 0);
        let labelling = labelling_covering(&[nid, x]);
        let fact = proto_fact(Multiplicity::Linear, "P", vec![v("x", LSort::Msg)]);
        let goal = |solved| {
            vec![(
                Goal::Action(nid, fact.clone()),
                GoalStatus {
                    solved,
                    ..Default::default()
                },
            )]
        };

        let open = canonicalize_goals(&goal(false), labelling.theta());
        let solved = canonicalize_goals(&goal(true), labelling.theta());

        let expected_kind = CanonicalGoalKind::Action(
            LVar::new("tv", LSort::Node, 0),
            fact_to_term(&apply_renaming(labelling.theta(), fact.clone())),
        );
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].kind, expected_kind);
        assert!(!open[0].solved);
        assert_eq!(solved[0].kind, expected_kind);
        assert!(solved[0].solved);
        assert_ne!(open, solved, "solved-ness of an action goal must survive canonicalization");
    }

    /// System-level regression: an action goal's `solved` flag decides
    /// whether `candidate_methods` offers it, so two systems differing only
    /// in it are different proof-search states.
    #[test]
    fn systems_differing_only_in_an_action_goals_solved_status_are_not_alpha_eq() {
        if !bliss_available() {
            return;
        }
        let with_action_goal = |solved| {
            let mut sys = System::empty();
            sys.content_mut().goals = std::sync::Arc::new(vec![(
                Goal::Action(node(0), crate::fact::ku_fact(zero_ary_fun_term("m"))),
                GoalStatus {
                    solved,
                    ..Default::default()
                },
            )]);
            canonicalize_constraint_system(&sys)
                .unwrap_or_else(|e| panic!("canonicalize (solved={solved}): {e:?}"))
        };

        let open = with_action_goal(false);
        let solved = with_action_goal(true);

        assert_eq!(
            open.graph_part, solved.graph_part,
            "neither system has a node"
        );
        assert_ne!(
            open, solved,
            "an open and a solved action goal must not canonicalize identically"
        );
    }

    /// System-level regression for names being constants: `!KU(<'1', x>)`
    /// and `!KU(<'2', x>)` unify with different rule conclusions, so two
    /// systems differing only in the tag must not canonicalize (or
    /// fingerprint) identically -- while renaming `x` still merges.
    #[test]
    fn systems_differing_only_in_a_message_tag_are_not_alpha_eq() {
        if !bliss_available() {
            return;
        }
        use crate::canon_fingerprint::fingerprint_constraint_system;
        use tamarin_term::builtin::pair;
        use tamarin_term::lterm::pub_term;

        let with_ku_goal = |tag: &str, var: &str| {
            let mut sys = System::empty();
            sys.content_mut().goals = std::sync::Arc::new(vec![(
                Goal::Action(node(0), crate::fact::ku_fact(pair(pub_term(tag), v(var, LSort::Msg)))),
                GoalStatus::default(),
            )]);
            canonicalize_constraint_system(&sys)
                .unwrap_or_else(|e| panic!("canonicalize ('{tag}', {var}): {e:?}"))
        };

        let one_x = with_ku_goal("1", "x");
        let two_x = with_ku_goal("2", "x");
        let one_y = with_ku_goal("1", "y");

        assert_ne!(one_x, two_x, "'1' and '2' are different constants");
        assert_ne!(
            fingerprint_constraint_system(&one_x),
            fingerprint_constraint_system(&two_x)
        );
        assert_eq!(one_x, one_y, "renaming the variable still merges");
    }

    /// Two rules with the same facts but different names (SAPIC generates
    /// such pairs, e.g. `channels1`'s `incx_1_121`/`incx_3_121`): systems
    /// differing only in which one a node instantiates are different
    /// proof-search states. Renaming the variable still merges.
    #[test]
    fn systems_differing_only_in_a_rule_name_are_not_alpha_eq() {
        if !bliss_available() {
            return;
        }
        let with_rule = |name: &'static str, var: &str| {
            let mut sys = System::empty();
            let fact = |f: &str| proto_fact(Multiplicity::Linear, f, vec![v(var, LSort::Msg)]);
            sys.add_node(
                node(0),
                rule_ac_inst(
                    name,
                    vec![in_fact(v(var, LSort::Msg))],
                    vec![],
                    vec![fact("M")],
                ),
            );
            canonicalize_constraint_system(&sys)
                .unwrap_or_else(|e| panic!("canonicalize ({name}, {var}): {e:?}"))
        };
        assert_ne!(with_rule("A1", "x"), with_rule("A2", "x"));
        assert_eq!(with_rule("A1", "x"), with_rule("A1", "y"));
    }

    #[test]
    fn canonicalize_goals_canonicalizes_chain_via_theta() {
        let conc_nid = node(0);
        let prem_nid = node(1);
        let labelling = labelling_covering(&[conc_nid, prem_nid]);
        let goals = vec![(
            Goal::Chain((conc_nid, ConcIdx(0)), (prem_nid, PremIdx(1))),
            GoalStatus {
                solved: true,
                ..Default::default()
            },
        )];

        let out = canonicalize_goals(&goals, labelling.theta());

        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].kind,
            CanonicalGoalKind::Chain(
                LVar::new("tv", LSort::Node, 0),
                ConcIdx(0),
                LVar::new("tv", LSort::Node, 1),
                PremIdx(1),
            )
        );
        assert!(out[0].solved, "GoalStatus::solved must survive canonicalization");
    }

    #[test]
    fn canonicalize_goals_canonicalizes_premise_fact_via_theta() {
        let prem_nid = node(0);
        let x = LVar::new("x", LSort::Msg, 0);
        let labelling = labelling_covering(&[prem_nid, x]);
        let fact = in_fact(v("x", LSort::Msg));
        let goals = vec![(
            Goal::Premise((prem_nid, PremIdx(2)), fact.clone()),
            GoalStatus::default(),
        )];

        let out = canonicalize_goals(&goals, labelling.theta());

        assert_eq!(out.len(), 1);
        let expected_fact_term = fact_to_term(&apply_renaming(labelling.theta(), fact.clone()));
        assert_eq!(
            out[0].kind,
            CanonicalGoalKind::Premise(
                LVar::new("tv", LSort::Node, 0),
                PremIdx(2),
                expected_fact_term,
            )
        );
    }

    #[test]
    fn canonicalize_goals_canonicalizes_disj_and_preserves_alternative_multiplicity() {
        let x = LVar::new("x", LSort::Msg, 0);
        let labelling = labelling_covering(&[x]);
        // Two IDENTICAL alternatives -- must NOT collapse to one (REVISED
        // 2026-09-23: `sort_guarded` no longer dedups, same reasoning as
        // `eq_store.conj`'s multiset preservation -- a solver-level
        // duplicate is not assumed redundant).
        let goals = vec![(
            Goal::Disj(crate::constraint::constraints::Disj(vec![g("Foo(x)"), g("Foo(x)")])),
            GoalStatus::default(),
        )];

        let out = canonicalize_goals(&goals, labelling.theta());

        assert_eq!(out.len(), 1);
        match &out[0].kind {
            CanonicalGoalKind::Disj(alts) => {
                assert_eq!(
                    alts.len(),
                    2,
                    "two identical alternatives must NOT be deduplicated -- multiplicity \
                     must survive canonicalization"
                );
                assert_eq!(alts[0], canonicalize_guarded(&g("Foo(x)"), labelling.theta()));
                assert_eq!(alts[1], canonicalize_guarded(&g("Foo(x)"), labelling.theta()));
            }
            other => panic!("expected Disj, got {other:?}"),
        }
    }

    #[test]
    fn canonicalize_goals_canonicalizes_subterm_via_theta() {
        let x = LVar::new("x", LSort::Msg, 0);
        // `v(name, sort)` always builds idx 0 -- `y` must match that, or
        // `labelling_covering` seeds a DIFFERENT literal than the one
        // `v("y", ..)` below actually produces.
        let y = LVar::new("y", LSort::Msg, 0);
        let labelling = labelling_covering(&[x, y]);
        let goals = vec![(
            Goal::Subterm((v("x", LSort::Msg), v("y", LSort::Msg))),
            GoalStatus::default(),
        )];

        let out = canonicalize_goals(&goals, labelling.theta());

        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].kind,
            CanonicalGoalKind::Subterm(
                var_term(LVar::new("mv", LSort::Msg, 0)),
                var_term(LVar::new("mv", LSort::Msg, 1)),
            )
        );
    }

    /// The exhaustiveness contract every other field already has
    /// (`rename_var`/`apply_renaming` panic on a miss, not silently pass
    /// through -- see `rename_var`'s own doc comment in `tamarin-term`)
    /// applies to goals too: a `Chain` goal
    /// referencing a `NodeId` missing from `theta` must panic
    /// loudly, not silently drop the reference.
    #[test]
    #[should_panic(expected = "not covered by the canonical renaming")]
    fn canonicalize_goals_panics_on_a_node_id_missing_from_theta() {
        let uncovered_nid = node(99);
        let goals = vec![(
            Goal::Chain((uncovered_nid, ConcIdx(0)), (uncovered_nid, PremIdx(0))),
            GoalStatus::default(),
        )];
        let empty_theta = std::collections::BTreeMap::new();
        let _ = canonicalize_goals(&goals, &empty_theta);
    }

    /// THE regression test this whole feature exists for, at the
    /// `canonicalize_constraint_system` level (not just `canonicalize_goals`
    /// in isolation): a system with a genuine pending goal must NOT
    /// canonicalize identically to an otherwise-identical system without
    /// it. Confirmed via real wireguard.spthy data before this fix: two
    /// sibling `!KU(mac1)`/`!KU(mac2)` sub-goals (synthesized by
    /// `insert_goal_with_loop_flag`'s pair-decomposition, `reduction.rs`)
    /// were completely invisible to canonicalization -- their `NodeId`s
    /// fell back to bare content-free `Dummy` vertices, and their own
    /// term content (which SPECIFIC message was pending derivation)
    /// appeared NOWHERE in the canonical form at all. Uses an otherwise
    /// EMPTY graph part (ground `zero_ary_fun_term`s for the `Subterm`
    /// pair, so no literal needs `theta` coverage) specifically so this
    /// test needs no bliss subprocess -- matches
    /// `empty_graph_part_systems_with_different_formulas_are_not_alpha_eq`'s
    /// own precedent for the same reason.
    #[test]
    fn systems_differing_only_in_a_pending_goal_are_not_alpha_eq() {

        let mut sys_a = System::empty();
        sys_a.content_mut().goals = std::sync::Arc::new(vec![(
            Goal::Subterm((zero_ary_fun_term("small"), zero_ary_fun_term("big"))),
            GoalStatus::default(),
        )]);
        let sys_b = System::empty();

        let canon_a = canonicalize_constraint_system(&sys_a)
            .unwrap_or_else(|e| panic!("canonicalize a: {e:?}"));
        let canon_b = canonicalize_constraint_system(&sys_b)
            .unwrap_or_else(|e| panic!("canonicalize b: {e:?}"));

        assert_ne!(
            canon_a, canon_b,
            "a system with a genuine pending Subterm goal must not canonicalize \
             identically to one without it"
        );
    }

    // -- canonicalize_proof_method --

    /// The same goal under two different raw namings, each renamed through
    /// its own system's labelling, must give the same canonical method.
    #[test]
    fn alpha_renamed_solve_goals_canonicalize_equally() {
        let solve = |nid: LVar, var: &str| {
            let x = LVar::new(var, LSort::Msg, 0);
            let method = ProofMethod::SolveGoal(Goal::Premise(
                (nid, PremIdx(0)),
                in_fact(v(var, LSort::Msg)),
            ));
            canonicalize_proof_method(&method, &System::empty(), &labelling_covering(&[nid, x]))
        };
        assert_eq!(solve(node(0), "x"), solve(node(5), "y"));
        assert_ne!(
            solve(node(0), "x"),
            canonicalize_proof_method(
                &ProofMethod::Simplify,
                &System::empty(),
                &CanonLabelling::empty()
            )
        );
    }

    /// A split goal is identified by its `EqDisj`'s canonicalized content,
    /// not by its raw split id (which the canonical form drops).
    #[test]
    fn split_methods_canonicalize_by_their_disjunctions_content() {
        use crate::tools::equation_store::SplitId;
        let x = LVar::new("x", LSort::Msg, 7);
        let solve_split = |id: i64, witness_idx: u64| {
            let mut sys = System::empty();
            sys.eq_store_mut().conj.push(EqDisj {
                split_id: SplitId(id),
                substs: vec![LNSubstVFresh::from_list(vec![(
                    x,
                    var_term(LVar::new("w", LSort::Msg, witness_idx)),
                )])],
            });
            canonicalize_proof_method(
                &ProofMethod::SolveGoal(Goal::Split(SplitId(id))),
                &sys,
                &labelling_covering(&[x]),
            )
        };
        let a = solve_split(0, 100);
        assert!(matches!(a, CanonicalProofMethod::SolveSplit(_)));
        assert_eq!(a, solve_split(3, 500));
    }

    #[test]
    #[should_panic(expected = "names no EqDisj")]
    fn a_split_method_naming_no_disjunction_panics() {
        let _ = canonicalize_proof_method(
            &ProofMethod::SolveGoal(Goal::Split(crate::tools::equation_store::SplitId(9))),
            &System::empty(),
            &CanonLabelling::empty(),
        );
    }

    /// Which contradiction instance `is_finished` reports first depends on
    /// raw iteration order, so only its kind may count.
    #[test]
    fn finished_methods_ignore_the_contradiction_payload() {
        let finished = |c: Contradiction| {
            canonicalize_proof_method(
                &ProofMethod::Finished(FinishedResult::Contradictory(Some(c))),
                &System::empty(),
                &CanonLabelling::empty(),
            )
        };
        assert_eq!(
            finished(Contradiction::NodeAfterLast(node(0), node(1))),
            finished(Contradiction::NodeAfterLast(node(7), node(3)))
        );
        assert_ne!(
            finished(Contradiction::NodeAfterLast(node(0), node(1))),
            finished(Contradiction::Cyclic)
        );
    }

    // -- canonicalize_constraint_system: systems without variables --

    /// A system without free variables (the root of every proof, before
    /// any rule instance; induction's `empty_trace` base case) is its own
    /// canonical form: it canonicalizes without bliss, which refuses an
    /// empty graph. Deliberately does NOT gate on `bliss_available()`.
    #[test]
    fn empty_graph_part_canonicalizes_without_bliss() {
        use crate::guarded::gtrue;
        let mut sys = System::empty();
        sys.formulas_mut().push(std::sync::Arc::new(gtrue()));

        let canon = canonicalize_constraint_system(&sys)
            .unwrap_or_else(|e| panic!("canonicalize_constraint_system: {e:?}"));

        assert_eq!(canon.formulas, vec![gtrue()]);
    }

    /// Two DIFFERENT empty-graph-part systems (different formulas) must
    /// still canonicalize to DIFFERENT `CanonicalSystem`s -- skipping bliss
    /// must not collapse everything with no nodes into one bucket.
    #[test]
    fn empty_graph_part_systems_with_different_formulas_are_not_alpha_eq() {
        use crate::guarded::{gfalse, gtrue};

        let mut sys_a = System::empty();
        sys_a.formulas_mut().push(std::sync::Arc::new(gtrue()));
        let canon_a = canonicalize_constraint_system(&sys_a).expect("canonicalize a");

        let mut sys_b = System::empty();
        sys_b.formulas_mut().push(std::sync::Arc::new(gfalse()));
        let canon_b = canonicalize_constraint_system(&sys_b).expect("canonicalize b");

        assert_ne!(canon_a, canon_b);
    }
}
