// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! JSON API for the TamRL reinforcement-learning client (feature `rl`).
//!
//! (No HS analog upstream.)  A port of the endpoints the TamRL fork of the
//! Haskell server adds (`GET /` as JSON, `initialcs`, `executemethod`,
//! `checkproof` in its `Web/Handler.hs`), served under `/api/v1` next to the
//! GUI instead of replacing it:
//!
//! ```text
//! GET  /api/v1/theories                                         fork: GET /
//! GET  /api/v1/theories/<idx>/lemmas/<lemma>/initial-system     fork: initialcs
//! POST /api/v1/theories/<idx>/lemmas/<lemma>/execute            fork: executemethod
//! POST /api/v1/theories/<idx>/lemmas/<lemma>/check-proof        fork: checkproof
//! ```
//!
//! Bodies are the fork's: a system is `{proofMethods, encodedMethods,
//! encodedSys, status}` (status `"Undertermined"` spelled as the fork spells
//! it); `execute` takes the form fields `proofmethod` and `system` and
//! answers `[{caseName, system}]` sorted by case name; `check-proof` takes
//! the form field `proof` (`{action, children}`, `action` a method token,
//! `"solved"`, `"contradictory"` or `null`) and answers `{proofStatus,
//! proofSize, proofFile}`.  Bad input is a 400 with the fork's message, an
//! unknown theory a 404.  Tokens come from
//! [`tamarin_theory::system_codec`]; they are only valid for the build
//! that wrote them.
//!
//! The API serves the theories loaded at startup, under the indices the GUI
//! gave them, and keeps its own state per theory: one prover session, a
//! pool of Maude processes, and per lemma a context with saturated sources
//! plus the start system's response, each built on first use.  Every request
//! runs on a blocking thread and does its proof work on a Maude process of
//! its own, with a private fresh-variable counter, so parallel requests do
//! not wait on each other (the fork's server runs them in parallel too).
//! SAPIC theories are listed but their lemma routes answer 400: the codec
//! does not carry the SAPIC process of a rule.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Form, Json, Router};
use parking_lot::Mutex;
use serde_json::{json, Map, Value};

use tamarin_term::lterm::{LSort, LVar};
use tamarin_term::maude_proc::{MaudeHandle, MaudePool, SharedMaudeCaches};
use tamarin_term::pretty::pretty_lnterm_styled;
use tamarin_theory::check_proof::{check_proof, MethodTree};
use tamarin_theory::constraint::solver::context::{IntrRuleCache, ProofContext};
use tamarin_theory::constraint::solver::contradictions::Contradiction;
use tamarin_theory::constraint::solver::proof_method::{
    check_and_exec_proof_method, ProofMethod, Result as MethodResult,
};
use tamarin_theory::constraint::solver::search::{
    proof_size, proof_status, ranked_methods, NodeStatus, ProofNode, ProofStatus,
};
use tamarin_theory::constraint::system::System;
use tamarin_theory::elaborate::{
    collect_user_funs_for_theory, elaborate, set_user_funs_from_collected, CollectedUserFuns,
};
use tamarin_theory::pretty_ml::{goal_type_and_args, ML_TERM};
use tamarin_theory::pretty_theory::{pretty_proof_body, ProvedLemma};
use tamarin_theory::prove::ProverSession;
use tamarin_theory::system_codec::{
    decode_proof_method, decode_system, encode_proof_method, encode_system,
};

use crate::state::{AppState, TheoryEntry, TheoryStore};
use crate::ServerConfig;

/// The API's state: the startup theories by index.
pub struct ApiState {
    theories: BTreeMap<usize, Arc<ApiTheory>>,
}

impl ApiState {
    /// Snapshot the theories currently in `store` (the startup ones).
    pub fn from_store(store: &TheoryStore) -> Self {
        ApiState {
            theories: store
                .list()
                .into_iter()
                .map(|entry| {
                    let theory = ApiTheory {
                        entry,
                        prover: Mutex::new(None),
                    };
                    (theory.entry.idx, Arc::new(theory))
                })
                .collect(),
        }
    }
}

struct ApiTheory {
    entry: TheoryEntry,
    /// Built by the first lemma request.  A failed build is not kept: it
    /// can fail for passing reasons (Maude not starting), and the next
    /// request tries again.
    prover: Mutex<Option<Arc<Prover>>>,
}

struct Prover {
    user_funs: CollectedUserFuns,
    session: ProverSession,
    pool: Arc<MaudePool>,
    lemmas: Mutex<BTreeMap<String, Arc<OnceLock<Result<Arc<LemmaState>, String>>>>>,
}

struct LemmaState {
    /// The lemma's context (sources saturated); requests run on copies of
    /// it that carry their own Maude process.
    ctx: ProofContext,
    /// The `initial-system` response.
    initial: Value,
}

/// An error response: 400 with the fork's message, 404, or 500.
enum ApiError {
    BadRequest(String),
    NotFound,
    Internal(String),
}

type ApiResult<T> = std::result::Result<T, ApiError>;

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError::BadRequest(msg.into())
}

impl IntoResponse for ApiError {
    /// Yesod's default error bodies for a JSON client.
    fn into_response(self) -> Response {
        let (status, body) = match self {
            ApiError::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                json!({"message": "Invalid Arguments", "errors": [msg]}),
            ),
            ApiError::NotFound => (StatusCode::NOT_FOUND, json!({"message": "Not Found"})),
            ApiError::Internal(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"message": "Internal Server Error", "errors": [msg]}),
            ),
        };
        (status, Json(body)).into_response()
    }
}

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/theories", get(theories))
        .route(
            "/theories/:idx/lemmas/:lemma/initial-system",
            get(initial_system),
        )
        .route("/theories/:idx/lemmas/:lemma/execute", post(execute))
        .route(
            "/theories/:idx/lemmas/:lemma/check-proof",
            post(check_proof_route),
        )
        .fallback(|| async { ApiError::NotFound })
}

/// `{name: {theoryIndex, theoryName, theoryKind}}`, keyed by theory name.
async fn theories(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut out = Map::new();
    for (idx, theory) in &state.api.theories {
        let name = &theory.entry.name;
        out.insert(
            name.clone(),
            json!({"theoryIndex": idx, "theoryName": name, "theoryKind": "trace"}),
        );
    }
    Json(Value::Object(out))
}

async fn initial_system(
    State(state): State<Arc<AppState>>,
    Path((idx, lemma)): Path<(String, String)>,
) -> ApiResult<Json<Value>> {
    let theory = state.api.theory(&idx)?;
    blocking(move || {
        theory.check_lemma(&lemma)?;
        let prover = theory.prover(&state.cfg)?;
        Ok(Json(prover.lemma(&lemma)?.initial.clone()))
    })
    .await
}

async fn execute(
    State(state): State<Arc<AppState>>,
    Path((idx, lemma)): Path<(String, String)>,
    Form(params): Form<BTreeMap<String, String>>,
) -> ApiResult<Json<Value>> {
    let theory = state.api.theory(&idx)?;
    let (Some(method), Some(sys)) = (params.get("proofmethod"), params.get("system")) else {
        return Err(bad("Missing method or system parameter"));
    };
    let (method, sys) = (method.clone(), sys.clone());
    blocking(move || {
        let method = decode_proof_method(&method)
            .map_err(|e| bad(format!("Failed to decode proof method: {e}")))?;
        let sys = decode_system(&sys).map_err(|e| bad(format!("Failed to decode system: {e}")))?;
        theory.check_lemma(&lemma)?;
        let prover = theory.prover(&state.cfg)?;
        let lemma = prover.lemma(&lemma)?;
        prover.with_step_context(&lemma, |ctx| {
            let mut cases = check_and_exec_proof_method(ctx, &method, &sys)
                .ok_or_else(|| bad("Proof method could not be applied to the given system"))?;
            cases.sort_by(|a, b| a.0.cmp(&b.0));
            let cases = cases
                .iter()
                .map(|(name, sys)| json!({"caseName": name, "system": system_json(ctx, sys)}))
                .collect();
            Ok(Json(Value::Array(cases)))
        })
    })
    .await
}

async fn check_proof_route(
    State(state): State<Arc<AppState>>,
    Path((idx, lemma)): Path<(String, String)>,
    Form(params): Form<BTreeMap<String, String>>,
) -> ApiResult<Json<Value>> {
    let theory = state.api.theory(&idx)?;
    let proof = params
        .get("proof")
        .ok_or_else(|| bad("Missing proof parameter"))?;
    let proof =
        proof_object(proof).map_err(|e| bad(format!("Invalid JSON in proof parameter: {e}")))?;
    blocking(move || {
        theory.check_lemma(&lemma)?;
        let tree = method_tree(&proof).map_err(bad)?;
        let prover = theory.prover(&state.cfg)?;
        let lemma_state = prover.lemma(&lemma)?;
        let start = prover
            .session
            .lemma_initial_system(&lemma)
            .map_err(|e| ApiError::Internal(e.to_string()))?;
        let checked = prover.with_step_context(&lemma_state, |ctx| check_proof(ctx, start, &tree));
        Ok(Json(json!({
            "proofStatus": status_name(proof_status(&checked)),
            "proofSize": proof_size(&checked),
            "proofFile": proof_file(&theory.entry, &lemma, &checked),
        })))
    })
    .await
}

/// Run `f` on a blocking thread; a panic there (e.g. a failing oracle)
/// becomes a 500.
async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> ApiResult<T> + Send + 'static,
) -> ApiResult<T> {
    tokio::task::spawn_blocking(f)
        .await
        .unwrap_or_else(|e| Err(ApiError::Internal(e.to_string())))
}

impl ApiState {
    fn theory(&self, idx: &str) -> ApiResult<Arc<ApiTheory>> {
        idx.parse::<usize>()
            .ok()
            .and_then(|idx| self.theories.get(&idx).cloned())
            .ok_or(ApiError::NotFound)
    }
}

impl ApiTheory {
    fn check_lemma(&self, lemma: &str) -> ApiResult<()> {
        if self.entry.typed_theory.is_sapic {
            return Err(bad("SAPIC theories are not supported by this API"));
        }
        if self.entry.typed_theory.lookup_lemma(lemma).is_none() {
            return Err(bad("Lemma not found"));
        }
        Ok(())
    }

    /// The theory's prover; concurrent first requests wait for one build.
    fn prover(&self, cfg: &ServerConfig) -> ApiResult<Arc<Prover>> {
        let mut slot = self.prover.lock();
        if let Some(prover) = &*slot {
            return Ok(prover.clone());
        }
        let prover = Arc::new(Prover::build(&self.entry, cfg).map_err(ApiError::Internal)?);
        *slot = Some(prover.clone());
        Ok(prover)
    }
}

impl Prover {
    /// The theory's prover session, set up as the GUI's (`ProofState::new`)
    /// but with a pool of Maude processes for the proof steps.
    fn build(entry: &TheoryEntry, cfg: &ServerConfig) -> Result<Prover, String> {
        let user_funs = collect_user_funs_for_theory(&entry.parser_theory);
        let _user_funs_guard = set_user_funs_from_collected(&user_funs);
        let cut =
            crate::handlers::proof_tree::effective_cut(&entry.parser_theory, cfg.stop_on_trace)?;
        let typed =
            elaborate(&entry.parser_theory).map_err(|e| format!("elaborate: {}", e.message))?;
        tamarin_theory::prove::validate_cli_heuristic(&cfg.heuristic, &typed.tactic)?;
        let sig = typed.signature.maude_sig;
        // One memo set for the session's Maude and the pool's, as `--prove`
        // does; they all load the same signature.
        let caches = Arc::new(SharedMaudeCaches::default());
        let maude = MaudeHandle::start_with_caches(&cfg.maude_path, sig.clone(), caches.clone())
            .map_err(|e| format!("maude start: {e:?}"))?;
        let pool = Arc::new(
            MaudePool::new(&cfg.maude_path, sig, cfg.maude_processes.max(1), caches)
                .map_err(|e| format!("maude start: {e:?}"))?,
        );
        let ndc_cache = entry.ndc_cache.clone().map(IntrRuleCache::from);
        let session = ProverSession::build_with_in_file_and_heuristic(
            &entry.parser_theory,
            maude,
            Some(pool.clone()),
            &entry.origin.label(),
            cfg.heuristic.clone(),
            cut,
            ndc_cache.as_ref(),
        )
        .map_err(|e| e.to_string())?;
        Ok(Prover {
            user_funs,
            session,
            pool,
            lemmas: Mutex::new(BTreeMap::new()),
        })
    }

    /// The lemma's state, built once; concurrent first requests wait for
    /// the one build.
    fn lemma(&self, lemma: &str) -> ApiResult<Arc<LemmaState>> {
        let cell = self
            .lemmas
            .lock()
            .entry(lemma.to_string())
            .or_default()
            .clone();
        cell.get_or_init(|| {
            let _user_funs_guard = set_user_funs_from_collected(&self.user_funs);
            let ctx = self
                .session
                .lemma_context(lemma)
                .map_err(|e| e.to_string())?;
            let start = self
                .session
                .lemma_initial_system(lemma)
                .map_err(|e| e.to_string())?;
            let initial = system_json(&ctx, &start);
            Ok(Arc::new(LemmaState { ctx, initial }))
        })
        .clone()
        .map_err(ApiError::Internal)
    }

    /// Run `f` with a copy of the lemma's context that uses a Maude process
    /// from the pool (waiting for a free one) and a fresh-variable counter
    /// of its own.
    fn with_step_context<R>(&self, lemma: &LemmaState, f: impl FnOnce(&ProofContext) -> R) -> R {
        let _user_funs_guard = set_user_funs_from_collected(&self.user_funs);
        let maude = self.pool.acquire();
        let ctx = lemma
            .ctx
            .with_swapped_maude(maude.handle().with_fresh_counter_next(0));
        f(&ctx)
    }
}

/// The fork's system object.  As in the fork, the methods are the ranking
/// at depth 0 (`useHeuristic heuristic 0`), and a finished system's only
/// method is `Finished`, which is where its status comes from.
fn system_json(ctx: &ProofContext, sys: &System) -> Value {
    let methods = ranked_methods(sys, ctx, 0);
    let status = match methods.as_slice() {
        [ProofMethod::Finished(MethodResult::Contradictory(_))] => "Contradictory",
        [ProofMethod::Finished(MethodResult::Solved)] => "Solved",
        [ProofMethod::Finished(MethodResult::Unfinishable)] => "Unfinishable",
        // Sic: the fork's spelling, which clients compare against.
        _ => "Undertermined",
    };
    json!({
        "proofMethods": methods.iter().map(|m| method_json(m, sys)).collect::<Vec<_>>(),
        "encodedMethods": methods.iter().map(encode_proof_method).collect::<Vec<_>>(),
        "encodedSys": encode_system(sys),
        "status": status,
    })
}

fn method_json(method: &ProofMethod, sys: &System) -> Value {
    match method {
        ProofMethod::Simplify => json!({"name": "Simplify"}),
        ProofMethod::Induction => json!({"name": "Induction"}),
        ProofMethod::Sorry(_) => json!({"name": "Sorry"}),
        ProofMethod::Invalidated => json!({"name": "Invalidated"}),
        ProofMethod::SolveGoal(goal) => {
            let (kind, args) = goal_type_and_args(goal, sys);
            json!({"name": "SolveGoal", "goal": {"type": kind, "args": args}})
        }
        ProofMethod::Finished(result) => json!({"name": "Finished", "result": result_json(result)}),
        // Display-only; never ranked.
        ProofMethod::RawSolve(_) => json!({"name": "RawSolve"}),
    }
}

/// aeson's generic encoding of the fork's `Result` and `Contradiction`
/// (`{"tag", "contents"}`).  Clients do not read it; a term inside is
/// printed as a goal argument would be.
fn result_json(result: &MethodResult) -> Value {
    match result {
        MethodResult::Solved => json!({"tag": "Solved"}),
        MethodResult::Unfinishable => json!({"tag": "Unfinishable"}),
        MethodResult::Contradictory(c) => {
            json!({"tag": "Contradictory", "contents": c.as_ref().map(contradiction_json)})
        }
    }
}

fn contradiction_json(c: &Contradiction) -> Value {
    let tag = |t: &str| json!({"tag": t});
    match c {
        Contradiction::Cyclic => tag("Cyclic"),
        Contradiction::SubtermCyclic => tag("SubtermCyclic"),
        Contradiction::NonNormalTerms => tag("NonNormalTerms"),
        Contradiction::ForbiddenExp => tag("ForbiddenExp"),
        Contradiction::ForbiddenBP => tag("ForbiddenBP"),
        Contradiction::ForbiddenKD => tag("ForbiddenKD"),
        Contradiction::ImpossibleChain => tag("ImpossibleChain"),
        Contradiction::ForbiddenChain => tag("ForbiddenChain"),
        Contradiction::ForbiddenACConstrChain => tag("ForbiddenACConstrChain"),
        Contradiction::IncompatibleEqs => tag("IncompatibleEqs"),
        Contradiction::FormulasFalse => tag("FormulasFalse"),
        Contradiction::NonInjectiveFactInstance(a, b, c) => json!({
            "tag": "NonInjectiveFactInstance",
            "contents": [lvar_json(a), lvar_json(b), lvar_json(c)],
        }),
        Contradiction::SuperfluousLearn(t, v) => json!({
            "tag": "SuperfluousLearn",
            "contents": [pretty_lnterm_styled(t, &ML_TERM), lvar_json(v)],
        }),
        Contradiction::NodeAfterLast(a, b) => json!({
            "tag": "NodeAfterLast",
            "contents": [lvar_json(a), lvar_json(b)],
        }),
    }
}

fn lvar_json(v: &LVar) -> Value {
    let sort = match v.sort {
        LSort::Pub => "LSortPub",
        LSort::Fresh => "LSortFresh",
        LSort::Msg => "LSortMsg",
        LSort::Node => "LSortNode",
        LSort::Nat => "LSortNat",
    };
    json!({"lvarName": v.name, "lvarSort": sort, "lvarIdx": v.idx})
}

/// The `proof` field: a JSON object, or a JSON string holding one (the
/// fork accepts both).
fn proof_object(text: &str) -> std::result::Result<Map<String, Value>, String> {
    const SHAPE: &str = "expected proof to be a JSON object or a JSON-encoded object string";
    match parse_json(text)? {
        Value::Object(obj) => Ok(obj),
        Value::String(inner) => match parse_json(&inner)? {
            Value::Object(obj) => Ok(obj),
            _ => Err(SHAPE.into()),
        },
        _ => Err(SHAPE.into()),
    }
}

/// `serde_json::from_str` without its nesting limit of 128: every proof
/// step nests two levels (the node and its `children`), so the limit would
/// refuse proofs deeper than 63 steps.  The fork's parser has no limit.
fn parse_json(text: &str) -> std::result::Result<Value, String> {
    use serde::Deserialize;
    let mut de = serde_json::Deserializer::from_str(text);
    de.disable_recursion_limit();
    let value = Value::deserialize(&mut de).map_err(|e| e.to_string())?;
    de.end().map_err(|e| e.to_string())?;
    Ok(value)
}

/// The fork's `jsonToProof`, with its error messages.
fn method_tree(obj: &Map<String, Value>) -> std::result::Result<MethodTree, String> {
    let method = match obj.get("action") {
        Some(Value::String(s)) if s == "solved" => ProofMethod::Finished(MethodResult::Solved),
        Some(Value::String(s)) if s == "contradictory" => {
            ProofMethod::Finished(MethodResult::Contradictory(None))
        }
        Some(Value::String(token)) => {
            decode_proof_method(token).map_err(|e| format!("Failed to decode proof method: {e}"))?
        }
        Some(Value::Null) => ProofMethod::Finished(MethodResult::Contradictory(None)),
        _ => return Err("Invalid proof JSON: missing or invalid 'action' field".into()),
    };
    let Some(Value::Object(children)) = obj.get("children") else {
        return Err("Invalid proof JSON: missing or invalid 'children' field".into());
    };
    let cases = children
        .iter()
        .map(|(name, child)| match child {
            Value::Object(child) => Ok((name.clone(), method_tree(child)?)),
            _ => Err("Invalid proof JSON: child case must be an object".to_string()),
        })
        .collect::<std::result::Result<_, _>>()?;
    Ok(MethodTree { method, cases })
}

/// HS `ProofStatus` constructor names (Theory/Proof.hs:397-407).
fn status_name(status: ProofStatus) -> &'static str {
    match status {
        ProofStatus::Undetermined => "UndeterminedProof",
        ProofStatus::Complete => "CompleteProof",
        ProofStatus::Incomplete => "IncompleteProof",
        ProofStatus::TraceFound => "TraceFound",
        ProofStatus::Unfinishable => "UnfinishableProof",
        ProofStatus::Invalidated => "InvalidatedProof",
    }
}

/// The theory with the checked proof for `lemma` and every other lemma's
/// proof replaced by `sorry "removed"`, as the fork's `checkproof` renders
/// it.
fn proof_file(entry: &TheoryEntry, lemma: &str, checked: &ProofNode) -> String {
    let removed = pretty_proof_body(&ProofNode {
        method: ProofMethod::Sorry(Some("removed".into())),
        sys: System::default(),
        children: BTreeMap::new(),
        status: NodeStatus::Sorry,
        annotated: true,
    });
    let proved: Vec<ProvedLemma> = entry
        .typed_theory
        .lemmas()
        .map(|l| ProvedLemma {
            name: l.name.clone(),
            proof_body: Some(if l.name == lemma {
                pretty_proof_body(checked)
            } else {
                removed.clone()
            }),
        })
        .collect();
    crate::handlers::theory::render_closed_theory(entry, &proved)
}
