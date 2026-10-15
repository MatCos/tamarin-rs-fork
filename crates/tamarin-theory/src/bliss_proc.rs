// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! (No HS analog.) A subprocess driver for `bliss` — graph canonical
//! labeling / automorphism-group computation — which names a constraint
//! system's variables through the canonical labeling of its term graph
//! (`canon_term_graph`).
//!
//! Mirrors `tamarin_term::maude_proc`'s pattern of spawning and driving
//! an external tool over stdin/stdout rather than binding via FFI, but
//! MUCH simpler: bliss is a one-shot batch tool (write the whole input
//! graph, close stdin, read all of stdout until the process exits), not
//! a persistent line-oriented REPL, so there is no protocol/prompt
//! parsing and no need for a process pool — every call spawns a fresh
//! `bliss` process and waits for it to finish. `TAM_ALLOW_NO_BLISS=1`
//! mirrors `TAM_ALLOW_NO_MAUDE`'s escape hatch (see
//! [`bliss_available`]).
//!
//! ## Input format
//!
//! bliss's own DIMACS-like graph format, verified directly against
//! bliss 0.77's own source (`src/digraph.cc`'s `Digraph::read_dimacs`)
//! — see [`to_dimacs`]:
//!
//! ```text
//! p edge <N> <M>
//! n <vertex 1-indexed> <color>        (one line per vertex)
//! e <src 1-indexed> <tgt 1-indexed>   (exactly M lines)
//! ```
//!
//! ## Output format
//!
//! With `-directed -can -v=0` (verified both against bliss 0.77's
//! source, `src/bliss.cc`/`src/utils.cc`'s `print_permutation`, AND
//! empirically against hand-built graphs — see this module's tests):
//! zero or more automorphism-GENERATOR lines, each followed by exactly
//! one canonical-labeling line:
//!
//! ```text
//! Generator: (1,2,3)(4,5)
//! Canonical labeling: (1,3)
//! ```
//!
//! Both are permutations of the 1-indexed vertex numbers in CYCLE
//! notation: `(a,b,c)` means `a -> b -> c -> a`. A vertex that maps to
//! itself is a FIXED POINT and is omitted from every cycle entirely
//! (not printed as a length-1 cycle `(a)`); an entirely-identity
//! permutation prints as the literal empty `()`. Reconstructing the
//! FULL permutation therefore requires knowing the vertex count `N` up
//! front and defaulting every unmentioned vertex to itself — see
//! [`Permutation::from_cycle_notation`].
//!
//! ## Caveats investigated before writing any of this
//!
//! - **Directed, not undirected.** bliss defaults to undirected graphs;
//!   a term graph's edges are directed (parent to child), so every call
//!   here passes `-directed` and uses `Digraph::read_dimacs`'s format, not
//!   `Graph::read_dimacs`'s.
//! - **Colors are an equivalence signal, not a magnitude.** bliss's
//!   partition refinement only cares which vertices share a color, not
//!   the numeric VALUE.
//! - **`-v=0` is required, not optional**: bliss's default verbosity is
//!   1 (`static unsigned int verbose_level = 1;`, `src/bliss.cc`), which
//!   interleaves human-readable statistics into stdout — without
//!   `-v=0` the output would not even be reliably line-parseable as
//!   just `Generator:`/`Canonical labeling:` lines. [`run_bliss_with_stats`]
//!   wants those statistics (`|Aut|`, search nodes), so it runs with `-v=1`
//!   and picks lines by prefix instead.
//! - **Duplicate edges are dropped by bliss.** A caller must not encode
//!   anything in edge multiplicity (the term graph gives every variable
//!   occurrence its own vertex for this reason).
//! - **Empty graphs are rejected by bliss** (`Digraph::read_dimacs`
//!   errors on `nof_vertices <= 0`) — [`to_dimacs`] returns a clear
//!   [`BlissError::EmptyGraph`] instead of letting bliss fail with a
//!   cryptic parse error.
//! - **Vertex numbering does not affect the canonical graph.** Two
//!   isomorphic graphs, numbered differently, give the same relabeled
//!   colors and edges once each is relabeled by its own canonical labeling
//!   (`differently_numbered_isomorphic_graphs_agree_after_canonicalizing`).
//!   When the graph has nontrivial automorphisms, which of the equivalent
//!   labelings bliss reports can depend on the input numbering.

use std::fmt;
use std::fmt::Write as _;
use std::io::Write as _;
use std::process::{Command, Stdio};

/// A vertex color. bliss only compares colors for equality.
pub type Color = u32;

const ALLOW_NO_BLISS_ENV: &str = "TAM_ALLOW_NO_BLISS";

/// Errors that can arise from the bliss bridge.
#[derive(Debug)]
pub enum BlissError {
    /// stdin/stdout I/O failure talking to the bliss subprocess.
    Io(std::io::Error),
    /// The `bliss` binary could not be launched (e.g. not found on PATH).
    Spawn(String),
    /// bliss's output could not be parsed as expected.
    Parse(String),
    /// bliss exited with a non-zero status.
    NonZeroExit { status: String, stderr: String },
    /// The graph has no vertices — bliss's own `read_dimacs` rejects
    /// this (`nof_vertices <= 0`), so this is caught here first with a
    /// clearer message.
    EmptyGraph,
}

impl fmt::Display for BlissError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlissError::Io(e) => write!(f, "io error: {e}"),
            BlissError::Spawn(s) => write!(f, "spawn error: {s}"),
            BlissError::Parse(s) => write!(f, "parse error: {s}"),
            BlissError::NonZeroExit { status, stderr } => {
                write!(f, "bliss exited with {status}: {stderr}")
            }
            BlissError::EmptyGraph => write!(f, "cannot canonicalize an empty graph (0 vertices)"),
        }
    }
}

impl std::error::Error for BlissError {}

/// The `bliss` binary to invoke: `$BLISS_PATH` if set, else bare
/// `bliss` (resolved via `$PATH`) — mirrors
/// `tamarin-prover/tests/common/mod.rs`'s `maude_path`/`MAUDE_PATH`
/// convention.
fn bliss_binary() -> String {
    std::env::var("BLISS_PATH").unwrap_or_else(|_| "bliss".to_string())
}

/// True when the `bliss` executable is runnable (`bliss -version` exits
/// successfully). PANICS (not a silent `false`/skip) when it is not,
/// unless `TAM_ALLOW_NO_BLISS=1` is set — mirrors
/// `tamarin-prover/tests/common/mod.rs`'s `maude_available()` test-gating
/// precedent: a machine where bliss is simply absent should fail loudly
/// when a bliss-backed test runs, not let every such test report a
/// vacuous green by silently skipping. Every bliss-backed test in this
/// crate calls this first.
pub fn bliss_available() -> bool {
    let bin = bliss_binary();
    let ok = Command::new(&bin)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if ok {
        return true;
    }
    assert!(
        std::env::var(ALLOW_NO_BLISS_ENV).as_deref() == Ok("1"),
        "no working `bliss` executable found (tried `{bin} -version`) -- install \
         bliss (https://users.aalto.fi/~tjunttil/bliss/download.html) and put it \
         on PATH, or point BLISS_PATH at it. Set {ALLOW_NO_BLISS_ENV}=1 to skip \
         bliss-backed tests deliberately."
    );
    false
}

// =============================================================================
// Permutation
// =============================================================================

/// A permutation of `0..n` vertex indices (0-indexed internally, even
/// though bliss's own textual format is 1-indexed — the `+1`/`-1`
/// conversion is entirely contained in [`Self::from_cycle_notation`]/
/// [`to_dimacs`]). `perm[i]` is where vertex `i` maps TO.
///
/// Used for both bliss's canonical labeling (`perm[old_vertex] =
/// new_vertex`) and each automorphism generator (`perm[v] = v'`, a
/// relabeling under which the graph maps to itself) — the same
/// underlying mathematical object serves both roles; only the caller's
/// interpretation differs.
///
/// Entries are stored as `u32`, not `usize`, to halve the size; a graph
/// never comes close to `u32::MAX` vertices (asserted on construction).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Permutation(Vec<u32>);

impl Permutation {
    pub fn identity(n: usize) -> Self {
        Permutation((0..vertex_index(n)).collect())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Where vertex `v` maps to.
    pub fn image_of(&self, v: usize) -> usize {
        self.0[v] as usize
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.0
    }

    /// Parses bliss's 1-indexed cycle notation (e.g. `"(1,2,3)(4,5)"`,
    /// or `"()"` for the identity), given the total vertex count `n`
    /// (needed because fixed points are omitted from the text — see the
    /// module docs).
    fn from_cycle_notation(s: &str, n: usize) -> Result<Self, BlissError> {
        let mut perm: Vec<u32> = (0..vertex_index(n)).collect();
        let s = s.trim();
        if s.is_empty() || s == "()" {
            return Ok(Permutation(perm));
        }
        for cycle in s.split(')') {
            let cycle = cycle.trim();
            if cycle.is_empty() {
                continue;
            }
            let cycle = cycle.strip_prefix('(').ok_or_else(|| {
                BlissError::Parse(format!("expected '(' to start a cycle in {s:?}"))
            })?;
            let elems: Vec<usize> = cycle
                .split(',')
                .map(|tok| {
                    tok.trim().parse::<usize>().map_err(|e| {
                        BlissError::Parse(format!("bad vertex number {tok:?} in {s:?}: {e}"))
                    })
                })
                .collect::<Result<_, _>>()?;
            if elems.is_empty() {
                continue;
            }
            for i in 0..elems.len() {
                let from = elems[i].checked_sub(1).ok_or_else(|| {
                    BlissError::Parse(format!("vertex 0 is invalid (bliss is 1-indexed): {s:?}"))
                })?;
                let to = elems[(i + 1) % elems.len()].checked_sub(1).ok_or_else(|| {
                    BlissError::Parse(format!("vertex 0 is invalid (bliss is 1-indexed): {s:?}"))
                })?;
                if from >= n || to >= n {
                    return Err(BlissError::Parse(format!(
                        "vertex out of range [1,{n}] in cycle notation: {s:?}"
                    )));
                }
                perm[from] = vertex_index(to);
            }
        }
        Ok(Permutation(perm))
    }
}

/// `v` as a stored [`Permutation`] entry.
fn vertex_index(v: usize) -> u32 {
    u32::try_from(v).expect("graph with more than u32::MAX vertices")
}

// =============================================================================
// DIMACS writer
// =============================================================================

/// Renders a colored digraph -- vertex `i` has color `colors[i]`, `edges`
/// are `(src, tgt)` pairs of 0-based vertex indices -- as bliss's
/// DIMACS-like input (see the module docs). bliss drops duplicate edges, so
/// a caller must not rely on edge multiplicity.
pub fn to_dimacs(colors: &[Color], edges: &[(usize, usize)]) -> Result<String, BlissError> {
    if colors.is_empty() {
        return Err(BlissError::EmptyGraph);
    }
    let mut out = String::new();
    writeln!(out, "p edge {} {}", colors.len(), edges.len()).ok();
    for (idx, color) in colors.iter().enumerate() {
        writeln!(out, "n {} {}", idx + 1, color).ok();
    }
    for (src, tgt) in edges {
        writeln!(out, "e {} {}", src + 1, tgt + 1).ok();
    }
    Ok(out)
}

// =============================================================================
// Running bliss
// =============================================================================

/// The result of running bliss with `-directed -can` on a graph:
/// bliss's own canonical vertex numbering plus a generating set for the
/// graph's automorphism group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlissResult {
    pub canonical_labeling: Permutation,
    pub generators: Vec<Permutation>,
}

/// Runs `bliss -directed -can -v=0` once on `dimacs_input`, parsing its
/// stdout. Spawns a FRESH process every call — no pool, no persistent
/// process (unlike `maude_proc::MaudeHandle`): bliss is a one-shot batch
/// tool, not an interactive REPL, so there is no protocol state to keep
/// alive between calls, and the caller doesn't need one either.
///
/// The vertex count is recovered by scanning `dimacs_input` for its own
/// `p edge N M` line, rather than taking it as a separate parameter —
/// one less thing for a caller to keep in sync with the input it just
/// built.
pub fn run_bliss(dimacs_input: &str) -> Result<BlissResult, BlissError> {
    let n = parse_vertex_count(dimacs_input)?;
    let stdout = run_bliss_process(dimacs_input, "-v=0")?;
    parse_bliss_stdout(&stdout, n)
}

/// bliss's own search statistics for one run (its `-v=1` summary).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BlissStats {
    /// `|Aut|` exactly as bliss prints it: an integer that can be far
    /// beyond `u64` (k interchangeable sessions give k!).
    pub aut_size: String,
    /// Nodes of bliss's search tree -- its own measure of the work done.
    pub search_nodes: u64,
    pub generators: u64,
    pub max_level: u64,
}

impl BlissStats {
    /// `|Aut|` as a float, for statistics (exact up to 2^53).
    pub fn aut_size_f64(&self) -> f64 {
        self.aut_size.parse().unwrap_or(f64::NAN)
    }
}

/// [`run_bliss`], also returning bliss's search statistics: runs
/// `bliss -directed -can -v=1`, whose summary lines follow the
/// `Generator:`/`Canonical labeling:` lines on stdout.
pub fn run_bliss_with_stats(dimacs_input: &str) -> Result<(BlissResult, BlissStats), BlissError> {
    let n = parse_vertex_count(dimacs_input)?;
    let stdout = run_bliss_process(dimacs_input, "-v=1")?;
    let mut stats = BlissStats::default();
    let mut result_lines = String::new();
    for line in stdout.lines() {
        let number = |rest: &str| -> Result<u64, BlissError> {
            rest.trim().parse().map_err(|e| {
                BlissError::Parse(format!("bad number in bliss stats line {line:?}: {e}"))
            })
        };
        if let Some(rest) = line.strip_prefix("|Aut|:") {
            stats.aut_size = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("Nodes:") {
            stats.search_nodes = number(rest)?;
        } else if let Some(rest) = line.strip_prefix("Generators:") {
            stats.generators = number(rest)?;
        } else if let Some(rest) = line.strip_prefix("Max level:") {
            stats.max_level = number(rest)?;
        } else if line.starts_with("Generator: ") || line.starts_with("Canonical labeling: ") {
            result_lines.push_str(line);
            result_lines.push('\n');
        }
        // Every other line is a statistic not recorded here (leaf and bad
        // nodes, canonical-representative updates, total time).
    }
    if stats.aut_size.is_empty() {
        return Err(BlissError::Parse(format!(
            "no '|Aut|:' line in bliss -v=1 output: {stdout:?}"
        )));
    }
    Ok((parse_bliss_stdout(&result_lines, n)?, stats))
}

/// Runs `bliss -directed -can <verbosity>` on `dimacs_input`, returning its
/// stdout.
fn run_bliss_process(dimacs_input: &str, verbosity: &str) -> Result<String, BlissError> {
    let bin = bliss_binary();

    let mut child = Command::new(&bin)
        .args(["-directed", "-can", verbosity])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| BlissError::Spawn(format!("{bin}: {e}")))?;

    {
        // Scoped so the `ChildStdin` is dropped (closing the pipe) before
        // `wait_with_output` reads stdout. Writing the whole input before
        // reading anything cannot deadlock even when it exceeds a pipe
        // buffer: bliss reads all of stdin before it writes anything.
        let stdin = child.stdin.as_mut().expect("piped stdin");
        stdin
            .write_all(dimacs_input.as_bytes())
            .map_err(BlissError::Io)?;
    }

    let output = child.wait_with_output().map_err(BlissError::Io)?;
    if !output.status.success() {
        return Err(BlissError::NonZeroExit {
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn parse_vertex_count(dimacs_input: &str) -> Result<usize, BlissError> {
    for line in dimacs_input.lines() {
        if let Some(rest) = line.strip_prefix("p edge ") {
            let n: usize = rest
                .split_whitespace()
                .next()
                .ok_or_else(|| BlissError::Parse(format!("malformed 'p edge' line: {line:?}")))?
                .parse()
                .map_err(|e| BlissError::Parse(format!("bad vertex count in {line:?}: {e}")))?;
            return Ok(n);
        }
    }
    Err(BlissError::Parse(
        "no 'p edge N M' line found in DIMACS input".to_string(),
    ))
}

fn parse_bliss_stdout(stdout: &str, n: usize) -> Result<BlissResult, BlissError> {
    let mut generators = Vec::new();
    let mut canonical_labeling = None;
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("Generator: ") {
            generators.push(Permutation::from_cycle_notation(rest, n)?);
        } else if let Some(rest) = line.strip_prefix("Canonical labeling: ") {
            canonical_labeling = Some(Permutation::from_cycle_notation(rest, n)?);
        } else if !line.trim().is_empty() {
            return Err(BlissError::Parse(format!(
                "unexpected bliss output line: {line:?}"
            )));
        }
    }
    let canonical_labeling = canonical_labeling.ok_or_else(|| {
        BlissError::Parse(format!(
            "no 'Canonical labeling:' line in bliss output -- was -can passed? \
             full stdout: {stdout:?}"
        ))
    })?;
    Ok(BlissResult {
        canonical_labeling,
        generators,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_parses_empty_cycle_notation_as_identity() {
        let p = Permutation::from_cycle_notation("()", 4).unwrap();
        assert_eq!(p.as_slice(), &[0, 1, 2, 3]);
    }

    #[test]
    fn permutation_parses_a_single_cycle() {
        // (1,2,3) on 3 vertices: 0-indexed 0->1->2->0.
        let p = Permutation::from_cycle_notation("(1,2,3)", 3).unwrap();
        assert_eq!(p.as_slice(), &[1, 2, 0]);
    }

    #[test]
    fn permutation_parses_multiple_disjoint_cycles_and_keeps_fixed_points() {
        // (1,3)(4,5) on 5 vertices: 0<->2 swap, 3<->4 swap, 1 fixed.
        let p = Permutation::from_cycle_notation("(1,3)(4,5)", 5).unwrap();
        assert_eq!(p.as_slice(), &[2, 1, 0, 4, 3]);
    }

    #[test]
    fn permutation_rejects_out_of_range_vertices() {
        let err = Permutation::from_cycle_notation("(1,9)", 3).unwrap_err();
        assert!(matches!(err, BlissError::Parse(_)));
    }

    // ---------------------------------------------------------------
    // Live bliss tests -- gated on `bliss_available()`.
    // ---------------------------------------------------------------

    #[test]
    fn bliss_reports_version() {
        // `bliss_available()` already asserted this succeeds (or skips
        // cleanly under `TAM_ALLOW_NO_BLISS=1`); nothing further to
        // check here beyond "didn't panic".
        bliss_available();
    }

    #[test]
    fn triangle_with_uniform_color_has_a_nontrivial_automorphism() {
        if !bliss_available() {
            return;
        }
        let dimacs = "p edge 3 3\nn 1 0\nn 2 0\nn 3 0\ne 1 2\ne 2 3\ne 3 1\n";
        let result = run_bliss(dimacs).expect("run_bliss");
        assert!(
            !result.generators.is_empty(),
            "a uniformly-colored directed triangle has rotational automorphisms"
        );
    }

    /// The manual experiment this module's design was validated against
    /// (see the module docs): two hand-built, differently-numbered but
    /// isomorphic colored digraphs must give the same relabeled colors and
    /// edges.
    #[test]
    fn differently_numbered_isomorphic_graphs_agree_after_canonicalizing() {
        if !bliss_available() {
            return;
        }
        // Graph A: 1(color5) -> 2(color7) -> 3(color9).
        let a = "p edge 3 2\nn 1 5\nn 2 7\nn 3 9\ne 1 2\ne 2 3\n";
        // Graph B: same graph, relabeled (v1=color7 mid, v2=color9 sink,
        // v3=color5 src; edges src->mid->sink = 3->1, 1->2).
        let b = "p edge 3 2\nn 1 7\nn 2 9\nn 3 5\ne 3 1\ne 1 2\n";

        let ra = run_bliss(a).expect("run_bliss a");
        let rb = run_bliss(b).expect("run_bliss b");
        assert!(ra.generators.is_empty());
        assert!(rb.generators.is_empty());

        let colors_a = [5u32, 7, 9];
        let colors_b = [7u32, 9, 5];
        let edges_a = [(0usize, 1usize), (1, 2)];
        let edges_b = [(2usize, 0usize), (0, 1)];

        let relabel = |colors: &[u32], edges: &[(usize, usize)], labeling: &Permutation| {
            let mut vc = vec![0u32; colors.len()];
            for (i, c) in colors.iter().enumerate() {
                vc[labeling.image_of(i)] = *c;
            }
            let es: std::collections::BTreeSet<(usize, usize)> = edges
                .iter()
                .map(|(s, t)| (labeling.image_of(*s), labeling.image_of(*t)))
                .collect();
            (vc, es)
        };

        let canon_a = relabel(&colors_a, &edges_a, &ra.canonical_labeling);
        let canon_b = relabel(&colors_b, &edges_b, &rb.canonical_labeling);
        assert_eq!(canon_a, canon_b);
    }

    /// `-v=1` adds bliss's statistics without changing the result, and
    /// `|Aut|` is kept exactly even far beyond `u64` (25 interchangeable
    /// vertices give 25!).
    #[test]
    fn run_bliss_with_stats_reports_the_group_size() {
        if !bliss_available() {
            return;
        }
        let triangle = "p edge 3 3\nn 1 0\nn 2 0\nn 3 0\ne 1 2\ne 2 3\ne 3 1\n";
        let (result, stats) = run_bliss_with_stats(triangle).expect("run_bliss_with_stats");
        assert_eq!(result, run_bliss(triangle).expect("run_bliss"));
        assert_eq!(
            stats.aut_size, "3",
            "a directed triangle has exactly its 3 rotations"
        );
        assert_eq!(stats.generators, result.generators.len() as u64);

        let colors = vec![0; 25];
        let (_, stats) =
            run_bliss_with_stats(&to_dimacs(&colors, &[]).expect("dimacs")).expect("run");
        assert_eq!(stats.aut_size, "15511210043330985984000000");
        assert!((stats.aut_size_f64() - 1.5511210043330986e25).abs() < 1e12);
    }

    #[test]
    fn to_dimacs_rejects_an_empty_graph() {
        assert!(matches!(to_dimacs(&[], &[]), Err(BlissError::EmptyGraph)));
    }

    // -----------------------------------------------------------------
    // Real bliss run against bliss's OWN documented automorphism example
    // (https://users.aalto.fi/~tjunttil/bliss/definitions.html, also
    // work.tex's $G_1$ example, \Cref{ex:graph_iso}): 4 vertices, vertex
    // 1 colored distinctly from 2/3/4, edges {1,2},{1,3},{1,4},{2,3},{2,4}
    // (encoded here as a symmetric pair of directed edges per undirected
    // edge, since `bliss_proc` always runs bliss with `-directed`).
    // Documented automorphism group: `{id, (3 4)}`, generated by the
    // single transposition swapping vertices 3 and 4.
    // -----------------------------------------------------------------

    #[test]
    fn bliss_g1_example_has_the_documented_automorphism_group() {
        if !bliss_available() {
            return;
        }
        let dimacs = "p edge 4 10\n\
                      n 1 1\n\
                      n 2 0\n\
                      n 3 0\n\
                      n 4 0\n\
                      e 1 2\ne 2 1\n\
                      e 1 3\ne 3 1\n\
                      e 1 4\ne 4 1\n\
                      e 2 3\ne 3 2\n\
                      e 2 4\ne 4 2\n";
        let result = run_bliss(dimacs).expect("run_bliss");

        assert_eq!(
            result.generators.len(),
            1,
            "bliss's own documented generating set for this graph is the single \
             transposition (3 4)"
        );

        assert_eq!(
            result.generators[0],
            Permutation::from_cycle_notation("(3,4)", 4).unwrap(),
            "Aut(G_1) is documented as exactly {{id, (3 4)}}"
        );
    }
}
