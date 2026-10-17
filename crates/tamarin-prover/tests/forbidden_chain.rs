//! (No HS analog.) Regression test for the forbidden-chain contradiction.
//!
//! It once also treated two message variables as equal when one case of an
//! open split mapped both to terms with the same outer function symbol.
//! That closed the branch holding the only trace of this exists-trace lemma,
//! so tamarin-rs answered "falsified" where tamarin-prover finds a trace.

mod common;

use common::{maude_available, run_binary};
use std::path::PathBuf;

#[test]
fn forbidden_chain_does_not_close_a_branch_with_a_trace() {
    if !maude_available() {
        eprintln!("skipping: maude not on path");
        return;
    }
    let theory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tamarin-prover/examples/ccs18-5G/5G-AKA-bindingChannel/5G_AKA_fix.spthy");
    let (code, stdout, stderr) = run_binary(
        &["--prove=executability_desync", "--heuristic=s"],
        &[theory.as_path()],
    );
    assert_eq!(code, 0, "stderr:\n{stderr}");
    assert!(
        stdout.contains("executability_desync (exists-trace): verified"),
        "expected a trace for executability_desync; got:\n{stdout}"
    );
}
