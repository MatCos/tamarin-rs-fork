// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! Auto-sources on the web load path.
//!
//! HS's interactive mode closes every theory it loads with the same
//! `closeTheory version thyLoadOptions` the batch mode uses
//! (Interactive.hs:135-136), so auto-sources is on when the CLI has
//! `--auto-sources` or the theory's `configuration:` block does
//! (`configAutoSources`, TheoryLoader.hs:764-765).  The fixture's `Recv`
//! rule outputs a variable it only got from inside an encryption, a partial
//! deconstruction, so auto-sources adds the `AUTO_typing` lemma (as
//! `tamarin-rs auto_sources_config.spthy` does).
//!
//! One test, because the CLI switch is process-wide.

mod common;

use common::{fixture_path, maude_available, maude_path, start_server_with_theory};
use tamarin_server::theory_io;

fn has_auto_typing(entry: &tamarin_server::TheoryEntry) -> bool {
    entry.typed_theory.lookup_lemma("AUTO_typing").is_some()
}

#[tokio::test(flavor = "multi_thread")]
async fn auto_sources_from_the_config_block_or_the_cli() {
    if !maude_available() {
        return;
    }
    tamarin_server::init_process_globals();
    let maude = maude_path();
    let with_block = fixture_path("auto_sources_config.spthy");

    // The configuration block turns it on without the CLI switch.
    theory_io::set_auto_sources(false);
    let entry = theory_io::load_from_path(&with_block, &maude, 0).expect("fixture loads");
    assert!(has_auto_typing(&entry));
    // Both halves are rewritten: the parser theory, which the proof state
    // and the RL API re-elaborate, carries the lemma too.
    assert!(entry
        .parser_theory
        .items
        .iter()
        .any(|item| format!("{item:?}").contains("AUTO_typing")));

    // Without the block, only the CLI switch turns it on.
    let src = std::fs::read_to_string(&with_block).unwrap();
    let without_block = src.replace("configuration: \"--auto-sources\"\n", "");
    assert_ne!(src, without_block);
    let origin = || tamarin_server::state::TheoryOrigin::Upload("no_block.spthy".into());
    let entry = theory_io::load_from_source(&without_block, origin(), &maude, 0).unwrap();
    assert!(!has_auto_typing(&entry));
    theory_io::set_auto_sources(true);
    let entry = theory_io::load_from_source(&without_block, origin(), &maude, 0).unwrap();
    assert!(has_auto_typing(&entry));
    theory_io::set_auto_sources(false);

    // The GUI shows the lemma and the annotated rules.
    let s = start_server_with_theory("auto_sources_config.spthy").await;
    let source = s
        .client
        .get(s.url("/thy/trace/1/source"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(source.contains("lemma AUTO_typing [sources]:"), "{source}");
    assert!(source.contains("AUTO_IN_TERM_1_0_0__Recv"), "{source}");
}
