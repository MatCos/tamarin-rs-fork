//! (No HS analog.) Encodes a constraint `System` or a `ProofMethod` as an
//! opaque text token for the RL server, which hands it to TamRL as
//! `encodedSys` / `encodedMethods` and gets it back in the next request.
//!
//! Format: the serde derives, written with postcard, then gzip (level 3), then
//! base64url with padding.  Level 3 writes about twice as fast as level 6
//! (the default of Haskell's zlib) for tokens 5-12% larger; reading costs the
//! same at both.  The TamRL Haskell fork wraps GHC's derived `Binary` the same
//! way; tokens of the two are not interchangeable.  The token is only meant to be read back by the
//! build that wrote it; it is not a stable file format.  Decoding trusts its
//! input: it rebuilds values without re-running the smart constructors.
//!
//! Not written: `RuleAttributes.process` (the SAPIC process of a rule), so a
//! system from a SAPIC theory does not round-trip.  The RL server refuses
//! SAPIC theories.

use std::io::{Read, Write};

use base64::alphabet::URL_SAFE;
use base64::engine::general_purpose::{GeneralPurpose, PAD};
use base64::engine::DecodePaddingMode;
use base64::Engine;
use flate2::bufread::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::constraint::solver::ProofMethod;
use crate::constraint::system::System;

/// base64url, written with padding (as Haskell's
/// `Data.ByteString.Base64.URL.encode` does), read with or without it.
const B64: GeneralPurpose = GeneralPurpose::new(
    &URL_SAFE,
    PAD.with_decode_padding_mode(DecodePaddingMode::Indifferent),
);

/// Why a token could not be read back.
#[derive(Debug)]
pub enum CodecError {
    Base64(base64::DecodeError),
    Gzip(std::io::Error),
    Postcard(postcard::Error),
    /// Bytes left over after the value: most likely a token of the other
    /// kind (a system read as a method), whose first bytes happen to read
    /// as a value of this kind.
    TrailingBytes(usize),
}

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodecError::Base64(e) => write!(f, "invalid base64url: {e}"),
            CodecError::Gzip(e) => write!(f, "invalid gzip data: {e}"),
            CodecError::Postcard(e) => write!(f, "invalid postcard data: {e}"),
            CodecError::TrailingBytes(n) => write!(f, "{n} bytes left after the value"),
        }
    }
}

impl std::error::Error for CodecError {}

fn encode<T: Serialize>(value: &T) -> String {
    // postcard into a `Vec` fails only for a sequence of unknown length,
    // which serde's impls for std collections never produce.
    let raw = postcard::to_allocvec(value).expect("postcard encoding into a Vec");
    let mut gz = GzEncoder::new(Vec::new(), Compression::new(3));
    gz.write_all(&raw).expect("gzip into a Vec");
    B64.encode(gz.finish().expect("gzip into a Vec"))
}

fn decode<T: DeserializeOwned>(token: &str) -> Result<T, CodecError> {
    let gz = B64.decode(token).map_err(CodecError::Base64)?;
    let mut raw = Vec::new();
    GzDecoder::new(&gz[..])
        .read_to_end(&mut raw)
        .map_err(CodecError::Gzip)?;
    let (value, rest) = postcard::take_from_bytes(&raw).map_err(CodecError::Postcard)?;
    if !rest.is_empty() {
        return Err(CodecError::TrailingBytes(rest.len()));
    }
    Ok(value)
}

/// Encode a system as a token (postcard, gzip, base64url).
pub fn encode_system(sys: &System) -> String {
    encode(sys)
}

/// Read back a token written by [`encode_system`].
pub fn decode_system(token: &str) -> Result<System, CodecError> {
    decode(token)
}

/// Encode a proof method as a token (postcard, gzip, base64url).
pub fn encode_proof_method(method: &ProofMethod) -> String {
    encode(method)
}

/// Read back a token written by [`encode_proof_method`].
pub fn decode_proof_method(token: &str) -> Result<ProofMethod, CodecError> {
    decode(token)
}

#[cfg(test)]
#[path = "system_codec_tests.rs"]
mod tests;
