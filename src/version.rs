use std::{borrow::Cow, sync::OnceLock};

use reth_node_core::version::{RethCliVersionConsts, try_init_version_metadata, version_metadata};

pub fn init_reth_hl_version() {
    let cargo_pkg_version = env!("CARGO_PKG_VERSION").to_string();

    let short = env!("RETH_HL_SHORT_VERSION").to_string();
    let long = format!(
        "{}\n{}\n{}\n{}\n{}",
        env!("RETH_HL_LONG_VERSION_0"),
        env!("RETH_HL_LONG_VERSION_1"),
        env!("RETH_HL_LONG_VERSION_2"),
        env!("RETH_HL_LONG_VERSION_3"),
        env!("RETH_HL_LONG_VERSION_4"),
    );
    let p2p = env!("RETH_HL_P2P_CLIENT_VERSION").to_string();

    let meta = RethCliVersionConsts {
        name_client: Cow::Borrowed("reth_hl"),
        cargo_pkg_version: Cow::Owned(cargo_pkg_version.clone()),
        vergen_git_sha_long: Cow::Owned(env!("VERGEN_GIT_SHA").to_string()),
        vergen_git_sha: Cow::Owned(env!("VERGEN_GIT_SHA_SHORT").to_string()),
        vergen_build_timestamp: Cow::Owned(env!("VERGEN_BUILD_TIMESTAMP").to_string()),
        vergen_cargo_target_triple: Cow::Owned(env!("VERGEN_CARGO_TARGET_TRIPLE").to_string()),
        vergen_cargo_features: Cow::Owned(env!("VERGEN_CARGO_FEATURES").to_string()),
        short_version: Cow::Owned(short),
        long_version: Cow::Owned(long),
        build_profile_name: Cow::Owned(env!("RETH_HL_BUILD_PROFILE").to_string()),
        p2p_client_version: Cow::Owned(p2p),
        extra_data: Cow::Owned(format!("reth_hl/v{}/{}", cargo_pkg_version, std::env::consts::OS)),
    };

    let _ = try_init_version_metadata(meta);
}

/// Returns a stable generator identifier for RPC provenance metadata.
pub fn rpc_generator() -> &'static str {
    static GENERATOR: OnceLock<String> = OnceLock::new();
    GENERATOR
        .get_or_init(|| format!("nanoreth/{}", version_metadata().short_version))
        .as_str()
}
