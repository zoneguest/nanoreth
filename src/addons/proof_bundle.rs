use crate::node::rpc::{HlEthApi, HlRpcNodeCore};
use alloy_consensus::Header;
use alloy_eips::BlockId;
use alloy_primitives::{Address, B256, Bytes, keccak256};
use alloy_rpc_types_eth::EIP1186AccountProofResponse;
use alloy_serde::JsonStorageKey;
use jsonrpsee::proc_macros::rpc;
use jsonrpsee_core::{RpcResult, async_trait};
use reth_chainspec::{ChainSpecProvider, EthChainSpec};
use reth::rpc::result::internal_rpc_err;
use reth_rpc_convert::RpcConvert;
use reth_rpc_eth_api::{FromEvmError, RpcNodeCore, helpers::EthState};
use reth_rpc_eth_types::EthApiError;
use reth_storage_api::{BlockIdReader, BlockNumReader, HeaderProvider, ProviderHeader};
use serde::Serialize;
use tracing::trace;

const SYNTHETIC_ROOT_TYPE: &str = "synthetic_mpt_v1";
const PROOF_FORMAT: &str = "eip1186";
const PROOF_BUNDLE_VERSION: &str = "1";

/// Derives the proof root from the first account-proof node.
///
/// Under the EIP-1186 proof shape exposed by reth/alloy, `account_proof` is the
/// top-down Merkle-Patricia branch for the account, so the first node is the
/// root trie node whose hash identifies the proof bundle. If that contract ever
/// changes upstream, this endpoint's `proofRoot` semantics must be revisited.
fn proof_root_from_account_proof(
    proof: &EIP1186AccountProofResponse,
    block_hash: B256,
) -> RpcResult<B256> {
    let first_node = proof.account_proof.first().ok_or_else(|| {
        internal_rpc_err(format!(
            "eth_getProof returned an empty accountProof for address {} at block {block_hash}",
            proof.address
        ))
    })?;
    Ok(keccak256(first_node))
}

/// The storage-trie root the returned `storageProof` entries verify against, or `None` when no
/// storage keys were requested (an empty `storageProof`).
///
/// This is the account's `storageHash`. Unlike `proofRoot` — a synthetic, unauthenticated
/// account-trie root — the storage root is authenticated by the account proof against `proofRoot`,
/// so storage-slot inclusion/exclusion binds transitively to the same anchor.
fn storage_root_from_proof(proof: &EIP1186AccountProofResponse) -> Option<B256> {
    (!proof.storage_proof.is_empty()).then_some(proof.storage_hash)
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HlProofBundleResponse {
    pub version: &'static str,
    #[serde(with = "alloy_serde::quantity")]
    pub chain_id: u64,
    pub block: HlProofBundleBlock,
    /// Canonical EVM header for `block`, RLP-encoded, with its committed sub-roots surfaced.
    ///
    /// `rlp` is the RLP of the *inner* `alloy_consensus::Header` — `HlHeader.extras` are not part
    /// of the block hash — so `keccak256(rlp) == block.hash`. `stateRoot` is expected to be zero on
    /// HyperEVM; `transactionsRoot`/`receiptsRoot` are the canonical roots committed via the block
    /// hash (and therefore, transitively, via `evm_db.block_hashes` in the HyperCore app hash),
    /// which is the anchor for future tx/receipt/log inclusion proofs. `None` only when the header
    /// could not be loaded (e.g. a pending block).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub header: Option<HlProofBundleHeader>,
    pub provenance: HlProofBundleProvenance,
    pub account_proof: EIP1186AccountProofResponse,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HlProofBundleBlock {
    pub hash: B256,
    #[serde(with = "alloy_serde::quantity")]
    pub number: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HlProofBundleHeader {
    /// RLP of the canonical `alloy_consensus::Header`; `keccak256(rlp) == block.hash`.
    pub rlp: Bytes,
    /// Expected to be zero on HyperEVM (headers carry `stateRoot = 0`).
    pub state_root: B256,
    pub transactions_root: B256,
    pub receipts_root: B256,
}

/// How strongly this bundle is anchored to HyperCore consensus.
///
/// Computed per response from the anchoring data actually present — never a hardcoded constant.
/// Today the endpoint carries no HyperCore anchor, so it resolves to [`AnchorType::SingleNode`];
/// once the co-located hl-node app hash + quorum certificate are wired in (Tier A2), a recent
/// finalized block resolves to [`AnchorType::ConsensusQuorum`], while unfinalized or
/// beyond-retention blocks continue to resolve to [`AnchorType::SingleNode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorType {
    /// No HyperCore anchor present; only the serving node vouches for `proofRoot`.
    SingleNode,
    /// Signed by an external attestor committee (Path D). Reserved; not yet produced.
    AttestorQuorum,
    /// Backed by the hl-node quorum-signed app hash for this block.
    ConsensusQuorum,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HlProofBundleProvenance {
    pub proof_root: B256,
    pub root_type: &'static str,
    pub generator: &'static str,
    pub proof_format: &'static str,
    /// Storage-trie root (`account.storageHash`) that the returned `storageProof` entries verify
    /// against. Present only when storage keys were requested. Authenticated by the account proof
    /// against `proofRoot` (unlike `proofRoot` itself, which is a synthetic unauthenticated root),
    /// so it carries the storage-proof root with explicit provenance rather than as a bare hash in
    /// the EIP-1186 body. `proofFormat` already describes the (EIP-1186) storage-proof encoding, so
    /// no separate storage-format tag is emitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage_root: Option<B256>,
    /// Per-response trust discriminant (see [`AnchorType`]).
    pub anchor_type: AnchorType,
    /// EVM blocks elapsed since `block.number` (`best_block_number - block.number`).
    ///
    /// HyperCore commits EVM block hashes in `evm_db.block_hashes` with a 256-block window, so a
    /// consumer uses this to tell whether the `H_evm ∈ block_hashes` link is still provable.
    #[serde(with = "alloy_serde::quantity")]
    pub core_commitment_depth: u64,
    /// `BLAKE3(ConciseLtHashes)` — HyperCore's canonical EVM state commitment for this block.
    ///
    /// Sourced from the co-located hl-node ABCI state (Tier A2); `None` until that reader lands.
    /// Deliberately not recomputed locally: a byte-exact value requires hl-node's canonical entry
    /// serialization, which is not yet reproduced, and a mismatching value would be worse than
    /// absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evm_state_commitment: Option<B256>,
    /// The three HyperCore LtHash accumulator digests `{accounts, contracts, storage}`.
    ///
    /// Same source and caveat as [`Self::evm_state_commitment`]; `None` until Tier A2.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concise_lt_hashes: Option<[B256; 3]>,
}

/// A Hyper-specific proof endpoint that exposes explicit provenance metadata.
///
/// Method name: `hl_getProofBundle`
///
/// Scope and trust model (HyperEVM headers carry `stateRoot = 0`):
/// - `proofRoot` identifies the serving node's synthetic keccak-MPT root
///   (`rootType = "synthetic_mpt_v1"`). Clients can validate the enclosed EIP-1186 proof
///   against it, but cannot authenticate it against the HL header or hl-node.
/// - Non-pending requests resolve to a canonical block hash before proof generation, so
///   `block` and `accountProof` are hash-pinned to the same historical state.
/// - `pending` is passed through to `eth_getProof` unchanged. Its returned block metadata is
///   resolved independently and may not describe the exact pending state used for the proof.
/// - Passing storage `keys` returns EIP-1186 `storageProof` entries (inclusion or exclusion) for a
///   contract's slots. They verify against `storageRoot` (the account's `storageHash`), which the
///   account proof in turn authenticates against `proofRoot` — so contract-state proofs share the
///   account proof's trust anchor. `storageRoot` is surfaced in `provenance` only when keys are
///   requested.
/// - Nanoreth does not guarantee trie state for deep-historical blocks, so this endpoint
///   shares the `--experimental-eth-get-proof` gate with `eth_getProof`.
#[rpc(server, namespace = "hl")]
#[async_trait]
pub trait HlProofBundleApi {
    #[method(name = "getProofBundle")]
    async fn get_proof_bundle(
        &self,
        address: Address,
        keys: Vec<JsonStorageKey>,
        block: Option<BlockId>,
    ) -> RpcResult<HlProofBundleResponse>;
}

pub struct HlProofBundleExt<N: HlRpcNodeCore, Rpc: RpcConvert> {
    eth_api: HlEthApi<N, Rpc>,
}

impl<N: HlRpcNodeCore, Rpc: RpcConvert> HlProofBundleExt<N, Rpc> {
    pub fn new(eth_api: HlEthApi<N, Rpc>) -> Self {
        Self { eth_api }
    }
}

#[async_trait]
impl<N, Rpc> HlProofBundleApiServer for HlProofBundleExt<N, Rpc>
where
    N: HlRpcNodeCore,
    EthApiError: FromEvmError<N::Evm>,
    N::Provider: BlockIdReader + ChainSpecProvider + HeaderProvider,
    ProviderHeader<N::Provider>: Into<Header>,
    Rpc: RpcConvert<Primitives = N::Primitives, Error = EthApiError>,
{
    async fn get_proof_bundle(
        &self,
        address: Address,
        keys: Vec<JsonStorageKey>,
        block: Option<BlockId>,
    ) -> RpcResult<HlProofBundleResponse> {
        let requested_block = block.unwrap_or_default();
        trace!(target: "rpc::hl", ?address, ?keys, ?requested_block, "Serving hl_getProofBundle");

        let provider = self.eth_api.provider();
        let (proof_block, block_hash) = if requested_block.is_pending() {
            // Preserve eth_getProof pending-state handling. Metadata is resolved separately,
            // so do not treat `block` as an atomic snapshot of the pending proof state.
            let block_hash = provider
                .block_hash_for_id(requested_block)
                .map_err(|err| internal_rpc_err(format!("Failed to resolve block hash: {err}")))?
                .ok_or_else(|| EthApiError::HeaderNotFound(requested_block))?;
            (requested_block, block_hash)
        } else {
            // Resolve tags such as `latest` once, then use the hash for every later read.
            let block_hash = provider
                .block_hash_for_id(requested_block)
                .map_err(|err| internal_rpc_err(format!("Failed to resolve block hash: {err}")))?
                .ok_or_else(|| EthApiError::HeaderNotFound(requested_block))?;
            (BlockId::Hash(block_hash.into()), block_hash)
        };
        let block_number = provider
            .block_number_for_id(proof_block)
            .map_err(|err| internal_rpc_err(format!("Failed to resolve block number: {err}")))?
            .ok_or_else(|| EthApiError::HeaderNotFound(proof_block))?;

        let proof = EthState::get_proof(&self.eth_api, address, keys, Some(proof_block))?.await?;
        let proof_root = proof_root_from_account_proof(&proof, block_hash)?;
        let storage_root = storage_root_from_proof(&proof);
        let chain_id = provider.chain_spec().chain_id();

        // Depth against the current tip. HyperCore commits EVM block hashes in `evm_db.block_hashes`
        // for a 256-block window, so this bounds whether `block_hash` is still provably committed.
        let best_block_number = provider.best_block_number().map_err(|err| {
            internal_rpc_err(format!("Failed to resolve best block number: {err}"))
        })?;
        let core_commitment_depth = best_block_number.saturating_sub(block_number);

        // Canonical header for the pinned block. We RLP-encode the inner `Header` (the `HlHeader`
        // extras are excluded from the block hash), so `keccak256(rlp) == block_hash`, and surface
        // the committed sub-roots. `None` if the header is not loadable (e.g. a pending block).
        let header = match provider
            .header(&block_hash)
            .map_err(|err| internal_rpc_err(format!("Failed to load header: {err}")))?
        {
            Some(provider_header) => {
                let eth_header: Header = provider_header.into();
                let rlp = Bytes::from(alloy_rlp::encode(&eth_header));
                debug_assert_eq!(
                    keccak256(rlp.as_ref()),
                    block_hash,
                    "canonical header RLP must hash to the pinned block hash"
                );
                Some(HlProofBundleHeader {
                    rlp,
                    state_root: eth_header.state_root,
                    transactions_root: eth_header.transactions_root,
                    receipts_root: eth_header.receipts_root,
                })
            }
            None => None,
        };

        Ok(HlProofBundleResponse {
            version: PROOF_BUNDLE_VERSION,
            chain_id,
            block: HlProofBundleBlock { hash: block_hash, number: block_number },
            header,
            provenance: HlProofBundleProvenance {
                proof_root,
                root_type: SYNTHETIC_ROOT_TYPE,
                generator: crate::version::rpc_generator(),
                proof_format: PROOF_FORMAT,
                storage_root,
                anchor_type: AnchorType::SingleNode,
                core_commitment_depth,
                evm_state_commitment: None,
                concise_lt_hashes: None,
            },
            account_proof: proof,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HlHeader;
    use crate::node::primitives::header::HlHeaderExtras;
    use alloy_primitives::{Bytes, Sealable, U256, address, b256};
    use alloy_rpc_types_eth::EIP1186StorageProof;
    use reth_node_core::version::version_metadata;
    use serde_json::Value;

    struct TestProofBundleServer;

    #[async_trait]
    impl HlProofBundleApiServer for TestProofBundleServer {
        async fn get_proof_bundle(
            &self,
            _address: Address,
            _keys: Vec<JsonStorageKey>,
            _block: Option<BlockId>,
        ) -> RpcResult<HlProofBundleResponse> {
            Ok(HlProofBundleResponse {
                version: PROOF_BUNDLE_VERSION,
                chain_id: 999,
                block: HlProofBundleBlock {
                    hash: b256!(
                        "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    ),
                    number: 7,
                },
                header: Some(HlProofBundleHeader {
                    rlp: Bytes::from_static(&[0xc0]),
                    state_root: B256::ZERO,
                    transactions_root: b256!(
                        "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                    ),
                    receipts_root: b256!(
                        "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
                    ),
                }),
                provenance: HlProofBundleProvenance {
                    proof_root: b256!(
                        "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                    ),
                    root_type: SYNTHETIC_ROOT_TYPE,
                    generator: crate::version::rpc_generator(),
                    proof_format: PROOF_FORMAT,
                    // Matches `storage_hash` below; present because a storage key was requested.
                    storage_root: Some(b256!(
                        "0xdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
                    )),
                    anchor_type: AnchorType::SingleNode,
                    core_commitment_depth: 5,
                    evm_state_commitment: None,
                    concise_lt_hashes: None,
                },
                account_proof: EIP1186AccountProofResponse {
                    address: address!("0x0000000000000000000000000000000000000011"),
                    balance: U256::from(42),
                    code_hash: b256!(
                        "0xcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
                    ),
                    nonce: 3,
                    storage_hash: b256!(
                        "0xdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
                    ),
                    account_proof: vec![Bytes::from_static(&[0x01, 0x02, 0x03])],
                    storage_proof: vec![EIP1186StorageProof {
                        value: U256::from(0x7b),
                        proof: vec![Bytes::from_static(&[0x11, 0x22])],
                        ..Default::default()
                    }],
                },
            })
        }
    }

    #[test]
    fn proof_root_is_derived_from_first_account_proof_node() {
        let proof = EIP1186AccountProofResponse {
            address: Address::ZERO,
            balance: U256::ZERO,
            code_hash: B256::ZERO,
            nonce: 0,
            storage_hash: B256::ZERO,
            account_proof: vec![Bytes::from_static(&[0xde, 0xad, 0xbe, 0xef])],
            storage_proof: vec![],
        };

        let root = proof_root_from_account_proof(&proof, B256::ZERO).unwrap();
        assert_eq!(root, keccak256([0xde, 0xad, 0xbe, 0xef]));
    }

    #[test]
    fn proof_root_from_empty_account_proof_returns_contextual_rpc_error() {
        let proof = EIP1186AccountProofResponse {
            address: address!("0x0000000000000000000000000000000000000011"),
            balance: U256::ZERO,
            code_hash: B256::ZERO,
            nonce: 0,
            storage_hash: B256::ZERO,
            account_proof: vec![],
            storage_proof: vec![],
        };

        let err = proof_root_from_account_proof(
            &proof,
            b256!("0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        )
        .unwrap_err();

        assert_eq!(
            err.message(),
            "eth_getProof returned an empty accountProof for address 0x0000000000000000000000000000000000000011 at block 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        );
    }

    #[test]
    fn exclusion_proof_for_absent_account_still_yields_proof_root() {
        // EIP-1186 exclusion proof: the account does not exist, but eth_getProof
        // still returns a non-empty `account_proof` (the branch proving the key is
        // absent) with zeroed balance/nonce and the canonical empty code/storage
        // hashes. This is the shape a ZK zero-balance proof depends on, so it must
        // derive a valid `proofRoot` rather than tripping the empty-proof guard
        // above. The distinction is: empty proof vec -> error; non-empty exclusion
        // branch for a zero account -> ok.
        let root_node = Bytes::from_static(&[0xf8, 0x51, 0x80, 0x80]);
        let proof = EIP1186AccountProofResponse {
            address: address!("0x0000000000000000000000000000000000000011"),
            balance: U256::ZERO,
            nonce: 0,
            code_hash: keccak256(b""),       // empty code: keccak256("")
            storage_hash: keccak256([0x80]), // empty storage trie: keccak256(rlp(""))
            account_proof: vec![root_node.clone()],
            storage_proof: vec![],
        };

        let root = proof_root_from_account_proof(&proof, B256::ZERO).unwrap();
        assert_eq!(root, keccak256(root_node));
    }

    #[tokio::test]
    async fn rpc_module_serves_hl_get_proof_bundle() {
        crate::version::init_reth_hl_version();
        let module = TestProofBundleServer.into_rpc();
        let request = r#"{"jsonrpc":"2.0","method":"hl_getProofBundle","params":["0x0000000000000000000000000000000000000011",[], "latest"],"id":1}"#;

        let (raw, _rx) = module.raw_json_request(request, 1).await.unwrap();
        let response: Value = serde_json::from_str(raw.get()).unwrap();
        let result = &response["result"];

        let expected_generator = format!("nanoreth/{}", version_metadata().short_version);

        assert_eq!(result["version"], PROOF_BUNDLE_VERSION);
        assert_eq!(result["chainId"], "0x3e7");
        assert_eq!(result["block"]["hash"], "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        assert_eq!(result["block"]["number"], "0x7");
        assert_eq!(
            result["provenance"]["proofRoot"],
            "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        );
        assert_eq!(result["provenance"]["rootType"], SYNTHETIC_ROOT_TYPE);
        assert_eq!(result["provenance"]["proofFormat"], PROOF_FORMAT);
        assert_eq!(result["provenance"]["generator"], expected_generator);
        assert_eq!(result["provenance"]["anchorType"], "single_node");
        assert_eq!(result["provenance"]["coreCommitmentDepth"], "0x5");
        // Absent (skip_serializing_if) until the Tier A2 hl-node reader populates them.
        assert!(result["provenance"]["evmStateCommitment"].is_null());
        assert!(result["provenance"]["conciseLtHashes"].is_null());
        assert_eq!(result["header"]["rlp"], "0xc0");
        assert_eq!(
            result["header"]["stateRoot"],
            "0x0000000000000000000000000000000000000000000000000000000000000000"
        );
        assert_eq!(
            result["header"]["transactionsRoot"],
            "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        );
        assert_eq!(
            result["header"]["receiptsRoot"],
            "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        );
        assert_eq!(result["accountProof"]["address"], "0x0000000000000000000000000000000000000011");
        assert_eq!(result["accountProof"]["balance"], "0x2a");
        assert_eq!(result["accountProof"]["nonce"], "0x3");
        assert_eq!(result["accountProof"]["accountProof"][0], "0x010203");
        // B2: storage proof surfaced through the envelope, with its authenticated root.
        assert_eq!(
            result["provenance"]["storageRoot"],
            "0xdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
        );
        assert_eq!(result["accountProof"]["storageProof"][0]["value"], "0x7b");
        assert_eq!(result["accountProof"]["storageProof"][0]["proof"][0], "0x1122");
    }

    #[test]
    fn storage_root_present_when_storage_proof_returned() {
        let proof = EIP1186AccountProofResponse {
            address: address!("0x0000000000000000000000000000000000000011"),
            balance: U256::ZERO,
            code_hash: B256::ZERO,
            nonce: 0,
            storage_hash: b256!(
                "0xdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
            ),
            account_proof: vec![Bytes::from_static(&[0x01])],
            storage_proof: vec![EIP1186StorageProof {
                value: U256::from(1),
                proof: vec![Bytes::from_static(&[0x02])],
                ..Default::default()
            }],
        };

        assert_eq!(
            storage_root_from_proof(&proof),
            Some(b256!("0xdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"))
        );
    }

    #[test]
    fn storage_root_absent_when_no_storage_keys() {
        let proof = EIP1186AccountProofResponse {
            address: address!("0x0000000000000000000000000000000000000011"),
            balance: U256::ZERO,
            code_hash: B256::ZERO,
            nonce: 0,
            storage_hash: b256!(
                "0xdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
            ),
            account_proof: vec![Bytes::from_static(&[0x01])],
            storage_proof: vec![],
        };

        assert_eq!(storage_root_from_proof(&proof), None);
    }

    #[test]
    fn canonical_header_rlp_hashes_to_block_hash() {
        // The block hash is keccak(rlp(inner Header)); the HlHeader.extras are excluded. So the
        // bundle's `header.rlp` must encode the inner header (matching block.hash), not the
        // wrapper. This is the exact invariant get_proof_bundle relies on.
        let inner = Header { number: 100, gas_limit: 30_000_000, ..Default::default() };
        let hl = HlHeader {
            inner,
            extras: HlHeaderExtras {
                logs_bloom_with_system_txs: Default::default(),
                system_tx_count: 3,
            },
        };

        let block_hash = Sealable::hash_slow(&hl);

        let eth_header: Header = hl.clone().into();
        let canonical_rlp = alloy_rlp::encode(&eth_header);
        assert_eq!(keccak256(&canonical_rlp), block_hash, "inner-header RLP must match block hash");

        // Encoding the HlHeader wrapper (which appends extras) must NOT match the block hash.
        let wrapper_rlp = alloy_rlp::encode(&hl);
        assert_ne!(keccak256(&wrapper_rlp), block_hash, "wrapper RLP must not match block hash");
    }

    #[test]
    fn anchor_type_serializes_snake_case() {
        assert_eq!(serde_json::to_value(AnchorType::SingleNode).unwrap(), "single_node");
        assert_eq!(serde_json::to_value(AnchorType::AttestorQuorum).unwrap(), "attestor_quorum");
        assert_eq!(serde_json::to_value(AnchorType::ConsensusQuorum).unwrap(), "consensus_quorum");
    }
}
