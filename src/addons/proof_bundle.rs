use crate::node::rpc::{HlEthApi, HlRpcNodeCore};
use alloy_eips::BlockId;
use alloy_primitives::{Address, B256, keccak256};
use alloy_rpc_types_eth::EIP1186AccountProofResponse;
use alloy_serde::JsonStorageKey;
use jsonrpsee::proc_macros::rpc;
use jsonrpsee_core::{RpcResult, async_trait};
use reth_chainspec::{ChainSpecProvider, EthChainSpec};
use reth::rpc::result::internal_rpc_err;
use reth_rpc_convert::RpcConvert;
use reth_rpc_eth_api::{FromEvmError, RpcNodeCore, helpers::EthState};
use reth_rpc_eth_types::EthApiError;
use reth_storage_api::BlockIdReader;
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

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HlProofBundleResponse {
    pub version: &'static str,
    #[serde(with = "alloy_serde::quantity")]
    pub chain_id: u64,
    pub block: HlProofBundleBlock,
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
pub struct HlProofBundleProvenance {
    pub proof_root: B256,
    pub root_type: &'static str,
    pub generator: &'static str,
    pub proof_format: &'static str,
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
    N::Provider: BlockIdReader + ChainSpecProvider,
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
        let chain_id = provider.chain_spec().chain_id();

        Ok(HlProofBundleResponse {
            version: PROOF_BUNDLE_VERSION,
            chain_id,
            block: HlProofBundleBlock { hash: block_hash, number: block_number },
            provenance: HlProofBundleProvenance {
                proof_root,
                root_type: SYNTHETIC_ROOT_TYPE,
                generator: crate::version::rpc_generator(),
                proof_format: PROOF_FORMAT,
            },
            account_proof: proof,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{Bytes, U256, address, b256};
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
                provenance: HlProofBundleProvenance {
                    proof_root: b256!(
                        "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                    ),
                    root_type: SYNTHETIC_ROOT_TYPE,
                    generator: crate::version::rpc_generator(),
                    proof_format: PROOF_FORMAT,
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
                    storage_proof: vec![],
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
        assert_eq!(result["accountProof"]["address"], "0x0000000000000000000000000000000000000011");
        assert_eq!(result["accountProof"]["balance"], "0x2a");
        assert_eq!(result["accountProof"]["nonce"], "0x3");
        assert_eq!(result["accountProof"]["accountProof"][0], "0x010203");
    }
}
