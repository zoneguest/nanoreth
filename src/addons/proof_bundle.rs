use crate::HlBlock;
use crate::addons::inclusion_proof::{HlInclusionProof, ordered_trie_inclusion_proof};
use crate::node::rpc::{HlEthApi, HlRpcNodeCore};
use crate::node::types::{ReadPrecompileCalls, ReadPrecompileResult};
use alloy_consensus::{Header, TxReceipt, transaction::TxHashRef};
use alloy_eips::{BlockId, Encodable2718};
use alloy_primitives::{Address, B256, Bytes, keccak256};
use alloy_rpc_types_eth::EIP1186AccountProofResponse;
use alloy_serde::JsonStorageKey;
use jsonrpsee::proc_macros::rpc;
use jsonrpsee_core::{RpcResult, async_trait};
use reth_chainspec::{ChainSpecProvider, EthChainSpec};
use reth::rpc::result::internal_rpc_err;
use reth_ethereum_primitives::Receipt;
use reth_rpc_convert::RpcConvert;
use reth_rpc_eth_api::{FromEvmError, RpcNodeCore, helpers::EthState};
use reth_rpc_eth_types::EthApiError;
use reth_storage_api::{
    BlockIdReader, BlockNumReader, BlockReader, HeaderProvider, ProviderHeader, ReceiptProvider,
};
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

/// Build transaction and receipt inclusion proofs for `target_hash` in a block.
///
/// Both root-defining sets are reconstructed exactly as the block was built
/// (`src/node/evm/config.rs`): transactions exclude system txs (`!is_system_transaction()`);
/// receipts keep only `cumulative_gas_used() != 0` (which drops the zero-gas system-tx receipts —
/// verified against live data). Those two filters select the same positions in order, so the tx's
/// index doubles as the receipt index — asserted via a length check. Each recomputed root is
/// checked against the header before returning, so a wrong set/order/encoding errors out rather
/// than yielding a bad proof.
fn tx_and_receipt_proofs(
    block: &HlBlock,
    receipts: &[Receipt],
    target_hash: B256,
    tx_root: B256,
    receipts_root: B256,
) -> RpcResult<(HlInclusionProof, HlInclusionProof)> {
    let non_system: Vec<_> =
        block.body.inner.transactions.iter().filter(|t| !t.is_system_transaction()).collect();
    let target = non_system.iter().position(|t| *t.tx_hash() == target_hash).ok_or_else(|| {
        internal_rpc_err(format!("transaction {target_hash} not found among non-system txs"))
    })?;

    let tx_items: Vec<Vec<u8>> = non_system.iter().map(|t| t.encoded_2718()).collect();
    let (tx_proof, tx_computed) = ordered_trie_inclusion_proof(&tx_items, target)
        .ok_or_else(|| internal_rpc_err("failed to build transaction inclusion proof"))?;
    if tx_computed != tx_root {
        return Err(internal_rpc_err(format!(
            "recomputed transactionsRoot {tx_computed} does not match header {tx_root}"
        )));
    }

    // receipts_for_root: `cumulative_gas_used != 0`, aligned 1:1 (and in order) with non-system txs.
    let rc_for_root: Vec<_> = receipts.iter().filter(|r| r.cumulative_gas_used() != 0).collect();
    if rc_for_root.len() != non_system.len() {
        return Err(internal_rpc_err(format!(
            "non-system tx count ({}) != root-defining receipt count ({}); cannot align receipt proof",
            non_system.len(),
            rc_for_root.len()
        )));
    }
    let rc_items: Vec<Vec<u8>> =
        rc_for_root.iter().map(|r| r.with_bloom_ref().encoded_2718()).collect();
    let (receipt_proof, rc_computed) = ordered_trie_inclusion_proof(&rc_items, target)
        .ok_or_else(|| internal_rpc_err("failed to build receipt inclusion proof"))?;
    if rc_computed != receipts_root {
        return Err(internal_rpc_err(format!(
            "recomputed receiptsRoot {rc_computed} does not match header {receipts_root}"
        )));
    }

    Ok((tx_proof, receipt_proof))
}

/// Flatten a block's recorded read-precompile calls into a per-read list (B3, unanchored).
///
/// These are the HyperCore values (SpotBalance/Position/OraclePx/…) the EVM consumed at this block,
/// as recorded by this node in `BlockReadPrecompileCalls`. Single-node trust — not a HyperCore
/// commitment (see proof-bundle-enrichment.md §6.1).
fn core_reads_from_calls(calls: &ReadPrecompileCalls) -> Vec<HlCoreRead> {
    calls
        .0
        .iter()
        .flat_map(|(precompile, pairs)| {
            pairs.iter().map(move |(input, result)| {
                let (status, value) = match result {
                    ReadPrecompileResult::Ok { bytes, .. } => ("ok", Some(bytes.clone())),
                    ReadPrecompileResult::OutOfGas => ("outOfGas", None),
                    ReadPrecompileResult::Error => ("error", None),
                    ReadPrecompileResult::UnexpectedError => ("unexpectedError", None),
                };
                HlCoreRead {
                    precompile: *precompile,
                    input: input.input.clone(),
                    gas_limit: input.gas_limit,
                    status,
                    value,
                }
            })
        })
        .collect()
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
    /// (B1) Transaction inclusion proof against `header.transactionsRoot`, present only when a tx
    /// selector (`tx` hash) was supplied. That root is canonical — committed via the block hash →
    /// `evm_db.block_hashes` → `app_hash` → ed25519 quorum — so this is a trustless cross-chain
    /// claim, unlike the account `proofRoot`. Verify with
    /// `alloy_trie::proof::verify_proof(header.transactionsRoot, key, Some(value), proof)`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tx_proof: Option<HlInclusionProof>,
    /// (B1) Receipt inclusion proof against `header.receiptsRoot`, present only when a `tx` selector
    /// was supplied. Same canonical anchor as `txProof`; the receipt sits at the same index as the
    /// transaction. Verify with
    /// `alloy_trie::proof::verify_proof(header.receiptsRoot, key, Some(value), proof)`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub receipt_proof: Option<HlInclusionProof>,
    /// (Tier A2) HyperCore consensus anchor — present only when the co-located hl-node core-state
    /// source can supply the quorum-signed app hash for this block; when present it drives
    /// `provenance.anchorType` to `consensus_quorum`. `None` today (reader not yet wired), so
    /// `anchorType` stays `single_node`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub core_anchor: Option<HlCoreAnchor>,
    /// (B3, unanchored) HyperCore read-precompile values the EVM consumed at this block, as recorded
    /// by this node (`BlockReadPrecompileCalls`). Present only when requested via the `coreReads`
    /// param. **Single-node trust** — `anchorType` stays `single_node`; this is NOT a HyperCore
    /// commitment, only a corroboration source alongside `txProof`/`receiptProof` (see §6.1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub core_state_reads: Option<Vec<HlCoreRead>>,
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

impl AnchorType {
    /// Compute the per-response trust discriminant from the anchoring data actually present.
    ///
    /// A present [`HlCoreAnchor`] carries the hl-node quorum-signed app hash, so it resolves to
    /// [`AnchorType::ConsensusQuorum`]; its absence (unfinalized/beyond-retention block, or no
    /// co-located core-state source) resolves to [`AnchorType::SingleNode`]. Never hardcoded.
    fn from_core_anchor(anchor: Option<&HlCoreAnchor>) -> Self {
        match anchor {
            Some(_) => Self::ConsensusQuorum,
            None => Self::SingleNode,
        }
    }
}

/// (Tier A2) HyperCore consensus anchor for this EVM block, sourced from the co-located hl-node.
///
/// Populated by a future core-state reader (hl-node persists this via periodic ABCI checkpoints —
/// `abci_checkpoint.rs`/`visor_abci_state.rs` — including an `lt_hashes.json` for the LtHash state).
/// When present, the account/tx/receipt proofs in this bundle are backed by the HyperCore validator
/// quorum, not just the serving node. Verifying consumers check the ed25519 `quorumSignatures`
/// (≥ threshold stake) over `coreAppHash`, and that this EVM block's hash is in `blockHashWindow`
/// (which HyperCore commits within a 256-block window) at `blockHashIndex`.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HlCoreAnchor {
    /// The quorum-signed HyperCore app hash for the round that committed this EVM block.
    pub core_app_hash: B256,
    /// HyperCore height whose app hash committed this EVM block.
    #[serde(with = "alloy_serde::quantity")]
    pub core_height: u64,
    /// The committed EVM block-hash window (≤256), for proving `H_evm ∈ evm_db.block_hashes`.
    pub block_hash_window: Vec<B256>,
    /// Index of this block's hash within `blockHashWindow`.
    #[serde(with = "alloy_serde::quantity")]
    pub block_hash_index: u64,
    /// Commitment to the ed25519 validator set that produced `quorumSignatures`.
    pub validator_set_hash: B256,
    /// ed25519 quorum signatures over `coreAppHash` (not aggregatable — one per validator).
    pub quorum_signatures: Vec<HlQuorumSig>,
    /// Aggregate stake behind the quorum, in basis points.
    #[serde(with = "alloy_serde::quantity")]
    pub quorum_stake_bps: u64,
}

/// A single validator's ed25519 signature over the core app hash.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HlQuorumSig {
    #[serde(with = "alloy_serde::quantity")]
    pub validator_index: u64,
    /// 64-byte ed25519 signature.
    pub signature: Bytes,
}

/// (B3, unanchored) One HyperCore read-precompile call the EVM made during a block, as recorded by
/// this node — a **single-node** attestation, not a HyperCore-committed proof (see §6.1 of the
/// design doc). Its value is corroboration: a read a contract consumed is reflected in the block's
/// canonical `receiptsRoot`, so it can be cross-checked against a `txProof`/`receiptProof`.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HlCoreRead {
    /// Read-precompile address — identifies the read kind (e.g. `0x…0800` = Position).
    pub precompile: Address,
    /// The precompile call input (the query bytes).
    pub input: Bytes,
    #[serde(with = "alloy_serde::quantity")]
    pub gas_limit: u64,
    /// Result status: `ok` | `outOfGas` | `error` | `unexpectedError`.
    pub status: &'static str,
    /// Raw HyperCore value bytes returned; present only when `status == "ok"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<Bytes>,
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
/// - Passing a `tx` hash adds `txProof` and `receiptProof`: inclusion proofs of that transaction
///   and its receipt against `header.transactionsRoot` / `header.receiptsRoot`. Unlike `proofRoot`,
///   those roots are canonical (committed via the block hash into the HyperCore app hash), so the
///   proofs are trustless. The `tx` must be a non-system transaction of the resolved block.
/// - Passing `coreReads = true` adds `coreStateReads`: the HyperCore read-precompile values the EVM
///   consumed at this block, as recorded by this node. This is **single-node** attested (it does not
///   change `anchorType`); its value is corroboration — a read a contract used is reflected in the
///   canonical receipts, so it can be cross-checked against `receiptProof`.
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
        tx: Option<B256>,
        core_reads: Option<bool>,
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
    N::Provider: BlockIdReader
        + ChainSpecProvider
        + HeaderProvider
        + BlockReader<Block = HlBlock>
        + ReceiptProvider<Receipt = Receipt>,
    ProviderHeader<N::Provider>: Into<Header>,
    Rpc: RpcConvert<Primitives = N::Primitives, Error = EthApiError>,
{
    async fn get_proof_bundle(
        &self,
        address: Address,
        keys: Vec<JsonStorageKey>,
        block: Option<BlockId>,
        tx: Option<B256>,
        core_reads: Option<bool>,
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

        // Load the block once if either tx/receipt proofs or core reads are requested.
        let full_block = if tx.is_some() || core_reads == Some(true) {
            Some(
                provider
                    .block_by_hash(block_hash)
                    .map_err(|err| internal_rpc_err(format!("Failed to load block: {err}")))?
                    .ok_or_else(|| EthApiError::HeaderNotFound(BlockId::Hash(block_hash.into())))?,
            )
        } else {
            None
        };

        // (B1) Optional transaction + receipt inclusion proofs against the canonical
        // `transactionsRoot` / `receiptsRoot`.
        let (tx_proof, receipt_proof) = match (tx, full_block.as_ref()) {
            (Some(target_hash), Some(block)) => {
                let hdr = header.as_ref().ok_or_else(|| {
                    internal_rpc_err("header unavailable for block; cannot build tx/receipt proofs")
                })?;
                let receipts = provider
                    .receipts_by_block(block_hash.into())
                    .map_err(|err| internal_rpc_err(format!("Failed to load receipts: {err}")))?
                    .ok_or_else(|| EthApiError::HeaderNotFound(BlockId::Hash(block_hash.into())))?;
                let (tx_p, receipt_p) = tx_and_receipt_proofs(
                    block,
                    &receipts,
                    target_hash,
                    hdr.transactions_root,
                    hdr.receipts_root,
                )?;
                (Some(tx_p), Some(receipt_p))
            }
            _ => (None, None),
        };

        // (B3, unanchored) Optional single-node HyperCore read-precompile snapshot for the block.
        let core_state_reads = (core_reads == Some(true)).then(|| {
            full_block
                .as_ref()
                .and_then(|b| b.body.read_precompile_calls.as_ref())
                .map(core_reads_from_calls)
                .unwrap_or_default()
        });

        // (Tier A2) HyperCore consensus anchor. `None` until the co-located hl-node core-state
        // reader is wired (hl-node persists this via periodic ABCI checkpoints / lt_hashes.json;
        // see proof-bundle-enrichment.md §5/§12). Its presence is what makes `anchorType`
        // resolve to `consensus_quorum`.
        let core_anchor: Option<HlCoreAnchor> = None;

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
                anchor_type: AnchorType::from_core_anchor(core_anchor.as_ref()),
                core_commitment_depth,
                evm_state_commitment: None,
                concise_lt_hashes: None,
            },
            account_proof: proof,
            tx_proof,
            receipt_proof,
            core_anchor,
            core_state_reads,
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
            _tx: Option<B256>,
            _core_reads: Option<bool>,
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
                tx_proof: Some(HlInclusionProof {
                    index: 1,
                    key: Bytes::from_static(&[0x01]),
                    value: Bytes::from_static(&[0xaa, 0xbb]),
                    proof: vec![Bytes::from_static(&[0xc0]), Bytes::from_static(&[0xc1])],
                }),
                receipt_proof: Some(HlInclusionProof {
                    index: 1,
                    key: Bytes::from_static(&[0x01]),
                    value: Bytes::from_static(&[0xcc, 0xdd]),
                    proof: vec![Bytes::from_static(&[0xd0])],
                }),
                core_anchor: None,
                core_state_reads: Some(vec![HlCoreRead {
                    precompile: address!("0x0000000000000000000000000000000000000800"),
                    input: Bytes::from_static(&[0x01, 0x02]),
                    gas_limit: 2000,
                    status: "ok",
                    value: Some(Bytes::from_static(&[0xbe, 0xef])),
                }]),
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
        // B1: transaction + receipt inclusion proofs surfaced in the envelope.
        assert_eq!(result["txProof"]["index"], "0x1");
        assert_eq!(result["txProof"]["key"], "0x01");
        assert_eq!(result["txProof"]["value"], "0xaabb");
        assert_eq!(result["txProof"]["proof"][0], "0xc0");
        assert_eq!(result["txProof"]["proof"][1], "0xc1");
        assert_eq!(result["receiptProof"]["index"], "0x1");
        assert_eq!(result["receiptProof"]["value"], "0xccdd");
        assert_eq!(result["receiptProof"]["proof"][0], "0xd0");
        // A2: no core anchor yet -> absent, and anchorType stays single_node.
        assert!(result["coreAnchor"].is_null());
        // B3 (unanchored): core-state reads surfaced when requested.
        assert_eq!(
            result["coreStateReads"][0]["precompile"],
            "0x0000000000000000000000000000000000000800"
        );
        assert_eq!(result["coreStateReads"][0]["input"], "0x0102");
        assert_eq!(result["coreStateReads"][0]["gasLimit"], "0x7d0"); // 2000
        assert_eq!(result["coreStateReads"][0]["status"], "ok");
        assert_eq!(result["coreStateReads"][0]["value"], "0xbeef");
    }

    #[test]
    fn core_reads_from_calls_flattens_and_maps_status() {
        use crate::node::types::ReadPrecompileInput;
        let calls = ReadPrecompileCalls(vec![(
            address!("0x0000000000000000000000000000000000000800"),
            vec![
                (
                    ReadPrecompileInput { input: Bytes::from_static(&[0xaa]), gas_limit: 5 },
                    ReadPrecompileResult::Ok { gas_used: 3, bytes: Bytes::from_static(&[0x11, 0x22]) },
                ),
                (
                    ReadPrecompileInput { input: Bytes::from_static(&[0xbb]), gas_limit: 5 },
                    ReadPrecompileResult::OutOfGas,
                ),
            ],
        )]);
        let reads = core_reads_from_calls(&calls);
        assert_eq!(reads.len(), 2);
        assert_eq!(reads[0].precompile, address!("0x0000000000000000000000000000000000000800"));
        assert_eq!(reads[0].status, "ok");
        assert_eq!(reads[0].value.as_ref().unwrap().as_ref(), &[0x11, 0x22]);
        assert_eq!(reads[1].status, "outOfGas");
        assert!(reads[1].value.is_none());
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

    fn sample_core_anchor() -> HlCoreAnchor {
        HlCoreAnchor {
            core_app_hash: b256!(
                "0x1111111111111111111111111111111111111111111111111111111111111111"
            ),
            core_height: 42,
            block_hash_window: vec![b256!(
                "0x2222222222222222222222222222222222222222222222222222222222222222"
            )],
            block_hash_index: 0,
            validator_set_hash: b256!(
                "0x3333333333333333333333333333333333333333333333333333333333333333"
            ),
            quorum_signatures: vec![HlQuorumSig {
                validator_index: 7,
                signature: Bytes::from_static(&[0xab, 0xcd]),
            }],
            quorum_stake_bps: 6700,
        }
    }

    #[test]
    fn anchor_type_computed_from_core_anchor() {
        // A2: SingleNode when absent, ConsensusQuorum when a core anchor is present (never hardcoded).
        assert_eq!(AnchorType::from_core_anchor(None), AnchorType::SingleNode);
        let anchor = sample_core_anchor();
        assert_eq!(AnchorType::from_core_anchor(Some(&anchor)), AnchorType::ConsensusQuorum);
    }

    #[test]
    fn core_anchor_serializes_camel_case() {
        let v = serde_json::to_value(sample_core_anchor()).unwrap();
        assert_eq!(
            v["coreAppHash"],
            "0x1111111111111111111111111111111111111111111111111111111111111111"
        );
        assert_eq!(v["coreHeight"], "0x2a");
        assert_eq!(v["blockHashIndex"], "0x0");
        assert_eq!(v["quorumStakeBps"], "0x1a2c"); // 6700
        assert_eq!(
            v["validatorSetHash"],
            "0x3333333333333333333333333333333333333333333333333333333333333333"
        );
        assert_eq!(v["quorumSignatures"][0]["validatorIndex"], "0x7");
        assert_eq!(v["quorumSignatures"][0]["signature"], "0xabcd");
        assert!(v["blockHashWindow"].is_array());
    }
}
