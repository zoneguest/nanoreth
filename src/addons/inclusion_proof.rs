//! Ordered-trie (transaction / receipt) inclusion proofs — the B1 building block.
//!
//! HyperEVM zeroes `stateRoot`, but `transactionsRoot`/`receiptsRoot` are real keccak-MPT roots
//! committed via the block hash → `evm_db.block_hashes` (256-block window) → `app_hash` → ed25519
//! quorum. So a proof against them is a *canonical*, non-single-node cross-chain claim, unlike the
//! account `proofRoot` (which is a synthetic node-local root).
//!
//! The trie construction mirrors `alloy_trie::root::ordered_trie_root_with_encoder` exactly: for
//! `n` items, leaf `i` has key `Nibbles::unpack(rlp(i))` and value = the item's canonical encoding.
//!
//! Caller responsibilities (the endpoint that will serve these):
//! - Transactions: HyperEVM's `transactionsRoot` is built over the block body **excluding system
//!   transactions** (`!is_system_transaction()`, per `HlBlockBody::calculate_tx_root`). Pass the
//!   already-filtered transactions, each encoded with `Encodable2718::encode_2718`, in that order.
//! - Receipts: encode with `with_bloom_ref().encode_2718` (per reth's `calculate_receipt_root`).
//! - After building, assert the returned root equals the header `transactionsRoot`/`receiptsRoot`
//!   before serving — a mismatch means the item set/order is wrong and the proof would be invalid.

use alloy_primitives::{B256, Bytes};
use alloy_trie::{HashBuilder, Nibbles, proof::ProofRetainer, root::adjust_index_for_rlp};
use serde::Serialize;

/// An ordered-trie inclusion proof: the trie key (`rlp(index)`), the leaf value, and the
/// root-to-leaf proof nodes. Verifiable with `alloy_trie::proof::verify_proof` against the
/// corresponding `transactionsRoot` / `receiptsRoot`.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HlInclusionProof {
    /// Position of the item within its (root-defining) list.
    #[serde(with = "alloy_serde::quantity")]
    pub index: u64,
    /// Trie key for the item: `rlp(index)`.
    pub key: Bytes,
    /// The leaf value (the item's canonical encoding).
    pub value: Bytes,
    /// Proof nodes from the root down to the leaf.
    pub proof: Vec<Bytes>,
}

/// Build an inclusion proof for `target` over the ordered trie of `items`, where each entry in
/// `items` is already encoded exactly as it contributes to the root.
///
/// Returns the proof together with the computed root, so the caller can assert it matches the
/// header root before serving. Returns `None` if `target` is out of range.
pub fn ordered_trie_inclusion_proof(
    items: &[Vec<u8>],
    target: usize,
) -> Option<(HlInclusionProof, B256)> {
    if target >= items.len() {
        return None;
    }

    let target_key = Nibbles::unpack(alloy_rlp::encode_fixed_size(&target).as_ref());
    let retainer = ProofRetainer::new(vec![target_key.clone()]);
    let mut hb = HashBuilder::default().with_proof_retainer(retainer);

    // Feed leaves in the RLP-adjusted order required for sorted-nibble insertion, matching
    // `ordered_trie_root_with_encoder`. The key is always `rlp(actual index)`.
    let len = items.len();
    for i in 0..len {
        let index = adjust_index_for_rlp(i, len);
        let index_buf = alloy_rlp::encode_fixed_size(&index);
        hb.add_leaf(Nibbles::unpack(index_buf.as_ref()), &items[index]);
    }

    let root = hb.root();
    let proof: Vec<Bytes> = hb
        .take_proof_nodes()
        .matching_nodes_sorted(&target_key)
        .into_iter()
        .map(|(_, node)| node)
        .collect();

    let proof = HlInclusionProof {
        index: target as u64,
        key: Bytes::copy_from_slice(alloy_rlp::encode_fixed_size(&target).as_ref()),
        value: Bytes::from(items[target].clone()),
        proof,
    };
    Some((proof, root))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_trie::{proof::verify_proof, root::ordered_trie_root_with_encoder};

    /// Distinct 40-byte values (long enough to hash into the trie like real tx/receipt entries).
    fn items(n: usize) -> Vec<Vec<u8>> {
        (0..n)
            .map(|i| {
                let mut v = vec![0u8; 40];
                v[0] = i as u8;
                v[1] = 0xab;
                v[39] = (i * 7) as u8;
                v
            })
            .collect()
    }

    #[test]
    fn inclusion_proof_verifies_against_root_for_every_index() {
        // Identity encoder: items are already-encoded bytes, matching the helper's add_leaf.
        for n in [1usize, 2, 3, 5, 16, 17, 130] {
            let its = items(n);
            let root = ordered_trie_root_with_encoder(&its, |it, buf| buf.extend_from_slice(it));
            for target in 0..n {
                let (p, computed_root) = ordered_trie_inclusion_proof(&its, target)
                    .unwrap_or_else(|| panic!("n={n} target={target}: None"));
                assert_eq!(computed_root, root, "n={n} target={target}: root mismatch");
                assert_eq!(p.value.as_ref(), its[target].as_slice());

                let key = Nibbles::unpack(alloy_rlp::encode_fixed_size(&target).as_ref());
                verify_proof(root, key, Some(its[target].clone()), p.proof.iter())
                    .unwrap_or_else(|e| panic!("n={n} target={target}: verify failed: {e:?}"));
            }
        }
    }

    #[test]
    fn out_of_range_target_returns_none() {
        assert!(ordered_trie_inclusion_proof(&items(3), 3).is_none());
        assert!(ordered_trie_inclusion_proof(&[], 0).is_none());
    }

    #[test]
    fn wrong_value_fails_verification() {
        // A proof must not verify against a different claimed value at the same key.
        let its = items(5);
        let root = ordered_trie_root_with_encoder(&its, |it, buf| buf.extend_from_slice(it));
        let (p, _) = ordered_trie_inclusion_proof(&its, 2).unwrap();
        let key = Nibbles::unpack(alloy_rlp::encode_fixed_size(&2usize).as_ref());
        assert!(verify_proof(root, key, Some(vec![0xde, 0xad]), p.proof.iter()).is_err());
    }
}
