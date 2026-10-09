# Historical psychometric outputs

Historical predictive numbers are useful when the execution inputs match.
They are not an oracle for a dataset whose sequence or bundle rule was
corrected later.

## EdNet IRT and AR-KT

The frozen files

`project_c/final_valid_pt/EdNet/IRT/ednet_irt_dev5000/state_train_only.npz`

and

`project_c/final_valid_pt/EdNet/AR-KT/ednet_ar_kt_dev5000/state_train_only.npz`

were produced by `train_ednet_irt_dev5000.py` and the matching AR-KT
runner. Their console log and `result.json` record
`preprocessing_version = ednet_kt1_preproc_v1`, protocol
`CP3_EDNET_KT1_V0`, and the split directory
`ednet_kt1_pilot_v1_from_v2_split_v1`. That runner reads
`ednet_split_v1_{train,valid}.parquet`. It does not read the corrected
KC-Learner dataset.

Those split files are materialized from `part-shard*` pilot chunks
(`outputs/ednet_make_split.py`), not from raw KT1 user-file order.
The saved state has 189 concept columns. The corrected publication
vocabulary has 188 real KCs because standalone `tags="-1"` is a missing
KC. Corrected execution uses raw eligible source order and
`(learner_id, solving_id)` bundles.

Replaying those states on `ednet_kt1_corrected_v1` is therefore a
comparison of different execution inputs. The Part 6A VALID differences
are an expected consequence of that correction. They are not a reason
to restore the old order. The migrated control flow is still
TRAIN-only state, then a VALID replay gate, then explicitly authorized
TEST. Corrected EdNet does not claim historical TEST numerical parity.

## ASSISTments IRT and AR-KT

VALID replay from `state_train_only.npz` matches the frozen VALID
metrics exactly at the historical tolerance `1e-9`, including AR-KT
`p_global`.

TEST metric-eligible count is `84792` on both sides. OOV rows stay in
the IRT / AR-KT sequence with problem index `0` and stay out of the
metrics. DKT / DKVMN still drop those rows from state.

The public TEST bundle is the frozen protocol key
`(student_id, timestamp)`. Prepare writes that key into canonical
`bundle_id`, and `canonical_interaction_hash_v1` hashes `bundle_id`.
The integer `group_id` column is only the streaming scan key. It is
absent from the logical hash; equal `bundle_id` values are the
scientific group. The historical TEST scripts never set
`group_id`. `ARTKTModel.run_streaming` then uses the row index, so
each TEST row updates before the next row. Giving every TEST row its
own group id reproduces the frozen TEST metrics exactly, including
AR-KT headline and `p_global`. The first same-timestamp pair in the
prepared TEST split is index 269 (learner `27`, items `839` and `837`).
There are 375 such timestamps.

The public path keeps the protocol bundle. It does not copy the
historical row-index fallback. Aggregate TEST differences of about
`1e-6` are that bundle difference, not an OOV-mask or vocabulary
mismatch.
