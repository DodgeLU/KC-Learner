# Logical dataset identity

Two versions are fixed by this note.

* Row serialization: `canonical_interaction_hash_v1`
* Dataset identity: `dataset_logical_identity_v1`

A prepared dataset's logical identity depends on the raw-data identity, the dataset protocol, the preprocessing version, and the cohort and split rules. It does not depend on the model, Q versus QC, hyperparameters, optimizer, hidden size, epochs, training seed, CPU or CUDA, checkpoint, filesystem path, clock, or Parquet bytes.

## Row bytes

Each canonical interaction is one JSON object. Keys appear in this order:

`interaction_id`, `learner_id`, `item_id`, `correct`, `kc_ids`, `order`, `timestamp`, `bundle_id`, `split`, `metric_mask`.

Rules:

* UTF-8.
* No insignificant whitespace.
* One LF after every record, including the last. The hash does not use the platform newline.
* Strings use JSON escaping.
* `kc_ids` is a JSON array in stored order. It is not sorted and it is not a set.
* Integers are decimal. Booleans are `true` and `false`. Missing `split` or timestamp is `null`.
* A finite float timestamp that is an integer inside `±2^53` is written as that integer. Other finite floats use 17-digit scientific notation. NaN and infinities are rejected.
* The hasher does not sort rows. Order is part of the identity.

`source_row` and `solving_id` are not generic interaction fields. Public EdNet preparation is present. The EdNet adapter stores the source index in `interaction_id` and `solving_id` in `bundle_id`, so `dataset_logical_identity_v1` sees those values only through those two canonical fields. `ednet_source_provenance_hash_v1`, below, records `source_row` and `solving_id` beside that identity.

## Ordered identifier hash

Cohort ids, item vocabulary, and KC vocabulary use SHA-256 of the UTF-8 string formed by joining the ids with LF and writing no trailing LF. The helper does not sort. Vocabulary construction must sort before hashing: items by lexicographic id, KCs by ascending integer rendered as a decimal string. That hash is not the per-row KC-order hash.

## Dataset hash

`dataset_logical_hash` is the SHA-256 of one compact JSON object whose keys are:

`identity_version`, `serialization_version`, `dataset_id`, `protocol_id`, `preprocessing_version`, `cohort_hash`, `splits`, `item_vocabulary_hash`, `kc_vocabulary_hash`.

Each split object contains `split_id`, `logical_hash`, and `row_count`, in the protocol's split order. Null cohort or vocabulary hashes are JSON null and still affect the dataset hash.

Parquet SHA-256 and the pyarrow version may be stored beside this object as encoding provenance. They are not inputs to `dataset_logical_hash`.

## EdNet 50k sampler

`ednet_pilot50k_sampler_v1` draws 50,000 ids from a strictly sorted user-id list with seed 42, then sorts the draw. The selection loop is the CPython 3.11.7 `Random.sample` sequence algorithm. It does not call `random.sample`. The frozen sample hash and the dev5000 prefix hash are gates. A mismatch stops; it does not create another cohort. The 70/10/20 split does not use this seed.

## EdNet source provenance

`ednet_source_provenance_hash_v1` hashes corrected rows in stored order. Each line is `learner_id,source_row,item_id,solving_id,timestamp,split` plus LF. It is recorded beside `dataset_logical_identity_v1` and is not one of that object's hashed fields. Raw-source provenance records the ordered filename-population hash, an aggregate of `user_id<TAB>file_sha256` over the selected 50k files, and the SHA-256 of `questions.csv`. Paths are not stored.

## ASSISTments order

Eligible rows are ordered by `(student_id, timestamp, source_row)`. `source_row` is the 0-based index in the provider file. Equal timestamps therefore keep provider-file order. This is the same order as the previous stable `(student_id, timestamp)` sort of file-order rows. It does not change the 70/10/20 cut, bundle closure, vocabulary, metric mask, or family-specific OOV rules. `source_row` is preparation provenance and is not a field of `dataset_logical_identity_v1`.
