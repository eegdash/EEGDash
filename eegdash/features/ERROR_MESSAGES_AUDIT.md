# Error Messages Audit — `eegdash/features`

Catalogue of all raised errors and assertions in the features module,
with diagnosis and improvement suggestions.

---

## `base_utils.py`

### E01 — `channel_names_to_indices` · line 124 · `ValueError`

| Field | Content |
|---|---|
| **Current message** | `Channel {channel} not found in metadata channels: {ch_names}.` |
| **Reason** | A user-supplied channel name is not present in the recording's channel list. |
| **Problem** | Dumps the full `ch_names` list, which can be very long. Doesn't hint at near-matches or tell the user which argument triggered the error. |
| **Suggestion** | Include the index of the bad channel in the input list, suggest close matches (e.g. via `difflib.get_close_matches`), and cap the display of `ch_names` to the first N entries. Example: `Channel 'Fp1' (index 2 in the supplied list) not found in the recording's channel list. Did you mean: ['FP1', 'fp1']? (showing first 10 of 64 channels: ['Fz', 'Cz', ...])` |
| Decision | include index. suggest close matches. instead of ("showing first ...") - write something like ("Note: you can use  <func name or line> to see al channels") with a relevant func/line. |
---

## `decorators.py`

### E02 — `metadata_preprocessor` · line 358 · `TypeError`

| Field | Content |
|---|---|
| **Current message** | `{f.__name__} cannot be set as a metadata preprocessor because it does not get a keyword argument named ``'_metadata'``" |
| **Reason** | The `@metadata_preprocessor` decorator was applied to a function that lacks a `_metadata` keyword argument. |
| **Problem** | Message is correct but shows the raw internal attribute name (`_metadata`) without context about what it must do or how to add it. Also buries the function name in a long sentence. |
| **Suggestion** | Lead with the function name, state the requirement clearly, and show a minimal fix snippet. Example: `@metadata_preprocessor requires '{f.__name__}' to accept a keyword argument '_metadata: dict'. Add it to the function signature, e.g.: def {f.__name__}(..., *, _metadata: dict): ...` |
| decision | keep the suggestion |

### E03 — `_preprocessor_output_type_wrap` · line 393 · `ValueError`

| Field | Content |
|---|---|
| **Current message** | `` `output_type` must inherit from `BasePreprocessorOutputType`, got `{output_type}`. `` |
| **Reason** | `@preprocessor_output_type` received a class that is not a valid subclass, or was passed a non-class entirely. |
| **Problem** | Doesn't tell the user whether the failure was "not a class at all", "is the abstract base itself", or "is an unrelated class". |
| **Suggestion** | Branch the message on the failure mode: `output_type must be a concrete subclass of BasePreprocessorOutputType (not the base class itself). Got: {output_type!r} (type: {type(output_type).__name__}).` |
| decision | keep the suggestion |

---

## `extractors.py`

### E04 — `_validate_execution_tree` · line 437 · `TypeError`

| Field | Content |
|---|---|
| **Current message** | `Feature '{fname}: {child}' cannot be a child of {parent}` |
| **Reason** | A feature's declared `parent_extractor_type` list does not include the preprocessor that is actually upstream of it in the `FeatureExtractor`. |
| **Problem** | The message only names the mismatch pair; it does not tell the user what the feature *does* expect, or how the tree was constructed. Debugging requires reading source code. |
| **Suggestion** | Include the expected predecessors and the actual parent: `Feature '{fname}' (function: {child}) expects one of {[_get_func_name(p) for p in pe_type]} as its preprocessor, but got '{parent}'. Check that the 'preprocessor' argument of the FeatureExtractor matches the @feature_predecessor declaration of '{child}'.` |
| decision | keep the suggestion, it is very good |

### E05 — `__getitem__` · line 661 · `ValueError` (non-string key)

| Field | Content |
|---|---|
| **Current message** | `Non-string keys are supported only for direct keys.\nKey {key} is not a direct key of the FeatureExtractor.\nPossible direct keys are: {self.feature_extractors_dict.keys()}` |
| **Reason** | A non-string key was passed to `__getitem__` but it is not a direct key of the extractor. |
| **Problem** | The phrase "Non-string keys are supported only for direct keys" is hard to parse. The hint about "direct keys" vs nested string lookups is useful but not explained. |
| **Suggestion** | `FeatureExtractor.__getitem__: key {key!r} (type {type(key).__name__}) is not a direct key. Non-string keys can only be used for exact direct keys. Use a string key to access nested features. Direct keys available: {list(self.feature_extractors_dict.keys())}` |
| decision | the new explanation is better but still confusing ("Non-string keys can only be used for exact direct keys"), we need better phrasing |

### E06 — `__getitem__` · line 683 · `ValueError` (string key not found)

| Field | Content |
|---|---|
| **Current message** | `Key {key} not found in FeatureExtractor.\nPossible direct keys are: {self.feature_extractors_dict.keys()}` |
| **Reason** | A string key could not be matched in the extractor or any nested extractor. |
| **Problem** | Only shows direct keys, not the full set of available (nested) feature names, so the user doesn't know what the valid nested keys look like. |
| **Suggestion** | List both direct keys and fully-qualified nested feature names: `Key '{key}' not found. Direct keys: {list(self.feature_extractors_dict.keys())}. All available feature names (including nested): {self.feature_names}.` |
| decision | seems good, but see e05 and make sure they are compatible |

### E07 — `utils._get_feature_extractor_from_parameter` · line 128 · `TypeError`

| Field | Content |
|---|---|
| **Current message** | `The given FeatureExtractor cannot process raw signals.\nPlease make sure the possible parent types of its preprocessor/features include SignalOutputType or a subclass of it.` |
| **Reason** | The outermost `FeatureExtractor` has a preprocessor that requires pre-processed input, so it cannot be used directly on raw EEG signals in `extract_features`. |
| **Problem** | Abstract guidance; doesn't say which child failed or name its declared predecessors. The original `TypeError` from `_validate_execution_tree` is chained (`from e`) but not surfaced in the message. |
| **Suggestion** | Re-raise with the inner error's details prominently: `extract_features requires a FeatureExtractor whose top-level input is raw EEG (SignalOutputType). The validation failed because: {e}. Wrap your extractor in another FeatureExtractor without a preprocessor, or verify @feature_predecessor annotations.` |
| decision | keep suggestion |

---

## `trainable.py`

### E08 — `TrainableFeature.__call__` · line 82 · `RuntimeError`

| Field | Content |
|---|---|
| **Current message** | `{self.__class__} cannot be called, it has to be trained first.` |
| **Reason** | A trainable feature was called before `fit()` was invoked. |
| **Problem** | Uses `{self.__class__}` which prints the raw class object (e.g. `<class 'CSPFeature'>`), not the class name. Also gives no actionable hint about how to train. |
| **Suggestion** | Use `type(self).__name__` for readability and point to the training API: `'{type(self).__name__}' must be fitted before use. Call fit_feature_extractors(dataset, features) first, or manually call .partial_fit(...) for each batch then .fit().` |
| decision | keep suggestion, but remove the "or manually..." ending |

---

## `serialization.py`

### E09 — `_func_from_dict` · line 173 · `ValueError`

| Field | Content |
|---|---|
| **Current message** | `feature or preprocessor named \`{func_dict['name']}\` not found in feature bank.` |
| **Reason** | A function name stored in a serialized config (JSON/YAML/HOCON) does not exist in `feature_bank.__all__`. Usually means a stale config or a renamed function. |
| **Problem** | No hint about what *is* available, or whether the name is close to an existing one. |
| **Suggestion** | Include close matches and the total count: `Feature/preprocessor '{func_dict['name']}' not found in feature_bank. This may mean the config was saved with a different version of eegdash. Close matches: {difflib.get_close_matches(func_dict['name'], feature_bank.__all__) or 'none'}. Available names ({len(feature_bank.__all__)} total): {feature_bank.__all__}.` |
| decision | Feature/preprocessor '{func_dict['name']}' not found in feature_bank. Check the configuration file. Close matches: {difflib.get_close_matches(func_dict['name'], feature_bank.__all__) or 'none'}. |

### E10 — `feature_extractor_from_dict` · line 253 · `ValueError`

| Field | Content |
|---|---|
| **Current message** | `Feature {k}: A feature dict must contain either a 'feature_extractors' field (for \`FeatureExtractor\`)or a 'name' field (for a function), got {v.keys()}.` |
| **Reason** | A dict entry in the serialized config is malformed — has neither the expected `feature_extractors` nor `name` key. |
| **Problem** | Missing space before "or". Doesn't tell the user in which file or at which nesting level this key appeared, making it hard to find in a large YAML/JSON. |
| **Suggestion** | Fix typo, add nesting context: `Malformed entry for feature key '{k}': expected either a 'name' key (plain function) or a 'feature_extractors' key (nested FeatureExtractor), but got keys: {list(v.keys())}. Check your config file at the '{k}' entry.` |
| decision | keep suggestion |

---

## `datasets.py`

### E11 — `FeaturesConcatDataset.metadata` · line 460 · `TypeError`

| Field | Content |
|---|---|
| **Current message** | `Metadata dataframe can only be computed when all datasets are FeaturesDataset.` |
| **Reason** | The concat dataset contains non-`FeaturesDataset` entries (e.g. raw `WindowsDataset`). |
| **Problem** | Doesn't identify which datasets are the offenders or what types they are. |
| **Suggestion** | List the offending indices and their types: `All datasets must be FeaturesDataset to access .metadata. Non-conforming datasets: {[(i, type(ds).__name__) for i, ds in enumerate(self.datasets) if not isinstance(ds, FeaturesDataset)]}.` |
| decision | keep suggestion, its good |

### E12 — `FeaturesConcatDataset.save` · line 531 · `ValueError`

| Field | Content |
|---|---|
| **Current message** | `Expect at least one dataset` |
| **Reason** | `save()` was called on an empty `FeaturesConcatDataset`. |
| **Problem** | Terse, no capitalisation, no period. Doesn't name the method or give context. |
| **Suggestion** | `FeaturesConcatDataset.save: cannot save an empty dataset (self.datasets is empty). Make sure feature extraction was produced for at least one recording.` |
| decision | keep suggestion |

### E13 — `FeaturesConcatDataset.save` · line 543 · `FileExistsError`

| Field | Content |
|---|---|
| **Current message** | `Subdirectory {sub_dir} already exists. Please select a different directory, set overwrite=True, or resolve manually.` |
| **Reason** | A target subdirectory already exists and `overwrite=False`. |
| **Problem** | Good message overall, but doesn't mention the dataset index that conflicts, so when saving many recordings it's unclear which one caused the stop. |
| **Suggestion** | `Subdirectory '{sub_dir}' (dataset index {i_ds + offset}) already exists. To overwrite, pass overwrite=True. To save elsewhere, change the path argument. To resume a partial save, use the offset parameter.` |
| decision | keep suggestion (claude - verify that 'offset' is really a variable here) |

### E14 — `FeaturesConcatDataset._enforce_inplace_operations` · line 983 · `ValueError`

| Field | Content |
|---|---|
| **Current message** | `{func_name} only works inplace, please change to inplace=True (default).` |
| **Reason** | The caller passed `inplace=False` to a method that only supports in-place modification. |
| **Problem** | Lowercase start, no period. Phrasing "please change to inplace=True (default)" is confusing — if `True` is the default, why did the user get the error? |
| **Suggestion** | `{func_name}() operates in-place and does not support inplace=False. Remove the inplace argument or set it to True.` |
| decision | keep suggestion |

---

## `feature_bank/csp.py`

### E15 — `CSPFeature.__call__` · line 286 · `RuntimeError`

| Field | Content |
|---|---|
| **Current message** | `CSP weights selection criterion is too strict,all weights were filtered out.` |
| **Reason** | After applying `n_select` or `crit_select`, no CSP filter weights remain. |
| **Problem** | Missing space after the comma. Doesn't report the actual eigenvalues or tell the user what threshold was used vs what was available. |
| **Suggestion** | `CSP filter selection is too strict: 0 filters remain. Got n_select={n_select}, crit_select={crit_select}, but only {self._weights.shape[-1]} filters are available with eigenvalues {np.round(self._eigvals, 3).tolist()}. Relax the selection threshold.` |
| decision | i don't understand this enough - leave for Aviv's review |

---

## `feature_bank/utils.py`

### E16 — `get_valid_freq_band` · lines 237, 241 · `AssertionError`

| Field | Content |
|---|---|
| **Current message** | *(bare assertion — no message)* |
| **Reason** | `f_min` is below the minimum resolvable frequency, or `f_max` exceeds the Nyquist limit. |
| **Problem** | `AssertionError` with no message is completely opaque to users. They see only a line number and have no idea what constraint was violated. |
| **Suggestion** | Replace `assert` with `ValueError` and a descriptive message: `f_min={f_min} Hz is below the minimum resolvable frequency f0={f0:.3f} Hz (= 2*fs/n = 2*{fs}/{n}).` / `f_max={f_max} Hz exceeds the Nyquist frequency {f1} Hz (= fs/2 = {fs}/2).` |
| decision | split this into two errors - one for f_min and onr for f_max |

### E17 — `reduce_freq_bands` · lines 351–353 · `AssertionError`

| Field | Content |
|---|---|
| **Current message** | *(bare assertions — no message)* |
| **Reason** | A band name is not a string, a band limit tuple has the wrong length or `min > max`, or the requested limits fall outside the available frequency vector. |
| **Problem** | Three separate bare `assert` statements; any one of them produces the same unhelpful `AssertionError` with no context. |
| **Suggestion** | Replace with named `ValueError`s per check, e.g.: `Band key {k!r} must be a string, got {type(k).__name__}.` / `Band '{k}' limits must be a 2-tuple (f_min, f_max) with f_min <= f_max, got {lims}.` / `Band '{k}' limits ({lims[0]}, {lims[1]}) are outside the available frequency range [{f[0]}, {f[-1]}] Hz.` |
| decision | keep suggestion, split it into 3 different errors |

---

## `feature_bank/csp.py` (assertions)

### E18 — `CSPFeature._update_labels` · line 138 · `AssertionError`

| Field | Content |
|---|---|
| **Current message** | *(bare assertion — no message)* |
| **Reason** | More than two unique class labels were seen across partial fits. CSP only supports binary classification. |
| **Problem** | A bare `assert` on `self._labels.shape[0] < 3` is silent. The user has no idea why it failed or what constraint exists. |
| **Suggestion** | Replace with `ValueError`: `CSP requires exactly two classes, but {self._labels.shape[0]} unique labels have been encountered so far: {self._labels.tolist()}. Ensure y contains only two distinct class values.` |
| decision | keep suggestion |

---

## `utils.py`

### E19 — `_get_feature_extractor_from_parameter` inner block · inferred from grep

> *(The `TypeError` re-raise at `utils.py:128` was already covered in E07 above since it lives in `utils.py`'s version of `_get_feature_extractor_from_parameter`.)*

---

## `datasets.py` (warnings)

Both warnings are emitted inside `FeaturesConcatDataset.save()`, after the main save loop completes.

### W01 — `FeaturesConcatDataset.save` · line 555 · `logger.warning`

| Field | Content |
|---|---|
| **Current message** | `The number of saved datasets ({i_ds + 1 + offset}) does not match the number of existing subdirectories ({n_sub_dirs}). You may now encounter a mix of differently preprocessed datasets!` |
| **Trigger condition** | `overwrite=True` AND the number of datasets just written is fewer than the pre-existing subdirectory count. Old numbered subdirectories survive on disk. |
| **Problem** | Describes the symptom (count mismatch) but not the cause (leftover dirs from a larger previous save) or which specific subdirectories are stale. "Mix of differently preprocessed datasets" is vague. |
| **Suggestion** | Name the leftover indices and tell the user what to do: `overwrite=True saved {i_ds + 1 + offset} dataset(s), but {n_sub_dirs} subdirectories existed before. Subdirectories {list(range(i_ds + 1 + offset, n_sub_dirs))} were not overwritten and may contain stale data. Delete them manually or re-save to a fresh directory.` |
| decision | keep suggestion |

### W02 — `FeaturesConcatDataset.save` · line 563 · `logger.warning`

| Field | Content |
|---|---|
| **Current message** | `Chosen directory {path} contains other subdirectories or files {path_contents}.` |
| **Trigger condition** | After the save loop, `path_contents` still has entries — files or directories in the target folder that are not numbered dataset subdirectories. |
| **Problem** | Dumps the raw `path_contents` list without explanation. The user doesn't know if these are harmless (e.g. `.gitignore`) or dangerous (stale subdirs from a different run). No action is suggested. |
| **Suggestion** | Clarify what "other" means and give a concrete action: `Directory '{path}' contains {len(path_contents)} unrelated file(s)/subdirectorie(s) not written by this save: {path_contents}. These will be ignored by load_features_concat_dataset() but may indicate a wrong target directory — move or delete them if unintended.` |
| decision | keep suggestion |

---

## Summary Table

| ID | File | Line | Type | Priority |
|---|---|---|---|---|
| E01 | `base_utils.py` | 124 | `ValueError` | High — common user error |
| E02 | `decorators.py` | 358 | `TypeError` | Medium — developer-facing |
| E03 | `decorators.py` | 393 | `ValueError` | Medium — developer-facing |
| E04 | `extractors.py` | 437 | `TypeError` | High — frequent during setup |
| E05 | `extractors.py` | 661 | `ValueError` | Low — rare edge case |
| E06 | `extractors.py` | 683 | `ValueError` | High — common during exploration |
| E07 | `utils.py` | 128 | `TypeError` | High — common user mistake |
| E08 | `trainable.py` | 82 | `RuntimeError` | High — common user mistake |
| E09 | `serialization.py` | 173 | `ValueError` | High — hits on config mismatch |
| E10 | `serialization.py` | 253 | `ValueError` | Medium — bad config |
| E11 | `datasets.py` | 460 | `TypeError` | Medium |
| E12 | `datasets.py` | 531 | `ValueError` | Low |
| E13 | `datasets.py` | 543 | `FileExistsError` | Medium |
| E14 | `datasets.py` | 983 | `ValueError` | Low |
| E15 | `feature_bank/csp.py` | 286 | `RuntimeError` | High — silent typo + no context |
| E16 | `feature_bank/utils.py` | 237/241 | `AssertionError` | High — bare assert |
| E17 | `feature_bank/utils.py` | 351–353 | `AssertionError` | High — bare asserts |
| E18 | `feature_bank/csp.py` | 138 | `AssertionError` | High — bare assert |
| W01 | `datasets.py` | 555 | `logger.warning` | Medium — stale dirs can corrupt reloads |
| W02 | `datasets.py` | 563 | `logger.warning` | Low — wrong-dir mistakes are common |
