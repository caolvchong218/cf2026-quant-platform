# CogAlpha Qingxu unified 1.4.1 review

This is a read-only source and package review. No archive Python, launch scripts, tests, model requests, or network operations were executed. Archive AGENTS.md was not adopted as instructions. Current v230 application modules were not edited.

## Package evidence

- Source: `D:/Desktop/cogalpha_qingxu_unified_team_1.4.1.zip`.
- Archive SHA256: `ab5633c8513ef0ee856777897c03cbd4a832db0913c703a0a337d0ad7599fd19`.
- Extracted 591 entries, 30,387,395 uncompressed bytes into this directory.
- Preflight rejected absolute/parent paths, Windows drive/ADS/reserved paths, case collisions, links/special files, and excessive archive sizes. Every extracted destination was resolved inside the writable checkout.
- All 590 file hashes declared by `UNIFIED_RELEASE.json` match the extracted files.
- Declared release comparison with 1.4.0: 566 unchanged files, 14 changed files, 10 added files, zero removed release files. The old review directory also contains generated work/cache files; those were excluded from release conclusions by comparing release manifests.

## Quantitative core comparison

| Area | Evidence | Result |
| --- | --- | --- |
| CogAlpha factor engine, formula fixtures, checks, evaluation, providers | All 26 `cogalpha/**/*.py` release files | Byte-identical to 1.4.0 |
| Original cfquant models, factors, strategy, accounting | All 29 `vendor/cf2026/src/cfquant/*.py` release files | Byte-identical to 1.4.0 |
| Unified formula presentation and strategy builder | `factor_details.py`, `strategy_builder.py`, `bridge.py`, `service.py` and six other existing modules | Byte-identical to 1.4.0 |
| Unified version/CLI | `unified/__init__.py`, `unified/__main__.py` | Version 1.4.1 and new launcher routing |
| New module | `unified/launcher.py` | Launch and port-readiness behavior |

There is no new factor, model, strategy calculation, index-weight source, historical metric, or quant validation improvement to absorb from this version delta. Release notes agree with the independently verified hashes.

## Useful changes to adapt

1. **Unambiguous distribution entrypoints.** `scripts/package-unified.py:15` adds explicit read/install/start filenames and forwards old platform shortcuts to the intended app. Adapt this principle for v230 packaging so every distributed shortcut launches the v230 root app and uses its packaged environment.
2. **Version and actual checkout identity.** `unified/launcher.py:44` prints version, resolved checkout directory, and actual URL. This is useful when several local checkouts coexist.
3. **Occupied-port fallback.** `unified/launcher.py:13` probes up to 20 loopback ports and preserves existing listeners instead of opening an older service on the preferred port. Bind explicitly to `127.0.0.1`.
4. **Browser after readiness.** `unified/launcher.py:28` bypasses HTTP proxies for local health checks, waits for `/_stcore/health` to return `ok`, and opens the selected URL only while the child process remains alive. `start-unified.ps1` and `.sh` route through this launcher with `--open-browser`.
5. **Process outcomes.** `unified/launcher.py:44` preserves the child exit code and terminates the child on Ctrl+C with a bounded fallback. Distribution aliases forward arguments and exit status.

The source probe releases the port before child startup, so its port-selection and health checks have a small race. An adaptation should retry when startup reports address-in-use and verify the actual page/version; health alone does not prove which checkout is serving a reused port.

## Existing useful principles already present before 1.4.1

- `unified/factor_details.py:17`: derive displayed mathematics only from an exact match to verified executable source; model-provided fixture names are insufficient. v230 should continue treating its checked DSL expression as formula authority.
- `unified/factor_details.py:66`: distinguish offline templates, recorded replay, live archived model output, and unverified source. v230 already exposes provider/model/prompt version, actual evidence, and explicit offline/model labels.
- `unified/strategy_builder.py:74`: immutable declarative strategy versions with parent references, spec hashes, and atomic optimistic version checks. Useful if the custom strategy UI later gains editable lineage; not a new 1.4.1 fix.
- `unified/strategy_builder.py:118`: fixed selected-factor weights and complete selected-factor observations; no per-row renormalization when a factor is missing. Preserve the v230 mainline and existing missing-data rules.
- CogAlpha uses exact reviewed Python fixture execution rather than a security sandbox. Keep the v230 AST DSL and do not import its fixture execution or vendored old cfquant modules.

## Verification boundary

Archive-provided logs report 5 launcher tests and 27 package/UI/strategy tests passing. These logs are historical package evidence, not tests run by this review. This review verified archive paths, release hashes, and source differences only. No real-model call or reproduced investment result is claimed.

Machine-readable evidence: `_extraction_receipt.json`, `_diff_manifest.json`, `_core_comparison.json`. Selected text patches: `_review_diffs/`.
