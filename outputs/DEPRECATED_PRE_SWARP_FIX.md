# Deprecated pre-Swarp-fix artifacts

Hard rule: Do not cite, compare, publish, or use these pre-fix artifacts for
physical conclusions.

These outputs were generated before the `compact_sigmoid` orientation fix in
`warpfactory/utils/helpers.py`.

They should not be used as physical evidence against the Fuchs W1
`vWarp=0.02` recipe. The old runs used an inverted effective shift profile:
inside the transition they evaluated `Swarp ~ sig` instead of the intended
`Swarp ~ 1 - sig`.

## Reason for deprecation

At `r ~= 11.7004`, the old profile produced:

```text
S_eff ~= 0.045
g01   ~= -9.07e-4
```

After the fix, the same point produces:

```text
S_eff ~= 0.968
g01   ~= -1.94e-2
```

That difference changes the effective W1 shift profile and invalidates the old
negative energy-condition conclusions.

## Valid replacement result

Post-fix W1 original, `vWarp=0.02`, `christoffel`, `num_vecs=40`,
`num_time_vecs=10`:

```text
NEC min = 1.365148e39, frac = 0
WEC min = 1.365148e39, frac = 0
SEC min = 9.757740e38, frac = 0
DEC min = 3.744716e39, frac = 0
```

Known affected examples include directories/files with names such as:

- `codex_final_evidence_original_n4`
- `codex_final_evidence_original_n40`
- `codex_final_evidence_threshold_n40`
- `codex_paper_regression_original_v002`
- `codex_paper_regression_original_v002_n40`
- `codex_w1_critical_frame_tensor_audit`
- `codex_w1_critical_frame_tensor_audit_inner`
- `fuchs-original-shift-20260512-*`

Use post-fix artifacts instead, especially:

- `codex_final_evidence_after_swarp_fix_n4`
- `codex_final_evidence_after_swarp_fix_n40`
- `codex_paper_regression_after_swarp_fix_v002_n40`
- `codex_w1_shift_profile_after_fix`
- `codex_w1_critical_frame_tensor_after_swarp_fix`

These artifacts may still be useful historically for debugging the inverted
`Swarp` profile, but they must be treated as invalid for physics/regression
claims about the Fuchs W1 `vWarp=0.02` metric.
