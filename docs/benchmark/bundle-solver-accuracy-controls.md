# Bundle solver accuracy controls

This diagnostic separates inner solver accuracy from the added stereo image
rows in the [failed target-relative experiment](stereo-target-relative-bundle.md).
It does not change image measurements, Huber loss, variable scales, map gauge,
the 30-evaluation outer limit, or geometric acceptance limits. Ground truth
remains evaluator-only.

`--bundle-solver-accuracy default` preserves the previous behavior: ordinary
bundle adjustment uses the existing inner defaults, while valid owned image
augmentation already requests precise LSMR. `precise` applies the same
`atol=btol=1e-12` and `maxiter=max(500, variable_count)` to ordinary bundle
adjustment too. More precise steps can change poses and increase runtime.

The requested policy is saved in configuration and result identities. Each
bundle report declares its effective policy and inner options; a skipped solve
reports that it did not run. Precise results cannot reuse older results with
missing policy metadata. Existing historical results remain separate.

The first comparison uses fresh KITTI 04 frames 0–79 for three declared cases:

| Case | Owned image rows | Requested solver policy |
| --- | --- | --- |
| Default control | Off | default |
| Precision control | Off | precise |
| Augmentation control | On | precise |

All cases use stereo, CUDA matching, current retrieval, one OpenCV thread,
verified-fallback depth, stereo pose arbitration and raw-reference retry,
with loops off. No diagnostic cache or per-sequence parameter selection is
used. Compare the precision control with the default control, then the
augmentation control with the precision control. Later map states differ;
their bundle objectives are not interchangeable accuracy scores.

Use `scripts/evaluate_shared_slam.py` with calibrated KITTI image and pose roots,
`--stereo --sequence 04 --max-frames 80 --loop-mode off --matching-backend cuda
--retrieval current --opencv-threads 1 --stereo-depth-policy verified_fallback
--stereo-pose-arbitration --stereo-raw-reference-retry`, and a separate output
directory per case. Set the requested solver policy above and add
`--stereo-owned-image-bundle` only for the augmentation case. Include a
wall-clock deadline. Reference poses are opened after estimator shutdown.

Testing and fresh evaluation are pending for this revision. The control remains
experimental; full validation is paused and main is unchanged.
