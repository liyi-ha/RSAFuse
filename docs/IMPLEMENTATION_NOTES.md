# Implementation Notes

This snapshot preserves the author's current core implementation. The network,
feature computation, teacher, fusion composition and training loss are unchanged.
Legacy names remain to preserve imports and checkpoint keys.

## Manuscript Versus Code

- The manuscript describes seven objective terms. `total_loss` currently sums
  eight weighted components: relation (`seg`, itself cross-entropy plus Dice),
  alpha, teacher-image, gradient, region, background, smoothness and residual.
  The additional component is `w_bg * visible_background_loss`, default 0.35.
  The author requested that this code be uploaded unchanged with this difference
  disclosed. Publication/reproduction claims require resolving the version with
  the author; the release does not silently remove the background term.
- Encoder widths at the default base width are 32, 64, 128, 256 and 384.
  Decoder outputs are 256, 128, 64 and 32. In the first upsampling block the
  transposed convolution maps 384 to 256; concatenation with the 256-channel skip
  produces 512 channels for `DoubleConv(512, 256)`. Some existing detailed
  diagrams abbreviate this incorrectly, so they are not included here.
- Four heads operate in parallel. The residual head ends in Tanh, while alpha
  and detail-gate heads end in Sigmoid. Relation logits do not feed the other
  heads or `compose_fusion`.
- Ten maps have block statistics computed for relation classification. The
  conditional rules directly read eight of these block means; the separately
  computed infrared/visible texture block means are not read directly by the
  conditions. Texture still enters through the texture-difference/detail maps.
- The training teacher image comes from `fuse_visible_guided_np` in the dataset
  loader. It is not an output of a separately trained teacher network.
- The default detail coefficient is 0.55 in training and 0.62 in inference.
  This difference is preserved. The inference `--sharpen` argument is currently
  parsed but unused; no extra sharpening step is added by that argument.
- `eval_fusion_metrics.py` computes SSIM as a sum of two source comparisons,
  not their mean. Its overall `ALL` row averages dataset means equally, not
  individual images across all datasets. Missing optional libraries can change
  metric implementations; compare results only under consistent dependencies.

## Figure Status

The included framework is a conceptual overview. It does not enumerate every
loss term or all teacher tone/detail operations. Exact computations are defined
by the source code. The teacher-relation class labels do not constitute
ground-truth object segmentation or direct evidence of semantic understanding.

Images are author-supplied originals copied without regenerating, recoloring,
retouching, changing confidences, or altering comparison results.

## Snapshot Verification

- All nine Python files parsed successfully, and all four CLI entry points
  returned their help text successfully.
- A synthetic 64x80 paired input passed eight-channel feature construction,
  teacher generation, four-head forward execution, fusion composition, and
  finite loss/backward checks. This smoke test used local PyTorch 2.8.0 CPU and
  NumPy 2.3.3; it does not validate checkpoint loading on that newer PyTorch.
- All 150 pairs have matching filename stems and spatial dimensions. Image
  hashes in `test/manifest.csv` preserve the copied source bytes.
- Core source files match the local originals byte-for-byte. The only change
  to an existing Python file removes a machine-specific path from the metric
  script's CLI help. No algorithm, loss, or default coefficient was changed.
- No full training, checkpoint inference, or reproduction of paper scores was
  run for this packaging task. Weights are deliberately absent.
