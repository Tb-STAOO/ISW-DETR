# ISW-DETR training, comparator and statistical protocol

This note accompanies manuscript revision 20261102_01. It retains the detailed settings underlying the compressed Methods sections. It documents existing experiments; no model, input, parameter value or result was changed to prepare this note. `ORIGINAL_METHODS_English.tex` and `ORIGINAL_METHODS_Chinese.tex` preserve the preceding complete Methods, and `MOVED_EQUATIONS_English.tex` / `MOVED_EQUATIONS_Chinese.tex` retain the exact comparator expressions moved out of the main text.

## 1. Base visual training

| Setting | Recorded value |
| --- | --- |
| Dataset and input | ScaleVision-MM; 640 × 640 |
| Base-detector warm-start stage | 90 epochs |
| Classification / localization losses | Varifocal / L1 and generalized IoU |
| Optimizer | AdamW |
| Initial learning rates | 10^-4 for both the backbone and other trainable parameters |
| Adam betas / weight decay | (0.9, 0.999) / 10^-4 |
| Batch size / gradient clipping | 8 / 0.1 |
| Precision and weight averaging | Mixed-precision training; EMA decay 0.9999 |
| Online augmentation | Photometric distortion p=0.5; random zoom-out; random IoU crop p=0.8; horizontal flip p=0.5; box sanitization and resizing |
| Validation preprocessing | Resizing only |
| Training seeds | 42, 3407 and 2026 |

Inference variants belonging to the same seed share the corresponding base-detector weights. Checkpoint identifiers, stage logs and the available training-provenance record remain with the existing resources. The main-text learning-rate statement uses the actual equal backbone/other learning rates, preserving the previously corrected training description.

## 2. Auxiliary identity–state branch

Each seed starts from its highest-validation-AP visual checkpoint. Original visual, leaf-classification-head and localization parameters remain frozen while the added classification components train. The joint objective is the unchanged `eq:is-objective` in the main text: final-leaf Varifocal loss plus 0.2 times the identity loss and 0.1 times the state loss. The auxiliary losses are mean cross-entropies over Hungarian-matched foreground queries. State supervision is restricted to identities with paired states. Matching uses the frozen visual logits and boxes.

| Setting | Recorded value |
| --- | --- |
| Maximum added-branch training | 30 epochs |
| Learning rate / batch size | 10^-4 / 8 |
| Warm-up | First 100 iterations |
| Learning-rate reduction | Tenfold at epoch 22 |
| Validation early stopping | After epoch 12; patience 7 |
| Photometric distortion / horizontal flip | p=0.5 / p=0.5 |

## 3. Selection and fixed refinement settings

Model checkpoints and fusion coefficients are selected using validation AP. The original configuration comparison retained state-mean association and alpha_r=0.5; this setting is kept for internal, directly measured and public evaluation and for the ablation without the auxiliary branch. The expanded validation scan evaluates alpha_r={0,0.1,…,1} with the already trained models fixed. It contains eleven evaluated settings; connecting lines are visual guides. It is not used to replace the fixed 0.5 setting in the other comparisons.

Fusion comparisons use the shared training-fitted class-mass prior. Candidate scope follows the respective method definitions. Complete 300-query outputs, sigmoid top-300 postprocessing and the corresponding seed's original boxes are retained.

## 4. Gaussian, MAP-style and CVSR controls

On ScaleVision-MM, Gaussian scene fusion adds scene evidence to every auxiliary-branch query: z_i^G = z_i + lambda_G gamma a. Validation selects lambda_G=1.5 from {0,0.125,0.25,0.5,1,1.5,2,3}. This all-query operation is distinct from the candidate-restricted public scene control, whose coefficient is 2.

The CVSR reference uses the original visual logits for association and omits the final state-contrast reduction. It retains candidate restriction, transaction verification, beta=2 and eta=0.2. It is the direct within-framework ablation of the state-contrast treatment, rather than another visual baseline.

The MAP-style comparator combines log visual class probabilities and log record-mass probabilities in its association utility. It uses the common candidate set, transaction gate and Hungarian solver. Final scores add scene support 2a and, for an assigned query, alpha × clip{log p_w(c|e)+log C,−6,6}. Validation selects alpha=0.125 using the same coefficient grid as the Gaussian control. The MAP comparison therefore changes both association utility and record-score refinement. See `tools/run_private_controls.py` for the implementation.

## 5. Benefit-guided query routing

BQR is a study-defined learned control. It regulates how far a visual score follows the corresponding fixed CVSR proposal. The routing network has dimensions 20→32→16→1 and SiLU activations. Its features describe visual confidence, proposal disagreement, assignment margins and mass compatibility. Validation selects lambda=0.5 for each seed.

Scene-averaged classification, benefit, background and clean-consistency losses have weights 1,0.5,0.1,0.02. Benefit targets are 0.9 for a class correction, 0.1 for a class harm, and 0.5 when both visual output and proposal are correct; remaining targets use clipped visual-versus-proposal loss differences. Background supervision applies only to unmatched candidates whose maximum IoU with every ground-truth box is below 0.1. Duplicate-object queries are therefore not automatically background examples.

Class-logit perturbations provide additional training hypotheses while boxes and weight records remain fixed. BQR training uses AdamW, learning rate 10^-3, weight decay 10^-4, batch size 64 and 30 epochs, totaling 210 updates. The exact feature definitions, soft-target construction and loss normalization remain in the packaged implementation.

## 6. Confidence-guided control

A shared 20→32→16 representation feeds two sigmoid heads estimating visual and CVSR-proposal confidence. For a matched foreground query with true label y and logit vector u, the detached target is softmax(u)_y. For a true-background query it is exp{−C^-1 × sum_c log(1+exp(u_c))}. Squared confidence errors replace the benefit loss. The raw routing strength is 2h_i^C/(h_i^v+h_i^C), used in the same refinement path.

Validation selects lambda=0.5,1,1 for seeds 42, 3407 and 2026, respectively. Data order, perturbations, optimizer, update count and validation-search budget match BQR. This matches training conditions without asserting identical network heads or identical supervision.

## 7. Conditional and general score-transform controls

Association compares original, state-group mean and state-group maximum scores. Classification compares half-contrast shrinkage, group-mean replacement and group-maximum replacement. Mean/max replacement substitutes every state score by its identity group's mean/maximum at the indicated stage. Half contrast retains a coefficient of 0.5.

The no-auxiliary-branch control uses the original flat logits and their visually selected candidates. Additional shared-pool comparisons separate representation effects from candidate-membership effects. These controls keep the mass evidence and box coordinates fixed.

Global score scaling uses z*=g z for g in {0.5,0.75,1.25,1.5}. The LSE-preserving half-contrast control starts with Tz=(z+Pz)/2 and adds the identity-group common offset LSE_G(z)−LSE_G(Tz). LSE_G(u)=log(sum_{d in G} exp(u_d)). Both controls perform association and classification again with z*, retain the same scene/record terms, and leave non-candidates unchanged. These are validation controls.

## 8. Paired-scene bootstrap

The reported intervals compare the state-contrast control with BQR, original association under the same classification refinement, and maximum association under that same refinement. Sampling units are the 120 measured scenes. There are 10,000 draws with replacement using seed 2026102105. All methods and training-seed models use exactly the same sampled scene indices for each draw.

For every draw, duplicate sampled scenes receive distinct image-instance identifiers. Complete-scene COCO AP is recomputed separately for every fixed model. AP is averaged over the three training seeds before taking the method contrast. The 2.5th and 97.5th percentiles form the 95% interval; image-wise AP values are not averaged in place of recomputing detection AP.

The intervals describe scene-sampling variation for the fixed models on this collection. They do not account for method-selection effects from earlier repeated analyses of that collection. Session-specific AP uses the original three 40-scene subsets.

## 9. Class changes, ownership and examples

All 28 classes use fixed, label-independent query–object matches. Matching first maximizes the number of IoU≥0.5 matches, then their total IoU. Class corrections and harms are computed on the same matches before and after refinement. Every assigned edge enters the ownership denominator; an unmatched candidate cannot count as a correct owner. Placement order determines ownership offline. Ground-truth labels and record-to-object correspondences do not enter inference.

The 1,436 measured matched-query observations aggregate repeated predictions of the same scenes across three seed models. They are not 1,436 independent objects. The correction and harm examples are from seed 42 and are selected by image then query order. Each compares diagnostic argmax labels for original RT-DETRv2, BQR and ISW-DETR with the true label using the same query box. The existing examples remain image 3 (empty Wanglaoji → empty JDB, true empty JDB) and image 22 (full large Yili → empty large Yili, true full large Yili).

The OVL diagnostic uses the unchanged training-fitted Gaussian class priors, as defined in `eq:mass-overlap` in the main text. Observation stability enters separately through the original likelihood variance model; observation noise is not silently absorbed into the prior-overlap calculation.

## 10. Existing implementation entries

The separate core package retains `configs/scalevision/training_warm.yml`, `configs/scalevision/training_is.yml`, `tools/train_scalevision.py`, `tools/train_router_controls.py`, `reference_sources/train_benefit_router.py` and `reference_sources/confidence_loss.py`. Original resource archives and their fingerprints remain unchanged. This documentation does not rerun the experiments or constitute a new training/reproduction result.


## 11. Visual and weight interventions

Validation scenes, original boxes and candidate pools stay fixed. A 300×10 random sign field is generated with seed 2026102201, indexed by image and shared across methods and training-seed models. Candidate paired-state scores receive opposing offsets for delta={0,1,2,3,4}; singleton classes are unchanged. The common control uses the same signs with (+2,+2), matching the nominal query-wise norm only at delta=2. State scaling uses nu={0,2,−1}.

Interior stable-platform errors use fixed standard-normal draws with seed 2026102202 and sigma={0,0.25,1,2,3,4,5} g. Initial/final offsets are zero. The exact record transformation remains eq:platform-stress. Total/final references, record masks and stable-window statistics remain unchanged; event masses are redistributed while the total is conserved. The maximum offset produces three negative observed increments in this computed extreme-stress condition. A separate increment-only change of 10 g leaves its before/after readings unchanged. Evidence flattening uses tau={0,0.1,…,1}. These are computational interventions on existing observations.

## 12. Accuracy and computational evaluation

Accuracy uses FP32 predictions, batches of eight and the original visual-export ordering, including the final incomplete batch. Physical evidence and candidate sets are recomputed from raw transactions, followed by one fresh record assignment. Complete RGB query predictions agree with saved outputs on all 90 validation images per seed under that ordering. Missing/rejected transactions give exact visual fallback.

Timing covers original RT-DETRv2, the auxiliary visual branch, CVSR reference, BQR and state-contrast refinement. Each uses FP32 and one image per call on 18 validation scenes with 3–5 records: 24 warm-ups per path, 12 synchronized repeats per scene and 216 observations per path, with cyclic path order. Computation, intermediate device transfers and common top-300 postprocessing are included; sensor acquisition, decoding/resizing and initial device transfer are excluded.

Hardware/software: RTX 5060 8 GB, Ryzen 5 9600X, Windows 11, Python 3.12.12, PyTorch 2.11.0 and CUDA 12.8. ISW-DETR uses GPU visual inference and CPU FP32 fusion with SciPy assignment; CVSR and BQR retain their GPU fusion implementations. GPU allocation peak and increment above the common resident model/input are distinct. Counts include inference-active weights and exclude training-only denoising embeddings.

ISW-DETR has 20,182,616 active parameters, 72,344 above RT-DETRv2. Median latency is 17.69 ms, IQR 16.67–18.68 ms and P95 24.97 ms; BQR median is 35.07 ms. Single-image validation AP is 86.88%; batch-eight AP is 86.83%. Main AP comparisons retain batches of eight. These describe the evaluated implementations, not sensor-to-result latency or a hardware-independent speed ratio.

## 13. Detailed comparison values

These are the existing recorded means in AP percentages or AP percentage points as indicated. No AP was recomputed for this revision.

| Comparison | Recorded result |
| --- | --- |
| Internal/measured RT-DETRv2 | 75.25 / 65.34 AP |
| Internal/measured ISW-DETR | 89.01 / 82.86 AP; original-precision gains 13.77 / 17.53 points |
| Measured Gaussian / MAP / confidence | 77.23 / 78.10 / 80.16 AP |
| Measured gain over CVSR / BQR | 3.76 / 2.71 points |
| BQR paired 95% interval | [1.47, 3.71] points |
| Original-association half contrast | 88.98 internal / 81.97 measured AP; 2.18 / 2.87-point gains |
| Mean-association conditional increment | 0.03 internal / 0.89 measured points; paired interval [0.16, 1.59] |
| Mean versus max association | 86.83 versus 86.81 validation AP; 0.80 measured-point advantage, interval [0.38, 1.25] |
| Full averaging | 88.00 internal / 83.17 measured AP versus 89.01 / 82.86 for half contrast |
| Expanded validation scan | alpha_r=0.4: 86.92268; alpha_r=0.5: 86.83226 AP; difference 0.09042 point |
| No-auxiliary-branch refined model | 88.85 internal / 82.62 measured AP |
| Added auxiliary-branch mean effect | 0.16 internal / 0.24 measured points, varying across seeds; shared-pool measured increments about 0.24 |
| Global scaling maximum / LSE-preserving half / ISW-DETR | 85.32 / 86.46 / 86.83 validation AP |
| Session gains over BQR | 3.01 / 4.05 / 1.72 points; third session has the highest absolute AP for both methods |

The expanded scan does not replace the original fixed 0.5 configuration. Figure 3 preserves all eleven evaluated settings and paired intervals. Values condensed out of body prose remain here and in the existing figure/table source records.

## 14. Detailed intervention results

| Test | Recorded validation result |
| --- | --- |
| Opposing delta=2 | BQR 86.02→79.26; ISW-DETR 86.83→84.80 AP |
| Opposing delta=3 | ISW-DETR 82.43; BQR 74.08 AP |
| Opposing delta=4 | ISW-DETR 79.06; BQR 66.17; original association with the same half refinement 76.84 AP |
| State scaling nu=0,2,−1 | ISW-DETR 86.48, 86.02, 81.87; BQR at 2,−1: 78.80, 68.35 AP |
| Common offset 2 | ISW-DETR 77.48 AP; decline 9.36 points versus 2.03 under opposing offsets |
| Plateau sigma=0.25 g | ISW-DETR 86.57 AP |
| Plateau sigma=1,3,5 g | ISW-DETR 84.31, 71.24, 63.54; BQR 82.94, 70.11, 61.86 AP |
| Evidence tau=0 | Half contrast 74.08; unfused auxiliary visual 73.95; full mean 65.44 AP |
| Evaluated tau=0.1–1 | ISW-DETR 86.54–86.99 AP; unchanged tau=1: 86.83 AP |

Figure 4 retains every evaluated mean point, including the intermediate grids. Full averaging is more resistant to large opposing state offsets, and max association is competitive under some interventions. Exact visual fallback concerns the tested contradiction; similar AP under flat evidence does not mean identical-output fallback.


## Figure 5 example selection

The correction and harm examples use seed 42 and are image 3 (before 0, after 3, truth 3) and image 22 (before 21, after 20, truth 21). They compare the query-level class argmax under fixed boxes.
