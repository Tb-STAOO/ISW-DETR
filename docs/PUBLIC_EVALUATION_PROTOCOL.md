# C-CCM public evaluation protocol

## Dataset, class mapping and view split

The source is the public CORSMAL Containers with Contents (C-CCM) image/annotation resource, linked to configuration-level mass attributes from CORSMAL Containers Manipulation (CCM). Use the existing manuscript references sanchezmatilla2021cccm and xompero2020ccm and the separate licence note; this document does not relicense either dataset.

The fixed eligible set contains8,727 RGB frames for six container identities. Each identity has seven states: empty, pasta50%, pasta90%, rice50%, rice90%, water50% and water90%. These yield42joint identity--state classes. Camera views1/2/3 supply2,929training/2,829validation/2,969test frames respectively. These are known containers and configurations observed across camera views, not held-out object identities. Evaluation uses complete resized384×384images, not ground-truth crops.

Each frame is paired with the published total mass of its container/content configuration. These are configuration-level attributes, not an independent scale measurement of each frame. Priors are computed from training data only. Training membership, annotations, observations and the42-class taxonomy are fixed in the existing EXPERIMENT_LOCK.json and source metadata.

## Weight evidence and scoring

The class standard-deviation floor is1g; the observation scale is0.25g. Visual state means are computed across the seven content states of each container identity. The retained contrast coefficient is alpha_r=0.5, scene coefficient beta=2 and record-residual coefficient eta=0.2. These are the original fixed public settings, not retuned during this prose revision.

The Gaussian scene-only control adds class-mass scene evidence only to the shared candidate pool, with coefficient2. This all-candidate restriction is consequential: do not substitute the private all-query Gaussian control. ISW-DETR and its no-refinement control share the visual checkpoint, boxes, candidate scope and configuration-mass inputs.

This public task has a single object and a single weight record per frame. In the evaluated implementation:

- Candidate-restricted Gaussian scene fusion with coefficient2 and the CVSR reference produce identical detections.
- Original and mean association produce identical detections under the same half-contrast refinement.

Thus the public matched-input result directly evaluates classification-score refinement. Multi-record assignment and transaction-verification behavior are assessed with ScaleVision-MM; the public experiment does not establish their separate gain.

## Training stages and checkpoint selection

The random seeds are42,3407and2026. The shared RT-DETRv2-R18 detector starts from the official COCO initialization, trains20epochs and uses EMA. Each seed then has two10-epoch branches: continued fine-tuning of the flat detector and auxiliary identity--state classification training with the original detector frozen. Both second stages restart the optimizer/schedule and disable EMA. AdamW uses base learning rate1e-4, training batch4 and the original384×384input setup.

The retained checkpoint at each stage maximizes validation RGB AP. The additional comparison using the already-retained base-stage checkpoints was post-hoc; their validation AP exceeded that of the continued fine-tuning branch for every seed. This chronology is preserved even though the compact text no longer lists each training step. The9checkpoints (3stages×3seeds) and7scoring variants per checkpoint form63recorded evaluation combinations.

The base configuration uses weight decay1e-4 (norm/batch-normalization groups0), AdamW betas0.9/0.999, MultiStepLR milestone16 and factor0.1, and100warm-up updates. Its original configuration enables mixed-precision training, gradient clipping0.1 and EMA decay0.9999/warmups2000. These are training settings; they do not change the separate accuracy/timing protocol in the private-data Methods. Exact branch settings and augmentation definitions are supplied in configs/cccm/*_seed*.yml rather than inferred from these summary values.

## Metrics and results

COCO AP uses IoU thresholds0.50:0.05:0.95. Reported AP values are means of the three training-seed results, not a chosen seed. A separate joint metric asks whether each image's highest-scoring detection has the correct fine class and IoU≥0.5; this is distinct from COCO AP.

| Visual model / scoring | Mean test AP (%) |
| --- | --- |
| Base RT-DETRv2-R18, RGB |16.57101769999882|
| Base RT-DETRv2-R18, candidate scene fusion |54.494989092094706|
| Base RT-DETRv2-R18, mean/half refinement |58.01347889431479|
| Auxiliary-branch model, RGB |16.597399811639818|
| Auxiliary-branch model, no score refinement |53.76794458877553|
| Auxiliary-branch model, mean/half refinement (ISW-DETR adaptation) |57.10378168934954|

The original-precision mean gain of the complete adaptation over RGB-only base RT-DETRv2-R18 is40.53276398935072APpoints and includes adding weight information. Its matched-input score-refinement gain is3.335837100574009points. The corresponding gain for the original visual representation is3.5184898022200812points. Displayed rounded endpoints can differ by0.01point from the rounded original-precision gain.

The complete adaptation is0.9096972049652455points below the original detector with the same refinement. Auxiliary-branch RGB gain is0.02638211164099798point on average and is not positive for every seed. These stronger/neutral comparisons remain part of the evidence.

## Pilot exclusion and dependence

The earlier pilot inspected168test images. The pre-fixed sensitivity subset excludes these, leaving2,801frames. ISW-DETR's score-refinement AP gain there is3.317640104734769points (display3.32). The original frame membership is in previous_pilot_selection.json and SENSITIVITY_SUBSET_LOCK.json; do not replace this with a newly selected favorable subset.

The joint top-detection metric changes from82.77759065903223% to86.35904344897273%. It measures scoring/box selection from the original predicted box set, not improved box regression.

Original independent-recording identifiers were unavailable for this evaluation, so recording-level uncertainty was not estimated. Three training seeds and a pilot-excluded subset do not create new independent physical containers or recording sessions.

## Reproduction resources

Use the existing separate core package, public-metadata.zip, public-checkpoints.zip and public-pretrained.zip. Original images are obtained through the documented preparation process. The core README and tools/prepare_cccm.py, tools/run_cccm.py, tools/run_public_evaluation.py provide the actual entrypoints and resource rewriting. The archived configurations retain historical paths as evidence; they are not portable commands to copy blindly.



## Input provenance and class-information diagnostic

The weight observations are assigned using the image annotation's container identity, filling type and filling level to join the public CCM configuration total mass. For every included class, the12source rows share one total mass. Scalar-only fusion input files do not remove this upstream annotation dependency. These are configuration-level reference masses, not independent measurements for individual frames.

The42joint classes have40distinct masses. Nearest-training-class-mean classification on the fixed2,969test frames gives2,832correct(95.38565173459077%),with288distance ties resolved by the smallest class index. The diagnostic does not use images or predicted boxes and does not compute COCO AP. Training-only prior fitting does not turn the matched configuration values into independent sensor observations.

The original3.335837100574009-point matched-input AP gain remains a classification-score comparison under these public inputs; it is not newly recomputed here. The private measured evaluation continues to supply direct transaction evidence. The public view split concerns known containers/configurations and does not establish recording-level independence.

The configuration mapping, source hashes and diagnostic script are in the C-CCM mass-diagnostic release archive. The diagnostic is also available through `tools/run_mass_diagnostic.py`.
