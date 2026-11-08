# ScaleVision-MM data protocol

This note preserves the acquisition and data-preparation settings associated with the ISW-DETR manuscript. It consolidates details previously reported in section 3.1 of manuscript 20261101_02; it does not describe a new acquisition, training run or AP evaluation. The corresponding original English and Chinese LaTeX passages are retained in this directory, with hashes in SOURCE_PROVENANCE.json.

## Catalog and partitions

The catalog contains 18 product identities and 28 leaf classes. Ten identities have full/empty state pairs; the remaining eight each have one catalog class. Each acquired scene contains three to five sequentially placed products. The partitions and training-only derivatives are:

| Use | Images/scenes | Annotated objects |
| --- | ---: | ---: |
| Acquired training scenes |420|1,668|
| Validation scenes |90|357|
| Internal-evaluation scenes |90|356|
| Directly measured evaluation scenes |120|480|
| Training-only image derivatives |750|2,976|
| Total including training derivatives |1,470|5,837|

The 720 acquired scenes comprise 600 development scenes and 120 measured-evaluation scenes. The latter were obtained in three acquisition sessions of 40 scenes each. Derivatives are used only for training, not for evaluation or class-mass-prior fitting. Total image/instance counts include the derivatives; they are not counts of independently acquired scenes.

Development scenes use generated multi-product weight records. The directly measured collection uses acquired placement transactions. This distinction applies to interpretation of the internal and measured detection results.

## Acquisition system and checks

- Camera: MF-200 USB; acquired image resolution 1280×720 at 10 frames/s.
- Weight hardware: LAB-B-B 6-kg load cell and CMCU-07 module.
- Weight output: approximately 10 Hz with 0.1-g resolution.
- A nominal 500-g mass check preceded each session.
- Characterization over 0–3 kg gave a fitted RMSE of 0.182 g. This is the reported characterization fit, not an independent validation of every event mass.
- All measured sessions used the same camera–scale system.

The hardware/scene/trajectory illustrations and representative training-fitted distributions are shown in manuscript Figure 2.

## Stable-window extraction

A 2-g increase over the preceding stable anchor triggers placement detection. A stable window spans 0.8 s and contains at least six samples. When trimming 10% of the values from each tail leaves at least six values, this trimming is applied. The retained mean, population SD and range are calculated.

Drift is the absolute difference between the means of the first and final thirds of time-ordered values after winsorization to the retained limits. A placement window is accepted when:

| Statistic | Limit |
| --- | ---: |
| Range |≤1.5 g|
| Population SD |≤0.5 g|
| Drift |≤0.8 g|

The empty-platform window lasts 1.0 s. Its range, SD, drift and absolute-mean limits are 1.0,0.35,0.5 and 0.5 g, respectively. Successive accepted means supply the before/after readings used to form placement increments. The transaction verification equations and thresholds remain in manuscript section 2.2; this note does not replace that method definition.

## Measured transaction completion, snapshot and annotation

For measured evaluation, a separate final snapshot starts 0.5 s after the last placement and lasts 1.2 s with at least eight samples. The image is captured after that snapshot completes. The median linked host-timestamp difference is 0.797 ms and the mean total-mass residual is 0.081 g.

The measured evaluation comprises completed transactions. Sixty-seven aborted attempts were not included: 13 item timeouts and 54 manual resets. These acquisition outcomes are retained here even though the compact manuscript dataset paragraph no longer lists them. No scenes or detections were removed anew during the prose revision.

Bounding boxes were annotated in EasyData. The authors reviewed class, state and placement order for all 480 measured objects. Multiple evaluation units per class had not been used in development, but observation-to-physical-unit identifiers were not retained. The records therefore do not establish a fully traceable physical-unit-disjoint split. Class labels and placement-order correspondence are used for annotation/offline diagnostics; the manuscript method does not use them as inference inputs.

## Development-record generation

Source single-unit trajectories contain 2,381 records and 223,575 samples. Development records use split-isolated virtual units, training-source class centers and a 1.4826-MAD noise estimate. Normal classes use 16/4/4 virtual units for training/validation/internal evaluation; empty-package classes use 8/2/2.

Profiles specify coefficient of variation, noise floor and source-relative clipping:

| Profile | Coefficient of variation | Noise floor | Source-relative clipping |
| --- | ---: | ---: | ---: |
| Bottled water |0.007|2.0 g|0.02|
| Cartons/cans |0.010|1.0 g|0.025|
| Chips |0.025|1.0 g|0.06|
| Empty packages |0.035|0.3 g|0.08|

Segments are quantized to 0.1 g. All readings, statistics and references are recalculated with seed 20260707. Separate virtual units across development partitions describe the generation procedure; they are not evidence of independent real-product acquisition for those partitions.

The class-mass prior shared by the compared fusion methods is fitted only to the 1,668 objects in the acquired training scenes, with a 1-g floor on the class SD. Training derivatives, validation/internal data and measured-evaluation records are not added to this prior-fitting set. The effective observation scale and mass-compatibility formula remain defined in the manuscript.

## Training-only image derivatives

Offline derivatives use seed 20260708, conservative photometric distortion, optional horizontal flipping and one to three rectangular occlusions, covering at most 28% of each box. They retain source links and use resampled development records. These offline derivatives are distinct from the online training augmentations described in manuscript section 3.2.

## Scope and material access

The product-data acquisition did not involve human participants or animals. The manuscript retains that acquisition-scope statement under Funding and data sources; this note does not create a new ethics approval or a blanket exemption for unrelated datasets.

The images, annotations, transaction records and reference model files are indexed in the [public reproduction release](https://github.com/Tb-STAOO/ISW-DETR/releases/tag/sr-repro-20261108).
