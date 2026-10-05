# Fashion-MNIST ANN: Git, DVC and Google Drive

Assignment 3 implements a reproducible fully connected neural network for Fashion-MNIST. The Git repository is named MLOPS and contains the assignment in this folder, rather than using a separate fashion-ann-pipeline repository.

## Results

| Version | Dense units | Test accuracy | Test loss |
| --- | ---: | ---: | ---: |
| v1 | 128 | 88.37% | 0.338573 |
| v2 / final | 256 | 88.43% | 0.340889 |

The final model meets the assignment's 85% accuracy target. The original 60,000 training images are split into 48,000 training and 12,000 validation images; the official 10,000-image test set is retained.

## Pipeline

1. `src/prepare.py`: downloads Fashion-MNIST and saves raw NumPy arrays.
2. `src/preprocess.py`: normalizes pixels by 255 and creates a seeded, stratified validation split.
3. `src/train.py`: builds Flatten -> Dense(ReLU) -> Dropout -> Dense(10, softmax), then saves `models/model.h5` and `models/history.csv`.
4. `src/evaluate.py`: computes test loss/accuracy and generates a confusion matrix.

`params.yaml` supplies the validation fraction, seeds and training hyperparameters. `dvc.yaml` declares the stage graph; `dvc.lock` records the executed parameters and artifact hashes. `uv.lock` records the environment and is tracked as a stage dependency.

## Reproduce

Install uv, clone the repository, and run these commands from the repository root:

```powershell
cd assignment_3
uv sync
uv run dvc pull -r gdrive_storage
uv run dvc repro
```

You can also regenerate artifacts from source with `uv run dvc repro --force --no-run-cache`. This retrains the model and takes longer. Training seeds and deterministic TensorFlow operations support reproducibility on the same software/hardware environment; bitwise equivalence across different platforms is not guaranteed.

Each script can run independently using `uv run python src/<script>.py`; run preparation, preprocessing, training and evaluation in that order when bypassing DVC.

## Google Drive remote

Remote folder: https://drive.google.com/drive/folders/1cWQ1aXsd-Ov_v3R9FreQfBMd-2wlOD0W

The configured default remote is `gdrive_storage`. Google Drive stores hash-addressed objects under `files/md5`, not the original filenames. DVC restores filenames using the Git-tracked metadata.

The instructor's Google account must have access to that folder. Authentication is performed separately by each user. If the default OAuth app is blocked, create a Google Cloud Desktop OAuth client, enable the Drive API and add the login account as a test user. Configure client settings with `uv run dvc remote modify --local gdrive_storage gdrive_client_id "CLIENT_ID"` and the analogous `gdrive_client_secret` setting. Never commit `.dvc/config.local` or authentication tokens. See https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive.

## Tracking and conflict exercise

Part C first used standalone `.dvc` pointers for raw data, processed data, models and reports. Part D transferred their ownership to pipeline stages. An output cannot be owned by both definitions simultaneously.

For Part E, preprocessing-stage ownership was temporarily replaced by `data/processed.dvc`. `teammate-sim` used divisor 256; main used divisor 254 with clipping. Their merge produced source-code and processed-pointer conflicts. The resolution regenerated authoritative `/255.0` data, synchronized it with `dvc checkout`, and restored the complete four-stage pipeline.

Root `metrics.json` is a Git-tracked metric (`cache: false`); the identical `reports/metrics.json` and confusion matrix are cached/pushed by DVC. The v1/v2 tags retain measured experiment versions. Reset demonstration commits remain accessible through `reset-soft-demo` and `reset-hard-demo` tags.

## Report and evidence

The report is [../outputs/ML-Versioning-Report.pdf](../outputs/ML-Versioning-Report.pdf). Full command records and rendered output panels are under [../outputs/evidence](../outputs/evidence).

Original D3/D4 console logs were retained. Historical evidence comes from actual commits, tags and reflogs. Missing staging/stash/rebase/reset displays were reconstructed in an isolated disposable clone and explicitly labelled. PNG panels render real recorded output; they are not claimed to be original IDE screenshots. Conflict and final-state records were captured during completion.
