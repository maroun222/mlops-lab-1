# Lab 5 — CI/CD with GitHub Actions

## Question 1 — Why use a fake model in unit tests?

The API code can be correct while the MLflow server is offline or unreachable from the runner. The model can also be unavailable because the `champion` alias is missing, artifact permissions have changed, or the artifact download timed out. A fake model makes the API tests independent of the registry, network, model files, and GPU.

## Question 2 — What does the prediction test prove?

The test uploads a small real JPEG as multipart data and checks that the API decodes it, passes a batch of shape `(1, 3, 128, 128)` to the fake model, and returns HTTP 200 with `Bread` and confidence close to `0.90` in the JSON response. The fake returns logits chosen to give that confidence after softmax. This verifies the request and response path. It provides no evidence about the trained model's accuracy, calibration, or ability to load from MLflow.

## Question 3 — What detected the duplicate?

The cross-split hash check compares image bytes, so it catches a copy even when its filename changes. A corrupt image typically causes a visible decoding error. A duplicate can silently let a model see an evaluation example during training, which makes measured performance look better than performance on unseen images.

The original mini dataset contained 43 cross-split matches among 3,292 files. The other four checks passed. In a temporary copy I excluded overlapping files, retaining copies in evaluation before validation and training. All five checks passed on the resulting 3,249 files. Copying `training/Bread/0_0.jpg` to `evaluation/Bread/lab5_intentional_duplicate.jpg` caused only the cross-split check to fail and returned exit code 1. Removing that copy restored a passing report. The original DVC dataset was left in place.

## Question 4 — Where should the full data check run?

1. Run it immediately after pulling or preparing a dataset version, before publishing that data version from a workstation or a runner with DVC access.
2. Run it after data is staged in the training environment and before the training job starts. Make a failing report stop the job before it trains or registers a model.

The standard GitHub workflow uses a tiny checked-in fixture. A runner with no access to the DVC dataset cannot check the full dataset contract.

## Question 5 — What does `needs: test` do?

If lint or a unit test fails, `test` fails and `smoke` is skipped. The registry job also stays gated. Without `needs: test`, `smoke` could run independently and spend time building a container from a change that already failed a test.

## Question 6 — First and second GitHub Actions runs

Fill in the observed durations from the repository's **Actions** tab after pushing both commits. Record the total and the `uv sync --frozen` step for each run. The expected improvement is in dependency setup, because `setup-uv` can restore cached packages when the dependency lockfile stays the same. The publishing job also has a separate Docker layer cache. The plain `docker build` in the smoke job runs on its own runner.

| GitHub Actions run | Total duration | `uv sync --frozen` | Cache result |
| --- | --- | --- | --- |
| First push | Pending | Pending | Pending |
| Empty-commit push | Pending | Pending | Pending |

## Question 7 — Why tag images with the commit SHA?

The SHA tag identifies the code revision used to build the image. If one machine pulls `latest` in the morning and another pulls it after an afternoon push moves the tag, both can claim to run `latest` while running different code. Using the same commit SHA tag lets both machines request the image for the same revision and makes it easier to roll back to a known build.

## Question 8 — Why declare token permissions?

The publishing job needs repository read access to check out the code and package write access to publish the image. The test and smoke jobs only need repository read access. A compromised step with broad write permissions could alter repository contents, releases, issues, or packages where those permissions apply. Giving each job only what it needs limits the effects of a compromised step.

## Question 9 — What happens when a protected PR fails?

Expected result for the intentionally broken PR: `test` appears as failed, `smoke` is skipped, and the failed required check blocks merging when the rule applies to the user. The team gets one consistent, shared check for the exact proposed change. A laptop-only process relies on each person remembering to run the right checks on the right code.

Record the actual result from the PR after applying the `main` rule and pushing the intentionally failing change.

## Question 10 — Model promotion results

The gate uses the logged `test_accuracy` and `evaluation_images` from the runs associated with each model version. The required gain is `2 * sqrt(p * (1 - p) / n)`, where `p` is the current champion's accuracy and `n` is the evaluation split size. It compares runs with the same dataset setting and evaluation image count.

The original champion, Version 3, scored **76.3686%** over **1,096** evaluation images. Two standard errors are **2.5664 percentage points**, so the threshold was **78.9350%** for both candidates. The runs used the mini dataset, batch size 32, and seed 43.

| Candidate | Training settings | Test accuracy | Gate decision |
| --- | --- | --- | --- |
| Version 4 | 1 epoch, learning rate `0.000001` | 9.4891% | `challenger` |
| Version 5 | 5 epochs, learning rate `0.0001` | 76.6423% | `challenger` |

Version 5 improved by about 0.27 percentage points, below the required gain. `champion` stayed on Version 3. The `challenger` alias points to Version 5 after the second run, and each candidate has tags recording its own decision and both accuracies. The model versions and artifacts are in the Lab 4 MLflow Docker volume. The gate was exercised on the same original split as the champion to reproduce the lab comparison. Because Question 3 identified leakage in that dataset, these scores demonstrate the mechanics of the gate and should be reevaluated on clean data before a production decision.

## Question 11 — How could promotion run automatically?

1. Add the promotion command after training and registration in a local training job or self-hosted runner that can reach the MLflow server on the private network.
2. Use a reachable hosted MLflow tracking server and registry, such as the one offered by DagsHub. Give the training pipeline authenticated access through secrets, then invoke promotion after it registers the candidate. [DagsHub MLflow documentation](https://dagshub.com/docs/integration_guide/mlflow_tracking/)

## Question 12 — Pipeline diagram

See the [pipeline diagram and GitHub checklist](lab/lab5.md).
