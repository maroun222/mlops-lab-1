# Lab 2 — Training and MLflow tracking

## Question 1 — Dependency changes

`pyproject.toml` now lists MLflow, PyTorch (`torch`), torchvision and scikit-learn
as direct dependencies, alongside DVC and Pillow. The explicit `pytorch-cpu`
index and source entries select CPU builds of torch and torchvision for this
Intel-graphics laptop. `uv.lock` records the resolved package versions, their
transitive dependencies, download sources and hashes so `uv sync --locked` can
reproduce the environment. Installed versions are MLflow 3.16.0,
torch 2.14.0+cpu, torchvision 0.29.0+cpu and scikit-learn 1.9.1.

## Question 2 — Metadata and artifacts

`--backend-store-uri sqlite:///mlflow.db` chooses a SQLite database relative to
the server's working directory. It stores experiment/run identities, statuses,
parameters, metric histories and tags. `--default-artifact-root ./mlruns` sets
the default artifact location for newly created experiments. Artifacts are
files, such as serialized trained models, class mappings, plots and reports;
the database records their locations rather than embedding the model weights.

This server runs from `C:\Users\User\Desktop\LAB1\mlops-lab-1` on
http://127.0.0.1:5000. Its database is `mlflow.db` in that folder. The initial
Default experiment had ID 0; the new food11 experiment has ID 1 and artifact
root `file:C:/Users/User/Desktop/LAB1/mlops-lab-1/mlruns/1`.

## Question 3 — Why exclude local MLflow files?

The database is mutable binary runtime state and model artifacts can be large.
Putting these files in Git would create noisy, bulky commits and make concurrent
updates difficult to merge. In this lab MLflow already manages the relationship
between runs, metadata and artifacts, so tracking the same live store with DVC
would duplicate storage and require repeatedly snapshotting changing server
state. DVC remains responsible for the datasets; Git tracks the code and setup.
MLflow stores the runs. Backing up MLflow separately can still be useful.

Both `.gitignore` and `.dvcignore` exclude `mlflow.db`, SQLite sidecar files,
`mlruns/` and `mlartifacts/`.

## Question 4 — Creating an experiment

`mlflow.set_experiment("food11")` creates the experiment when it does not exist
and selects it as the destination for subsequent runs. Later calls reuse it.
This happened on the first training invocation: MLflow reported that it created
food11, and the server returned experiment ID 1.

## Question 5 — Parameters versus metrics

A parameter describes a fixed run setting, such as `lr=0.001`, `batch_size=32`,
or `epochs=5`. Log it once with `mlflow.log_param` or `mlflow.log_params`.
A metric is a numeric measurement such as training loss or validation accuracy.
It may change during a run, so repeated logging with `step=epoch` records a
history that the UI can plot. A parameter has no step because it represents one
fixed setting, not a time series.

## Training implementation

`src/food11/train.py` loads training, validation and evaluation folders using
ImageFolder and DataLoader. It keeps the lab's 128x128 resolution and uses
ImageNet channel normalization. It loads pretrained ResNet18 IMAGENET1K_V1
weights, replaces the final layer with 11 outputs and fine-tunes all layers with
Adam. It checks class mappings across splits and logs the mapping as an artifact.

Every run uses seed 42. Training is shuffled; validation and evaluation are not.
Losses are averaged over images, including the last partial batch. Every epoch
logs train_loss, val_loss and val_accuracy with steps 1 through 5. Evaluation is
the held-out test split and is only evaluated after training. The logged model
contains the final epoch's weights; run ranking uses final val_accuracy rather
than the separately logged peak best_val_accuracy. Test accuracy is reported as
required by the lab but is not used to choose the best run.

MLflow 3 uses `name="model"` as the modern equivalent of the handout's older
positional artifact-path argument. The trained PyTorch module is logged with
`mlflow.pytorch.log_model` using explicit pickle serialization.

## Run the server and training

Keep the server running in its own terminal from the repository root:

```powershell
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Open http://127.0.0.1:5000 and choose **Model training**, then **Experiments**.
In another terminal:

```powershell
uv run python src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
uv run python src/food11/train.py --dataset mini --epochs 5 --lr 0.01 --batch-size 32
uv run python src/food11/train.py --dataset mini --epochs 5 --lr 0.0001 --batch-size 32
uv run python src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 64
```

## Question 6 — UI and model artifact location

Open a run from **food11 → Runs**. Its **Overview** shows parameters and final
metric values; **Model metrics** shows the epoch charts. In MLflow 3.16 the
serialized model is linked under **Logged models** and has its own model page.
The run's **Artifacts** tab also contains `class_to_idx.json`.

The first run's model is stored at:

```text
C:\Users\User\Desktop\LAB1\mlops-lab-1\mlruns\1\models\m-661c95b532e64a0595ea9629034cdba5\artifacts\data\model.pth
```

The same artifact directory includes `MLmodel`, `requirements.txt`,
`python_env.yaml` and `conda.yaml`. This MLflow 3 model layout differs from older
versions that stored the model directly under the run's artifact folder.

The first run's training, test evaluation and model logging succeeded, but
Windows' redirected cp1252 console rejected an emoji while MLflow printed its
completion link. The script now configures UTF-8 output. After verifying the
saved model was READY and all metrics existed, the first run was finalized as
FINISHED; its completion note records this recovery.

### Q7
The best learning rate was `0.0001`, with a final validation accuracy of approximately `73.27%`. A higher learning rate did not always perform better: `0.01` produced the lowest validation accuracy.

### Q8
For batch size 32, validation accuracy improved as the learning rate decreased from `0.01` to `0.001` to `0.0001`. With learning rate `0.001`, batch size 64 achieved a higher final validation accuracy than batch size 32. This pattern is based only on the four tested configurations.

### Q9
The best run was:

- Run ID: `5969ea6a4f0047b896a408da97f9c951`
- Learning rate: `0.0001`
- Batch size: `32`
- Final validation accuracy: approximately `73.27%`
- Test accuracy: approximately `76.37%`

## Sources

- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pytorch.html
- https://mlflow.org/docs/latest/ml/tracking/
- https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
