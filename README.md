# Lab 1 — Git, DVC and Food-11 Preparation

This lab demonstrates how to version code with Git, version datasets with DVC, store data on DagsHub, and prepare Food-11 images for machine learning.

## Project setup

Python dependencies are managed with uv:

```powershell
uv sync
```

The preparation script is located at `src/food11/data.py`. When the raw dataset is available and the processed folders do not already exist, run:

```powershell
uv run python src/food11/data.py
```

The script preserves the raw images and all three dataset splits. It converts images to RGB, resizes them to 128×128, and organizes them into category folders.

The mini dataset contains the first 100 images in sorted filename order per category per split, or all available images if fewer than 100 exist. The script refuses to overwrite existing processed folders.

## Question 1 — Files created by uv init

In this project, `uv init` created:

- `pyproject.toml`: project metadata, required Python version, and dependencies.
- `.python-version`: the Python version used by the project, which is 3.12.
- `main.py`: a starter Python program.
- `README.md`: project documentation, initially empty.

The `.git` folder already existed from cloning the GitHub repository.

A root `.gitignore` was added later to exclude the virtual environment and temporary Python files. Adding dependencies also generated `uv.lock`, which records resolved dependency versions for reproducibility.

## Question 2 — Files created by dvc init

DVC initialization created:

- `.dvc/config`: shared DVC configuration, such as the remote URL.
- `.dvc/.gitignore`: rules excluding DVC cache, temporary state, and local configuration from Git.
- `.dvcignore`: rules controlling which files DVC ignores when scanning the project.

These three files should be committed to Git.

The following should not be committed:

- `.dvc/cache`: cached dataset contents.
- `.dvc/tmp`: temporary DVC state.
- `.dvc/config.local`: local settings that may contain credentials.

The `.dvc` directory stores configuration and internal state. A file such as `data.dvc` identifies a tracked dataset version.

## Question 3 — Credentials and configuration scopes

With `--global`, DVC stores settings in the current user's configuration. On Windows, the typical location is:

```text
%LOCALAPPDATA%\iterative\dvc\config
```

Other configuration scopes are:

- `--system`: settings for all users.
- `--project`: shared project settings in `.dvc/config`; this is the default scope.
- `--local`: local project settings in `.dvc/config.local`, which Git ignores.

Configuration precedence, from highest to lowest, is:

```text
local → project → global → system
```

Credentials must not be pushed to GitHub. The remote URL should be stored in the shared project configuration so a fresh clone knows where to find the data.

In this project, the authentication type, username, and password-prompt setting were configured locally. The DagsHub token was entered at the prompt.

Defining a remote only with `--global` would not share its URL with someone cloning the repository on another computer.

## Question 4 — Change to .gitignore

Running:

```powershell
uv run dvc add data
```

added `/data` to the root `.gitignore`.

Git therefore ignores the actual image directory, while DVC tracks its contents. The updated `.gitignore` and the `data.dvc` pointer are committed to Git.

## Question 5 — Contents of data.dvc

`data.dvc` is a small YAML file describing the tracked dataset.

Its `outs` entry includes:

- The tracked path: `data`.
- The hash algorithm and directory checksum.
- The total size in bytes.
- The number of files.

A directory checksum ends in `.dir` and identifies a cached manifest listing the directory's files and their hashes.

The pointer contains no image bytes. The remote URL is configured separately in `.dvc/config`.

The raw-only snapshot, recorded in commit `d43356b`, contains:

- 16,643 files.
- 1,188,442,712 bytes.
- Checksum: `a3a457d03c51ff8b037a833440f6ad13.dir`.

The raw dataset has:

| Split | Images |
|---|---:|
| Training | 9,866 |
| Evaluation | 3,347 |
| Validation | 3,430 |
| Total | 16,643 |

## Question 6 — GitHub and DagsHub

GitHub contains:

- Python code.
- Project documentation.
- Dependency configuration and lock files.
- Non-secret DVC configuration.
- `.gitignore` and `data.dvc`.

The actual image directory is not stored in GitHub.

DVC uses the checksum in `data.dvc` and the configured remote to identify and retrieve the correct dataset version.

The complete dataset was successfully uploaded to DagsHub using:

```powershell
uv run dvc push
```

The initial upload encountered a server disconnection near the end. A retry completed successfully and reported `3 files pushed`, uploading the remaining DVC objects.

Project links:

- GitHub: https://github.com/maroun222/mlops-lab-1
- DagsHub: https://dagshub.com/maroun222/mlops-lab-1

## Question 7 — Clone into a new folder

A fresh Git clone contains `data.dvc`, but it does not initially contain the `data` folder.

The repository was cloned into `lab1-download-check`:

```powershell
git clone https://github.com/maroun222/mlops-lab-1.git lab1-download-check
cd lab1-download-check
uv sync
```

Local authentication was then configured:

```powershell
uv run dvc remote modify --local origin auth basic
uv run dvc remote modify --local origin user maroun222
uv run dvc remote modify --local origin ask_password true
```

The data was downloaded with:

```powershell
uv run dvc pull
```

The command completed successfully and reported:

```text
32033 files fetched and 36578 files added
```

All 36,578 workspace files were restored, including:

```text
data/
├── food11_raw/
├── food11_processed/
└── food11_processed_mini/
```

The number of fetched objects differs from the number of restored files because DVC can reuse identical content for multiple file paths.

`dvc pull` downloads missing data into the cache and restores the workspace. `dvc checkout` alone restores from the local cache and cannot download missing data from the remote.

## Image preparation and ResNet

The preparation script produces:

```text
data/
├── food11_raw/
│   ├── training/
│   ├── evaluation/
│   └── validation/
├── food11_processed/
│   ├── training/
│   ├── evaluation/
│   └── validation/
└── food11_processed_mini/
    ├── training/
    ├── evaluation/
    └── validation/
```

Each processed split contains these category folders:

```text
Bread
Dairy product
Dessert
Egg
Fried food
Meat
Noodles-Pasta
Rice
Seafood
Soup
Vegetable-Fruit
```

For example:

```text
data/food11_processed/training/Bread/0_0.jpg
```

All 16,643 processed images were validated as RGB images with dimensions of 128×128.

The mini dataset contains:

| Split | Mini images |
|---|---:|
| Training | 1,100 |
| Evaluation | 1,096 |
| Validation | 1,096 |
| Total | 3,292 |

Every mini image was verified to match its corresponding processed image byte-for-byte.

The combined raw, processed, and mini dataset contains 36,578 files and 1,277,237,495 bytes. This snapshot was recorded in commit `11b082d`.

This category-folder structure is compatible with torchvision's `ImageFolder` loader. ResNet itself receives image tensors rather than directories.

`ImageFolder` assigns class indices alphabetically by folder name. For this dataset, the category names follow the original numerical label order alphabetically. Checking `class_to_idx` remains useful to confirm that all splits use the same mapping.

The stored images are 128×128 as required by the lab. When using pretrained ResNet weights later, apply the preprocessing required by the selected weights, including the appropriate resizing and normalization.

## Question 8 — Switching data versions

Commits affecting the data pointer can be listed with:

```powershell
git log --oneline -- data.dvc
```

To restore the raw-only version:

```powershell
git checkout d43356b
uv run dvc checkout
```

After these commands, `food11_processed` and `food11_processed_mini` disappear, leaving `food11_raw`.

Git changes the version of the pointer file, while DVC changes the actual workspace data to match that pointer.

To return to the current version:

```powershell
git checkout main
uv run dvc checkout
```

This exercise was tested successfully: the raw-only version removed both processed folders, and returning to `main` restored them from the DVC cache.

## DagsHub setup reference

The DagsHub repository is connected to the existing GitHub repository.

Its HTTPS DVC remote is:

```text
https://dagshub.com/maroun222/mlops-lab-1.dvc
```

The following commands document the initial setup. The remote is already configured in this project, so there is no need to add it again.

```powershell
uv run dvc remote add -d origin https://dagshub.com/maroun222/mlops-lab-1.dvc
uv run dvc remote modify --local origin auth basic
uv run dvc remote modify --local origin user maroun222
uv run dvc remote modify --local origin ask_password true

git add .dvc/config
git commit -m "Configure DagsHub DVC remote"
git push

uv run dvc push
```

Enter the DagsHub access token when prompted. Do not commit credentials to Git.

A fresh clone receives the shared remote URL through `.dvc/config`, but needs its own local authentication settings.

## Updating the project later

After changing dataset contents:

```powershell
uv run dvc add data
git add data.dvc .gitignore src/food11/data.py pyproject.toml uv.lock
git commit -m "Update Food-11 data and preparation"
git push
uv run dvc push
```

For documentation-only changes:

```powershell
git add README.md
git commit -m "Update lab documentation"
git push
```

## Completion status

- GitHub repository created and synchronized.
- Python environment managed with uv.
- DVC initialized and configured.
- Raw dataset tracked with DVC.
- Processed and mini datasets generated and validated.
- Complete dataset uploaded to DagsHub.
- Fresh-clone download completed successfully.
- Switching between data versions tested successfully.
- All eight lab questions answered.

## Preparation for the next session

The project includes `mlflow`, `torch`, `torchvision`, and `scikit-learn`.
PyTorch and torchvision use the explicit official CPU wheel index, as recommended
for the mini dataset on this laptop with Intel graphics.

Before class, open Docker Desktop and wait for its engine to start. Then open a
terminal in `mlops-lab-1` and run:

```powershell
uv sync --locked
docker info
docker compose version
```

Run Python scripts through `uv run python` so they use this project's environment.
The package is installed as `scikit-learn` and imported in Python as `sklearn`.
There is no need to regenerate or upload the datasets for this preparation step.

Verified on this laptop on 2026-09-14: Docker runs the `hello-world` container,
Docker Compose is available, all package dependencies are compatible, the mini
training split loads 1,100 images, ResNet completes a CPU training step, and MLflow
successfully records and retrieves a metric in a temporary local experiment.

## References

- https://doc.dvc.org/command-reference/init
- https://doc.dvc.org/command-reference/config
- https://doc.dvc.org/command-reference/pull
- https://doc.dvc.org/command-reference/checkout
- https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
- https://pytorch.org/hub/pytorch_vision_resnet/
- https://dagshub.com/docs/integration_guide/dvc/
