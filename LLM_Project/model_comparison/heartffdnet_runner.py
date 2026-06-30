from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def _first_nifti(image_folder: Path) -> Path | None:
    image_dir = image_folder / "ct_test"
    for pattern in ("*.nii.gz", "*.nii", "*.vti"):
        matches = sorted(image_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _conda_executable() -> str:
    return os.getenv("CONDA_EXE") or shutil.which("conda") or "conda"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--conda-env", default=os.getenv("HEART_LEGACY_CONDA_ENV", "heart_legacy"))
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    conda_exe = _conda_executable()
    command = [
        conda_exe,
        "run",
        "-n",
        args.conda_env,
        "python",
        "predict.py",
        "--image",
        args.image_folder,
        "--mesh_dat",
        r"examples\examples\example_dat_of_template_with_veins.dat",
        "--attr",
        "_test",
        "--mesh_tmplt",
        r"examples\examples\template_with_veins_original_normalized.vtp",
        "--model",
        r"examples\examples\weights_gcn.hdf5",
        "--output",
        args.output_folder,
        "--modality",
        "ct",
        "--seg_id",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "--size",
        "128",
        "128",
        "128",
        "--mode",
        "test",
        "--amplify_factor",
        "0.1",
        "--num_mesh",
        "7",
        "--num_seg",
        "1",
        "--num_block",
        "3",
        "--if_swap_mesh",
        "--compare_seg",
    ]
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("TF") or key.startswith("TPU_ML"):
            env.pop(key, None)
    env.pop("PYTHONPATH", None)
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    result = subprocess.run(command, cwd=str(repo), env=env, check=False)
    if result.returncode != 0:
        return int(result.returncode)

    image_folder = Path(args.image_folder)
    if not image_folder.is_absolute():
        image_folder = repo / image_folder
    output_folder = Path(args.output_folder)
    if not output_folder.is_absolute():
        output_folder = repo / output_folder

    input_image = _first_nifti(image_folder)
    if input_image is None:
        return 2

    output_stem = input_image.name.split(".")[0]
    mesh_path = output_folder / f"block2_{output_stem}.vtp"
    if not mesh_path.exists():
        return 3

    helper_path = Path(__file__).resolve().with_name("heartffdnet_mesh_mask.py")
    postprocess_command = [
        conda_exe,
        "run",
        "-n",
        args.conda_env,
        "python",
        str(helper_path),
        "--repo",
        str(repo),
        "--image",
        str(input_image),
        "--mesh",
        str(mesh_path),
        "--output",
        str(output_folder / f"{output_stem}_heartffdnet_mask.npy"),
    ]
    postprocess = subprocess.run(postprocess_command, cwd=str(repo), env=env, check=False)
    if postprocess.returncode != 0:
        return int(postprocess.returncode)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
