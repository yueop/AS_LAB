from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import yaml


def _first_nifti(image_folder: Path) -> Path | None:
    image_dir = image_folder / "ct_test"
    for pattern in ("*.nii.gz", "*.nii", "*.vti"):
        matches = sorted(image_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("TF") or key.startswith("TPU_ML"):
            env.pop(key, None)
    env.pop("PYTHONPATH", None)
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "ITK_NIFTI_SFORM_PERMISSIVE": "1"})
    return env


def _conda_executable() -> str:
    return os.getenv("CONDA_EXE") or shutil.which("conda") or "conda"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--mesh-dat", required=True)
    parser.add_argument("--mesh-template", required=True)
    parser.add_argument("--conda-env", default=os.getenv("HEART_LEGACY_CONDA_ENV", "heart_legacy"))
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    image_folder = Path(args.image_folder).expanduser().resolve()
    output_folder = Path(args.output_folder).expanduser().resolve()
    mesh_dat = Path(args.mesh_dat).expanduser().resolve()
    mesh_template = Path(args.mesh_template).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    config = {
        "network": {
            "num_blocks": 3,
            "num_seg_class": 1,
            "rescale_factor": 0.1,
            "input_size": [128, 128, 128],
            "hidden_dim": 128,
            "coord_emb_dim": 384,
        },
        "prediction": {
            "model_weights_filename": str(repo / "pretrained" / "task1_mmwhs.hdf5"),
            "image": {
                "image_folder": str(image_folder),
                "image_folder_attr": "_test",
                "modality": ["ct"],
            },
            "mesh": {
                "mesh_dat_filemame": str(mesh_dat),
                "mesh_tmplt_filename": str(mesh_template),
                "swap_bc_coordinates": None,
                "num_mesh": 7,
            },
            "output_folder": str(output_folder),
            "mode": "test",
        },
    }

    config_path = output_folder / "heartdeform_task1_mmwhs_adapter.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    env = _clean_env()
    conda_exe = _conda_executable()
    command = [conda_exe, "run", "-n", args.conda_env, "python", "predict.py", "--config", str(config_path)]
    result = subprocess.run(command, cwd=str(repo), env=env, check=False)
    if result.returncode != 0:
        return int(result.returncode)

    input_image = _first_nifti(image_folder)
    if input_image is None:
        return 2

    output_stem = input_image.name.split(".")[0]
    mesh_path = output_folder / f"block_2_{output_stem}.vtp"
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
        str(output_folder / f"{output_stem}_heartdeform_mask.npy"),
    ]
    postprocess = subprocess.run(postprocess_command, cwd=str(repo), env=env, check=False)
    return int(postprocess.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
