from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def _add_repo_paths(repo: Path) -> None:
    for relative in ("src", "template", "external"):
        sys.path.append(str(repo / relative))


def _read_polydata(mesh_path: Path) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(mesh_path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0 or poly.GetNumberOfCells() == 0:
        raise RuntimeError(f"HeartFFDNet mesh is empty: {mesh_path}")
    return poly


def _mesh_to_mask(poly: vtk.vtkPolyData, ref_im: vtk.vtkImageData) -> np.ndarray:
    ones = vtk.vtkImageData()
    ones.DeepCopy(ref_im)
    ones.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    ones.GetPointData().GetScalars().Fill(1)

    poly_to_stencil = vtk.vtkPolyDataToImageStencil()
    poly_to_stencil.SetTolerance(0.05)
    poly_to_stencil.SetInputData(poly)
    poly_to_stencil.SetInformationInput(ref_im)
    poly_to_stencil.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(ones)
    stencil.SetStencilData(poly_to_stencil.GetOutput())
    stencil.ReverseStencilOff()
    stencil.SetBackgroundValue(0)
    stencil.Update()

    output = stencil.GetOutput()
    dims = output.GetDimensions()
    flat = vtk_to_numpy(output.GetPointData().GetScalars())
    return flat.reshape(dims[::-1]).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    _add_repo_paths(repo)
    from vtk_utils.vtk_utils import load_vtk_image

    image_path = Path(args.image).expanduser().resolve()
    mesh_path = Path(args.mesh).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    ref_im = load_vtk_image(str(image_path))
    poly = _read_polydata(mesh_path)
    mask = _mesh_to_mask(poly, ref_im)
    if int(mask.sum()) == 0:
        raise RuntimeError(f"Voxelized HeartFFDNet mesh is empty: {mesh_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, mask)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
