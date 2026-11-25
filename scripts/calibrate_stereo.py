#!/usr/bin/env python3
"""Chessboard stereo camera calibration utility exporting YAML/JSON for StereoCalibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2 as cv
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realtime.frame_source import StereoCalibration


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image


def calibrate_stereo_pairs(
    left_images: list[np.ndarray],
    right_images: list[np.ndarray],
    *,
    board_size: tuple[int, int] = (9, 6),
    square_size_m: float = 0.025,
    alpha: float = 0.0,
    min_valid_pairs: int = 5,
) -> tuple[StereoCalibration, dict[str, Any]]:
    """Run OpenCV chessboard stereo calibration and return StereoCalibration + diagnostics."""
    if len(left_images) != len(right_images) or not left_images:
        raise ValueError("left_images and right_images must be non-empty lists of equal length")
    cols, rows = int(board_size[0]), int(board_size[1])
    if cols < 3 or rows < 3 or square_size_m <= 0.0:
        raise ValueError("board_size must be >= (3, 3) and square_size_m > 0")

    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_size_m)

    criteria_subpix = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    obj_points: list[np.ndarray] = []
    img_points_l: list[np.ndarray] = []
    img_points_r: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    flags_find = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
    for img_l, img_r in zip(left_images, right_images):
        gl = _to_gray(img_l)
        gr = _to_gray(img_r)
        if gl.shape != gr.shape:
            raise ValueError(f"Mismatched stereo image shapes: {gl.shape} vs {gr.shape}")
        h, w = gl.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif image_size != (w, h):
            raise ValueError(f"Inconsistent image sizes across pairs: {image_size} vs {(w, h)}")

        ok_l, corners_l = cv.findChessboardCorners(gl, (cols, rows), flags_find)
        ok_r, corners_r = cv.findChessboardCorners(gr, (cols, rows), flags_find)
        if not (ok_l and ok_r):
            continue
        corners_l = cv.cornerSubPix(gl, corners_l, (11, 11), (-1, -1), criteria_subpix)
        corners_r = cv.cornerSubPix(gr, corners_r, (11, 11), (-1, -1), criteria_subpix)
        obj_points.append(objp)
        img_points_l.append(corners_l)
        img_points_r.append(corners_r)

    if len(obj_points) < int(min_valid_pairs) or image_size is None:
        raise ValueError(
            f"Only {len(obj_points)} valid chessboard pairs detected (need at least {min_valid_pairs})"
        )

    rms_l, K_l, dist_l, _, _ = cv.calibrateCamera(
        obj_points, img_points_l, image_size, None, None
    )
    rms_r, K_r, dist_r, _, _ = cv.calibrateCamera(
        obj_points, img_points_r, image_size, None, None
    )
    rms_stereo, K_l, dist_l, K_r, dist_r, R, T, E, F = cv.stereoCalibrate(
        obj_points,
        img_points_l,
        img_points_r,
        K_l,
        dist_l,
        K_r,
        dist_r,
        image_size,
        criteria=criteria_stereo,
        flags=cv.CALIB_FIX_INTRINSIC,
    )
    _ = (E, F)

    calib = StereoCalibration(
        K_left=K_l,
        K_right=K_r,
        dist_left=dist_l.ravel(),
        dist_right=dist_r.ravel(),
        R=R,
        T=T.ravel(),
        image_size=image_size,
        is_rectified=False,
    )
    calib.ensure_rectification(image_size=image_size, alpha=float(alpha))

    # Measure mean vertical epipolar alignment error after rectification
    epi_errors: list[float] = []
    for cl, cr in zip(img_points_l, img_points_r):
        ul = cv.undistortPoints(cl, calib.K_left, calib.dist_left, R=calib.R1, P=calib.P1).reshape(-1, 2)
        ur = cv.undistortPoints(cr, calib.K_right, calib.dist_right, R=calib.R2, P=calib.P2).reshape(-1, 2)
        epi_errors.append(float(np.mean(np.abs(ul[:, 1] - ur[:, 1]))))

    report = {
        "pairs_input": len(left_images),
        "pairs_used": len(obj_points),
        "image_size": list(image_size),
        "board_size": [cols, rows],
        "square_size_m": float(square_size_m),
        "rms_left_px": round(float(rms_l), 4),
        "rms_right_px": round(float(rms_r), 4),
        "rms_stereo_px": round(float(rms_stereo), 4),
        "mean_epipolar_y_error_px": round(float(np.mean(epi_errors)), 4),
        "baseline_m": round(float(calib.baseline), 6),
    }
    return calib, report


def _load_image_pairs(
    left_dir: Path | None,
    right_dir: Path | None,
    sbs_dir: Path | None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    left_imgs: list[np.ndarray] = []
    right_imgs: list[np.ndarray] = []
    if sbs_dir is not None:
        files = sorted(p for p in sbs_dir.iterdir() if p.suffix.lower() in exts)
        for p in files:
            img = cv.imread(str(p), cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            half = img.shape[1] // 2
            left_imgs.append(np.ascontiguousarray(img[:, :half]))
            right_imgs.append(np.ascontiguousarray(img[:, half : 2 * half]))
        return left_imgs, right_imgs

    if left_dir is None or right_dir is None:
        raise ValueError("Provide either --sbs-dir or both --left-dir and --right-dir")
    l_files = sorted(p for p in left_dir.iterdir() if p.suffix.lower() in exts)
    r_files = sorted(p for p in right_dir.iterdir() if p.suffix.lower() in exts)
    if len(l_files) != len(r_files):
        raise ValueError(f"Mismatched image counts: {len(l_files)} left vs {len(r_files)} right")
    for lp, rp in zip(l_files, r_files):
        il = cv.imread(str(lp), cv.IMREAD_GRAYSCALE)
        ir = cv.imread(str(rp), cv.IMREAD_GRAYSCALE)
        if il is not None and ir is not None:
            left_imgs.append(il)
            right_imgs.append(ir)
    return left_imgs, right_imgs


def _capture_live_pairs(
    camera_id: int,
    right_camera_id: int | None,
    layout: str,
    board_size: tuple[int, int],
    num_pairs: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    cap_l = cv.VideoCapture(camera_id)
    if not cap_l.isOpened():
        raise OSError(f"Failed to open camera {camera_id}")
    cap_r = None
    if layout == "dual":
        if right_camera_id is None:
            cap_l.release()
            raise ValueError("Dual layout requires --right-camera-id")
        cap_r = cv.VideoCapture(right_camera_id)
        if not cap_r.isOpened():
            cap_l.release()
            raise OSError(f"Failed to open right camera {right_camera_id}")

    left_imgs: list[np.ndarray] = []
    right_imgs: list[np.ndarray] = []
    print(f"Capturing {num_pairs} chessboard pairs. Press SPACE to capture, ESC/q to finish.")
    try:
        while len(left_imgs) < num_pairs:
            ok_l, raw_l = cap_l.read()
            if not ok_l or raw_l is None:
                continue
            if layout == "sbs":
                half = raw_l.shape[1] // 2
                il = np.ascontiguousarray(raw_l[:, :half])
                ir = np.ascontiguousarray(raw_l[:, half : 2 * half])
            else:
                assert cap_r is not None
                ok_r, ir = cap_r.read()
                if not ok_r or ir is None:
                    continue
                il = raw_l
            gl, gr = _to_gray(il), _to_gray(ir)
            ok_cl, cl = cv.findChessboardCorners(gl, board_size, None)
            ok_cr, cr = cv.findChessboardCorners(gr, board_size, None)
            vis_l = il.copy() if il.ndim == 3 else cv.cvtColor(il, cv.COLOR_GRAY2BGR)
            vis_r = ir.copy() if ir.ndim == 3 else cv.cvtColor(ir, cv.COLOR_GRAY2BGR)
            if ok_cl:
                cv.drawChessboardCorners(vis_l, board_size, cl, ok_cl)
            if ok_cr:
                cv.drawChessboardCorners(vis_r, board_size, cr, ok_cr)
            preview = np.hstack([vis_l, vis_r])
            cv.putText(
                preview,
                f"Captured {len(left_imgs)}/{num_pairs} | SPACE=capture ESC=done",
                (16, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if (ok_cl and ok_cr) else (0, 140, 255),
                2,
            )
            cv.imshow("Stereo Chessboard Calibration", preview)
            key = cv.waitKey(10) & 0xFF
            if key == ord(" ") and ok_cl and ok_cr:
                left_imgs.append(gl)
                right_imgs.append(gr)
                print(f"Captured pair {len(left_imgs)}/{num_pairs}")
            elif key in (27, ord("q")):
                break
    finally:
        cap_l.release()
        if cap_r is not None:
            cap_r.release()
        cv.destroyAllWindows()
    return left_imgs, right_imgs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-dir", type=Path, default=None, help="Directory of left chessboard images")
    parser.add_argument("--right-dir", type=Path, default=None, help="Directory of right chessboard images")
    parser.add_argument("--sbs-dir", type=Path, default=None, help="Directory of side-by-side stereo chessboard images")
    parser.add_argument("--camera-id", type=int, default=None, help="Live capture camera ID")
    parser.add_argument("--right-camera-id", type=int, default=None, help="Live capture right camera ID (dual mode)")
    parser.add_argument("--layout", choices=("sbs", "dual"), default="sbs")
    parser.add_argument("--num-pairs", type=int, default=20, help="Target number of live capture pairs")
    parser.add_argument("--board-cols", type=int, default=9, help="Inner chessboard corner columns")
    parser.add_argument("--board-rows", type=int, default=6, help="Inner chessboard corner rows")
    parser.add_argument("--square-size", type=float, default=0.025, help="Chessboard square size in meters")
    parser.add_argument("--alpha", type=float, default=0.0, help="cv.stereoRectify alpha (0=crop, 1=full)")
    parser.add_argument("--output", type=Path, required=True, help="Output calibration path (.yaml/.yml or .json)")
    args = parser.parse_args(argv)

    board = (args.board_cols, args.board_rows)
    if args.camera_id is not None:
        left_imgs, right_imgs = _capture_live_pairs(
            args.camera_id, args.right_camera_id, args.layout, board, args.num_pairs
        )
    else:
        left_imgs, right_imgs = _load_image_pairs(args.left_dir, args.right_dir, args.sbs_dir)

    calib, report = calibrate_stereo_pairs(
        left_imgs,
        right_imgs,
        board_size=board,
        square_size_m=args.square_size,
        alpha=args.alpha,
    )
    if args.output.suffix.lower() == ".json":
        calib.save_json(args.output)
    else:
        calib.save_yaml(args.output)
    print(json.dumps({"output": str(args.output), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
