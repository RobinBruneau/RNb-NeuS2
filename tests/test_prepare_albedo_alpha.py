"""Repro: 8-bit normal + 16-bit albedo -> albedo alpha must stay opaque.

Reproduces the bug where prepare.py derives mask bit-depth from the normal
image (8-bit PNG) and pastes a 0/255 alpha onto a 16-bit albedo, making the
albedo ~fully transparent (alpha 255/65535).
"""
import os
import sys
import tempfile

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rnb_neus2.prepare import prepare_testbed_data


class _Log:
    def info(self, m): print("[INFO]", m)
    def warning(self, m): print("[WARN]", m)


def main():
    tmp = tempfile.mkdtemp(prefix="albedo_alpha_")
    h, w = 16, 16

    # 8-bit normal PNG (like normalMaps.sfm -> .png)
    normal = np.full((h, w, 3), 128, dtype=np.uint8)
    normal_path = os.path.join(tmp, "n0.png")
    cv2.imwrite(normal_path, normal)

    # 16-bit albedo (like albedoMaps.sfm -> .exr, read as uint16 here)
    albedo = np.full((h, w, 3), 30000, dtype=np.uint16)
    albedo_path = os.path.join(tmp, "a0.png")
    cv2.imwrite(albedo_path, albedo)

    K = np.eye(3, dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[2, 3] = 3.0
    data = {
        "views": [{
            "c2w": c2w, "K": K,
            "normal_path": normal_path,
            "albedo_path": albedo_path,
            "mask_path": None,  # -> full mask
        }],
        "landmarks": None,
        "image_width": w, "image_height": h,
    }

    out = os.path.join(tmp, "prepared")
    prepare_testbed_data(data, out, _Log(), scaling_mode="cameras")

    prep = cv2.imread(os.path.join(out, "albedos", "00000.png"),
                      cv2.IMREAD_UNCHANGED)
    print("prepared albedo:", prep.shape, prep.dtype)
    alpha_max = int(prep[:, :, 3].max())
    expected = 65535 if prep.dtype == np.uint16 else 255
    print("alpha max =", alpha_max, "/ expected (opaque) =", expected)

    if alpha_max == expected:
        print("PASS: albedo mask is opaque")
        return 0
    print("FAIL: albedo mask is ~transparent (alpha {}/{})".format(
        alpha_max, 65535 if prep.dtype == np.uint16 else 255))
    return 1


if __name__ == "__main__":
    sys.exit(main())
