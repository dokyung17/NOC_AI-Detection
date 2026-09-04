"""Slim pyVHR pipeline: video → rPPG (POS) → BPM (Welch) only."""

import os
from importlib import import_module
from inspect import getmembers, isfunction

import pyVHR
import pyVHR.BVP.methods
from pyVHR.BPM.BPM import BVP_to_BPM, multi_est_BPM_median
from pyVHR.BVP.BVP import RGB_sig_to_BVP
from pyVHR.BVP.filters import apply_filter, rgb_filter_th
from pyVHR.extraction.sig_processing import SignalProcessing, SignalProcessingParams
from pyVHR.extraction.skin_extraction_methods import SkinExtractionConvexHull, SkinProcessingParams
from pyVHR.extraction.utils import get_fps, sig_windowing


class Pipeline:
    """Run rPPG on a single video file (ConvexHull + patches + POS + Welch)."""

    def run_on_video(
        self,
        videoFileName,
        roi_approach="patches",
        fixed_roi=None,
        method="cpu_POS",
        bpm_type="welch",
        pre_filt=False,
        post_filt=True,
        verb=False,
        return_bvp=False,
    ):
        ldmks_list = [
            2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50, 54, 58,
            67, 68, 69, 71, 92, 93, 101, 103, 104, 108, 109, 116, 117, 118, 123, 132,
            134, 135, 138, 139, 142, 148, 149, 150, 151, 152, 182, 187, 188, 193, 197,
            201, 205, 206, 207, 210, 211, 212, 216, 234, 248, 251, 262, 265, 266, 273,
            277, 278, 280, 284, 288, 297, 299, 322, 323, 330, 332, 333, 337, 338, 345,
            346, 361, 363, 364, 367, 368, 371, 377, 379, 411, 412, 417, 421, 425, 426,
            427, 430, 432, 436,
        ]
        assert os.path.isfile(videoFileName), f"Video not found: {videoFileName}"

        available_methods = [name for name, _ in getmembers(pyVHR.BVP.methods, isfunction)]
        assert method in available_methods, f"Unknown rPPG method: {method}"
        assert roi_approach in ("patches", "hol", "crop"), (
            "roi_approach must be 'patches', 'hol', or 'crop'"
        )
        assert bpm_type == "welch", "Only bpm_type='welch' is supported in this build"

        sig_processing = SignalProcessing()
        use_face_mesh = roi_approach != "crop"

        if use_face_mesh:
            sig_processing.set_skin_extractor(SkinExtractionConvexHull())

        if roi_approach == "patches":
            sig_processing.set_landmarks(ldmks_list)
            sig_processing.set_square_patches_side(28.0)

        SignalProcessingParams.RGB_LOW_TH = 75
        SignalProcessingParams.RGB_HIGH_TH = 230
        SkinProcessingParams.RGB_LOW_TH = 75
        SkinProcessingParams.RGB_HIGH_TH = 230

        if verb:
            print("Processing video:", videoFileName)
            if roi_approach == "crop":
                print("FaceMesh: OFF | fixed_roi:", fixed_roi or "full_frame")

        fps = get_fps(videoFileName)
        sig_processing.set_total_frames(0)

        if roi_approach == "hol":
            sig = sig_processing.extract_holistic(videoFileName)
        elif roi_approach == "crop":
            sig = sig_processing.extract_crop(videoFileName, fixed_roi)
        else:
            sig = sig_processing.extract_patches(videoFileName, "squares", "mean")

        windowed_sig, timesES = sig_windowing(sig, 6, 1, fps)

        if roi_approach == "patches":
            windowed_sig = apply_filter(
                windowed_sig,
                rgb_filter_th,
                params={"RGB_LOW_TH": 75, "RGB_HIGH_TH": 230},
            )

        module = import_module("pyVHR.BVP.methods")
        method_to_call = getattr(module, method)
        pars = {"fps": "adaptive"} if "POS" in method else {}

        bvps = RGB_sig_to_BVP(
            windowed_sig,
            fps,
            device_type="cpu",
            method=method_to_call,
            params=pars,
        )

        if post_filt:
            bpfilter = getattr(import_module("pyVHR.BVP.filters"), "BPfilter")
            bvps = apply_filter(
                bvps,
                bpfilter,
                fps=fps,
                params={"minHz": 0.65, "maxHz": 4.0, "fps": "adaptive", "order": 6},
            )

        bpmES = BVP_to_BPM(bvps, fps, minHz=0.65, maxHz=4.0)
        median_bpmES, mad_bpmES = multi_est_BPM_median(bpmES)

        if return_bvp:
            return timesES, median_bpmES, mad_bpmES, bvps, fps
        return timesES, median_bpmES, mad_bpmES
