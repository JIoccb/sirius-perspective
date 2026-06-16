from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from page_curl_upd import PageCurl


@dataclass(frozen=True)
class PerspectiveWarperConfig:
    canvas_scale: float = 1.2
    doc_scale_range: Tuple[float, float] = (0.9, 1.1)
    jitter_ratio_range: Tuple[float, float] = (0.05, 0.12)
    margin_px: int = 50
    page_curl_prob: float = 1.0
    page_curl_amount_range: Tuple[float, float] = (0.7, 0.9)
    page_curl_height_range: Tuple[float, float] = (0.15, 0.35)
    page_curl_corner_width_range: Tuple[float, float] = (0.7, 1.0)
    page_curl_falloff_range: Tuple[float, float] = (0.6, 0.9)
    p_perspective: float = 0.7
    p_rot: float = 0.7
    p_curl: float = 0.7


class PerspectiveWarper:
    def __init__(self, config: PerspectiveWarperConfig | None = None) -> None:
        self.config = config or PerspectiveWarperConfig()

    def apply(
        self,
        page_np: np.ndarray,
        background_np: np.ndarray,
        p_perspective: float | None = None,
        p_rot: float | None = None,
        p_curl: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> Tuple[np.ndarray, List[List[float]], List[List[float]], int, int, np.ndarray]:
        rng = rng or np.random.default_rng()
        p_perspective = self.config.p_perspective if p_perspective is None else p_perspective
        p_rot = self.config.p_rot if p_rot is None else p_rot
        p_curl = self.config.p_curl if p_curl is None else p_curl

        canvas_w = max(1, int(page_np.shape[1] * self.config.canvas_scale))
        canvas_h = max(1, int(page_np.shape[0] * self.config.canvas_scale))

        bg_h, bg_w = background_np.shape[:2]
        scale_bg = max(canvas_w / bg_w, canvas_h / bg_h)
        new_w = max(int(bg_w * scale_bg), canvas_w)
        new_h = max(int(bg_h * scale_bg), canvas_h)
        bg_resized = cv2.resize(
            background_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

        x0_bg = max((new_w - canvas_w) // 2, 0)
        y0_bg = max((new_h - canvas_h) // 2, 0)
        canvas_np = bg_resized[y0_bg: y0_bg +
                               canvas_h, x0_bg: x0_bg + canvas_w].copy()

        ph_orig, pw_orig = page_np.shape[:2]

        original_keypoints = np.array(
            [
                [0, 0],              # TL
                [pw_orig-1, 0],      # TR
                [pw_orig-1, ph_orig-1],  # BR
                [0, ph_orig-1],      # BL
            ],
            dtype=np.float32,
        )

        doc_scale = float(rng.uniform(*self.config.doc_scale_range))
        target_w = pw_orig * doc_scale
        target_h = ph_orig * doc_scale

        src_pts = np.array([
            [0, 0],
            [pw_orig-1, 0],
            [pw_orig-1, ph_orig-1],
            [0, ph_orig-1],
        ], dtype=np.float32)

        hw, hh = target_w / 2.0, target_h / 2.0
        quad = np.array(
            [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]],
            dtype=np.float32,
        )

        if rng.random() < p_perspective:
            jitter_base = min(target_w, target_h)
            jitter_px = jitter_base * \
                float(rng.uniform(*self.config.jitter_ratio_range))
            noise = rng.uniform(-jitter_px, jitter_px,
                                size=(4, 2)).astype(np.float32)
            quad += noise

        min_x, max_x = np.min(quad[:, 0]), np.max(quad[:, 0])
        min_y, max_y = np.min(quad[:, 1]), np.max(quad[:, 1])
        bb_w, bb_h = max_x - min_x, max_y - min_y

        margin = self.config.margin_px
        avail_w, avail_h = canvas_w - 2 * margin, canvas_h - 2 * margin

        if bb_w > avail_w or bb_h > avail_h:
            scale_fit = min(
                avail_w / bb_w if bb_w > 0 else 1.0,
                avail_h / bb_h if bb_h > 0 else 1.0,
            ) * 0.99
            quad *= scale_fit
            min_x, max_x = min_x * scale_fit, max_x * scale_fit
            min_y, max_y = min_y * scale_fit, max_y * scale_fit

        min_sx, max_sx = margin - min_x, canvas_w - margin - max_x
        min_sy, max_sy = margin - min_y, canvas_h - margin - max_y

        sx = rng.uniform(min_sx, max_sx) if max_sx >= min_sx else canvas_w / 2
        sy = rng.uniform(min_sy, max_sy) if max_sy >= min_sy else canvas_h / 2

        dst_pts = quad + np.array([sx, sy], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts.astype(np.float32))

        h, w = page_np.shape[:2]

        angle = np.random.randint(-45, 45) if rng.random() < p_rot else 0
        scale = 1.0
        center = (canvas_w / 2, canvas_h / 2)
        M_rotation_affine = cv2.getRotationMatrix2D(center, angle, scale)

        M_rotation_persp = np.vstack([M_rotation_affine, [0, 0, 1]])

        M_final = M_rotation_persp @ M

        warped_doc = cv2.warpPerspective(
            page_np,
            M_final,
            (canvas_w, canvas_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        mask_base = np.full((ph_orig, pw_orig), 255, dtype=np.uint8)
        warped_mask = cv2.warpPerspective(
            mask_base,
            M_final,
            (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,),
        )

        kpts_after_warp = cv2.perspectiveTransform(
            src_pts.reshape(-1, 1, 2),
            M_final,
        ).reshape(-1, 2)
        current_kpts = kpts_after_warp.copy()

        if rng.random() < p_curl and self.config.page_curl_prob > 0:
            amount = float(rng.uniform(*self.config.page_curl_amount_range))
            curl_height = float(rng.uniform(
                *self.config.page_curl_height_range))
            corner_width = float(rng.uniform(
                *self.config.page_curl_corner_width_range))
            falloff_x = float(rng.uniform(
                *self.config.page_curl_falloff_range))
            side = rng.choice(["top", "bottom", "left", "right"])

            curl_aug = PageCurl(
                side=side,
                amount=amount,
                curl_height=curl_height,
                corner_width=corner_width,
                falloff_x=falloff_x,
                p=1.0,
            )

            res = curl_aug(
                image=warped_doc,
                mask=warped_mask,
            )

            warped_doc = res.get("image", warped_doc)
            warped_mask = res.get("mask", warped_mask)

            _, warped_mask = cv2.threshold(
                warped_mask, 240, 255, cv2.THRESH_BINARY)

            current_kpts = curl_aug.apply_to_keypoints(
                keypoints=current_kpts.tolist(),
                shape=(canvas_h, canvas_w)
            )

        kernel_erode = np.ones((3, 3), np.uint8)
        warped_mask_clean = cv2.erode(warped_mask, kernel_erode, iterations=1)
        warped_mask_soft = cv2.GaussianBlur(warped_mask_clean, (3, 3), 0)

        mask_3ch = cv2.merge(
            [warped_mask_soft, warped_mask_soft, warped_mask_soft]
        )
        mask_float = mask_3ch.astype(np.float32) / 255.0

        doc_float = warped_doc.astype(np.float32)
        bg_float = canvas_np.astype(np.float32)

        composite = (
            doc_float * mask_float + bg_float * (1.0 - mask_float)
        ).astype(np.uint8)

        kpts_norm_orig = self._normalize_keypoints(
            src_pts, pw_orig, ph_orig
        )
        kpts_norm_final = self._normalize_keypoints(
            current_kpts, canvas_w, canvas_h
        )

        return (
            composite,
            kpts_norm_orig,
            kpts_norm_final,
            canvas_w,
            canvas_h,
            warped_mask_clean,
        )

    @staticmethod
    def _normalize_keypoints(
        points: np.ndarray, w: int, h: int
    ) -> List[List[float]]:
        w, h = max(1, w), max(1, h)
        return [[float(p[0] / w), float(p[1] / h)] for p in points]


def warp_with_seed(page_np, background_np, seed, config=None):
    warper = PerspectiveWarper(config=config)
    rng = np.random.default_rng(seed)
    return warper.apply(page_np, background_np, rng=rng)
