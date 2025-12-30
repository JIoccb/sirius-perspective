import math
import cv2
import numpy as np
import albumentations as A


class PageCurl(A.DualTransform):
    """
    Имитация закручивания края листа.

    Параметры:
        side: "top" | "bottom" | "left" | "right"
        amount: [0..1] – сила закрутки. 0 => тождественное отображение.
        curl_height: (0..1) – относительная высота/ширина полосы от края.
        corner_width: [0..1] – доля ширины листа, где есть заметный эффект
                      (0 => эффект только в самой маленькой зоне у угла,
                       1 => по всей ширине, модифицируется falloff_x).
        falloff_x: [0..1] – как быстро эффект затухает от угла к середине:
                      0 => плавно / почти равномерно,
                      1 => очень резкий локальный загиб только в углах.
        max_angle_deg: максимальный угол загиба при amount=1 (в градусах),
                      по умолчанию 90°.
        gamma_min, gamma_max: диапазон для степени затухания вдоль края.
                      Итоговое gamma = gamma_min + (gamma_max - gamma_min) * falloff_x.
    """

    def __init__(
            self,
            side: str = "top",
            amount: float = 0.7,
            curl_height: float = 0.75,
            corner_width: float = 1.0,
            falloff_x: float = .4,
            max_angle_deg: float = 110.0,
            gamma_min: float = 1.0,
            gamma_max: float = 5.0,
            always_apply: bool = True,
            p: float = 0.5,
    ):
        super().__init__(p=p)

        assert side in ("top", "bottom", "left", "right")
        assert 0.0 <= amount <= 1.0
        assert 0.0 <= curl_height <= 1.0
        assert 0.0 <= corner_width <= 1.0
        assert 0.0 <= falloff_x <= 1.0
        assert gamma_min > 0.0
        assert gamma_max >= gamma_min

        self.side = side
        self.amount = float(amount)
        self.curl_height = float(curl_height)
        self.corner_width = float(corner_width)
        self.falloff_x = float(falloff_x)

        self.max_angle_deg = float(max_angle_deg)
        self.max_angle_rad = math.radians(self.max_angle_deg)

        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)

    def apply(self, img, **params):
        return self._warp(img, is_mask=False)

    def apply_to_mask(self, mask, **params):
        return self._warp(mask, is_mask=True)

    def get_transform_init_args_names(self):
        return (
            "side",
            "amount",
            "curl_height",
            "corner_width",
            "falloff_x",
            "max_angle_deg",
            "gamma_min",
            "gamma_max",
        )

    def apply_to_keypoints(self, keypoints, shape):
        """Transform keypoints using the same curl transformation."""
        h, w = shape[:2]
        if self.amount <= 0.0 or self.curl_height <= 0.0:
            return np.array(keypoints, dtype=np.float32)

        side = self.side
        kpts = np.array(keypoints, dtype=np.float32).copy()

        def _forward_map_centroid(map_x, map_y):
            src_x = np.clip(np.rint(map_x).astype(np.int32), 0, map_x.shape[1] - 1)
            src_y = np.clip(np.rint(map_y).astype(np.int32), 0, map_y.shape[0] - 1)
            dest_x = np.tile(np.arange(map_x.shape[1], dtype=np.float32), (map_x.shape[0], 1))
            dest_y = np.tile(np.arange(map_x.shape[0], dtype=np.float32).reshape(-1, 1), (1, map_x.shape[1]))

            sum_x = np.zeros_like(map_x, dtype=np.float64)
            sum_y = np.zeros_like(map_y, dtype=np.float64)
            counts = np.zeros_like(map_x, dtype=np.int32)

            np.add.at(sum_x, (src_y, src_x), dest_x)
            np.add.at(sum_y, (src_y, src_x), dest_y)
            np.add.at(counts, (src_y, src_x), 1)

            forward_x = np.full_like(map_x, -1.0, dtype=np.float32)
            forward_y = np.full_like(map_y, -1.0, dtype=np.float32)
            mask = counts > 0
            forward_x[mask] = (sum_x[mask] / counts[mask]).astype(np.float32)
            forward_y[mask] = (sum_y[mask] / counts[mask]).astype(np.float32)
            return forward_x, forward_y, counts

        if side in ("top", "bottom"):
            if side == "bottom":
                kpts[:, 1] = h - 1 - kpts[:, 1]

            map_x, map_y = self._build_top_curl_map(
                h=h,
                w=w,
                amount=self.amount,
                curl_height=self.curl_height,
                corner_width=self.corner_width,
                falloff_x=self.falloff_x,
                max_angle_rad=self.max_angle_rad,
                gamma_min=self.gamma_min,
                gamma_max=self.gamma_max,
            )

            fwd_x, fwd_y, counts = _forward_map_centroid(map_x, map_y)

            for i in range(len(kpts)):
                x, y = kpts[i]
                x_int, y_int = int(round(x)), int(round(y))
                x_int = np.clip(x_int, 0, w - 1)
                y_int = np.clip(y_int, 0, h - 1)
                if counts[y_int, x_int] > 0:
                    kpts[i] = [float(fwd_x[y_int, x_int]), float(fwd_y[y_int, x_int])]
                else:
                    new_x, new_y = self._find_destination(map_x, map_y, x_int, y_int, w, h)
                    kpts[i] = [new_x, new_y]

            if side == "bottom":
                kpts[:, 1] = h - 1 - kpts[:, 1]

        else:
            if side == "left":
                kpts_rot = np.zeros_like(kpts)
                kpts_rot[:, 0] = kpts[:, 1]
                kpts_rot[:, 1] = w - 1 - kpts[:, 0]
                kpts = kpts_rot
                h_rot, w_rot = w, h
            else:
                kpts_rot = np.zeros_like(kpts)
                kpts_rot[:, 0] = h - 1 - kpts[:, 1]
                kpts_rot[:, 1] = kpts[:, 0]
                kpts = kpts_rot
                h_rot, w_rot = w, h

            map_x, map_y = self._build_top_curl_map(
                h=h_rot,
                w=w_rot,
                amount=self.amount,
                curl_height=self.curl_height,
                corner_width=self.corner_width,
                falloff_x=self.falloff_x,
                max_angle_rad=self.max_angle_rad,
                gamma_min=self.gamma_min,
                gamma_max=self.gamma_max,
            )

            fwd_x, fwd_y, counts = _forward_map_centroid(map_x, map_y)

            for i in range(len(kpts)):
                x, y = kpts[i]
                x_int, y_int = int(round(x)), int(round(y))
                x_int = np.clip(x_int, 0, w_rot - 1)
                y_int = np.clip(y_int, 0, h_rot - 1)
                if counts[y_int, x_int] > 0:
                    kpts[i] = [float(fwd_x[y_int, x_int]), float(fwd_y[y_int, x_int])]
                else:
                    new_x, new_y = self._find_destination(map_x, map_y, x_int, y_int, w_rot, h_rot)
                    kpts[i] = [new_x, new_y]

            if side == "left":
                kpts_orig = np.zeros_like(kpts)
                kpts_orig[:, 0] = w - 1 - kpts[:, 1]
                kpts_orig[:, 1] = kpts[:, 0]
                kpts = kpts_orig
            else:
                kpts_orig = np.zeros_like(kpts)
                kpts_orig[:, 0] = kpts[:, 1]
                kpts_orig[:, 1] = h - 1 - kpts[:, 0]
                kpts = kpts_orig

        return kpts

    @staticmethod
    def _find_destination(map_x, map_y, src_x, src_y, w, h):
        marker = np.zeros((h, w), dtype=np.uint8)
        y_min, y_max = max(0, src_y - 1), min(h, src_y + 2)
        x_min, x_max = max(0, src_x - 1), min(w, src_x + 2)
        marker[y_min:y_max, x_min:x_max] = 255

        transformed = cv2.remap(
            marker,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        moments = cv2.moments(transformed)
        if moments['m00'] > 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            return cx, cy
        return float(np.clip(src_x, 0, w - 1)), float(np.clip(src_y, 0, h - 1))

    def _warp(self, img, is_mask: bool):
        h, w = img.shape[:2]

        if self.amount <= 0.0 or self.curl_height <= 0.0:
            return img

        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        border_mode = cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REPLICATE
        border_value = 0

        side = self.side

        if side in ("top", "bottom"):
            if side == "bottom":
                img_proc = np.flipud(img)
            else:
                img_proc = img

            map_x, map_y = self._build_top_curl_map(
                h=img_proc.shape[0],
                w=img_proc.shape[1],
                amount=self.amount,
                curl_height=self.curl_height,
                corner_width=self.corner_width,
                falloff_x=self.falloff_x,
                max_angle_rad=self.max_angle_rad,
                gamma_min=self.gamma_min,
                gamma_max=self.gamma_max,
            )

            warped = cv2.remap(
                img_proc,
                map_x,
                map_y,
                interpolation=interpolation,
                borderMode=border_mode,
                borderValue=border_value,
            )

            if side == "bottom":
                warped = np.flipud(warped)

            return warped

        else:

            if side == "left":
                img_rot = np.rot90(img, k=1)
            else:
                img_rot = np.rot90(img, k=-1)

            map_x, map_y = self._build_top_curl_map(
                h=img_rot.shape[0],
                w=img_rot.shape[1],
                amount=self.amount,
                curl_height=self.curl_height,
                corner_width=self.corner_width,
                falloff_x=self.falloff_x,
                max_angle_rad=self.max_angle_rad,
                gamma_min=self.gamma_min,
                gamma_max=self.gamma_max,
            )

            warped_rot = cv2.remap(
                img_rot,
                map_x,
                map_y,
                interpolation=interpolation,
                borderMode=border_mode,
                borderValue=border_value,
            )

            if side == "left":
                warped = np.rot90(warped_rot, k=-1)
            else:
                warped = np.rot90(warped_rot, k=1)

            return warped

    @staticmethod
    def _build_top_curl_map(
            h: int,
            w: int,
            amount: float,
            curl_height: float,
            corner_width: float,
            falloff_x: float,
            max_angle_rad: float,
            gamma_min: float,
            gamma_max: float,
    ):
        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))

        L = min(int(curl_height * h), h - 1)
        if L < 2 or amount <= 0.0:
            return map_x, map_y

        weights = PageCurl._build_horizontal_weights(
            w=w,
            corner_width=corner_width,
            falloff_x=falloff_x,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )

        y_dest = np.arange(L + 1, dtype=np.float32)

        effective_amount = min(1.0, amount * 1.3)

        for x in range(w):
            col_amount = effective_amount * float(weights[x])
            if col_amount <= 1e-4:
                map_y[: L + 1, x] = y_dest
                continue

            y_src = PageCurl._build_column_y_src(
                L=L,
                amount=col_amount,
                max_angle_rad=max_angle_rad,
            )
            map_y[: L + 1, x] = y_src

        # Lateral squeeze near the curled edge to make corners move inward (soft)
        center = 0.5 * (w - 1)
        y_indices = np.arange(h, dtype=np.float32).reshape(-1, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            profile = np.ones_like(y_indices, dtype=np.float32)
            if L > 0:
                strength = (1.0 - y_indices / float(max(L, 1)))
                strength = np.clip(strength, 0.0, 1.0)
                squeeze_strength = 0.20 * effective_amount  # stronger curl pull
                profile = 1.0 - squeeze_strength * (strength ** 2)
                profile = np.clip(profile, 0.60, 1.0)
            map_x = center + (map_x - center) / profile

        map_x = np.clip(map_x, 0, float(w - 1)).astype(np.float32)
        map_y = np.clip(map_y, 0, float(h - 1)).astype(np.float32)
        return map_x, map_y

    @staticmethod
    def _build_column_y_src(
            L: int,
            amount: float,
            max_angle_rad: float,
    ) -> np.ndarray:

        if L < 2 or amount <= 0.0:
            return np.arange(L + 1, dtype=np.float32)

        angle_max = float(amount) * max_angle_rad
        if angle_max < 1e-4:
            return np.arange(L + 1, dtype=np.float32)

        y0 = np.linspace(0.0, float(L), num=L + 1, dtype=np.float32)
        s = float(L) - y0
        phi = angle_max * (s / float(L))
        R = float(L) / angle_max

        y_prime = R * np.sin(phi)
        y_d = float(L) - y_prime

        y_min = float(y_d[0])
        denom = float(L) - y_min
        if abs(denom) < 1e-6:
            return np.arange(L + 1, dtype=np.float32)

        scale = float(L) / denom
        y_d_norm = (y_d - y_min) * scale

        y_dest = np.arange(L + 1, dtype=np.float32)
        y_src = np.interp(y_dest, y_d_norm, y0)
        return y_src.astype(np.float32)

    @staticmethod
    def _build_horizontal_weights(
            w: int,
            corner_width: float,
            falloff_x: float,
            gamma_min: float,
            gamma_max: float,
    ) -> np.ndarray:
        if w <= 1:
            return np.ones(1, dtype=np.float32)

        cw = float(np.clip(corner_width, 0.0, 1.0))
        fo = float(np.clip(falloff_x, 0.0, 1.0))

        if cw >= 1.0 and fo <= 0.0 and gamma_min == gamma_max == 1.0:
            return np.ones(w, dtype=np.float32)

        xs = np.arange(w, dtype=np.float32)
        d = np.minimum(xs, (w - 1) - xs)

        half = 0.5 * (w - 1)
        region = max(cw * half, 1e-3)

        t = d / region
        t = np.clip(t, 0.0, 1.0)

        gamma = gamma_min + (gamma_max - gamma_min) * fo

        weights = (1.0 - t) ** gamma

        if cw < 1.0:
            mask = (d <= region).astype(np.float32)
            weights *= mask

        return weights.astype(np.float32)

