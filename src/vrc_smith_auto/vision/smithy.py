from pathlib import Path

import cv2
import numpy as np

_IMAGES = Path(__file__).resolve().parents[1] / "images"
# Apparent sizes of in-world controls relative to the height-normalized reference.
_BUTTON_SCALE_PERCENTS = range(48, 153, 4)
_SEARCH_DOWNSAMPLE = 4
_POINTER_OUTLINE = np.array(((0, 0), (19, 6), (11, 10), (7, 19)), dtype=np.int32)
# Slot centers measured in each reference. The title and footer locate the panel
# without relying on the count, item icons, or item colors.
_PANELS = (
    ("empty_storage", (43, 98, 154, 209), (61, 117, 172), (13, 4, 123, 27), (49, 203, 208, 238)),
    ("full_storage", (35, 91, 146, 201), (57, 112, 166), (5, 3, 115, 23), (48, 198, 200, 233)),
)


def _load(name: str) -> np.ndarray:
    image = cv2.imread(str(_IMAGES / f"{name}.png"))
    if image is None:
        raise ValueError(f"Could not read Smithy reference image: {name}.png")
    return image


def _match(image: np.ndarray, template: np.ndarray) -> tuple[float, tuple[int, int]]:
    if not template.size or any(length < size for length, size in zip(image.shape[:2], template.shape[:2])):
        return 1.0, (0, 0)
    score, _, origin, _ = cv2.minMaxLoc(cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED))
    return score, origin


def _match_contrast(image: np.ndarray, template: np.ndarray) -> tuple[float, tuple[int, int]]:
    if any(length < size for length, size in zip(image.shape[:2], template.shape[:2])):
        return 1.0, (0, 0)
    _, correlation, _, origin = cv2.minMaxLoc(cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED))
    return 1.0 - correlation, origin


class SmithyVision:
    def __init__(self, auto_sell: bool, *, perspective_storage: bool = False) -> None:
        names = ["hit", "hit_unavailable", "fire_up", "fire_up_unavailable"]
        if auto_sell:
            names.append("smithy")
        self._buttons = {name: cv2.cvtColor(_load(name), cv2.COLOR_BGR2GRAY) for name in names}
        self._panels = {name: _load(name) for name, *_ in _PANELS} if auto_sell else {}
        self._perspective_storage = perspective_storage
        self._storage_bounds: tuple[int, int, int, int] | None = None
        self._storage_track: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if auto_sell and perspective_storage:
            self._storage_sift = cv2.SIFT_create(contrastThreshold=0.012)
            self._storage_matcher = cv2.BFMatcher()
            self._storage_reference = cv2.cvtColor(self._panels["empty_storage"], cv2.COLOR_BGR2GRAY)
            self._storage_features = []
            _, xs, ys, _, _ = _PANELS[0]
            for name, ref_xs, ref_ys, _, _ in _PANELS:
                gray = cv2.cvtColor(self._panels[name], cv2.COLOR_BGR2GRAY)
                points, descriptors = self._storage_sift.detectAndCompute(gray, None)
                # Both screenshots use the same grid with different crop offsets.
                scale = np.array([(xs[-1] - xs[0]) / (ref_xs[-1] - ref_xs[0]),
                                  (ys[-1] - ys[0]) / (ref_ys[-1] - ref_ys[0])])
                canonical = (np.float32([point.pt for point in points]) - (ref_xs[0], ref_ys[0])) * scale
                self._storage_features.append((canonical + (xs[0], ys[0]), descriptors))
        self._background = self._panels["empty_storage"][61, 43] if auto_sell else None
        self._size = (0, 0)
        self._scale_templates: dict[tuple[str, bool], list[tuple[np.ndarray, np.ndarray]]] = {}
        self._button_interiors = {}
        for name, image in self._buttons.items():
            h, w = image.shape
            # Keep the center while removing variable borders and shadows.
            trim = 0.0 if name == "fire_up_unavailable" else 0.14
            x, y = round(w * trim), round(h * trim)
            self._button_interiors[name] = image[y:h - y, x:w - x]

    def prepare(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        self._frame = frame
        self._gray = None
        self._coarse = None
        self._scale_search = None
        self._cursor_result = None
        self._cursor_checked = False
        if (width, height) != self._size:
            self._size = (width, height)
            self._storage_bounds = None
            self._storage_track = None
            self._sx, self._sy = width / 1920, height / 1080
            self._scaled_buttons = {
                name: tuple(
                    self._resize(image, sx, self._sy) for sx in dict.fromkeys((self._sx, self._sy))
                )
                for name, image in self._buttons.items()
            }
            self._coarse_hits = tuple(
                self._resize(image, 0.5, 0.5) for image in self._scaled_buttons["hit_unavailable"]
            )
            self._scale_templates.clear()
            self._scaled_panels = {
                name: self._resize(image, self._sx, self._sy) for name, image in self._panels.items()
            }

    @staticmethod
    def _resize(image: np.ndarray, sx: float, sy: float) -> np.ndarray:
        h, w = image.shape[:2]
        return cv2.resize(image, (max(1, round(w * sx)), max(1, round(h * sy))),
                          interpolation=cv2.INTER_AREA if max(sx, sy) <= 1 else cv2.INTER_LINEAR)

    def button(self, name: str) -> tuple[int, int] | None:
        matches = []
        for index, template in enumerate(self._scaled_buttons[name]):
            coarse = self._coarse_hits[index] if name == "hit_unavailable" else None
            match = self._button_match(template, coarse)
            if match is not None:
                matches.append(match)
        if matches:
            return min(matches)[1]
        point = self._match_button_at_other_scales(name)
        if point is None and name == "fire_up":
            point = self._match_button_at_other_scales(name, use_border=True)
        return point

    def _match_button_at_other_scales(
        self, name: str, *, use_border: bool = False
    ) -> tuple[int, int] | None:
        factor = _SEARCH_DOWNSAMPLE
        reference = self._buttons[name] if use_border else self._button_interiors[name]
        key = (name, use_border)
        if key not in self._scale_templates:
            templates = [self._resize(reference, self._sy * percent / 100, self._sy * percent / 100)
                         for percent in _BUTTON_SCALE_PERCENTS]
            self._scale_templates[key] = [
                (template, self._resize(template, 1 / factor, 1 / factor)) for template in templates
            ]
        templates = self._scale_templates[key]

        if self._gray is None:
            self._gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)
        if self._scale_search is None:
            self._scale_search = self._resize(self._gray, 1 / factor, 1 / factor)

        # Rank on a smaller image and verify only the best candidates at full resolution.
        candidates = []
        for index, (_, small) in enumerate(templates):
            score, (x, y) = _match(self._scale_search, small)
            if score <= 0.20:
                candidates.append((score, index, x, y))

        matches = []
        threshold = 0.12 if name == "fire_up" else 0.10
        for _, index, x, y in sorted(candidates)[:6]:
            candidate, _ = templates[index]
            h, w = candidate.shape
            x, y = x * factor, y * factor
            x0, y0 = max(0, x - 2 * factor), max(0, y - 2 * factor)
            patch = self._gray[y0:y + h + 2 * factor, x0:x + w + 2 * factor]
            percent = _BUTTON_SCALE_PERCENTS[index]
            for fine_percent in range(percent - 2, percent + 3):
                scale = self._sy * fine_percent / 100
                template = self._resize(reference, scale, scale)
                score, (dx, dy) = _match(patch, template)
                if score <= threshold:
                    th, tw = template.shape
                    matches.append((score, (x0 + dx + tw // 2, y0 + dy + th // 2)))
        return min(matches)[1] if matches else None

    def _button_match(
        self, template: np.ndarray, coarse: np.ndarray | None
    ) -> tuple[float, tuple[int, int]] | None:
        width, height = self._size
        h, w = template.shape
        x0 = y0 = 0
        if coarse is not None:
            if self._coarse is None:
                small = cv2.resize(self._frame, (max(1, width // 2), max(1, height // 2)),
                                   interpolation=cv2.INTER_AREA)
                self._coarse = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            coarse_score, (cx, cy) = _match(self._coarse, coarse)
            if coarse_score > 0.17:
                return None
            x0, y0 = max(0, 2 * cx - 6), max(0, 2 * cy - 6)
            patch = self._frame[y0:2 * cy + h + 6, x0:2 * cx + w + 6]
            image = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            if self._gray is None:
                self._gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)
            image = self._gray
        score, (x, y) = _match(image, template)
        if score > 0.10:
            return None
        return score, (x0 + x + w // 2, y0 + y + h // 2)

    def cursor(self) -> tuple[int, int] | None:
        """Locate the tip of VRChat's white-outlined desktop pointer."""
        if self._cursor_checked:
            return self._cursor_result
        self._cursor_checked = True
        dark = cv2.inRange(self._frame, (0, 0, 0), (32, 32, 32))
        contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        matches = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (11 * self._sy <= w <= 21 * self._sy and 11 * self._sy <= h <= 21 * self._sy):
                continue
            # The dark interior stays visible when the white outline overlaps button text.
            cx = x + w // 2
            halo_y = min(y + h + max(1, round(2 * self._sy)), self._frame.shape[0] - 1)
            blue, _, red = self._frame[halo_y, cx]
            if int(blue) - int(red) < 18:
                continue
            score = cv2.matchShapes(contour, _POINTER_OUTLINE, cv2.CONTOURS_MATCH_I1, 0)
            if score < 0.18:
                matches.append((score, (max(0, x - 1), max(0, y - 1))))
        self._cursor_result = min(matches)[1] if matches else None
        if self._cursor_result is None:
            # In camera mode VRChat aims at a fixed reticle and shows a small
            # cyan arrow beside it when the world accepts pointer interaction.
            width, height = self._size
            cx, cy = width // 2, height // 2
            radius = max(1, round(28 * self._sy))
            center = self._frame[max(0, cy - radius):cy + radius, max(0, cx - radius):cx + radius]
            cyan = cv2.inRange(center, (118, 101, 0), (240, 230, 90))
            if cv2.countNonZero(cyan) >= max(4, round(26 * self._sy ** 2)):
                self._cursor_result = (cx, cy)
        return self._cursor_result

    def hovered_button(self, name: str, hit: tuple[int, int] | None = None) -> tuple[int, int] | None:
        """Match a button under the pointer while masking the pointer itself."""
        cursor = self.cursor()
        if cursor is None and hit is None:
            return None
        ref = self._buttons[name]
        h, w = ref.shape
        radius_x, radius_y = round(w * self._sy), round(h * self._sy)
        smithy = name == "smithy"
        if smithy:
            # The wide border is distorted by camera yaw. Match the centered
            # label and mask the pointer before testing the remaining pixels.
            margin_x, margin_y = round(w * 0.26), round(h * 0.14)
            ref = ref[margin_y:h - margin_y, margin_x:w - margin_x]
            radius_x = round(radius_x * 0.75)
        if hit is None:
            x, y = cursor
            x0, y0 = max(0, x - radius_x), max(0, y - radius_y)
            frame = self._frame[y0:y + radius_y, x0:x + radius_x]
        else:
            x, y = hit
            x0, y0 = max(0, x - 3 * radius_x), max(0, y - radius_y)
            frame = self._frame[y0:y + radius_y, x0:x]
        if frame.size == 0:
            return None
        weights = np.ones(frame.shape[:2], np.float32)
        if cursor is not None:
            rim = max(1, round(4 * self._sy))
            tip_x, tip_y = cursor[0] - x0, cursor[1] - y0
            weights[max(0, tip_y - rim):max(0, tip_y + 5 * rim),
                    max(0, tip_x - rim):max(0, tip_x + 5 * rim)] = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        weighted = weights * gray
        best = None
        for percent in _BUTTON_SCALE_PERCENTS:
            scale = self._sy * percent / 100
            template = self._resize(ref, scale, scale).astype(np.float32)
            th, tw = template.shape
            if th > frame.shape[0] or tw > frame.shape[1]:
                continue
            ones = np.ones_like(template)
            energy = cv2.matchTemplate(weighted * gray, ones, cv2.TM_CCORR)
            cross = cv2.matchTemplate(weighted, template, cv2.TM_CCORR)
            reference = cv2.matchTemplate(weights, template * template, cv2.TM_CCORR)
            coverage = cv2.matchTemplate(weights, ones, cv2.TM_CCORR)
            score = (energy + reference - 2 * cross) / np.sqrt(np.maximum(1, energy * reference))
            score[coverage < tw * th * 0.52] = 1
            cx = x0 + tw // 2 + np.arange(score.shape[1])
            cy = y0 + th // 2 + np.arange(score.shape[0])
            if hit is None:
                # For Smithy the template is only the label. Test containment
                # using the full button footprint, with room inside its edges.
                limit_x = w * scale * 0.36 if smithy else tw * 0.33
                limit_y = h * scale * 0.30 if smithy else th * 0.33
                score[:, np.abs(cx - x) > limit_x] = 1
                score[np.abs(cy - y) > limit_y, :] = 1
            else:
                # Fire Up and Hit share a row on the forge panel. Their screen-space
                # separation scales with the projected size of the controls.
                score[:, ((x - cx) < tw * 0.9) | ((x - cx) > tw * 1.85)] = 1
                score[np.abs(cy - y) > th * 0.55, :] = 1
            value, _, (dx, dy), _ = cv2.minMaxLoc(score)
            point = cursor if smithy else (x0 + dx + tw // 2, y0 + dy + th // 2)
            threshold = 0.16 if name == "smithy" else 0.26 if hit is not None else 0.21
            if value <= threshold and (best is None or value < best[0]):
                best = (value, point)
        return best[1] if best is not None else None

    def storage(self) -> list[tuple[int, int]] | None:
        """Return occupied slot centers, or None if the storage panel is unconfirmed."""
        return self._storage_by_labels() if not self._perspective_storage else self._storage_by_grid()

    def _storage_by_grid(self) -> list[tuple[int, int]] | None:
        """Find the storage panel and inspect its rectified slot borders."""
        if self._gray is None:
            self._gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)
        width, height = self._size
        bounds = self._storage_bounds or (0, 0, width, height)
        transform = self._find_storage_transform(bounds)
        if transform is None:
            transform = self._track_storage_transform()
        if transform is None and self._storage_bounds is not None:
            transform = self._find_storage_transform((0, 0, width, height))
        if transform is None:
            return None

        ref_height, ref_width = self._storage_reference.shape
        projected = self._storage_corners(transform)
        if projected is None:
            return None
        left, top, panel_width, panel_height = cv2.boundingRect(projected)
        mask = np.zeros_like(self._gray)
        cv2.fillConvexPoly(mask, np.round(projected).astype(np.int32), 255)
        features = cv2.goodFeaturesToTrack(self._gray, 140, 0.01, 5, mask=mask)
        if features is not None and len(features) >= 12:
            self._storage_track = (self._gray, features, transform)
        margin = max(24, round(110 * self._sy))
        self._storage_bounds = (
            max(0, left - margin), max(0, top - margin),
            min(width, left + panel_width + margin), min(height, top + panel_height + margin),
        )
        panel = cv2.warpPerspective(
            self._frame, transform, (ref_width, ref_height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )
        _, xs, ys, _, _ = _PANELS[0]
        occupied = []
        bright = np.max(panel, axis=2) > 100
        for y in ys:
            for x in xs:
                # Occupied slots have a bright frame in every item tier. The
                # pointer can light up one edge of an otherwise empty cell.
                sides = (
                    bright[y - 15:y + 16, x - 25:x - 15],
                    bright[y - 15:y + 16, x + 15:x + 25],
                    bright[y - 25:y - 15, x - 15:x + 16],
                    bright[y + 15:y + 25, x - 15:x + 16],
                )
                if sum(np.mean(side) > 0.17 for side in sides) >= 3:
                    occupied.append((x, y))
        if not occupied:
            return []
        points = cv2.perspectiveTransform(np.float32([occupied]), transform)[0]
        return [tuple(point) for point in np.round(points).astype(int).tolist()]

    def _storage_corners(self, transform: np.ndarray) -> np.ndarray | None:
        width, height = self._size
        ref_height, ref_width = self._storage_reference.shape
        corners = np.float32([[[0, 0], [ref_width, 0], [ref_width, ref_height], [0, ref_height]]])
        projected = cv2.perspectiveTransform(corners, transform)[0]
        if not cv2.isContourConvex(projected) or np.any(projected < 0) or np.any(projected >= (width, height)):
            return None
        _, _, panel_width, panel_height = cv2.boundingRect(projected)
        if (0.45 * ref_width * self._sy <= panel_width <= 1.8 * ref_width * self._sy
                and 0.45 * ref_height * self._sy <= panel_height <= 1.8 * ref_height * self._sy):
            return projected
        return None

    def _track_storage_transform(self) -> np.ndarray | None:
        if self._storage_track is None:
            return None
        previous, features, transform = self._storage_track
        current, found, _ = cv2.calcOpticalFlowPyrLK(previous, self._gray, features, None)
        if current is None:
            return None
        returned, checked, _ = cv2.calcOpticalFlowPyrLK(self._gray, previous, current, None)
        if returned is None:
            return None
        error = np.linalg.norm((features - returned).reshape(-1, 2), axis=1)
        valid = (found.ravel() != 0) & (checked.ravel() != 0) & (error < 1.5)
        if np.count_nonzero(valid) < 12:
            return None
        motion, inliers = cv2.findHomography(features[valid], current[valid], cv2.RANSAC, 2.0)
        if inliers is None:
            return None
        anchors = features[valid][inliers.ravel() != 0, 0, :]
        ref_height, ref_width = self._storage_reference.shape
        normalized = cv2.perspectiveTransform(anchors.reshape(1, -1, 2), np.linalg.inv(transform))[0]
        if len(anchors) < 12 or np.any(np.ptp(normalized, axis=0) < (ref_width * 0.45, ref_height * 0.55)):
            return None
        updated = motion @ transform
        return updated if self._storage_corners(updated) is not None else None

    def _find_storage_transform(self, bounds: tuple[int, int, int, int]) -> np.ndarray | None:
        left, top, right, bottom = bounds
        points, descriptors = self._storage_sift.detectAndCompute(self._gray[top:bottom, left:right], None)
        if descriptors is None or len(descriptors) < 2:
            return None
        ref_height, ref_width = self._storage_reference.shape
        for reference_points, reference_descriptors in self._storage_features:
            pairs = self._storage_matcher.knnMatch(reference_descriptors, descriptors, k=2)
            matches = [first for first, second in pairs if first.distance < 0.75 * second.distance]
            if len(matches) < 12:
                continue
            source = reference_points[[match.queryIdx for match in matches]]
            target = np.float32([points[match.trainIdx].pt for match in matches]) + (left, top)
            transform, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
            if inliers is None:
                continue
            aligned = source[inliers.ravel() != 0]
            if (len(aligned) >= 12 and np.all(np.ptp(aligned, axis=0) >= (ref_width * 0.45, ref_height * 0.65))
                    and self._storage_corners(transform) is not None):
                return transform
        return None

    def _storage_by_labels(self) -> list[tuple[int, int]] | None:
        matches = []
        if self._gray is None:
            self._gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)
        for name, xs, ys, title, footer in _PANELS:
            panel = self._scaled_panels[name]
            gray_panel = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
            h, w = panel.shape[:2]
            tx0, ty0, tx1, ty1 = self._scaled_bounds(title)
            fx0, fy0, fx1, fy1 = self._scaled_bounds(footer)
            score, (x, y) = _match_contrast(self._gray, gray_panel[ty0:ty1, tx0:tx1])
            x, y = x - tx0, y - ty0
            if score > 0.48 or x < 0 or y < 0 or x + w > self._size[0] or y + h > self._size[1]:
                continue
            footer_template = gray_panel[fy0:fy1, fx0:fx1]
            # Independently confirm the footer near the location predicted by the title.
            margin = max(3, round(8 * self._sy))
            footer_image = self._gray[max(0, y + fy0 - margin):y + fy1 + margin,
                                       max(0, x + fx0 - margin):x + fx1 + margin]
            footer_score, _ = _match_contrast(footer_image, footer_template)
            if footer_score <= 0.43:
                matches.append((score + footer_score, x, y, xs, ys))
        if not matches:
            return None

        _, x, y, xs, ys = min(matches)
        occupied = []
        for cy in ys:
            for cx in xs:
                x0, y0, x1, y1 = self._scaled_bounds((cx - 13, cy - 13, cx + 14, cy + 14))
                patch = self._frame[y + y0:y + y1, x + x0:x + x1]
                if not patch.size:
                    return None
                difference = np.max(np.abs(patch.astype(np.int16) - self._background), axis=2)
                if np.mean(difference > 17) > 0.07:
                    occupied.append((x + round(cx * self._sx), y + round(cy * self._sy)))
        return occupied

    def _scaled_bounds(self, bounds: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = bounds
        return round(x0 * self._sx), round(y0 * self._sy), round(x1 * self._sx), round(y1 * self._sy)
