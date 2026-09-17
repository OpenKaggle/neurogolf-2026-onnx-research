import numpy as np

H30, W30 = 30, 30


def _label_4conn(mask):
    """4-connected component labeling. Returns (labeled, n_features)."""
    h, w = mask.shape
    labeled = np.zeros((h, w), dtype=int)
    current = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] and labeled[r, c] == 0:
                current += 1
                stack = [(r, c)]
                labeled[r, c] = current
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labeled[nr, nc] == 0:
                            labeled[nr, nc] = current
                            stack.append((nr, nc))
    return labeled, current


def solver_task366(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape

    if H >= W:
        half = H // 2
        half_a = grid[:half].copy()
        half_b = grid[half:2*half].copy()
    else:
        half = W // 2
        half_a = grid[:, :half].copy()
        half_b = grid[:, half:2*half].copy()

    def get_bg(g):
        vals = g.ravel()
        counts = np.bincount(vals, minlength=10)
        return int(np.argmax(counts))

    bg_a = get_bg(half_a)
    bg_b = get_bg(half_b)

    # Source side has more non-background cells (patterns vs isolated anchors)
    nb_a = (half_a != bg_a).sum()
    nb_b = (half_b != bg_b).sum()
    src_is_a = nb_a > nb_b

    if src_is_a:
        src, src_bg = half_a, bg_a
        dst, dst_bg = half_b, bg_b
    else:
        src, src_bg = half_b, bg_b
        dst, dst_bg = half_a, bg_a

    out = dst.copy()

    # Find source patterns (connected components of area > 1)
    src_mask = (src != src_bg)
    src_labeled, n_src = _label_4conn(src_mask)

    patterns = []
    for feat_id in range(1, n_src + 1):
        mask = (src_labeled == feat_id)
        coords = np.argwhere(mask)
        if len(coords) <= 1:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        patch = src[r0:r1 + 1, c0:c1 + 1].copy()
        patch_mask = mask[r0:r1 + 1, c0:c1 + 1]
        patch[~patch_mask] = 0
        colors = set(src[r, c] for r, c in coords)
        patterns.append({
            'patch': patch,
            'colors_used': colors,
        })

    # Find destination anchors (all non-bg cells in destination)
    dst_mask = (dst != dst_bg)

    anchors = {}
    for r in range(dst.shape[0]):
        for c in range(dst.shape[1]):
            if dst_mask[r, c]:
                color = int(dst[r, c])
                anchors.setdefault(color, []).append((r, c))

    # Match patterns to anchors and place (most key cells first)
    patterns.sort(key=lambda p: -sum((p['patch'] == c).sum() for c in p['colors_used'] & set(anchors.keys())))
    used_anchors = set()

    for pat in patterns:
        patch = pat['patch']
        ph, pw = patch.shape

        key_colors = pat['colors_used'] & set(anchors.keys())
        if not key_colors:
            continue

        for kc in key_colors:
            p_cells = list(zip(*np.where(patch == kc)))
            if not p_cells:
                continue

            avail = [a for a in anchors.get(kc, []) if a not in used_anchors]
            if len(avail) < len(p_cells):
                continue

            anchor_set = set(avail)
            found = False
            for anchor in avail:
                dr = anchor[0] - p_cells[0][0]
                dc = anchor[1] - p_cells[0][1]
                ok = True
                matched = []
                for pr, pc in p_cells:
                    ar, ac = pr + dr, pc + dc
                    if (ar, ac) in anchor_set and (ar, ac) not in used_anchors:
                        matched.append((ar, ac))
                    else:
                        ok = False
                        break
                if ok and len(matched) == len(p_cells):
                    for r in range(ph):
                        for c in range(pw):
                            if patch[r, c] != 0:
                                rr, cc = dr + r, dc + c
                                if 0 <= rr < dst.shape[0] and 0 <= cc < dst.shape[1]:
                                    out[rr, cc] = patch[r, c]
                    for m in matched:
                        used_anchors.add(m)
                    found = True
                    break
            if found:
                break

    return out
