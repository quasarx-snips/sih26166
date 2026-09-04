"""
Terrain modules for LunaX -- ported from lunar_map.ipynb.

Everything here consumes the SAME risk map main.generate_hazard_overlay()
already produced (see the *_risk.npy sidecar it now saves) -- nothing here
re-runs the AI model or re-tiles the image.

  - compute_terrain_masks()  : notebook Cell 0's morphology, run once on the
                                full image (no tiling -- these are cheap CV
                                ops, only the AI inference needed tiling).
                                Supplies crater/boulder/steep masks + gradient
                                magnitude that Module 2 needs but the tiled
                                hazard pipeline doesn't expose.
  - compute_landing_sites()  : notebook Cell 1 (LUNAX 5-factor suitability
                                engine), parameterized instead of using
                                globals, returning JSON-safe data (no plots).
  - astar_pathfinding()      : notebook Cell 2's A*, unchanged.
  - compute_routes()         : notebook Cell 2's 3-strategy route generation
                                (Distance Priority / Balanced / Max Safety),
                                using a dropped start/goal pin instead of
                                ipywidgets sliders.
  - compute_terrain_3d()     : notebook Cell 3's heightmap + route projection,
                                returning plain arrays for Plotly.js on the
                                frontend instead of Python-side plotly figures.
"""

import heapq
import numpy as np
import cv2

from main import (
    CRATER_SCALES, CRATER_PERCENTILE, CRATER_MIN_AREA_FRAC, CRATER_CIRCULARITY,
    DARK_SHADOW_THRESH, DARK_MIN_AREA_FRAC, DARK_CIRCULARITY,
    STEEP_PERCENTILE, MIN_STEEP_AREA, BOULDER_MIN_AREA, BOULDER_MAX_AREA,
)

NUM_LANDING_SITES = 5
LANDER_RADIUS_PX = 4
MIN_SITE_SPACING_PX = 60


# ---------------------------------------------------------------------------
# Module 2 support: full-image (untiled) morphological masks
# ---------------------------------------------------------------------------
def compute_terrain_masks(baseline_bgr: np.ndarray) -> dict:
    """Crater/boulder/steep masks + gradient magnitude, full-image, no AI."""
    gray_u8 = cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray_u8.shape
    diag = float(np.sqrt(H ** 2 + W ** 2))
    den = cv2.medianBlur(gray_u8, 3)

    # --- Craters (multi-scale blackhat) ---
    crater_radii = [max(3, int(diag * f)) for f in CRATER_SCALES]
    blackhat_stack = np.zeros((H, W), dtype=np.float32)
    for r in crater_radii:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        bh = cv2.morphologyEx(den, cv2.MORPH_BLACKHAT, k).astype(np.float32)
        blackhat_stack = np.maximum(blackhat_stack, bh)

    thr = np.percentile(blackhat_stack, CRATER_PERCENTILE)
    crater_bin = (blackhat_stack > thr).astype(np.uint8)
    crater_bin = cv2.morphologyEx(crater_bin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crater_bin, connectivity=8)
    crater_mask = np.zeros((H, W), dtype=bool)
    min_area = CRATER_MIN_AREA_FRAC * diag ** 2
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        comp = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        if 4 * np.pi * area / (perim ** 2) > CRATER_CIRCULARITY:
            crater_mask |= (labels == i)

    # --- Dark shadow craters ---
    _, dark_bin = cv2.threshold(den, DARK_SHADOW_THRESH, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(dark_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dark_min_area = DARK_MIN_AREA_FRAC * diag ** 2
    for c in contours:
        area = cv2.contourArea(c)
        if area < dark_min_area:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        if 4 * np.pi * area / (perim ** 2) > DARK_CIRCULARITY:
            tmp = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(tmp, [c], -1, 1, -1)
            crater_mask |= tmp.astype(bool)

    # --- Boulders ---
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat = cv2.morphologyEx(den, cv2.MORPH_TOPHAT, k_small)
    boulder_bin = (tophat > np.percentile(tophat, 99.2)).astype(np.uint8)
    n_labels_b, labels_b, stats_b, _ = cv2.connectedComponentsWithStats(boulder_bin, connectivity=8)
    boulder_mask = np.zeros((H, W), dtype=bool)
    for i in range(1, n_labels_b):
        area = stats_b[i, cv2.CC_STAT_AREA]
        if BOULDER_MIN_AREA <= area <= BOULDER_MAX_AREA:
            boulder_mask |= (labels_b == i)
    boulder_mask &= ~crater_mask

    # --- Steep slopes ---
    gray_f = gray_u8.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=2.5)
    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    steep_raw = grad_mag > np.percentile(grad_mag, STEEP_PERCENTILE)
    steep_clean = cv2.morphologyEx(steep_raw.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(steep_clean, connectivity=8)
    steep_mask = np.zeros((H, W), dtype=bool)
    for i in range(1, n_labels2):
        if stats2[i, cv2.CC_STAT_AREA] >= MIN_STEEP_AREA:
            steep_mask |= (labels2 == i)
    steep_mask &= ~crater_mask

    return {
        "gray_u8": gray_u8,
        "crater_mask": crater_mask,
        "boulder_mask": boulder_mask,
        "steep_mask": steep_mask,
        "grad_mag": grad_mag,
    }


# ---------------------------------------------------------------------------
# Module 2: LUNAX 5-factor landing-site suitability engine
# ---------------------------------------------------------------------------
def compute_landing_sites(gray_u8, crater_mask, boulder_mask, steep_mask, total_risk, grad_mag,
                           num_sites=NUM_LANDING_SITES, lander_radius_px=LANDER_RADIUS_PX):
    h, w = gray_u8.shape
    
    # 1. Flatness Score (Inverse of gradient magnitude)
    flatness = 1.0 - np.clip(grad_mag / np.percentile(grad_mag, 95), 0, 1)
    
    # 2. Smoothness Score (Local Variance)
    gray_f = gray_u8.astype(np.float32) / 255.0
    mean_sq = cv2.blur(gray_f**2, (5, 5))
    sq_mean = cv2.blur(gray_f, (5, 5))**2
    variance = np.clip(mean_sq - sq_mean, 0, None)
    smoothness = 1.0 - np.clip(variance / np.percentile(variance, 95), 0, 1)
    
    # 3. Hazard Free Score
    hazard_free = 1.0 - total_risk
    
    # Combine suitability mathematically
    suitability = (flatness * 0.4) + (smoothness * 0.3) + (hazard_free * 0.3)
    
    # 4. Hard Constraints Clearance
    hard_mask = crater_mask | boulder_mask | steep_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lander_radius_px * 2 + 1, lander_radius_px * 2 + 1))
    hard_mask_dilated = cv2.dilate(hard_mask.astype(np.uint8), kernel).astype(bool)
    suitability[hard_mask_dilated] = 0.0

    # 5. Iterative local maxima discovery
    sites = []
    temp_suit = suitability.copy()
    site_names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]
    
    for i in range(num_sites):
        if temp_suit.max() == 0:
            break
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(temp_suit)
        x, y = max_loc
        
        threats = []
        if crater_mask[y, x]: threats.append("Crater edge")
        if boulder_mask[y, x]: threats.append("Boulder scatter")
        if steep_mask[y, x]: threats.append("Slope inclination")
        primary_threat = threats[0] if threats else "None"
        
        risk_val = total_risk[y, x]
        status = "SAFE" if risk_val < 0.3 else "EMERGENCY"
        color = "#4ade80" if status == "SAFE" else "#f87171"
        
        sites.append({
            "rank": i + 1,
            "x": int(x),
            "y": int(y),
            "site_id": f"Site {site_names[i] if i < len(site_names) else i}",
            "status": status,
            "color": color,
            "final_suitability": float(max_val),
            "safety_score": int((1.0 - risk_val) * 100),
            "primary_threat": primary_threat
        })
        
        # Suppress local neighborhood to avoid clustering points
        cv2.circle(temp_suit, max_loc, MIN_SITE_SPACING_PX, 0, -1)
        
    return sites


# ---------------------------------------------------------------------------
# Module 3: 3D Route Risk-Aware A* pathfinding
# ---------------------------------------------------------------------------
def astar_pathfinding(start, goal, risk_map, risk_weight=1.0):
    """8-connected grid A* algorithm utilizing the numeric risk map."""
    h, w = risk_map.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        x, y = current
        # 8-connected grid traversal
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                step_dist = np.hypot(dx, dy)
                # Cost function blends euclidean distance with AI risk output
                penalty = risk_weight * risk_map[ny, nx] * step_dist
                tentative_g = g_score[current] + step_dist + penalty

                neighbor = (nx, ny)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + np.hypot(goal[0] - nx, goal[1] - ny)
                    heapq.heappush(open_set, (f_score, neighbor))
    return []

def compute_routes(start_xy, goal_xy, total_risk):
    """Generates three traversal strategies connecting the dropped pins."""
    routes = []
    configs = [
        {"strategy": "Distance Priority", "weight": 0.5, "color": "#facc15", "style": "dash"},
        {"strategy": "Balanced", "weight": 5.0, "color": "#38bdf8", "style": "solid"},
        {"strategy": "Max Safety", "weight": 20.0, "color": "#4ade80", "style": "solid"},
    ]
    
    for cfg in configs:
        path = astar_pathfinding(start_xy, goal_xy, total_risk, risk_weight=cfg["weight"])
        if path:
            dist = sum(np.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]) for i in range(len(path)-1))
            avg_risk = np.mean([total_risk[y, x] for x, y in path])
            safety_score = int((1.0 - avg_risk) * 100)
            
            routes.append({
                "strategy": cfg["strategy"],
                "path": path,
                "color": cfg["color"],
                "distance_px": int(dist),
                "safety_score": safety_score,
                "style": cfg["style"]
            })
            
    return routes


# ---------------------------------------------------------------------------
# Module 4: Grounded 3D Terrain Generator
# ---------------------------------------------------------------------------
def compute_terrain_3d(gray_u8, total_risk, routes=None, start=None, goal=None):
    """Downsamples the terrain and projects computed routes into 3D space for Plotly.js."""
    h, w = gray_u8.shape
    
    # Restrict mesh size to keep browser rendering smooth
    max_dim = 150
    scale = min(1.0, max_dim / max(h, w))
    nw, nh = int(w * scale), int(h * scale)
    
    # Generate pseudo-elevation based on grayscale intensity 
    # (darker areas like craters naturally map lower)
    # Generate pseudo-elevation based on grayscale intensity 
    # (lighter areas like peaks naturally map higher)
    z_down = cv2.resize(gray_u8, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
    z_down = cv2.GaussianBlur(z_down, (5, 5), 0)
    z_down = (255.0 - z_down)*0.05 # Add this line to invert the elevation peaks/valleys
    # Setup coordinates
    x_grid, y_grid = np.meshgrid(np.linspace(0, w - 1, nw), np.linspace(0, h - 1, nh))
    
    res = {
        "x": x_grid.tolist(),
        "y": y_grid.tolist(),
        "z": z_down.tolist(),
        "texture": z_down.tolist(), # Surfacecolor will map against the Z-height
        "routes": []
    }
    
    def get_z_at(px, py):
        ix, iy = int(px * scale), int(py * scale)
        ix = max(0, min(nw - 1, ix))
        iy = max(0, min(nh - 1, iy))
        return float(z_down[iy, ix])
    
    if routes:
        for r in routes:
            res["routes"].append({
                "strategy": r["strategy"],
                "color": r["color"],
                "x": [p[0] for p in r["path"]],
                "y": [p[1] for p in r["path"]],
                "z": [get_z_at(px, py) for px, py in r["path"]]
            })
            
    if start:
        res["start"] = {"x": start[0], "y": start[1], "z": get_z_at(start[0], start[1])}
    if goal:
        res["goal"] = {"x": goal[0], "y": goal[1], "z": get_z_at(goal[0], goal[1])}
        
    return res