#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import os
import cv2
import numpy as np


@dataclass
class LineModel:
    # parametric line: (x, y) = (x0, y0) + t*(vx, vy)
    x0: float
    y0: float
    vx: float
    vy: float


@dataclass
class CellPoly:
    id: int
    polygon: List[List[int]]  # [[x,y],...]


# -----------------------------
# Geometry helpers
# -----------------------------
def x_at_y(line: LineModel, y: float) -> Optional[float]:
    # Solve y = y0 + t*vy  =>  t = (y - y0)/vy  => x = x0 + t*vx
    if abs(line.vy) < 1e-6:
        return None
    t = (y - line.y0) / line.vy
    return line.x0 + t * line.vx


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# -----------------------------
# LSD-based divider extraction
# -----------------------------
def extract_vertical_dividers_lsd(
    img_bgr: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],
    angle_tol_deg: float = 20.0,
    min_seg_len: int = 120,
    merge_tol_px: float = 25.0,
) -> List[LineModel]:
    """
    Detect line segments with LSD, keep near-vertical ones, cluster by x (at mid-y),
    then fit one line per cluster with cv2.fitLine.
    """
    h, w = img_bgr.shape[:2]
    if roi is None:
        x1, y1, x2, y2 = 0, 0, w - 1, h - 1
    else:
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

    crop = img_bgr[y1:y2 + 1, x1:x2 + 1]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # LSD works best on reasonable-contrast grayscale
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]  # Nx1x4 or None

    if lines is None:
        return []

    # Collect candidate segments (in full-image coordinates)
    cand = []
    ang_tol = np.deg2rad(angle_tol_deg)

    for l in lines.reshape(-1, 4):
        xA, yA, xB, yB = l
        dx = float(xB - xA)
        dy = float(yB - yA)
        seg_len = (dx * dx + dy * dy) ** 0.5
        if seg_len < min_seg_len:
            continue

        # segment angle in [-pi..pi], vertical means ~ +/- pi/2
        ang = np.arctan2(dy, dx)
        # distance to vertical: min(|ang - pi/2|, |ang + pi/2|)
        vdist = min(abs(ang - np.pi / 2), abs(ang + np.pi / 2))
        if vdist > ang_tol:
            continue

        # shift back to full image coords
        xA2, yA2 = xA + x1, yA + y1
        xB2, yB2 = xB + x1, yB + y1

        # x position at mid-y (approx): just average endpoints x
        xmid = 0.5 * (xA2 + xB2)
        cand.append((xmid, (xA2, yA2, xB2, yB2)))

    if not cand:
        return []

    # sort by xmid and cluster
    cand.sort(key=lambda t: t[0])
    clusters: List[List[Tuple[float, float]]] = []  # list of point lists

    cur_pts: List[Tuple[float, float]] = []
    cur_center = cand[0][0]

    for xmid, (xA, yA, xB, yB) in cand:
        if abs(xmid - cur_center) <= merge_tol_px:
            cur_pts.extend([(xA, yA), (xB, yB)])
            # update running center (simple)
            cur_center = 0.7 * cur_center + 0.3 * xmid
        else:
            if len(cur_pts) >= 4:
                clusters.append(cur_pts)
            cur_pts = [(xA, yA), (xB, yB)]
            cur_center = xmid

    if len(cur_pts) >= 4:
        clusters.append(cur_pts)

    # fit a line per cluster
    models: List[LineModel] = []
    for pts in clusters:
        pts_np = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        vx, vy, x0, y0 = cv2.fitLine(pts_np, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)

        # enforce near-vertical (just in case)
        # direction angle:
        ang = np.arctan2(vy, vx)
        vdist = min(abs(ang - np.pi / 2), abs(ang + np.pi / 2))
        if vdist > ang_tol:
            continue

        models.append(LineModel(x0=x0, y0=y0, vx=vx, vy=vy))

    return models


# -----------------------------
# Build polygon strips between lines
# -----------------------------
def build_cells_between_lines(
    w: int,
    h: int,
    lines: List[LineModel],
    min_width_px: int = 120,
) -> List[CellPoly]:
    y_mid = (h - 1) / 2.0

    # sort by x at mid-height
    lines = sorted(lines, key=lambda L: (x_at_y(L, y_mid) if x_at_y(L, y_mid) is not None else 1e9))

    cells: List[CellPoly] = []
    cid = 0

    for i in range(len(lines) - 1):
        L = lines[i]
        R = lines[i + 1]

        xL_top = x_at_y(L, 0.0)
        xL_bot = x_at_y(L, float(h - 1))
        xR_top = x_at_y(R, 0.0)
        xR_bot = x_at_y(R, float(h - 1))
        xL_mid = x_at_y(L, y_mid)
        xR_mid = x_at_y(R, y_mid)

        if None in (xL_top, xL_bot, xR_top, xR_bot, xL_mid, xR_mid):
            continue

        if (xR_mid - xL_mid) < min_width_px:
            continue

        xL_top = clamp(xL_top, 0, w - 1)
        xL_bot = clamp(xL_bot, 0, w - 1)
        xR_top = clamp(xR_top, 0, w - 1)
        xR_bot = clamp(xR_bot, 0, w - 1)

        poly = [
            [int(round(xL_top)), 0],
            [int(round(xR_top)), 0],
            [int(round(xR_bot)), h - 1],
            [int(round(xL_bot)), h - 1],
        ]
        cells.append(CellPoly(cid, poly))
        cid += 1

    return cells


def draw_debug(img: np.ndarray, lines: List[LineModel], cells: List[CellPoly]) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    y_ref = (h - 1) / 2.0
    # draw lines (infinite-ish) using intersections at top/bottom
    valid_lines = []
    #print("Drawing lines...")
    for L in lines:
        xt = x_at_y(L, 0.0)
        xb = x_at_y(L, float(h - 1))
        if xt is None or xb is None:
            print("skipping line with undefined x at top/bottom {0}/{1}", xt, xb)
            continue
        p1 = (int(round(clamp(xt, 0, w - 1))), 0)
        p2 = (int(round(clamp(xb, 0, w - 1))), h - 1)
        cv2.line(out, p1, p2, (255, 0, 0), 5)

    # draw cell polygons
    if cells is not None:
        for c in cells:
            pts = np.array(c.polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, (0, 255, 0), 2)
            x0, y0 = c.polygon[0]
            cv2.putText(out, str(c.id), (x0 + 8, y0 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    return out

# def draw_line(img, line, color):
#     h, w = img.shape[:2]
#     xt = x_at_y(line, 0)
#     xb = x_at_y(line, h - 1)
#     if xt is None or xb is None:
#         return
#     p1 = (int(round(xt)), 0)
#     p2 = (int(round(xb)), h - 1)
#     cv2.line(img, p1, p2, color, 4)


# def normalize(v):
#     return v / np.linalg.norm(v)

# def offset_line(x1, y1, x2, y2, n, offset):
#     p1 = np.array([x1, y1]) + n * offset
#     p2 = np.array([x2, y2]) + n * offset
#     return p1.astype(int), p2.astype(int)

# def full_height_line(x_top, slope, H):
#     if slope is None:
#         # perfectly vertical
#         return (x_top, 0), (x_top, H)
#     else:
#         x_bottom = int(x_top + H / slope)
#         return (x_top, 0), (x_bottom, H)
    
def slanted_separator(x_top, slope, h, w):
    """
    Create a full-height slanted line parallel to the detected line.
    Works for both / and \ directions.
    """
    if slope is None:
        # Perfect vertical
        x_bot = x_top
    else:
        # y = slope * (x - x_top)
        # at y = h  =>  h = slope * (x_bot - x_top)
        x_bot = x_top + h / slope

    # Clamp to image width
    x_top = int(np.clip(x_top, 0, w))
    x_bot = int(np.clip(x_bot, 0, w))

    return (x_top, 0), (x_bot, h)
# -----------------------------
# Main
# -----------------------------
class SleeperDivider():
    def __init__(self):
        ap = argparse.ArgumentParser()

        self.angle_tol_deg = 30.0
        self.min_seg_len = 100
        self.merge_tol = 30.0
        self.min_cell_width = 100
    
    def get_gap_roi(self,img):
        
    
        #print(" working on image:", img_path)
        
        img = cv2.resize(img, (1024, 768))
        h, w = img.shape[:2]
        roi = (50, 0, w-50, h)

        # 1) detect divider lines (tilt-aware)
        lines = extract_vertical_dividers_lsd(
            img,
            roi=roi,
            angle_tol_deg=self.angle_tol_deg,
            min_seg_len=self.min_seg_len,
            merge_tol_px=self.merge_tol,
        )
        print(lines, " lines detected")
        if len(lines) > 0:
#            print("there is a line")
            # sort by x

            leftmost_line  = lines[0]
            #rightmost_line = valid[-1][1]

            #debug = img.copy()

            # reconstruct line endpoints
            x1 = int(leftmost_line.x0)
            y1 = int(leftmost_line.y0)
            x2 = int(leftmost_line.x0 + leftmost_line.vx * h)
            y2 = int(leftmost_line.y0 + leftmost_line.vy * h)

            dx = x2 - x1
            dy = y2 - y1

            if abs(dx) < 1e-6:
                slope = None
            else:
                slope = dy / dx

            # percentages
            x_sep1 = int(0.19 * w)             # L | G 
            x_sep2 = int((0.19 + 0.4) * w)    # G | R

            # separators (WORKS for / and \)
            S1_top, S1_bot = slanted_separator(x_sep1, slope, h, w)
            S2_top, S2_bot = slanted_separator(x_sep2, slope, h, w)

            # ROIs (full height)
            # left_roi = np.array([
            #     [0, 0],
            #     S1_top,
            #     S1_bot,
            #     [0, h]
            # ], dtype=np.int32)

            gap_roi = np.array([
                S1_top,
                S2_top,
                S2_bot,
                S1_bot
            ], dtype=np.int32)

            # right_roi = np.array([
            #     S2_top,
            #     [w, 0],
            #     [w, h],
            #     S2_bot
            # ], dtype=np.int32)

            # visualize
            #cv2.polylines(debug, [left_roi],  True, (0,255,0), 2)
            #cv2.polylines(debug, [gap_roi],   True, (0,0,255), 2)
            #cv2.polylines(debug, [right_roi], True, (255,0,0), 2)

            return gap_roi
        cells = None
        # 4) debug overlay
        
    def get_gap_roi_by_camera(self, camPath):
        if "1E2" in camPath:
            return np.array([[184,   0],
                [614,   0],
                [592, 768],
                [162, 768]])
        
        elif "3E2" in camPath:
            return np.array([[184,   0],
                            [614,   0],
                            [581, 768],
                            [151, 768]])

        elif "4E2" in camPath:
            return np.array([[184,   0],
                            [614,   0],
                            [570, 768],
                            [140, 768]])
        elif "5E2" in camPath:
            return np.array([[184,   0],
                            [614,   0],
                            [632, 768],
                            [192, 768]])
        
        elif "6E2" in camPath:
            return np.array([[184,  0],
                            [614,   0],
                            [607, 768],
                            [177, 768]])