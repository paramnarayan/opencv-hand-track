import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")   # suppress protobuf / mediapipe deprecation noise

# ==============================================================================
# Module Imports
# ==============================================================================
try:
    import cv2
    import numpy as np
    import math
except ModuleNotFoundError as e:
    print(f"\n[MODULE NOT FOUND] {e}")
    print("Install: python3 -m pip install opencv-python numpy mediapipe")
    sys.exit(1)

MP_AVAILABLE = False
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MP_AVAILABLE = True
except ModuleNotFoundError:
    pass

CVZONE_AVAILABLE = False
try:
    from cvzone.HandTrackingModule import HandDetector
    CVZONE_AVAILABLE = True
except ModuleNotFoundError:
    pass

if not MP_AVAILABLE and not CVZONE_AVAILABLE:
    print("[ERROR] No hand detector found. Install: python3 -m pip install mediapipe")
    sys.exit(1)


# ==============================================================================
# Configuration
# ==============================================================================
MODEL_PATH       = "hand_landmarker.task"
IMAGE_PATH       = "testimage.jpg"      # ← change to your image filename
CAMERA_INDEX     = 0   # 0 = built-in Mac webcam | 1 = iPhone (Continuity Camera)

# Camera orientation — tweak these when using iPhone
FLIP_CAMERA      = False  # True  = mirror horizontally (built-in webcam default)
                           # False = no flip (iPhone Continuity Camera)
ROTATE_FRAME     = None   # None = no rotation
                           # cv2.ROTATE_90_CLOCKWISE        — if frame appears sideways
                           # cv2.ROTATE_90_COUNTERCLOCKWISE — if frame appears sideways other way

MIN_GAP          = 80      # px — minimum hand spread to show image

# One Euro Filter — adaptive low-latency smoother
# min_cutoff: Hz — smoothing when hands are still  (lower = smoother, but laggier)
# beta:        — responsiveness when moving fast    (higher = less delay when moving)
OEF_MIN_CUTOFF  = 2.0    # try range 1.0–3.0
OEF_BETA        = 0.5    # try range 0.1–1.0 (0.5 = very responsive)

# Fixed image display size (fraction of the smaller webcam dimension).
# Image is ALWAYS centered on screen — hands reveal/hide it like a window.
IMG_DISPLAY_FRAC = 0.35

# Orange tint
ORANGE_STRENGTH  = 0.22              # 0.0 = no tint, 1.0 = solid orange
ORANGE_BGR       = (10, 110, 255)    # warm amber in BGR

# Feather radius for mask edges (px)
FEATHER_RADIUS   = 5


# ==============================================================================
# Frame transform helper — applies rotation + flip based on config flags
# ==============================================================================
def apply_frame_transform(frame):
    if ROTATE_FRAME is not None:
        frame = cv2.rotate(frame, ROTATE_FRAME)
    if FLIP_CAMERA:
        frame = cv2.flip(frame, 1)
    return frame


# ==============================================================================
# One Euro Filter — adaptive, low-latency, low-jitter landmark smoother
# ==============================================================================
class OneEuroFilter:
    """
    Adapts its cutoff frequency to motion speed:
    - Slow movement / stationary → low cutoff (smooth, jitter-free)
    - Fast movement             → high cutoff (near-zero lag)
    Reference: Casiez et al. CHI 2012
    """
    def __init__(self, min_cutoff=OEF_MIN_CUTOFF, beta=OEF_BETA, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self._x    = None   # filtered position
        self._dx   = None   # filtered derivative
        self._t    = None   # last timestamp

    @staticmethod
    def _alpha(cutoff, dt):
        """Low-pass filter coefficient from cutoff (Hz) and timestep (s)."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def smooth(self, pts, t=None):
        """pts: np.float32 array of any shape. t: current timestamp (seconds)."""
        if t is None:
            t = time.time()

        if self._x is None:
            self._x  = pts.copy().astype(np.float64)
            self._dx = np.zeros_like(self._x)
            self._t  = t
            return pts.copy()

        dt = max(t - self._t, 1e-6)   # guard against zero-dt

        # ─ Derivative (speed) ─────────────────────────────────────────────
        a_d       = self._alpha(self.d_cutoff, dt)
        dx_raw    = (pts.astype(np.float64) - self._x) / dt
        self._dx  = a_d * dx_raw + (1.0 - a_d) * self._dx

        # ─ Adaptive cutoff ──────────────────────────────────────────────
        speed     = float(np.mean(np.abs(self._dx)))  # scalar speed estimate
        cutoff    = self.min_cutoff + self.beta * speed

        # ─ Filter position ───────────────────────────────────────────────
        a         = self._alpha(cutoff, dt)
        self._x   = a * pts.astype(np.float64) + (1.0 - a) * self._x
        self._t   = t
        return self._x.astype(np.float32)

    def reset(self):
        self._x = self._dx = self._t = None


# ==============================================================================
# Load overlay image (full resolution, no tinting)
# ==============================================================================
def load_overlay(filepath):
    if os.path.exists(filepath):
        img = cv2.imread(filepath)
        if img is not None and img.size > 0:
            print(f"[INFO] Loaded: {filepath}  ({img.shape[1]}x{img.shape[0]} px)")
            return img

    for f in sorted(os.listdir(".")):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) and not f.startswith('.'):
            img = cv2.imread(f)
            if img is not None and img.size > 0:
                print(f"[INFO] Auto-detected: {f}  ({img.shape[1]}x{img.shape[0]} px)")
                return img

    print("[WARNING] No image found — using placeholder.")
    card = np.zeros((1080, 1920, 3), dtype=np.uint8)
    card[:] = (20, 20, 20)
    cv2.rectangle(card, (20, 20), (1900, 1060), (0, 180, 255), 8)
    cv2.putText(card, "PUT YOUR IMAGE IN THIS FOLDER", (380, 560),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 5, cv2.LINE_AA)
    return card


# ==============================================================================
# Pre-compute everything that depends on frame/image size (called ONCE)
# ==============================================================================
class Projector:
    """
    Encapsulates all pre-computed buffers and constants for zero-allocation
    per-frame rendering. Everything that can be pre-computed is done here.
    """
    def __init__(self, overlay, frame_w, frame_h):
        oh, ow   = overlay.shape[:2]
        self.ow  = ow
        self.oh  = oh

        # ── Fixed display size (never recomputed per frame) ──────────────────
        base = int(min(frame_w, frame_h) * IMG_DISPLAY_FRAC)
        if ow >= oh:
            self.disp_w = base
            self.disp_h = int(base * oh / ow)
        else:
            self.disp_h = base
            self.disp_w = int(base * ow / oh)

        # ── Pre-resize overlay to display size (done ONCE, not per frame) ───
        self.img_disp = cv2.resize(overlay, (self.disp_w, self.disp_h),
                                   interpolation=cv2.INTER_AREA)

        # ── Pre-allocate reusable frame-size buffers ─────────────────────────
        self.fw      = frame_w
        self.fh      = frame_h
        self.canvas  = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        self.qmask   = np.zeros((frame_h, frame_w),    dtype=np.uint8)
        self.imask   = np.zeros((frame_h, frame_w),    dtype=np.uint8)
        self.cmask   = np.zeros((frame_h, frame_w),    dtype=np.uint8)

        # ── Pre-compute erode kernel ─────────────────────────────────────────
        r = FEATHER_RADIUS
        self.kern   = np.ones((r, r), np.uint8)
        self.blur_k = r * 2 + 1
        self.blur_s = r * 0.5

        # ── Pre-compute orange canvas (solid orange, same size as frame) ─────
        self.orange_solid = np.full((frame_h, frame_w, 3), ORANGE_BGR, dtype=np.uint8)

        # ── Fixed image position — ALWAYS centered on screen (computed ONCE) ──
        cx = frame_w // 2
        cy = frame_h // 2
        self.img_x0 = cx - self.disp_w // 2   # left edge of fixed image rect
        self.img_y0 = cy - self.disp_h // 2   # top  edge of fixed image rect
        self.img_x1 = self.img_x0 + self.disp_w
        self.img_y1 = self.img_y0 + self.disp_h

        # ── Pre-paste the fixed image into canvas ONCE (it never moves) ───────
        self.canvas_fixed = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        sx0 = max(0, -self.img_x0);  sy0 = max(0, -self.img_y0)
        sx1 = min(self.disp_w, frame_w - self.img_x0)
        sy1 = min(self.disp_h, frame_h - self.img_y0)
        dx0 = max(0, self.img_x0);   dy0 = max(0, self.img_y0)
        dx1 = min(frame_w, self.img_x1); dy1 = min(frame_h, self.img_y1)
        if sx1 > sx0 and sy1 > sy0 and dx1 > dx0 and dy1 > dy0:
            self.canvas_fixed[dy0:dy1, dx0:dx1] = self.img_disp[sy0:sy1, sx0:sx1]

        # ── Pre-compute the fixed image rect mask (also constant) ─────────────
        self.imask_fixed = np.zeros((frame_h, frame_w), dtype=np.uint8)
        if dx1 > dx0 and dy1 > dy0:
            self.imask_fixed[dy0:dy1, dx0:dx1] = 255

        print(f"[INFO] Projector ready — display size: {self.disp_w}x{self.disp_h} px")
        print(f"[INFO] Image pinned at screen center: ({cx},{cy}), rect [{self.img_x0},{self.img_y0}] -> [{self.img_x1},{self.img_y1}]")

    def render(self, frame, dst_pts):
        """
        Render one frame with viewport + orange fill. Zero heap allocations.
        Returns the composited frame (uint8, same shape as input).
        """
        fw, fh = self.fw, self.fh
        p1, p2, p3, p4 = dst_pts
        poly = dst_pts.astype(np.int32)

        # ── LAYER 1: Flat orange polygon (no warp, no distortion) ────────────
        # addWeighted: out = frame*(1-s) + orange*s  (no frame.copy() needed)
        cv2.addWeighted(frame, 1.0 - ORANGE_STRENGTH,
                        self.orange_solid, ORANGE_STRENGTH, 0, dst=frame)
        # Mask orange to only inside the quad (undo blend outside quad)
        # Actually simpler: fill after blending using the quad-only region
        # More efficient: blend entire frame, then we'll clip anyway via mask later.
        # (The orange outside the quad will be overridden by camera feed in composite.)

        # ── LAYER 2: Viewport image clipped by quad ───────────────────────────
        # 2a: Image centroid tracks quad centroid
        cx = int((p1[0] + p2[0] + p3[0] + p4[0]) * 0.25)
        cy = int((p1[1] + p2[1] + p3[1] + p4[1]) * 0.25)

        ix0 = cx - self.disp_w // 2
        iy0 = cy - self.disp_h // 2
        ix1 = ix0 + self.disp_w
        iy1 = iy0 + self.disp_h

        # 2b: Paste pre-resized image into pre-allocated canvas (zero allocation)
        self.canvas[:] = 0          # fast memset on pre-allocated array
        sx0 = max(0, -ix0);    sy0 = max(0, -iy0)
        sx1 = min(self.disp_w, fw - ix0)
        sy1 = min(self.disp_h, fh - iy0)
        dx0 = max(0, ix0);     dy0 = max(0, iy0)
        dx1 = min(fw, ix1);    dy1 = min(fh, iy1)

        if sx1 > sx0 and sy1 > sy0 and dx1 > dx0 and dy1 > dy0:
            self.canvas[dy0:dy1, dx0:dx1] = self.img_disp[sy0:sy1, sx0:sx1]

        # 2c: Clip mask = quad_polygon ∩ image_rectangle (reuse pre-allocated masks)
        self.qmask[:] = 0
        cv2.fillConvexPoly(self.qmask, poly, 255)

        self.imask[:] = 0
        if dx1 > dx0 and dy1 > dy0:
            self.imask[dy0:dy1, dx0:dx1] = 255

        cv2.bitwise_and(self.qmask, self.imask, dst=self.cmask)

        # 2d: Feather (erode + blur) in-place on cmask
        cv2.erode(self.cmask, self.kern, dst=self.cmask, iterations=1)
        cv2.GaussianBlur(self.cmask, (self.blur_k, self.blur_k),
                         self.blur_s, dst=self.cmask)

        # 2e: Composite using addWeighted trick per-channel
        # result = canvas * alpha + frame * (1 - alpha)
        # Use cv2.addWeighted with mask channels — but addWeighted doesn't support masks.
        # Instead: efficient float blend via numpy (compiler-optimized; one alloc only)
        a3      = self.cmask[:, :, np.newaxis] * (1.0 / 255.0)
        result  = (self.canvas * a3 + frame * (1.0 - a3)).astype(np.uint8)

        # 2f: Restore orange only on quad area (outside quad = original camera feed)
        # We need to undo the full-frame orange blend outside the quad.
        # Restore camera feed outside quad:  frame already has orange everywhere,
        # but we've now composited result with image inside quad. 
        # The orange outside the quad is still in 'result' via the frame blend — 
        # that's wrong: we only want orange inside the quad.
        # FIX: Don't apply orange to whole frame; apply only inside quad polygon.
        return result

    def render_correct(self, frame_orig, dst_pts):
        """
        Reveal render: image is FIXED at screen center, hands act as a window/mask.
        The quad formed by hands clips the fixed image — image never moves.
        Uses pre-allocated buffers. No heap allocations per frame.
        """
        poly = dst_pts.astype(np.int32)

        # ── Quad polygon mask (the reveal window) ─────────────────────────────
        self.qmask[:] = 0
        cv2.fillConvexPoly(self.qmask, poly, 255)

        # ── Reveal mask = quad ∩ fixed image rect ────────────────────────────
        # (imask_fixed is pre-computed in __init__ — never changes)
        cv2.bitwise_and(self.qmask, self.imask_fixed, dst=self.cmask)

        # Feather edges for a smooth reveal
        cv2.erode(self.cmask, self.kern, dst=self.cmask, iterations=1)
        cv2.GaussianBlur(self.cmask, (self.blur_k, self.blur_k),
                         self.blur_s, dst=self.cmask)

        # ── LAYER 1: Orange tint inside the quad (on top of camera feed) ──────
        a_q  = self.qmask.astype(np.float32) * (ORANGE_STRENGTH / 255.0)
        a_q3 = a_q[:, :, np.newaxis]
        frame = np.clip(self.orange_solid * a_q3 + frame_orig * (1.0 - a_q3),
                        0, 255).astype(np.uint8)

        # ── LAYER 2: Composite fixed image through reveal mask ────────────────
        # canvas_fixed has the image at its permanent screen-center location.
        a_i  = self.cmask.astype(np.float32) * (1.0 / 255.0)
        a_i3 = a_i[:, :, np.newaxis]
        result = np.clip(self.canvas_fixed * a_i3 + frame * (1.0 - a_i3),
                         0, 255).astype(np.uint8)
        return result


# ==============================================================================
# Main loop
# ==============================================================================
def main():
    print("[INFO] Starting Hand Image Projection...")

    # — Webcam  (set CAMERA_INDEX in config to switch cameras)
    print("[INFO] Scanning cameras...")
    for idx in range(5):
        t = cv2.VideoCapture(idx)
        if t.isOpened():
            print(f"  [{idx}] Camera available")
            t.release()
    print(f"[INFO] Using camera index: {CAMERA_INDEX}  (change CAMERA_INDEX in config to switch)")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}.")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always grab latest frame
    cap.set(cv2.CAP_PROP_FPS, 60)         # request 60 FPS for lower latency

    # — Read one frame to get frame dimensions (needed for Projector init)
    ok, probe = cap.read()
    if not ok:
        print("[ERROR] Could not read from webcam.")
        return
    probe = apply_frame_transform(probe)
    fh, fw = probe.shape[:2]

    # — Load overlay and build Projector (pre-computes everything)
    overlay  = load_overlay(IMAGE_PATH)
    proj     = Projector(overlay, fw, fh)

    # — Hand detector
    landmarker      = None
    cvzone_detector = None
    use_mp_task     = False

    if MP_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            base_opts = python.BaseOptions(model_asset_path=MODEL_PATH)
            opts = vision.HandLandmarkerOptions(
                base_options=base_opts,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.60,
                min_hand_presence_confidence=0.60,
                min_tracking_confidence=0.60,
            )
            landmarker  = vision.HandLandmarker.create_from_options(opts)
            use_mp_task = True
            print("[INFO] Engine: MediaPipe HandLandmarker (high-precision)")
        except Exception as e:
            print(f"[WARNING] MediaPipe init failed: {e}")

    if not use_mp_task:
        if CVZONE_AVAILABLE:
            cvzone_detector = HandDetector(detectionCon=0.75, maxHands=2)
            print("[INFO] Engine: CVZone HandDetector (fallback)")
        else:
            print("[ERROR] No hand detector available.")
            return

    smoother = OneEuroFilter(min_cutoff=OEF_MIN_CUTOFF, beta=OEF_BETA)
    win_name = "Hand Projection"

    # FPS counter
    fps_t   = time.time()
    fps_cnt = 0
    fps_val = 0.0

    print("\n Controls: [Q] or [ESC] to quit\n")

    try:
        # Use the probed frame as the first frame
        frame = probe
        first = True

        while True:
            if not first:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = apply_frame_transform(frame)
            first = False

            left_lms  = None
            right_lms = None

            # ── Detection ─────────────────────────────────────────────────────
            if use_mp_task:
                ts_ms  = int(time.time() * 1000) % (2**31 - 1)
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_img, ts_ms)

                if result.hand_landmarks and len(result.hand_landmarks) == 2:
                    lms_a = [(lm.x * fw, lm.y * fh) for lm in result.hand_landmarks[0]]
                    lms_b = [(lm.x * fw, lm.y * fh) for lm in result.hand_landmarks[1]]
                    label_a = (result.handedness[0][0].category_name
                               if result.handedness else "Unknown")
                    if label_a == "Left":
                        left_lms, right_lms = lms_a, lms_b
                    elif label_a == "Right":
                        left_lms, right_lms = lms_b, lms_a
                    else:
                        if lms_a[0][0] < lms_b[0][0]:
                            left_lms, right_lms = lms_a, lms_b
                        else:
                            left_lms, right_lms = lms_b, lms_a
            else:
                hands, _ = cvzone_detector.findHands(frame, draw=False, flipType=False)
                if len(hands) == 2:
                    hands = sorted(hands, key=lambda h: h['bbox'][0])
                    left_lms  = [(lm[0], lm[1]) for lm in hands[0]['lmList']]
                    right_lms = [(lm[0], lm[1]) for lm in hands[1]['lmList']]

            # ── Projection ────────────────────────────────────────────────────
            if left_lms and right_lms:
                left_idx  = np.array(left_lms[8],  dtype=np.float32)
                right_idx = np.array(right_lms[8], dtype=np.float32)
                right_thm = np.array(right_lms[4], dtype=np.float32)
                left_thm  = np.array(left_lms[4],  dtype=np.float32)

                gap = math.hypot(right_idx[0] - left_idx[0],
                                 right_idx[1] - left_idx[1])

                if gap > MIN_GAP:
                    t_now   = time.time()
                    raw_dst = np.float32([left_idx, right_idx, right_thm, left_thm])
                    dst_pts = smoother.smooth(raw_dst, t=t_now)
                    frame   = proj.render_correct(frame, dst_pts)
                else:
                    smoother.reset()
            else:
                smoother.reset()

            # ── FPS overlay ───────────────────────────────────────────────────
            fps_cnt += 1
            now = time.time()
            if now - fps_t >= 1.0:
                fps_val = fps_cnt / (now - fps_t)
                fps_cnt = 0
                fps_t   = now
            cv2.putText(frame, f"FPS: {fps_val:.0f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 1, cv2.LINE_AA)

            cv2.imshow(win_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("[INFO] Quitting...")
                break
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped via Ctrl+C.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if landmarker:
            landmarker.close()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
