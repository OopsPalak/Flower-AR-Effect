import cv2
import mediapipe as mp
import numpy as np
import math

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# ── State ────────────────────────────────────────────────────────────────────
stem_progress = 0.0   # 0‒1  (left hand zoom controls this)
bloom_progress = 0.0  # 0‒1  (right hand zoom controls this)

STEMS = [
    dict(bx=0.38, by=0.98, cx1=0.35, cy1=0.75, cx2=0.32, cy2=0.58, ex=0.30, ey=0.42),
    dict(bx=0.42, by=0.98, cx1=0.43, cy1=0.73, cx2=0.41, cy2=0.55, ex=0.40, ey=0.38),
    dict(bx=0.46, by=0.99, cx1=0.48, cy1=0.75, cx2=0.51, cy2=0.57, ex=0.53, ey=0.40),
    dict(bx=0.50, by=0.99, cx1=0.54, cy1=0.74, cx2=0.60, cy2=0.56, ex=0.63, ey=0.39),
    dict(bx=0.54, by=0.99, cx1=0.59, cy1=0.76, cx2=0.67, cy2=0.59, ex=0.71, ey=0.43),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def bezier_point(t, p0, p1, p2, p3):
    mt = 1 - t
    return mt**3*p0 + 3*mt**2*t*p1 + 3*mt*t**2*p2 + t**3*p3


def get_tip(stem, prog, W, H):
    t = prog
    x = bezier_point(t, stem['bx']*W, stem['cx1']*W, stem['cx2']*W, stem['ex']*W)
    y = bezier_point(t, stem['by']*H, stem['cy1']*H, stem['cy2']*H, stem['ey']*H)
    return int(x), int(y)


def draw_stem(img, stem, prog, W, H):
    steps = 30
    pts = []
    for i in range(steps + 1):
        t = (i / steps) * prog
        x = int(bezier_point(t, stem['bx']*W, stem['cx1']*W, stem['cx2']*W, stem['ex']*W))
        y = int(bezier_point(t, stem['by']*H, stem['cy1']*H, stem['cy2']*H, stem['ey']*H))
        pts.append((x, y))
    for i in range(len(pts)-1):
        cv2.line(img, pts[i], pts[i+1], (120, 190, 110), 2, cv2.LINE_AA)


def draw_bud(overlay, cx, cy, bloom_t, tick):
    """Draw flower bud / bloom on an RGBA overlay."""
    stage = bloom_t

    if stage < 0.33:
        # ── closed bud ──
        s = 0.6 + stage * 1.2
        bud_h = int(22 * s)
        bud_w = int(9 * s)
        cv2.ellipse(overlay, (cx, cy - bud_h//2), (bud_w//2, bud_h//2),
                    0, 0, 360, (170, 130, 220, 200), -1, cv2.LINE_AA)
        cv2.ellipse(overlay, (cx, cy - int(bud_h*0.78)), (max(1,bud_w//3), max(1,int(bud_h*0.18))),
                    0, 0, 360, (140, 90, 190, 220), -1, cv2.LINE_AA)

    elif stage < 0.66:
        # ── half-open tulip ──
        t2 = (stage - 0.33) / 0.33
        n_petals = 5
        size = int(16 + t2 * 10)
        for i in range(n_petals):
            angle_rad = (i / n_petals) * 2 * math.pi + tick * 0.01
            spread = t2 * 0.55
            px = cx + int(math.sin(angle_rad) * spread * size * 0.6)
            py = cy - int(size * (0.7 + t2 * 0.3)) + int(math.cos(angle_rad) * spread * size * 0.3)
            petal_axes = (max(1, int(5 + t2*3)), max(1, int(12 + t2*5)))
            rot = math.degrees(angle_rad)
            alpha = int(180 + t2 * 50)
            cv2.ellipse(overlay, (px, py), petal_axes, rot,
                        0, 360, (180, 130, 210, alpha), -1, cv2.LINE_AA)
        # centre
        cv2.circle(overlay, (cx, cy - int(size*0.45)), 5,
                   (230, 200, 240, 230), -1, cv2.LINE_AA)

    else:
        # ── full lotus bloom ──
        t3 = (stage - 0.66) / 0.34
        outer_r = int(14 + t3 * 12)
        layers = [
            dict(count=8, r=outer_r,        pw=int(6+t3*3), ph=int(15+t3*7), color=(180,130,210,200)),
            dict(count=6, r=int(outer_r*.55), pw=int(5+t3*2), ph=int(11+t3*4), color=(210,170,230,215)),
            dict(count=4, r=int(outer_r*.25), pw=3,           ph=7,            color=(230,210,240,230)),
        ]
        for li, layer in enumerate(layers):
            for i in range(layer['count']):
                a = (i / layer['count']) * 2 * math.pi + tick * 0.008 + li * 0.25
                px = cx + int(math.sin(a) * layer['r'])
                py = cy - int(math.cos(a) * layer['r'])
                cv2.ellipse(overlay, (px, py),
                            (max(1,layer['pw']), max(1,layer['ph'])),
                            math.degrees(a), 0, 360,
                            layer['color'], -1, cv2.LINE_AA)
        # glowing centre
        cv2.circle(overlay, (cx, cy), int(5 + t3*4),
                   (245, 235, 180, 240), -1, cv2.LINE_AA)


def blend_overlay(frame, overlay):
    """Alpha-blend an BGRA overlay onto a BGR frame."""
    alpha = overlay[:, :, 3:4] / 255.0
    rgb   = overlay[:, :, :3]
    frame[:] = (frame * (1 - alpha) + rgb * alpha).astype(np.uint8)


def pinch_distance(lm, W, H):
    """Distance between thumb tip and index tip (normalised 0‒1)."""
    t = lm[4]
    i = lm[8]
    d = math.hypot((t.x - i.x)*W, (t.y - i.y)*H)
    return d / (W * 0.25)   # 0 = fully pinched, 1 = fully open


def hand_label(hand_result, idx):
    return hand_result.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'


# ── Main loop ─────────────────────────────────────────────────────────────────
# Try camera indices with DirectShow backend (Windows fix)
cap = None
for cam_idx in range(3):
    _cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if _cap.isOpened():
        cap = _cap
        print(f"Using camera index {cam_idx}")
        break
    _cap.release()
# Fallback: try without backend
if cap is None:
    for cam_idx in range(3):
        _cap = cv2.VideoCapture(cam_idx)
        if _cap.isOpened():
            cap = _cap
            print(f"Using camera index {cam_idx} (fallback)")
            break
        _cap.release()
if cap is None:
    print("No camera found. Check if another app is using it.")
    exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

tick = 0

print("Controls:")
print("  LEFT  hand pinch → zoom in/out = stem grows / shrinks")
print("  RIGHT hand pinch → zoom in/out = flowers open / close")
print("  Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]
    tick += 1

    # ── Hand detection ────────────────────────────────────────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for idx, lm_set in enumerate(result.multi_hand_landmarks):
            label = hand_label(result, idx)
            dist  = pinch_distance(lm_set.landmark, W, H)
            # dist ~0 = closed, ~1 = open → map to progress
            progress = min(1.0, max(0.0, dist))

            if label == 'Left':
                stem_progress  = stem_progress  * 0.8 + progress * 0.2   # smooth
            else:
                bloom_progress = bloom_progress * 0.8 + progress * 0.2

            # draw hand landmarks (subtle)
            mp.solutions.drawing_utils.draw_landmarks(
                frame, lm_set, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(200,200,200), thickness=1, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(150,150,150), thickness=1),
            )

    # ── Draw flowers ──────────────────────────────────────────────────────────
    overlay = np.zeros((H, W, 4), dtype=np.uint8)

    for i, stem in enumerate(STEMS):
        prog = min(1.0, max(0.0, stem_progress * 1.15 - i * 0.04))
        if prog <= 0:
            continue
        draw_stem(frame, stem, prog, W, H)
        if prog > 0.45:
            tx, ty = get_tip(stem, prog, W, H)
            bud_bloom = max(0.0, (bloom_progress - 0.08 * i) / 0.95)
            draw_bud(overlay, tx, ty, min(1.0, bud_bloom * 1.2), tick + i * 30)

    blend_overlay(frame, overlay)

    # ── HUD ───────────────────────────────────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = 10, H - 60, 180, 10
    # stem bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (60,60,60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+int(bar_w*stem_progress), bar_y+bar_h), (110,190,100), -1)
    cv2.putText(frame, f"Stem (L): {int(stem_progress*100)}%",
                (bar_x, bar_y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,255,180), 1, cv2.LINE_AA)
    # bloom bar
    by2 = bar_y + 28
    cv2.rectangle(frame, (bar_x, by2), (bar_x+bar_w, by2+bar_h), (60,60,60), -1)
    cv2.rectangle(frame, (bar_x, by2), (bar_x+int(bar_w*bloom_progress), by2+bar_h), (200,130,210), -1)
    cv2.putText(frame, f"Bloom (R): {int(bloom_progress*100)}%",
                (bar_x, by2-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230,180,255), 1, cv2.LINE_AA)

    cv2.putText(frame, "Q: quit", (W-80, H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1, cv2.LINE_AA)

    cv2.imshow("Flower AR Effect", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
