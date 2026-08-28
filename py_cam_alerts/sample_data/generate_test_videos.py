import cv2
import numpy as np
import os

def create_cradle_demo(output_path, fps=20, duration_sec=12):
    """Creates a synthetic video simulating a toddler in a crib climbing the rail."""
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = fps * duration_sec
    for frame_idx in range(total_frames):
        t = frame_idx / float(fps)
        img = np.ones((h, w, 3), dtype=np.uint8) * 230  # Soft light room background
        
        # Draw room floor & walls
        cv2.rectangle(img, (0, 350), (w, h), (180, 190, 200), -1)

        # Draw Crib / Cradle (x: 130..510, y: 150..400)
        cv2.rectangle(img, (130, 150), (510, 400), (120, 80, 50), 6)  # Wooden frame
        # Draw crib vertical rails
        for rx in range(160, 500, 30):
            cv2.line(img, (rx, 150), (rx, 400), (150, 100, 70), 3)

        # Toddler Movement simulation
        if t < 4.0:
            # Safe playing inside crib (center)
            tx = int(320 + 30 * np.sin(t * 2))
            ty = int(280 + 15 * np.cos(t * 2))
            tw, th = 70, 110
        elif t < 8.0:
            # Moving towards crib left wall
            progress = (t - 4.0) / 4.0
            tx = int(320 - progress * 170)
            ty = int(280 - progress * 60)
            tw, th = 80, 130
        else:
            # Climbing over the crib wall! (Danger stage)
            tx = int(140 - (t - 8.0) * 15)
            ty = int(200 - (t - 8.0) * 20)
            tw, th = 110, 80  # horizontal posture leaning out

        # Draw Toddler Body (head & torso)
        cv2.rectangle(img, (tx - tw//2, ty - th//2), (tx + tw//2, ty + th//2), (60, 130, 240), -1)  # Blue shirt
        cv2.circle(img, (tx, ty - th//2 - 15), 20, (220, 190, 170), -1)  # Head

        out.write(img)

    out.release()
    print(f"Generated {output_path}")

def create_fall_demo(output_path, fps=20, duration_sec=12):
    """Creates a synthetic video simulating a child playing and falling down."""
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = fps * duration_sec
    for frame_idx in range(total_frames):
        t = frame_idx / float(fps)
        img = np.ones((h, w, 3), dtype=np.uint8) * 240

        # Draw playroom floor
        cv2.rectangle(img, (0, 250), (w, h), (210, 220, 230), -1)
        # Draw toys/play mat
        cv2.rectangle(img, (50, 300), (250, 450), (140, 210, 140), -1)

        # Child 1: Standing/playing on left
        cv2.rectangle(img, (120, 260), (170, 390), (240, 120, 100), -1)
        cv2.circle(img, (145, 245), 18, (220, 190, 170), -1)

        # Child 2: Running then falling!
        if t < 5.0:
            # Walking right
            cx = int(250 + t * 40)
            cy = 280
            cw, ch = 60, 140
        elif t < 6.5:
            # Rapid fall downwards
            progress = (t - 5.0) / 1.5
            cx = int(450 + progress * 30)
            cy = int(280 + progress * 90)
            # Morph from vertical (60x140) to horizontal fallen (150x50)
            cw = int(60 + progress * 90)
            ch = int(140 - progress * 90)
        else:
            # Fallen on ground (lying horizontal)
            cx = 480
            cy = 370
            cw, ch = 150, 45

        # Draw Child 2
        cv2.rectangle(img, (cx - cw//2, cy - ch//2), (cx + cw//2, cy + ch//2), (90, 170, 240), -1)
        head_x = cx - cw//2 + 15 if cw > ch else cx
        head_y = cy - ch//2 - 12 if ch > cw else cy
        cv2.circle(img, (head_x, head_y), 18, (220, 190, 170), -1)

        out.write(img)

    out.release()
    print(f"Generated {output_path}")

def create_hazard_demo(output_path, fps=20, duration_sec=12):
    """Creates a synthetic video simulating a water spill zone near a child."""
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = fps * duration_sec
    for frame_idx in range(total_frames):
        t = frame_idx / float(fps)
        img = np.ones((h, w, 3), dtype=np.uint8) * 220

        # Draw room floor
        cv2.rectangle(img, (0, 200), (w, h), (180, 180, 180), -1)

        # Draw Liquid Spill Zone on floor (high bright specular highlight)
        cv2.ellipse(img, (380, 360), (70, 45), 15, 0, 360, (250, 245, 235), -1)
        cv2.ellipse(img, (380, 360), (70, 45), 15, 0, 360, (255, 255, 255), 2)

        # Draw Table with dangerous object (knife/scissors)
        cv2.rectangle(img, (500, 220), (620, 360), (100, 70, 40), -1)
        # Knife representation
        cv2.line(img, (530, 210), (590, 210), (200, 200, 210), 4)

        # Child moving towards liquid spill
        cx = int(180 + t * 18)
        cy = 270
        cv2.rectangle(img, (cx - 30, cy - 60), (cx + 30, cy + 60), (220, 100, 150), -1)
        cv2.circle(img, (cx, cy - 75), 18, (220, 190, 170), -1)

        out.write(img)

    out.release()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    os.makedirs("sample_data", exist_ok=True)
    create_cradle_demo("sample_data/cradle_demo.mp4")
    create_fall_demo("sample_data/fall_demo.mp4")
    create_hazard_demo("sample_data/hazard_demo.mp4")
