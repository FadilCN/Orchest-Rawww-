import cv2
import mediapipe as mp
import fluidsynth
import time
import threading
import pygame
import time
import numpy as np

pos_conv = None  # changed from 0 to None
running = True

def show_overlay():
    global pos_conv, running, chords
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    clock = pygame.time.Clock()

    # --- Load video instead of static background ---
    cap = cv2.VideoCapture("back60.mp4")
    if not cap.isOpened():
        print("Warning: Could not open video. Using placeholder background.")
        bg = pygame.Surface((640, 480)); bg.fill((50, 50, 50))
        use_video = False
    else:
        use_video = True

    # --- Load and scale images ---
    try:
        layers = [
            pygame.image.load("front.png").convert_alpha(),   # layer 0
            pygame.image.load("front2.png").convert_alpha(),  # layer 1
            pygame.image.load("front3.png").convert_alpha(),  # layer 2
            pygame.image.load("front4.png").convert_alpha()   # layer 3
        ]
        fore_layer = pygame.image.load("fore.png").convert_alpha()
    except pygame.error as e:
        print(f"Warning: {e}. Using placeholder surfaces.")
        layers = [pygame.Surface((640, 480), pygame.SRCALPHA) for _ in range(4)]
        layers[0].fill((255, 0, 0, 100)); layers[1].fill((0, 255, 0, 100))
        layers[2].fill((0, 0, 255, 100)); layers[3].fill((255, 255, 0, 100))
        fore_layer = pygame.Surface((640, 480), pygame.SRCALPHA)
    
    layers = [pygame.transform.smoothscale(layer, (640, 480)) for layer in layers]
    fore_layer = pygame.transform.smoothscale(fore_layer, (640, 480))

    # current Y positions for each layer
    current_ys = [0.0] * 4

    # min/max for linear mapping
    min_y = -28.8
    max_y = -180

    # Track the time movement starts AND the previous state
    start_time = None
    prev_pos_conv = None # Track the last value of pos_conv

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        current_time = pygame.time.get_ticks()

        # --- Read video frame ---
        if use_video:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bg = pygame.surfarray.make_surface(np.flipud(np.rot90(frame)))

        # --- Detect changes in pos_conv ---
        if pos_conv != prev_pos_conv:
            if pos_conv is not None:
                start_time = current_time
            else:
                start_time = None
            prev_pos_conv = pos_conv

        # update each layer's vertical position
        for i, layer in enumerate(layers):
            if pos_conv is not None and start_time is not None:
                delay_ms = i * 250  # delay between layers
                if (current_time - start_time) >= delay_ms:
                    mapped = chords[pos_conv][i]
                    target_y = min_y + (mapped - 50) * (max_y - min_y) / (70 - 50)
                    current_ys[i] += (target_y - current_ys[i]) * 0.05

        # --- DRAW ---
        screen.blit(bg, (0, 0))
        for i, layer in enumerate(layers):
            screen.blit(layer, (0, current_ys[i]))
        screen.blit(fore_layer, (0, 0))

        pygame.display.flip()
        clock.tick(60)

    if use_video:
        cap.release()
    pygame.quit()

sf2_path = "052_Florestan_Ahh_Choir.sf2"
fs = fluidsynth.Synth()
fs.start(driver="dsound")
sfid = fs.sfload(sf2_path)
fs.program_select(0, sfid, 0, 52)
chords = [
    [67, 60, 71, 64],
    [59, 67, 55, 62],
    [69, 64, 57, 60],
    [65, 58, 70, 62],
    [63, 52, 55, 59],
    [61, 54, 65, 57],
    [53, 60, 50, 57],
    [71, 62, 66, 59]
]

def play_chord(index_rec):
    index = index_rec

    fs.noteon(0, chords[index][0], 100)
    time.sleep(.5)
    fs.noteon(0, chords[index][1], 100)
    time.sleep(.5)
    fs.noteon(0, chords[index][2], 100)
    time.sleep(.5)
    fs.noteon(0, chords[index][3], 100)

    time.sleep(1)  # sustain
    time.sleep(.5)
    fs.noteoff(0, chords[index][1])
    time.sleep(.5)
    fs.noteoff(0, chords[index][2])
    time.sleep(.5)
    fs.noteoff(0, chords[index][3])
    fs.noteoff(0, chords[index][0])


is_playing = False

def play_chord_thread(index):
    global is_playing
    is_playing = True
    play_chord(index)
    is_playing = False


# Start Pygame overlay in its own thread
threading.Thread(target=show_overlay, daemon=True).start()

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands()

while running:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and not is_playing:
        hand = result.multi_hand_landmarks[0]
        pos = hand.landmark[12]
        pos_conv = int(pos.y * 8)
        print(pos_conv)

        threading.Thread(target=play_chord_thread, args=(pos_conv,)).start()

    cv2.imshow("Hand Chords", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        running = False
        break

cap.release()
cv2.destroyAllWindows()
