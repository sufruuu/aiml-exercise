import cv2
import mediapipe as mp
import pygame
import time

# -----------------------------------
#   initialize pygame and load sounds
# -----------------------------------
print("Initializing Pygame and loading sounds...")
pygame.init()
pygame.mixer.init()

# load sound effects (.wav files in your working directory)
beep_sound = pygame.mixer.Sound("beep.wav")
shutter_sound = pygame.mixer.Sound("shutter.wav")

# -----------------------------------
#   setup mediapipe hand detection
# -----------------------------------
print("Setting up MediaPipe hand detection...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# -----------------------------------
#   initialize opencv video capture
# -----------------------------------
print("Starting video capture...")
cap = cv2.VideoCapture(0)  # use 1 for external camera

# flags for gesture detection
thumbs_up_detected = False
thumbs_up_time = 0


# --------------------------------------------------
#   define function to check for thumbs up gesture
# --------------------------------------------------
def is_thumbs_up(hand_landmarks, image_height, image_width):
    lm = hand_landmarks.landmark

    # convert normalized landmarks to pixel coordinates
    thumb_tip_y = lm[4].y * image_height
    thumb_ip_y = lm[3].y * image_height

    index_tip_y = lm[8].y * image_height
    index_mcp_y = lm[5].y * image_height

    middle_tip_y = lm[12].y * image_height
    middle_mcp_y = lm[9].y * image_height

    ring_tip_y = lm[16].y * image_height
    ring_mcp_y = lm[13].y * image_height

    pinky_tip_y = lm[20].y * image_height
    pinky_mcp_y = lm[17].y * image_height

    # check gesture conditions:
    if (thumb_tip_y < thumb_ip_y and  # thumb is extended
            index_tip_y > index_mcp_y and  # index finger is folded
            middle_tip_y > middle_mcp_y and  # middle finger is folded
            ring_tip_y > ring_mcp_y and  # ring finger is folded
            pinky_tip_y > pinky_mcp_y):  # pinky is folded
        return True
    return False


# -----------------------------------
#   main loop for processing frames
# -----------------------------------
print("Entering main loop. Press 'q' to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        break

    # flip frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # convert the frame color from BGR to RGB for mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # process the frame to detect hands
    results = hands.process(rgb_frame)

    # get frame dimensions for coordinate calculations
    h, w, _ = frame.shape
    gesture_detected = False  # flag to check if any hand shows thumbs up

    # if any hands are detected, process each hand
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # draw hand landmarks on the frame for visualization
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # check if the current hand is showing a thumbs up gesture
            if is_thumbs_up(hand_landmarks, h, w):
                gesture_detected = True
                print("Thumbs up gesture detected!")

                # play beep sound only if this is the first detection in the current cycle
                if not thumbs_up_detected:
                    beep_sound.play()
                    thumbs_up_detected = True
                    thumbs_up_time = time.time()  # record the time of detection
                break  # process only the first detected thumbs up gesture

    # if a thumbs up gesture has been detected, check if 3 seconds have passed to capture snapshot
    if thumbs_up_detected:
        elapsed_time = time.time() - thumbs_up_time
        countdown = max(0, 3 - int(elapsed_time))
        cv2.putText(frame, f"Capturing in {countdown} sec", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if elapsed_time >= 3:
            # Capture the snapshot
            snapshot = frame.copy()
            filename = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(filename, snapshot)
            print(f"Snapshot captured and saved as {filename}")

            # play shutter sound
            shutter_sound.play()

            # reset detection flags to allow further captures
            thumbs_up_detected = False
            thumbs_up_time = 0
            time.sleep(1)  # delay to avoid immediate retriggers

    # display the video feed with annotations
    cv2.imshow("Camera Feed", frame)

    # exit loop if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program.")
        break

# -----------------------------------
# cleanup and release resources
# -----------------------------------
cap.release()
cv2.destroyAllWindows()
pygame.quit()
print("Program terminated gracefully.")
