import cv2
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hand module for my hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for visualization
mp_drawing = mp.solutions.drawing_utils

# Initialize the camera (0 is usually the default camera and 1 for thr secondary camera )
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the system audio volume control
def get_volume_control():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    return volume

# function for Setting the volume level
def set_volume(volume, level):
    volume.SetMasterVolumeLevelScalar(level, None)

# Get the current volume level from the speakers
def get_volume(volume):
    return volume.GetMasterVolumeLevelScalar()

volume = get_volume_control()
current_volume = get_volume(volume)
print(f"Current Volume: {current_volume}")

# Define variables for volume control
min_volume = 0.0
max_volume = 1.0
volume_step = 0.02  # Adjust this as needed on up to you

# Define coordinates and dimensions for the volume bar
bar_x = 30
bar_y = 30
bar_height = 300
bar_width = 20

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect both hands
    results = hands.process(rgb_frame)

    # If hands are detected,  then draw landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmarks for thumb and index finger
            thumb_tip = hand_landmarks.landmark[4]
            index_finger_tip = hand_landmarks.landmark[8]

            #  Euclidean distance calculate kregy fingur aur thumb ka coordinatre mai
            distance = ((thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5

            # Adjust volume based on distance (reverse logic)
            if distance < 0.05:
                current_volume -= volume_step  # Decrease volume when fingers are close
                if current_volume < min_volume:
                    current_volume = min_volume
            elif distance > 0.1:
                current_volume += volume_step  # Increase volume when fingers are far apart
                if current_volume > max_volume:
                    current_volume = max_volume

            # Set the new volume
            set_volume(volume, current_volume)

    # Draw the volume bar on the frame
    bar_length = int(bar_height * current_volume)
    cv2.rectangle(frame, (bar_x, bar_y + bar_height - bar_length), (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), -1)

    # Display the volume percentage on the frame
    cv2.putText(frame, f"Volume: {int(current_volume * 100)}%", (bar_x + bar_width + 10, bar_y + bar_height - bar_length), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with hand landmarks and the volume bar
    cv2.imshow("Camera with Volume Bar", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
