import cv2
import mediapipe as mp


cap = cv2.VideoCapture(0)
    
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def calculate_bounding_box(hand_landmarks, frame_shape):
    min_x, min_y, max_x, max_y = frame_shape[1], frame_shape[0], 0, 0

    for landmark in hand_landmarks.landmark:
        h, w, c = frame_shape
        x, y = int(landmark.x * w), int(landmark.y * h)

        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    return (min_x, min_y), (max_x, max_y)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

            # Calculate bounding box
            bbox = calculate_bounding_box(hand_landmarks, frame.shape)

            # Draw bounding box
            cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 2)

            # Write text on the rectangle
            text = f"{handedness.classification[0].label} Hand Detected"
            cv2.putText(frame, text, (bbox[0][0], bbox[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # You can further process the hand landmarks here
            # For example, you can extract coordinates of specific landmarks
            # For simplicity, we'll just print whether it's a left or right hand
            handedness = results.multi_handedness[0].classification[0].label
            print(f"Detected {handedness} hand")

            # Draw landmarks on the image
            for landmark in hand_landmarks.landmark:
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
