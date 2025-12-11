import cv2
import torch
import torch.nn as nn
import os
import sys

# --- CONFIG ---
VIDEO_PATH = "videos/openaigym.video.0.82530.video000000.mp4" # CHANGE THIS to your actual video file
CLASSIFIER_PATH = "cave_classifier.pth"

# (Must match your training script)
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Could not find video {VIDEO_PATH}")
        print("Check your 'videos' folder and copy the filename of a generated mp4.")
        return

    # Load Model
    device = torch.device("cpu")
    model = SimpleClassifier().to(device)
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(VIDEO_PATH)
    print("Playing video with AI Vision... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Preprocess for AI (Resize -> Tensor)
        # Note: The video is likely 64x64 already if recorded from MineRL
        img_input = cv2.resize(frame, (64, 64))
        img_tensor = torch.from_numpy(img_input).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            cave_prob = probs[0][1].item()

        # Visualize
        # Scale up for human viewing
        display_frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # Draw Bar
        color = (0, 255, 0) if cave_prob > 0.9 else (0, 0, 255)
        text = f"Cave Conf: {cave_prob*100:.1f}%"
        cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if cave_prob > 0.9:
            cv2.putText(display_frame, "REWARD!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Agent Vision", display_frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()