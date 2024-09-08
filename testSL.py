import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from googletrans import Translator
import random

# Initialize a device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the class names
class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

# Load the pre-trained ResNet model
model = models.resnet18(weights='DEFAULT')  # Updated to use weights parameter
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(class_names))  # Adjust the output layer for the number of classes
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Dataset Preparation
data_path = "C:\\Users\\Sumit\\OneDrive\\Desktop\\ideathon\\asl_alphabet_train\\asl_alphabet_train"

images = []
labels = []

for subfolder in os.listdir(data_path):
    subfolder_path = os.path.join(data_path, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    if subfolder not in class_names:
        print(f"Warning: {subfolder} is not in class_names. Skipping...")
        continue

    image_files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path)]
    
    # Sample 10% of images from each class
    sampled_images = random.sample(image_files, max(1, int(len(image_files) * 0.1)))
    
    for image_path in sampled_images:
        images.append(image_path)
        labels.append(class_names.index(subfolder))

if len(images) == 0 or len(labels) == 0:
    raise ValueError("No images or labels found. Please check your dataset directory structure.")

train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.2, random_state=123, stratify=labels)

class ASLDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def translate_and_save(text):
    translator = Translator()
    languages = {'en': 'English', 'hi': 'Hindi', 'ja': 'Japanese', 'zh-cn': 'Chinese (Simplified)'}
    translations = {}
    
    for lang_code, lang_name in languages.items():
        try:
            translated = translator.translate(text, dest=lang_code).text
            translations[lang_name] = translated
        except Exception as e:
            print(f"Error translating to {lang_name}: {e}")
            translations[lang_name] = "Error"

    print(f"Saving sentence: {text.strip()}")  # Debug print

    # Write to file using utf-8 encoding
    with open('translated_sentences.txt', 'a', encoding='utf-8') as f:
        f.write(f"Original: {text}\n")
        for lang_name, translation in translations.items():
            f.write(f"{lang_name}: {translation}\n")
        f.write("\n")
        
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = ASLDataset(train_images, train_labels, transform=transform)
valid_dataset = ASLDataset(valid_images, valid_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

def train_model(model, train_loader, valid_loader, epochs=1):
    model.train()
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Loss: {running_loss/len(train_loader)}')

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Validation Accuracy: {accuracy * 100:.2f}%')

    torch.save(model.state_dict(), './trained_model.pth')

def predict(image):
    image = transform(Image.fromarray(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
    _, predicted_class = torch.max(outputs, 1)
    return predicted_class.item()

# Train the model with the predefined dataset
train_model(model, train_loader, valid_loader, epochs=1)

# Load the trained model for real-time predictions
model.load_state_dict(torch.load('./trained_model.pth'))
model.eval()

# Access the camera
cap = cv2.VideoCapture(0)

sentence = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break
    
    resized_frame = cv2.resize(frame, (224, 224))
    predicted_class = predict(resized_frame)
    predicted_sign = class_names[predicted_class]
    
    sentence += predicted_sign + " "
    
    cv2.putText(frame, f'Predicted: {predicted_sign}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('ASL Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == 13:  # Enter key pressed
        translate_and_save(sentence.strip())
        sentence = ""  # Clear the sentence after saving

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Translate and save the final text
translate_and_save(sentence.strip())
