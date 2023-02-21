import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt


class PlayomgCardDS(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


class CardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(CardClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Linear(64 * 32 * 32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = CardClassifier()
model.load_state_dict(torch.load('model/card_classifier_model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def predict_card_class(image_path, model, transform, class_names):
    image_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class_index = torch.argmax(output, dim=1).item()
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

train_dataset = PlayomgCardDS('train', transform=transform)


image_path = '' # image to predict
predicted_card_class = predict_card_class(image_path, model, transform, train_dataset.classes)
print("Predicted card class index:", predicted_card_class)

image = Image.open(image_path)
plt.imshow(image)
plt.title(f"Predicted class: {predicted_card_class}")
plt.axis('off')
plt.show()
