# # import pandas as pd
# # import torch.nn as nn

# # class CNN(nn.Module):
# #     def __init__(self, K):
# #         super(CNN, self).__init__()
# #         self.conv_layers = nn.Sequential(
# #             # conv1
# #             nn.Conv2d(in_channels=3, out_channels=32,
# #                       kernel_size=3, padding=1),
# #             nn.ReLU(),
# #             nn.BatchNorm2d(32),
# #             nn.Conv2d(in_channels=32, out_channels=32,
# #                       kernel_size=3, padding=1),
# #             nn.ReLU(),
# #             nn.BatchNorm2d(32),
# #             nn.MaxPool2d(2),
# #             # conv2
# #             nn.Conv2d(in_channels=32, out_channels=64,
# #                       kernel_size=3, padding=1),
# #             nn.ReLU(),
# #             nn.BatchNorm2d(64),
# #             nn.Conv2d(in_channels=64, out_channels=64,
# #                       kernel_size=3, padding=1),
# #             nn.ReLU(),
# #             nn.BatchNorm2d(64),
# #             nn.MaxPool2d(2),
# #             # conv3
# #             nn.Conv2d(in_channels=64, out_channels=128,
# #                       kernel_size=3, padding=1),
# #             nn.ReLU(),
# #             nn.BatchNorm2d(128),
# #             nn.Conv2d(in_channels=128, out_channels=128,
# #                       kernel_size=3, padding=1),
# #             nn.ReLU(),
# #             nn.BatchNorm2d(128),
# #             nn.MaxPool2d(2),
# #             # conv4
# #             nn.Conv2d(in_channels=128, out_channels=256,
# #                       kernel_size=3, padding=1),
# #             nn.ReLU(),
# #             nn.BatchNorm2d(256),
# #             nn.Conv2d(in_channels=256, out_channels=256,
# #                       kernel_size=3, padding=1),
# #             nn.ReLU(),
# #             nn.BatchNorm2d(256),
# #             nn.MaxPool2d(2),
# #         )

# #         self.dense_layers = nn.Sequential(
# #             nn.Dropout(0.4),
# #             nn.Linear(50176, 1024),
# #             nn.ReLU(),
# #             nn.Dropout(0.4),
# #             nn.Linear(1024, K),
# #         )

# #     def forward(self, X):
# #         out = self.conv_layers(X)

# #         # Flatten
# #         out = out.view(-1, 50176)

# #         # Fully connected
# #         out = self.dense_layers(out)

# #         return out


# # idx_to_classes = {0: 'Apple___Apple_scab',
# #                   1: 'Apple___Black_rot',
# #                   2: 'Apple___Cedar_apple_rust',
# #                   3: 'Apple___healthy',
# #                   4: 'Background_without_leaves',
# #                   5: 'Blueberry___healthy',
# #                   6: 'Cherry___Powdery_mildew',
# #                   7: 'Cherry___healthy',
# #                   8: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
# #                   9: 'Corn___Common_rust',
# #                   10: 'Corn___Northern_Leaf_Blight',
# #                   11: 'Corn___healthy',
# #                   12: 'Grape___Black_rot',
# #                   13: 'Grape___Esca_(Black_Measles)',
# #                   14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
# #                   15: 'Grape___healthy',
# #                   16: 'Orange___Haunglongbing_(Citrus_greening)',
# #                   17: 'Peach___Bacterial_spot',
# #                   18: 'Peach___healthy',
# #                   19: 'Pepper,_bell___Bacterial_spot',
# #                   20: 'Pepper,_bell___healthy',
# #                   21: 'Potato___Early_blight',
# #                   22: 'Potato___Late_blight',
# #                   23: 'Potato___healthy',
# #                   24: 'Raspberry___healthy',
# #                   25: 'Soybean___healthy',
# #                   26: 'Squash___Powdery_mildew',
# #                   27: 'Strawberry___Leaf_scorch',
# #                   28: 'Strawberry___healthy',
# #                   29: 'Tomato___Bacterial_spot',
# #                   30: 'Tomato___Early_blight',
# #                   31: 'Tomato___Late_blight',
# #                   32: 'Tomato___Leaf_Mold',
# #                   33: 'Tomato___Septoria_leaf_spot',
# #                   34: 'Tomato___Spider_mites Two-spotted_spider_mite',
# #                   35: 'Tomato___Target_Spot',
# #                   36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
# #                   37: 'Tomato___Tomato_mosaic_virus',
# #                   38: 'Tomato___healthy'}



















# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image

# # --- CNN Model Definition ---
# class CNN(nn.Module):
#     def __init__(self, K):
#         super(CNN, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(2),

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(2),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(2),

#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             nn.MaxPool2d(2),
#         )

#         self.dense_layers = nn.Sequential(
#             nn.Dropout(0.4),
#             nn.Linear(50176, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(1024, K),
#         )

#     def forward(self, X):
#         out = self.conv_layers(X)
#         out = out.view(-1, 50176)  # Flatten
#         out = self.dense_layers(out)
#         return out


# # --- Index to Class Mapping ---
# idx_to_classes = {
#     0: 'Apple___Apple_scab',
#     1: 'Apple___Black_rot',
#     2: 'Apple___Cedar_apple_rust',
#     3: 'Apple___healthy',
#     4: 'Background_without_leaves',
#     5: 'Blueberry___healthy',
#     6: 'Cherry___Powdery_mildew',
#     7: 'Cherry___healthy',
#     8: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
#     9: 'Corn___Common_rust',
#     10: 'Corn___Northern_Leaf_Blight',
#     11: 'Corn___healthy',
#     12: 'Grape___Black_rot',
#     13: 'Grape___Esca_(Black_Measles)',
#     14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#     15: 'Grape___healthy',
#     16: 'Orange___Haunglongbing_(Citrus_greening)',
#     17: 'Peach___Bacterial_spot',
#     18: 'Peach___healthy',
#     19: 'Pepper,_bell___Bacterial_spot',
#     20: 'Pepper,_bell___healthy',
#     21: 'Potato___Early_blight',
#     22: 'Potato___Late_blight',
#     23: 'Potato___healthy',
#     24: 'Raspberry___healthy',
#     25: 'Soybean___healthy',
#     26: 'Squash___Powdery_mildew',
#     27: 'Strawberry___Leaf_scorch',
#     28: 'Strawberry___healthy',
#     29: 'Tomato___Bacterial_spot',
#     30: 'Tomato___Early_blight',
#     31: 'Tomato___Late_blight',
#     32: 'Tomato___Leaf_Mold',
#     33: 'Tomato___Septoria_leaf_spot',
#     34: 'Tomato___Spider_mites Two-spotted_spider_mite',
#     35: 'Tomato___Target_Spot',
#     36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
#     37: 'Tomato___Tomato_mosaic_virus',
#     38: 'Tomato___healthy'
# }


# # --- Load Model Once ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CNN(K=len(idx_to_classes))
# model.load_state_dict(torch.load("plant_disease_model_1_latest.pt", map_location=device))
# model.eval()
# model.to(device)

# def predict_disease(image_path):
#     # Example prediction logic
#     pred = 7   # Suppose model predicts class 7 (Orange disease)

#     # Map pred → full details
#     disease_info = {
#         7: {
#             "title": "Orange - Citrus Canker",
#             "desc": "Citrus canker is caused by bacteria, leading to lesions on leaves and fruit.",
#             "prevent": "Use copper-based sprays, remove infected leaves, maintain orchard hygiene.",
#             "simage": "/static/fertilizers/copper.png",
#             "sname": "Copper Fungicide",
#             "buy_link": "https://example.com/copper-fungicide"
#         },
#         # add other mappings...
#     }

#     return {
#         "pred": pred,
#         "title": disease_info[pred]["title"],
#         "desc": disease_info[pred]["desc"],
#         "prevent": disease_info[pred]["prevent"],
#         "simage": disease_info[pred]["simage"],
#         "sname": disease_info[pred]["sname"],
#         "buy_link": disease_info[pred]["buy_link"]
#     }

# # --- Prediction Function ---

# # def predict_disease(image_path):
# #     transform = transforms.Compose([
# #         transforms.Resize((224, 224)),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                              std=[0.229, 0.224, 0.225])
# #     ])

# #     image = Image.open(image_path).convert("RGB")
# #     image = transform(image).unsqueeze(0).to(device)

# #     with torch.no_grad():
# #         outputs = model(image)
# #         _, predicted = torch.max(outputs, 1)
# #         class_idx = predicted.item()
# #         return idx_to_classes[class_idx]   # ✅ return clean string


# # def predict_disease(image_path):
# #     transform = transforms.Compose([
# #         transforms.Resize((224, 224)),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                              std=[0.229, 0.224, 0.225])
# #     ])

# #     image = Image.open(image_path).convert("RGB")
# #     image = transform(image).unsqueeze(0).to(device)

# #     with torch.no_grad():
# #         outputs = model(image)
# #         _, predicted = torch.max(outputs, 1)
# #         class_idx = predicted.item()
# #         # return {"disease": idx_to_classes[class_idx]}
# #         return idx_to_classes[class_idx]





import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# --- CNN Model Definition ---
class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(-1, 50176)  # Flatten
        out = self.dense_layers(out)
        return out


# --- Index to Class Mapping ---
idx_to_classes = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Background_without_leaves',
    5: 'Blueberry___healthy',
    6: 'Cherry___Powdery_mildew',
    7: 'Cherry___healthy',
    8: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    9: 'Corn___Common_rust',
    10: 'Corn___Northern_Leaf_Blight',
    11: 'Corn___healthy',
    12: 'Grape___Black_rot',
    13: 'Grape___Esca_(Black_Measles)',
    14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    15: 'Grape___healthy',
    16: 'Orange___Haunglongbing_(Citrus_greening)',
    17: 'Peach___Bacterial_spot',
    18: 'Peach___healthy',
    19: 'Pepper,_bell___Bacterial_spot',
    20: 'Pepper,_bell___healthy',
    21: 'Potato___Early_blight',
    22: 'Potato___Late_blight',
    23: 'Potato___healthy',
    24: 'Raspberry___healthy',
    25: 'Soybean___healthy',
    26: 'Squash___Powdery_mildew',
    27: 'Strawberry___Leaf_scorch',
    28: 'Strawberry___healthy',
    29: 'Tomato___Bacterial_spot',
    30: 'Tomato___Early_blight',
    31: 'Tomato___Late_blight',
    32: 'Tomato___Leaf_Mold',
    33: 'Tomato___Septoria_leaf_spot',
    34: 'Tomato___Spider_mites Two-spotted_spider_mite',
    35: 'Tomato___Target_Spot',
    36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    37: 'Tomato___Tomato_mosaic_virus',
    38: 'Tomato___healthy'
}


# --- Disease Information Mapping (Expand this dictionary for all classes) ---

disease_info = {
    0: {
        "title": "Apple - Apple Scab",
        "desc": "A fungal disease that causes dark lesions on leaves, fruits, and twigs.",
        "prevent": "Use resistant apple varieties, prune affected branches, apply fungicides.",
        "simage": "/static/fertilizers/mancozeb.png",
        "sname": "Mancozeb Fungicide",
        "buy_link": "https://example.com/mancozeb"
    },
    1: {
        "title": "Apple - Black Rot",
        "desc": "Causes leaf spots, fruit rot, and branch cankers in apple trees.",
        "prevent": "Prune dead wood, spray copper-based fungicides, maintain orchard hygiene.",
        "simage": "/static/fertilizers/copper.png",
        "sname": "Copper Fungicide",
        "buy_link": "https://example.com/copper"
    },
    2: {
        "title": "Apple - Cedar Apple Rust",
        "desc": "Fungal disease forming bright orange spots on apple leaves and fruits.",
        "prevent": "Remove nearby juniper hosts, apply preventive fungicides.",
        "simage": "/static/fertilizers/propiconazole.png",
        "sname": "Propiconazole Fungicide",
        "buy_link": "https://example.com/propiconazole"
    },
    3: {
        "title": "Apple - Healthy",
        "desc": "No visible signs of disease. Tree is healthy.",
        "prevent": "Maintain good irrigation, balanced nutrients, and pruning.",
        "simage": "/static/fertilizers/general.png",
        "sname": "Plant Care Kit",
        "buy_link": "https://example.com/plant-care"
    },
    4: {
        "title": "Background - No Disease",
        "desc": "Image contains background with no plant disease.",
        "prevent": "Not applicable.",
        "simage": "/static/fertilizers/na.png",
        "sname": "N/A",
        "buy_link": "#"
    },
    5: {
        "title": "Blueberry - Healthy",
        "desc": "Blueberry leaf is healthy and green.",
        "prevent": "Provide proper irrigation and soil nutrients.",
        "simage": "/static/fertilizers/general.png",
        "sname": "Soil Nutrient Mix",
        "buy_link": "https://example.com/nutrient-mix"
    },
    6: {
        "title": "Cherry - Powdery Mildew",
        "desc": "Fungal disease producing white powdery coating on leaves.",
        "prevent": "Apply sulfur-based fungicides, improve air circulation.",
        "simage": "/static/fertilizers/sulfur.png",
        "sname": "Sulfur Fungicide",
        "buy_link": "https://example.com/sulfur"
    },
    7: {
        "title": "Cherry - Healthy",
        "desc": "No symptoms of disease on cherry leaves.",
        "prevent": "Maintain orchard hygiene and proper pruning.",
        "simage": "/static/fertilizers/general.png",
        "sname": "Plant Care Kit",
        "buy_link": "https://example.com/plant-care"
    },
    8: {
        "title": "Corn - Cercospora Leaf Spot",
        "desc": "Causes gray leaf spots leading to reduced yield.",
        "prevent": "Rotate crops, use resistant varieties, fungicide sprays.",
        "simage": "/static/fertilizers/strobilurin.png",
        "sname": "Strobilurin Fungicide",
        "buy_link": "https://example.com/strobilurin"
    },
    9: {
        "title": "Corn - Common Rust",
        "desc": "Orange pustules appear on corn leaves due to fungal infection.",
        "prevent": "Use resistant hybrids, apply fungicides.",
        "simage": "/static/fertilizers/tebuconazole.png",
        "sname": "Tebuconazole Fungicide",
        "buy_link": "https://example.com/tebuconazole"
    },
    10: {
        "title": "Corn - Northern Leaf Blight",
        "desc": "Elongated gray-green lesions caused by fungal infection.",
        "prevent": "Crop rotation, hybrid resistance, fungicides.",
        "simage": "/static/fertilizers/triazole.png",
        "sname": "Triazole Fungicide",
        "buy_link": "https://example.com/triazole"
    },
    11: {
        "title": "Corn - Healthy",
        "desc": "Corn plant is healthy.",
        "prevent": "Maintain soil fertility and irrigation.",
        "simage": "/static/fertilizers/general.png",
        "sname": "Soil Care Mix",
        "buy_link": "https://example.com/soil-care"
    },
    12: {
        "title": "Grape - Black Rot",
        "desc": "Causes dark brown leaf spots and shriveled fruits.",
        "prevent": "Remove infected debris, apply fungicides.",
        "simage": "/static/fertilizers/mancozeb.png",
        "sname": "Mancozeb Fungicide",
        "buy_link": "https://example.com/mancozeb"
    },
    13: {
        "title": "Grape - Esca (Black Measles)",
        "desc": "Complex fungal disease causing tiger-striped leaves.",
        "prevent": "Avoid pruning wounds, apply systemic fungicides.",
        "simage": "/static/fertilizers/systemic.png",
        "sname": "Systemic Fungicide",
        "buy_link": "https://example.com/systemic"
    },
    14: {
        "title": "Grape - Leaf Blight",
        "desc": "Irregular brown necrotic spots on grape leaves.",
        "prevent": "Improve vineyard sanitation, fungicide sprays.",
        "simage": "/static/fertilizers/carbendazim.png",
        "sname": "Carbendazim Fungicide",
        "buy_link": "https://example.com/carbendazim"
    },
    15: {
        "title": "Grape - Healthy",
        "desc": "No disease detected on grapevine.",
        "prevent": "Regular monitoring and preventive sprays.",
        "simage": "/static/fertilizers/general.png",
        "sname": "Plant Health Booster",
        "buy_link": "https://example.com/booster"
    },
    16: {
        "title": "Orange - Citrus Greening (HLB)",
        "desc": "Bacterial disease causing yellow shoots and bitter fruits.",
        "prevent": "Control psyllid insect, remove infected trees.",
        "simage": "/static/fertilizers/insecticide.png",
        "sname": "Imidacloprid Insecticide",
        "buy_link": "https://example.com/imidacloprid"
    },
    17: {
        "title": "Peach - Bacterial Spot",
        "desc": "Small water-soaked lesions on leaves and fruits.",
        "prevent": "Use resistant varieties, apply copper sprays.",
        "simage": "/static/fertilizers/copper.png",
        "sname": "Copper Spray",
        "buy_link": "https://example.com/copper"
    },
    18: {
        "title": "Peach - Healthy",
        "desc": "No symptoms found on peach leaves.",
        "prevent": "Proper irrigation and soil nutrition.",
        "simage": "/static/fertilizers/general.png",
        "sname": "Fruit Tree Care",
        "buy_link": "https://example.com/tree-care"
    },
    19: {
        "title": "Pepper - Bacterial Spot",
        "desc": "Causes dark, water-soaked spots on pepper leaves.",
        "prevent": "Use disease-free seeds, copper fungicides.",
        "simage": "/static/fertilizers/copper.png",
        "sname": "Copper Fungicide",
        "buy_link": "https://example.com/copper"
    },
    20: {
        "title": "Pepper - Healthy",
        "desc": "Pepper plant shows no sign of disease.",
        "prevent": "Maintain irrigation and pest control.",
        "simage": "/static/fertilizers/general.png",
        "sname": "Plant Care Kit",
        "buy_link": "https://example.com/plant-care"
    },
    21: {
        "title": "Potato - Early Blight",
        "desc": "Dark concentric spots on lower leaves.",
        "prevent": "Rotate crops, apply chlorothalonil fungicides.",
        "simage": "/static/fertilizers/chlorothalonil.png",
        "sname": "Chlorothalonil Fungicide",
        "buy_link": "https://example.com/chlorothalonil"
    },
    22: {
        "title": "Potato - Late Blight",
        "desc": "Irregular water-soaked spots spreading rapidly.",
        "prevent": "Destroy infected plants, spray metalaxyl fungicide.",
        "simage": "/static/fertilizers/metalaxyl.png",
        "sname": "Metalaxyl Fungicide",
        "buy_link": "https://example.com/metalaxyl"
    },
    23: {
        "title": "Potato - Healthy",
        "desc": "Potato leaf is healthy.",
        "prevent": "Maintain soil fertility and irrigation.",
        "simage": "/static/fertilizers/general.png",
        "sname": "Soil Nutrient Mix",
        "buy_link": "https://example.com/nutrient-mix"
    },
    24: {
        "title": "Raspberry - Healthy",
        "desc": "No signs of disease detected.",
        "prevent": "Maintain good soil health.",
        "simage": "/static/fertilizers/general.png",
        "sname": "Plant Health Kit",
        "buy_link": "https://example.com/health-kit"
    },
    25: {
        "title": "Soybean - Healthy",
        "desc": "Soybean leaf is green and healthy.",
        "prevent": "Ensure irrigation and pest management.",
        "simage": "/static/fertilizers/general.png",
        "sname": "Soil Nutrient Kit",
        "buy_link": "https://example.com/nutrient-kit"
    },
    26: {
        "title": "Squash - Powdery Mildew",
        "desc": "White powdery growth on squash leaves.",
        "prevent": "Improve air flow, apply sulfur fungicides.",
        "simage": "/static/fertilizers/sulfur.png",
        "sname": "Sulfur Fungicide",
        "buy_link": "https://example.com/sulfur"
    },
    27: {
        "title": "Strawberry - Leaf Scorch",
        "desc": "Red-brown edges on leaves leading to drying.",
        "prevent": "Use resistant varieties, apply fungicides.",
        "simage": "/static/fertilizers/fungicide.png",
        "sname": "Fungicide Spray",
        "buy_link": "https://example.com/fungicide"
    },
    28: {
        "title": "Strawberry - Healthy",
        "desc": "Strawberry leaf looks healthy.",
        "prevent": "Good irrigation and pest-free environment.",
        "simage": "/static/fertilizers/general.png",
        "sname": "Plant Care Pack",
        "buy_link": "https://example.com/plant-pack"
    },
    29: {
        "title": "Tomato - Bacterial Spot",
        "desc": "Small, dark, water-soaked spots on leaves.",
        "prevent": "Use disease-free seeds, copper sprays.",
        "simage": "/static/fertilizers/copper.png",
        "sname": "Copper Fungicide",
        "buy_link": "https://example.com/copper"
    },
    30: {
        "title": "Tomato - Early Blight",
        "desc": "Dark concentric lesions on tomato leaves.",
        "prevent": "Crop rotation, fungicide applications.",
        "simage": "/static/fertilizers/chlorothalonil.png",
        "sname": "Chlorothalonil Spray",
        "buy_link": "https://example.com/chlorothalonil"
    },
    31: {
        "title": "Tomato - Late Blight",
        "desc": "Irregular water-soaked lesions spreading rapidly.",
        "prevent": "Destroy infected plants, apply metalaxyl.",
        "simage": "/static/fertilizers/metalaxyl.png",
        "sname": "Metalaxyl Fungicide",
        "buy_link": "https://example.com/metalaxyl"
    },
    32: {
        "title": "Tomato - Leaf Mold",
        "desc": "Yellow spots on upper side with mold underneath.",
        "prevent": "Improve ventilation, fungicide sprays.",
        "simage": "/static/fertilizers/copper.png",
        "sname": "Copper Fungicide",
        "buy_link": "https://example.com/copper"
    },
    33: {
        "title": "Tomato - Septoria Leaf Spot",
        "desc": "Small circular spots with dark borders.",
        "prevent": "Remove infected leaves, apply fungicides.",
        "simage": "/static/fertilizers/fungicide.png",
        "sname": "General Fungicide",
        "buy_link": "https://example.com/fungicide"
    },
    34: {
        "title": "Tomato - Spider Mites",
        "desc": "Causes leaf yellowing and webbing.",
        "prevent": "Spray miticides, neem oil.",
        "simage": "/static/fertilizers/miticide.png",
        "sname": "Miticide Spray",
        "buy_link": "https://example.com/miticide"
    },
    35: {
        "title": "Tomato - Target Spot",
        "desc": "Circular lesions with concentric rings.",
        "prevent": "Apply preventive fungicides, crop rotation.",
        "simage": "/static/fertilizers/fungicide.png",
        "sname": "Preventive Fungicide",
        "buy_link": "https://example.com/fungicide"
    },
    36: {
        "title": "Tomato - Yellow Leaf Curl Virus",
        "desc": "Viral disease causing curling of leaves.",
        "prevent": "Control whiteflies, use resistant varieties.",
        "simage": "/static/fertilizers/insecticide.png",
        "sname": "Whitefly Control",
        "buy_link": "https://example.com/whitefly"
    },
    37: {
        "title": "Tomato - Mosaic Virus",
        "desc": "Leaves become mottled with light and dark patches.",
        "prevent": "Remove infected plants, sanitize tools.",
        "simage": "/static/fertilizers/general.png",
        "sname": "Plant Virus Management",
        "buy_link": "https://example.com/virus-control"
    },
    38: {
        "title": "Tomato - Healthy",
        "desc": "Tomato plant is healthy and green.",
        "prevent": "Maintain irrigation and soil nutrients.",
        "simage": "/static/fertilizers/general.png",
        "sname": "Tomato Care Kit",
        "buy_link": "https://example.com/tomato-care"
    }
}

# disease_info = {
#     0: {
#         "title": "Apple - Apple Scab",
#         "desc": "Apple scab is a fungal disease causing dark, scabby lesions on leaves and fruits.",
#         "prevent": "Use resistant varieties, apply fungicides, and remove fallen leaves.",
#         "simage": "/static/fertilizers/fungicide.png",
#         "sname": "Mancozeb Fungicide",
#         "buy_link": "https://example.com/mancozeb"
#     },
#     1: {
#         "title": "Apple - Black Rot",
#         "desc": "Black rot affects apple trees with leaf spots, fruit rot, and branch cankers.",
#         "prevent": "Prune infected branches, use fungicide sprays, and maintain orchard sanitation.",
#         "simage": "/static/fertilizers/copper.png",
#         "sname": "Copper Fungicide",
#         "buy_link": "https://example.com/copper-fungicide"
#     },
#     16: {
#         "title": "Orange - Citrus Greening (HLB)",
#         "desc": "Citrus greening is a bacterial disease causing yellowing of leaves and misshapen fruits.",
#         "prevent": "Control psyllid vectors, remove infected trees, and apply systemic insecticides.",
#         "simage": "/static/fertilizers/insecticide.png",
#         "sname": "Imidacloprid Insecticide",
#         "buy_link": "https://example.com/imidacloprid"
#     },
#     21: {
#         "title": "Potato - Early Blight",
#         "desc": "Early blight causes dark spots with concentric rings on potato leaves and stems.",
#         "prevent": "Rotate crops, use resistant varieties, and apply fungicides.",
#         "simage": "/static/fertilizers/chlorothalonil.png",
#         "sname": "Chlorothalonil Fungicide",
#         "buy_link": "https://example.com/chlorothalonil"
#     },
#     38: {
#         "title": "Tomato - Healthy",
#         "desc": "This tomato leaf is healthy with no signs of disease.",
#         "prevent": "Maintain regular watering, balanced fertilization, and good field hygiene.",
#         "simage": "/static/fertilizers/general.png",
#         "sname": "General Plant Care",
#         "buy_link": "https://example.com/plant-care"
#     }
# }


# --- Load Model Once ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(K=len(idx_to_classes))
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt", map_location=device))
model.eval()
model.to(device)


# --- Prediction Function ---
def predict_disease(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()

    # Fallback if class not in disease_info
    info = disease_info.get(class_idx, {
        "title": idx_to_classes[class_idx],
        "desc": "No detailed description available.",
        "prevent": "No prevention details available.",
        "simage": "/static/fertilizers/default.png",
        "sname": "General Advice",
        "buy_link": "#"
    })

    return {
        "pred": class_idx,
        "title": info["title"],
        "desc": info["desc"],
        "prevent": info["prevent"],
        "simage": info["simage"],
        "sname": info["sname"],
        "buy_link": info["buy_link"]
    }

