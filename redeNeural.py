
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import copy
import torch, gc
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import pickle
import matplotlib.pyplot as plt


def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    since = time.time()
    
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        
        for phase in ['treino', 'validacao']:
            if phase == 'treino':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'treino'):
                    outputs = model(inputs)  
                    _, preds = torch.max(outputs, 1)  
                    loss = criterion(outputs, labels)  

                    if phase == 'treino':
                        loss.backward()  
                        optimizer.step()  

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            
            if phase == 'validacao' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Treinamento concluído em {time_elapsed // 60}m {time_elapsed % 60}s')

    
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "treinoUf05.pth")
    return model

def test_model(model, dataloader):
    model.eval()  
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

    accuracy = running_corrects.double() / len(dataloader.dataset)
    print(f'Accuracy: {accuracy:.4f}')

def save_confusion_matrix(model, dataloader, class_names, device, save_path):

    model.eval()
    

    all_labels = []
    all_preds = []
    

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
        
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) 
            
        
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    

    cm = confusion_matrix(all_labels, all_preds)
    

    cm_data_path = save_path + "_matrix.pkl"
    with open(cm_data_path, "wb") as f:
        pickle.dump({"confusion_matrix": cm, "class_names": class_names}, f)
    print(f"Matriz de confusão salva em {cm_data_path}")
    

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    fig.savefig(save_path + "_figure.png")
    print(f"Figura da matriz de confusão salva em {save_path}_figure.png")
    plt.close(fig)


imgDim = 32

dataTransforms = {
    "treino": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((imgDim,imgDim)),
        transforms.ToTensor()
    ]),
    "validacao":  transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((imgDim,imgDim)),
        transforms.ToTensor()
    ]),
    "teste":  transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((imgDim,imgDim)),
        transforms.ToTensor()
    ])
}

gc.collect()
torch.cuda.empty_cache()

sets = ["treino", "validacao", "teste"]

dataDir = "./datasets"

#imageDatasets = {x: datasets.ImageFolder(root = f'{dataDir}/{x}', transform=dataTransforms[x]) for x in sets}
#dataloaders = {x: DataLoader(imageDatasets[x], batch_size=32, shuffle=True, num_workers=2) for x in sets}

imageDatasetsReal = datasets.ImageFolder(root = "./PKLot/PKLotSegmented/puc", transform=dataTransforms["teste"])
dataloadersReal = DataLoader(imageDatasetsReal, batch_size=32, shuffle=True, num_workers=2)

model = models.convnext_base(weights="ConvNeXt_Base_Weights.DEFAULT")
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
model.features[0][0] = nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)



datasetSizes =  len(imageDatasetsReal)
classNames = imageDatasetsReal.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

checkPointPath = "./treinoUf0590.pth"
model.load_state_dict(torch.load(checkPointPath, map_location=device,weights_only=True))

#save_confusion_matrix(model = model, dataloader = dataloadersReal, class_names = classNames, device = device, save_path="matrizConfusaoUf05x2")
#model = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)

#test_model(model, dataloaders['teste'])

#print("Dataloaders:", dataloaders["treino"])
#print("Device empregado:", device)
print("Tamanhos dos datasets:", datasetSizes)
#print("Classes:", classNames)

