
# Fichier : resnet18_entraineur.py
import os
import json
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchvision import vgg19, VGG19_Weights, transforms
from datetime import datetime

# Configuration GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device}")

foldername = '/data/calcul0/jakoekis/Vgg19'

# === Dataset CACAO ===
class CACAO(Dataset):
    def __init__(self, data_root, rand_st, training=True):
        self.training = training
        suffix = "training" if training else "test"
        self.imgs = joblib.load(os.path.join(data_root, f"{suffix}_{rand_st}.pkl"))
        self.mc = joblib.load(os.path.join(data_root, f"mc_{suffix if training else 'test'}_{rand_st}.pkl"))

        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transforms = {
            'original': self.base_transform,
            'hflip': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=0.5),  # Force flip
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'vflip': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.RandomVerticalFlip(p=0.5),  # Force flip
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'rotate90': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.RandomRotation(degrees=(90, 90)),  # Exact 90°
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'rotate180': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.RandomRotation(degrees=(180, 180)),  # Exact 180°
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

    def __len__(self):
        return len(self.mc)

    def __getitem__(self, index):
        img = self.imgs[index]
        mc = torch.tensor([float(str(self.mc[index]).replace(',', '.').split()[0])], dtype=torch.float32)
        if self.training:
            # Apply all transformations
            transformed = {
                'original': self.transforms['original'](img),
                'hflip': self.transforms['hflip'](img),
                'vflip': self.transforms['vflip'](img),
                'rotate90': self.transforms['rotate90'](img),
                'rotate180': self.transforms['rotate180'](img),
                'mc': mc
            }
        else:
            # Validation - only original
            transformed = {
                'original': self.transforms['original'](img),
                'mc': mc
            }
            
        return transformed

# === Initialisation du modèle ===
def initialize_model():
    model = vgg19(weights=VGG19_Weights.DEFAULT)
    model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
    for param in model.parameters():
            param.requires_grad = True
    model.classifier = torch.nn.Linear(512, 1)
    return model

def save_checkpoint(model, optimizer, epoch, loss, filename=os.path.join(foldername,'checkpoint.pth')):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint sauvegardé à l'epoch {epoch}")

def load_checkpoint(model, optimizer, filename=os.path.join(foldername,'checkpoint.pth')):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint chargé - Reprise à l'epoch {epoch} avec loss {loss}")
        return epoch, loss
    return 0, np.inf

def mae_fn(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))
# === Entraînement ===
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch_index, writer):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    nbr_batch = len(dataloader)
    
    for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch_index+1}")):
        mc = data['mc'].to(device).float()
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass for each transformation
        losses = []
        for transform_name in ['original', 'hflip', 'vflip', 'rotate90', 'rotate180']:
            inputs = data[transform_name].to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, mc)
            losses.append(loss)
            # Only compute metrics on original
            if transform_name == 'original':
                mae = mae_fn(outputs, mc)
                running_mae += mae.item()
        
        # Combined loss (average of all transformations)
        combined_loss = torch.mean(torch.stack(losses))
        combined_loss.backward()
        optimizer.step()
        
        running_loss += combined_loss.item()

    avg_loss = running_loss / nbr_batch
    avg_mae = running_mae / nbr_batch

    tb_x = epoch_index * len(training) + i + 1
    writer.add_scalar('Loss/train', avg_loss, tb_x)
    writer.add_scalar('MAE/train', avg_mae, tb_x)
    return avg_loss, avg_mae

# === Validation ===
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    counter = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Validating"):
            inputs = data['original'].to(device)
            mc = data['mc'].to(device).float()
            
            outputs = model(inputs.float())
            loss = criterion(outputs, mc)
            mae = mae_fn(outputs, mc)
            
            running_loss += loss.item()
            running_mae += mae.item()
            counter += 1
    avg_loss = running_loss / counter
    avg_mae = running_mae / counter
    return avg_loss, avg_mae

# === Boucle principale ===
print("boucle principale")
def main():
    global model, optimizer, training, testing, nbr_batch
    model = initialize_model(True)
    model = model.to(device)

    rand_st = 10
    data_root = os.path.join("/data/calcul0/jakoekis/ft_dossierok")
    print('chargement des donnees')

    training_data = CACAO(data_root, rand_st)
    testing_data = CACAO(data_root, rand_st, False)
    training = torch.utils.data.DataLoader(training_data, batch_size=2, shuffle=True)
    testing = torch.utils.data.DataLoader(testing_data, batch_size=1, shuffle=True)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    writer = SummaryWriter(os.path.join(foldername,f"Cacao_Joel_trainer_{timestamp}")) 
    epoch_number, best_loss = load_checkpoint(model, optimizer)
    epoch_number = 0
    EPOCHS = 200
    best_loss = np.inf
    best_mae = np.inf
    history = {
        'avg_loss': [],
        'avg_vloss': []
    }
    avg_mae = []
    avg_vmae = []

    best_model_path = os.path.join(foldername,f"best_model_resnet18_ft_{timestamp}.pt") 
    print('entrainement du model ')

    try:
        for epoch in range(EPOCHS):
            print(f'EPOCH {epoch_number + 1}:')
            model.train(True)
            tr_avg_loss, tr_avg_mae = train_one_epoch(model, training, optimizer, loss_fn, device, epoch_number, writer)
            vld_avg_vloss, vld_avg_vmae = validate_one_epoch(model, testing, loss_fn, device)

            print(f"Training loss: {tr_avg_loss:.3f}, training MAE: {tr_avg_mae:.3f}")
            print(f"Validation loss: {vld_avg_vloss:.3f}, validation MAE: {vld_avg_vmae:.3f}")

            avg_mae.append(tr_avg_mae)
            avg_vmae.append(vld_avg_vmae)
            history['avg_loss'].append(tr_avg_loss)
            history['avg_vloss'].append(vld_avg_vloss)

            if vld_avg_vloss < best_loss:
                best_loss = vld_avg_vloss
                best_mae = vld_avg_vmae
                torch.save(model.state_dict(), best_model_path)
                print(f"--> Meilleur modèle sauvegardé avec une perte de validation de {best_loss:.3f}")

            if (epoch + 1) % 5 == 0:
                save_checkpoint(model, optimizer, epoch + 1, best_loss)

            writer.flush()
            epoch_number += 1
    except KeyboardInterrupt:
        print("\nEntraînement interrompu. Sauvegarde du checkpoint en cours...")
        save_checkpoint(model, optimizer, epoch, best_loss)

    model.load_state_dict(torch.load(best_model_path))
    print("\nMeilleures performances du modèle:")
    print(f"MSE: {best_loss:.2f}")
    print(f"MAE: {best_mae:.2f}")
    print(f"RMSE: {np.sqrt(best_loss):.2f}")

    # Sauvegarde des meilleures métriques dans un fichier JSON
    best_metrics = {
        "MSE": best_loss,
        "MAE": best_mae,
        "RMSE": np.sqrt(best_loss),
    }
    metrics_filename = os.path.join(foldername,f"best_metrics_{timestamp}.json")
    with open(metrics_filename, 'w') as f:
        json.dump(best_metrics, f, indent=4)
    print(f"Meilleures métriques sauvegardées dans {metrics_filename}")

    plt.plot(history['avg_loss'], label='Training Loss')
    plt.plot(history['avg_vloss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    plt.savefig(os.path.join(foldername,'loss_curve.jpg'))

if __name__ == "__main__":
    main()
