import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

from model.model_bin import NetworkTrafficMLP, NetworkTrafficDataset, save_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model, hyperparams, device, class_weights=None):
        self.model = model.to(device)
        self.hyperparams = hyperparams
        self.device = device
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams['lr'],
            weight_decay=hyperparams['weight_decay']
        )
        
        if class_weights is not None:
            class_weights = class_weights.to(device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            logger.info(f"Usando BCEWithLogitsLoss con pos_weight: {class_weights.item():.3f}")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            logger.info("Usando BCEWithLogitsLoss senza class weights")
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
    def add_noise_to_input(self, x, noise_factor):
        if self.model.training and noise_factor > 0:
            noise = torch.randn_like(x) * noise_factor
            return x + noise
        return x
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        total_actual_attacks = 0
        total_predicted_attacks = 0
        total_correct_attacks = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, (batch_x, batch_y) in enumerate(progress_bar):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            if batch_idx == 0:
                actual_attacks_in_batch = (batch_y == 1).sum().item()
                logger.info(f"Primo batch epoca {epoch}: {actual_attacks_in_batch}/{len(batch_y)} attacchi")
                if actual_attacks_in_batch == 0:
                    logger.warning("⚠️  Il primo batch non contiene attacchi!")
            
            if 'noise_factor' in self.hyperparams:
                batch_x = self.add_noise_to_input(batch_x, self.hyperparams['noise_factor'])
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x).squeeze()
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            batch_actual_attacks = (batch_y == 1).sum().item()
            batch_predicted_attacks = (predicted == 1).sum().item()
            batch_correct_attacks = ((predicted == 1) & (batch_y == 1)).sum().item()
            
            total_actual_attacks += batch_actual_attacks
            total_predicted_attacks += batch_predicted_attacks
            total_correct_attacks += batch_correct_attacks
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'Pred_Att': f'{batch_predicted_attacks}/{batch_actual_attacks}'
            })
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                           f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%, '
                           f'Pred Attacks: {batch_predicted_attacks}/{batch_actual_attacks}')
                
                if total_predicted_attacks == 0 and total_actual_attacks > 0:
                    logger.warning(f"⚠️  Dopo {batch_idx} batch, il modello non ha ancora predetto NESSUN attacco!")
                    logger.warning(f"Range probabilità: {probabilities.min():.4f} - {probabilities.max():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        logger.info(f"Epoca {epoch} completata:")
        logger.info(f"  Attacchi visti: {total_actual_attacks:,}")
        logger.info(f"  Attacchi predetti: {total_predicted_attacks:,}")
        logger.info(f"  Attacchi corretti: {total_correct_attacks:,}")
        
        if total_actual_attacks > 0:
            attack_recall = total_correct_attacks / total_actual_attacks
            logger.info(f"  Attack Recall: {attack_recall:.4f}")
        
        if total_predicted_attacks > 0:
            attack_precision = total_correct_attacks / total_predicted_attacks
            logger.info(f"  Attack Precision: {attack_precision:.4f}")
        
        return avg_loss, accuracy

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_x).squeeze()
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)
    
    def train(self, train_loader, val_loader, epochs=100, patience=15):
        logger.info(f"Inizio addestramento per {epochs} epochs...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Parametri del modello: {sum(p.numel() for p in self.model.parameters())}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            train_loss, train_acc = self.train_epoch(train_loader, epoch+1)
            logger.info(f"Terminata l'epoca: {epoch+1}")
            
            val_loss, val_acc, val_pred, val_true, val_prob = self.validate(val_loader)
            self.scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            precision, recall, f1, _ = precision_recall_fscore_support(val_true, val_pred, average='binary', zero_division=0)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"Val Precision: {precision:.4f}, Val Recall: {recall:.4f}, Val F1: {f1:.4f}")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                logger.info("✓ Nuovo miglior modello salvato!")
            else:
                patience_counter += 1
                logger.info(f"Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logger.info(f"Early stopping dopo {epoch+1} epochs (patience: {patience})")
                    break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Caricato il miglior modello per la valutazione finale.")
        
        return self.model
    
    def plot_training_history(self, save_path=None):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.train_losses) + 1)
        
        ax1.plot(epochs, self.train_losses, label='Training Loss', color='blue')
        ax1.plot(epochs, self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, self.val_accuracies, label='Validation Accuracy', color='green')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        ax3.plot(epochs, self.learning_rates, label='Learning Rate', color='orange')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
        
        loss_diff = [val - train for val, train in zip(self.val_losses, self.train_losses)]
        ax4.plot(epochs, loss_diff, label='Val Loss - Train Loss', color='purple')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Overfitting Check')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Difference')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot salvato in: {save_path}")
        plt.show()


def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    logger.info("Valutazione sul test set...")
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze()
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    probabilities = np.array(all_probabilities)
    
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='binary', zero_division=0)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"RISULTATI TEST SET")
    logger.info(f"{'='*50}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    
    print(classification_report(targets, predictions, target_names=['Benign', 'Attack']))
    
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Attack'], 
                yticklabels=['Benign', 'Attack'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j+0.5, i+0.7, f'({cm_percent[i, j]:.1%})', 
                    ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.show()
    
    return accuracy, precision, recall, f1, predictions, targets, probabilities


def main_pipeline_bin(model_size="small"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        dataset_manager = NetworkTrafficDataset(model_size=model_size)
        input_dim = dataset_manager.load_data(
            train_path="resources/datasets/train.csv",
            val_path="resources/datasets/val.csv",
            test_path="resources/datasets/test.csv"
        )
        
        train_loader, val_loader, test_loader = dataset_manager.create_dataloaders()
        
        model = NetworkTrafficMLP(
            input_dim, 
            dataset_manager.hyperparams, 
            class_weights=dataset_manager.class_weights
        )
        logger.info(f"Modello creato con {sum(p.numel() for p in model.parameters())} parametri")
        
        trainer = ModelTrainer(
            model, 
            dataset_manager.hyperparams, 
            device, 
            class_weights=dataset_manager.class_weights
        )
        
        trained_model = trainer.train(train_loader, val_loader, epochs=100, patience=15)
        os.makedirs("plots", exist_ok=True)
        trainer.plot_training_history('plots/training_history.png')
        
        accuracy, precision, recall, f1, predictions, targets, probabilities = evaluate_model(
            trained_model, test_loader, device
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/mlp_classifier_{timestamp}.pth"
        os.makedirs("models", exist_ok=True)
        save_model(
            trained_model, 
            dataset_manager.hyperparams, 
            dataset_manager.feature_columns, 
            model_path,
            class_weights=dataset_manager.class_weights
        )
        
        logger.info(f"\n{'='*50}")
        logger.info(f"ADDESTRAMENTO COMPLETATO!")
        logger.info(f"{'='*50}")
        logger.info(f"Miglior accuratezza: {accuracy:.4f}")
        logger.info(f"Modello salvato in: {model_path}")
        
    except Exception as e:
        logger.error(f"Errore durante l'addestramento: {str(e)}")
        raise


if __name__ == "__main__":
    main_pipeline_bin()