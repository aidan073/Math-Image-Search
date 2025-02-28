from submodules.Long_CLIP.model import longclip
import json
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from colorama import Fore, Style
from torch.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from adabelief_pytorch import AdaBelief


class MathDataset(Dataset):
    def __init__(self, metadata_path:str, caption_field_name:str, transform=None):
        self.transform = transform
        self.caption_field_name = caption_field_name
        with open(metadata_path,'r',encoding='utf-8') as file:
            self.metadata=list(json.loads(file.read()).values())
        self.image_paths = []
        for sample in self.metadata:
            self.image_paths.append(sample["file_path"])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)

        caption = self.metadata[idx][self.caption_field_name]
        if(isinstance(caption, list)): # If caption is list of sentences (in case of artpedia), make into single caption
            caption = " ".join(caption)
        text = longclip.tokenize(caption, truncate=True) # Tokenize the caption

        return image, text.squeeze(0) # Remove the extra dimension
    
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, logits_per_image, logits_per_text):
        # Normalize the features to avoid overflow or underflow
        logits_per_image = torch.nn.functional.normalize(logits_per_image, p=2, dim=1)
        logits_per_text = torch.nn.functional.normalize(logits_per_text, p=2, dim=1)

        # Calculate logits
        logits = torch.matmul(logits_per_image, logits_per_text.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        # Calculate loss as the mean of the two cross-entropy losses
        loss_img = self.criterion(logits, labels)
        loss_txt = self.criterion(logits.t(), labels)

        return (loss_img + loss_txt) / 2

class finetune:
    """
    Finetune longclip checkpoints
    """
    def __init__(self, val_path:str, train_path:str, caption_field_name:str, checkpoint_output_path:str, epochs:int=6, save_min_loss:bool=True, checkpoint_input_path:str='submodules/longclip/checkpoints/longclip-L.pt', **kwargs):
        """
        Args:
            val_path: Path to validation set
            train_path: Path to train set
            caption_field_name: Name of caption field within metadata file
            checkpoint_output_path: Desired path to output finetuned checkpoint files
            checkpoint_input_path: Path of checkpoints to be finetuned
            save_min_loss: Only save checkpoints when the validation loss is lower than all previous checkpoints
        """
        self.caption_field_name = caption_field_name
        self.checkpoint_input_path = checkpoint_input_path
        self.checkpoint_output_path = checkpoint_output_path
        self.epochs = epochs
        self.save_min_loss = save_min_loss

        # Save training plots with matplotlib to:
        self.plots_folder = kwargs.get('plots_folder', 'ft-plots')
        os.makedirs(self.plots_folder, exist_ok=True)
        # Save model .pt files to: 
        self.ft_checkpoints_folder = kwargs.get('ft_checkpoints_folder', 'ft-checkpoints')
        os.makedirs(self.ft_checkpoints_folder, exist_ok=True)
        # Save verbose text / training logs to:
        self.text_logs_folder = kwargs.get('text_logs_folder', 'ft-logs')
        os.makedirs(self.text_logs_folder, exist_ok=True)

        # Load model and preprocessing - path to Long-CLIP model file:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = longclip.load(checkpoint_input_path, device=self.device)

        self.train_dataset = ArtCaptionDataset(train_path, self.caption_field_name, transform=self.preprocess)
        self.val_dataset = ArtCaptionDataset(val_path, self.caption_field_name, transform=self.preprocess)

    def trainloop(self):
        """
        Complete training loop. Outputs finetuned checkpoints in designated directory, as well as additional logs
        """
        loss_func = ContrastiveLoss()
        training_losses = []
        validation_losses = []

        """CONFIG"""
        EPOCHS = self.epochs
        learning_rate = 1e-7
        batch_size = 30
        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True)
        total_steps = len(train_dataloader) * EPOCHS
        unfreeze_all = True
        scaler = GradScaler()
        optimizer = AdaBelief(self.model.parameters(), lr=learning_rate, eps=1e-16, betas=(0.9, 0.995), weight_decay=1e-3, weight_decouple=False, rectify=True, print_change_log = False)
        scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps, pct_start=0.1, anneal_strategy='linear')
        
        min_val_loss = 0 # save only min val loss checkpoints if save_min_loss arg == True
        model = self.model.float()
        print(f"Precision: {model.dtype}")
        print(f'Total batches: {len(train_dataloader)} @ Batch Size: {batch_size}')
        print("== START == \n")
        for epoch in range(EPOCHS):
            gradient_norms = {}
            self._unfreeze_layers(model, epoch, total_layers=24, unfreeze_all=unfreeze_all)
            model.train()
            total_train_loss = 0.0
            progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{EPOCHS}', leave=True)
            for batch_idx, (images, texts) in progress_bar:
                images, texts = images.to(self.device), texts.to(self.device)
                
                optimizer.zero_grad()
                with torch.autocast(device_type=self.device):
                    logits_per_image, logits_per_text = model(images, texts)
                    total_loss = loss_func(logits_per_image, logits_per_text)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                        
                # Store gradient norms for plot
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        grad_norm = parameter.grad.norm().item()
                        gradient_norms.setdefault(name, []).append(grad_norm)
                
                # OPTIONAL DEBUG
                # use this line to debug (and be spammed with red messages about exploding and vanishing gradients):
                # monitor_gradient_norms(gradient_norms)
                
                total_train_loss += total_loss.item()

                progress_bar.set_postfix({'loss': f'{total_train_loss / (batch_idx + 1):.4f}'})
                with open(f"{self.text_logs_folder}/log_details_train.txt", "a", encoding='utf-8') as f:
                    f.write(f"Epoch {epoch + 1}/{EPOCHS}, Batch: {batch_idx + 1}/{len(train_dataloader)}, Loss: {total_loss.item():.4f}\n")
    
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            training_losses.append(avg_train_loss)
            self._plot_gradient_norms(gradient_norms, epoch)

            # Validation
            model.eval()    
            print("Running Validation...")
            min_flag = False
            val_total_loss = 0
            with torch.no_grad():
                for images, texts in val_dataloader:
                    images, texts = images.to(self.device), texts.to(self.device)
                    images = model.encode_image(images)
                    texts = model.encode_text(texts)
                    val_total_loss += loss_func(images, texts).item()

            avg_val_loss = val_total_loss / len(val_dataloader)
            validation_losses.append(avg_val_loss)

            if epoch==0:
                min_val_loss = avg_val_loss
            else:
                if avg_val_loss <= min_val_loss:
                    min_val_loss = avg_val_loss
                    min_flag = True
            
            if epoch >= 1:
                # Plot losses
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, epoch + 2), training_losses, label='Training Loss')
                plt.plot(range(1, epoch + 2), validation_losses, label='Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss Over Epochs')
                plt.legend()
                plt.savefig(f"{self.plots_folder}/loss_plot_epoch_{epoch + 1}.png")
                plt.close()        
            
            
            print(Fore.YELLOW + "======================== STATS =============================")
            print(Fore.YELLOW + f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
            print(Fore.YELLOW + "============================================================" + Style.RESET_ALL)
            
            with open(f"{self.text_logs_folder}/log_training.txt", "a", encoding='utf-8') as f:
                f.write("======================== STATS =============================\n")
                f.write(f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}\n")
                f.write("============================================================\n")

            # Save model every <> epochs + save final model
            if (self.save_min_loss == False or self.save_min_loss == True and min_flag == True) and (epoch + 1) % 1 == 0 or epoch == EPOCHS - 1:
                output_path = f"{self.ft_checkpoints_folder}/{self.checkpoint_output_path}{epoch+1}.pt"
                torch.save(model.state_dict(), output_path)      
                print(Fore.GREEN + f"Checkpoint saved at: {output_path}" + Style.RESET_ALL)

    def _adjust_unfreeze_rate(self, epoch, adjust_after=12, increase_rate=2):
        """
        Adjusts the rate of unfreezing after a certain number of epochs.

        Args:
            epoch: Current epoch number.
            adjust_after: Epoch after which to increase unfreezing rate.
            increase_rate: How many layers to unfreeze per epoch after adjust_after.

        Returns: 
            Number of layers to unfreeze per epoch.
        """
        if epoch < adjust_after:
            return 1  # Initial slower unfreeze rate
        else:
            return increase_rate  # Increased rate after initial pass

    def _unfreeze_layers(self, model, epoch, total_layers=24, unfreeze_all=False):
        if unfreeze_all:
            for param in model.parameters():
                param.requires_grad = True
        else:
            unfreeze_every_n_epochs = self._adjust_unfreeze_rate(epoch)
            layers_to_unfreeze = (epoch // unfreeze_every_n_epochs) % total_layers
            layers_to_unfreeze = min(layers_to_unfreeze, total_layers)
            for i, (name, param) in enumerate(model.named_parameters()):
                if i >= total_layers - layers_to_unfreeze:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def _monitor_gradient_norms(self, gradient_norms, threshold=1e-5):
        alert_messages = []
        for name, norms in gradient_norms.items():
            mean_norm = sum(norms) / len(norms)
            if mean_norm < threshold:  # Vanishing gradient
                alert_messages.append(Fore.RED + f"Vanishing gradient detected in {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
            elif mean_norm > 1000:  # Exploding gradient
                alert_messages.append(Fore.RED + f"Exploding gradient detected in {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
        if alert_messages:
            for message in alert_messages:
                print(message)

    def _plot_gradient_norms(self, gradient_norms, epoch, use_log_scale=True):
        plt.figure(figsize=(20, 10))
        
        # Choose a colormap
        cmap = plt.get_cmap('Spectral')
        
        # Sort the layers by the maximum gradient norm value, descending
        sorted_layers = sorted(gradient_norms.items(), key=lambda item: max(item[1]), reverse=True)
        
        # Generate distinct colors from the colormap
        colors = cmap(range(len(sorted_layers)))
        
        for (layer_name, norms), color in zip(sorted_layers, colors):
            plt.plot(norms, label=layer_name, color=color)

        plt.xlabel('Batch')
        plt.ylabel('Gradient Norm')
        #plt.title(f'Gradient Norms for Epoch {epoch}{" - Log Scale" if use_log_scale else ""}')
        
        # Adjust legend: position at top right with smaller font size
        plt.legend(loc='upper right', fontsize='small')
        
        # If log scale is requested, change the y-axis to logarithmic
        if use_log_scale:
            plt.yscale('log')
            plt.title(f'Gradient Norms for Epoch {epoch}{" - Log Scale" if use_log_scale else ""}')
            plt.savefig(f"{self.plots_folder}/gradient_norms_epoch_{epoch}_log.png")
        else:
            plt.savefig(f"{self.plots_folder}/gradient_norms_epoch_{epoch}.png")
        
        plt.close()

if __name__ == '__main__':

    finetuner1 = finetune(val_split_path, train_split_path, "LlamaCaptioner", "Llama-ft",
                        plots_folder="llama-ft-plots", ft_checkpoints_folder="llama-ft-checkpoints",
                        text_logs_folder="llama-ft-logs")
    finetuner1.trainloop()