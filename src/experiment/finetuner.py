from submodules.Long_CLIP.model import longclip

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from colorama import Fore, Style
from torch.amp import GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from adabelief_pytorch import AdaBelief

class MathDataset(Dataset):
    def __init__(self, metadata:list[list[str]], corrupted_ids:set[str], transform=None):
        self.metadata = [sample for sample in metadata if sample[0] not in corrupted_ids] # get only non-corrupted samples
        self.corrupted_ids = corrupted_ids
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_path = self.metadata[idx][2]
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)

        caption = self.metadata[idx][1]
        text = longclip.tokenize(caption, truncate=True) # Tokenize the caption

        return image, text.squeeze(0) # Remove the extra dimension

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(finetune.ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = torch.nn.functional.cross_entropy()

    def forward(self, sim_i2t, sim_t2i, rank):
        # Calculate loss as the mean of the two cross-entropy losses
        bs = sim_i2t.size(0)
        targets = torch.linspace(rank * bs,rank * bs + bs - 1, bs, dtype=torch.long).to(sim_i2t.device)

        loss_img = self.criterion(sim_i2t, targets, label_smoothing=0.1)
        loss_txt = self.criterion(sim_t2i, targets, label_smoothing=0.1)

        return (loss_img + loss_txt) / 2


# TO DO: Probably use text.encode and image.encode instead of standard longclip() forward call. Previously i modified the forward call, but this time i will need the forward for distributed. Or just setup distributed now.
class finetune:
    """
    Finetune longclip checkpoints
    """
        
    def __init__(self, distributed:bool, splits_path:str, corrupted_path:str, checkpoint_input_path:str, output_path:str, epochs:int=6, batch_size:int=30, save_min_loss:bool=False, **kwargs):
        """
        Args:
            distributed: If True, fine-tuning will be performed using multiple GPUs
            splits_path: Path to the folder containing the split up metadata
            corrupted_path: Path to the .txt file that contains the corrupted image ids
            checkpoint_input_path: Path of checkpoints to be finetuned
            output_path: Desired folder path to output finetuned checkpoints and figures/logs
            epochs: Number of epochs to finetune for
            save_min_loss: Only save checkpoints when the validation loss is lower than all previous checkpoints
        """

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = 1e-7
        self.scaler = GradScaler()
        self.distributed = distributed
        self.checkpoint_input_path = checkpoint_input_path
        self.output_path = output_path
        self.save_min_loss = save_min_loss

        # Save training plots with matplotlib
        self.plots_folder = kwargs.get('plots_folder', 'ft-plots')
        os.makedirs(self.plots_folder, exist_ok=True)
        # Save model .pt files
        self.ft_checkpoints_folder = kwargs.get('ft_checkpoints_folder', 'ft-checkpoints')
        os.makedirs(self.ft_checkpoints_folder, exist_ok=True)
        # Save verbose text / training logs
        self.text_logs_folder = kwargs.get('text_logs_folder', 'ft-logs')
        os.makedirs(self.text_logs_folder, exist_ok=True)

        # Load model and preprocessing - path to Long-CLIP model file
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = longclip.load(checkpoint_input_path, device=self.device)

        # Gather corrupted ids
        corrupted_ids = []
        with open(corrupted_path, 'r', encoding='utf-8') as f:
            corrupted_ids = f.readlines()
        corrupted_ids = set(corrupted_ids)

        # Get datasets 
        train_split = np.load(os.path.join(splits_path, "train_split.npy"))
        val_split = np.load(os.path.join(splits_path, "val_split.npy"))
        self.train_dataset = MathDataset(train_split, corrupted_ids, self.caption_field_name, transform=self.preprocess)
        self.val_dataset = MathDataset(val_split, corrupted_ids, self.caption_field_name, transform=self.preprocess)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True)
        self.total_steps = (len(self.train_dataloader) // batch_size) * epochs

        # Set up distributed environment
        if self.distributed:
            self.world_size = torch.cuda.device_count()
            assert self.world_size >= 2, f"Distributed requires at least 2 GPUs to run, but got {self.world_size}"
            self.total_steps = self.total_steps // self.world_size # simulating batch_size of batch_size*world_size, so you need less steps
            mp.spawn(self.dist_train, args=(self.world_size, self.train_dataset, self.world_size), nprocs=self.world_size, join=True)
        else:
            self.world_size = 1

    def dist_train(self, rank:int, world_size:int):
        # Set up GPU communication
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12354"
        dist.init_process_group("nccl", rank=rank, world_size=self.world_size)
        
        # Move model to current GPU
        self.model.to(rank)
        self.model = DDP(self.model, device_ids=[rank]).module

        # Ensures each GPU sees a different batch shard
        train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=rank)
        self.train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.batch_size, shuffle=True)

        self.trainloop(rank)


    def trainloop(self, rank:int=0):
        """
        Complete training loop. Outputs finetuned checkpoints in designated directory, as well as additional logs
        """
        
        self.optimizer = AdaBelief(self.model.parameters(), lr=self.learning_rate, eps=1e-16, betas=(0.9, 0.995), weight_decay=1e-3, weight_decouple=False, rectify=True, print_change_log = False)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=self.learning_rate, total_steps=self.total_steps, pct_start=0.1, anneal_strategy='linear')

        unfreeze_all = True
        training_losses = []
        validation_losses = []
        
        min_val_loss = 0 # save only min val loss checkpoints if save_min_loss arg == True
        model = self.model.float()
        print(f"Precision: {model.dtype}")
        print(f'Total batches: {len(self.train_dataloader)} @ Batch Size: {self.batch_size}')
        print("== START == \n")
        for epoch in range(self.epochs):
            if self.distributed:
                self.train_dataloader.sampler.set_epoch(epoch)

            gradient_norms = {}
            self._unfreeze_layers(model, epoch, total_layers=24, unfreeze_all=unfreeze_all)
            model.train()
            total_train_loss = 0.0
            progress_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc=f'Epoch {epoch + 1}/{self.epochs}', leave=True, disable=(self.rank != 0))

            for batch_idx, (images, texts) in progress_bar:
                images, texts = images.to(rank), texts.to(rank)
                
                with torch.autocast(device_type=self.device):
                    sim_i2t, sim_t2i = model(images, texts)
                    total_loss = ContrastiveLoss(sim_i2t, sim_t2i, rank)

                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                
                # Once per batch
                if self.rank==0:
                    # Store gradient norms for plot
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            grad_norm = parameter.grad.norm().item()
                            gradient_norms.setdefault(name, []).append(grad_norm)
                
                    # OPTIONAL DEBUG
                    # use this line to debug (and be spammed with red messages about exploding and vanishing gradients):
                    # monitor_gradient_norms(gradient_norms)
                    
                    if self.distributed:
                        # Sum the losses across all processes
                        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                        total_loss /= self.world_size
                    total_train_loss += total_loss.item()

                    progress_bar.set_postfix({'loss': f'{total_train_loss / (batch_idx + 1):.4f}'})
                    with open(f"{self.text_logs_folder}/log_details_train.txt", "a", encoding='utf-8') as f:
                        f.write(f"Epoch {epoch + 1}/{self.epochs}, Batch: {batch_idx + 1}/{len(self.train_dataloader)}, Loss: {total_loss.item():.4f}\n")

            # Once per epoch
            if self.rank==0:
                avg_train_loss = total_train_loss / (len(self.train_dataloader) / self.world_size)
                training_losses.append(avg_train_loss)
                self._plot_gradient_norms(gradient_norms, epoch)

                # Validation
                model.eval()    
                print("Running Validation...")
                min_flag = False
                val_total_loss = 0
                with torch.no_grad():
                    for images, texts in self.val_dataloader:
                        images, texts = images.to(self.device), texts.to(self.device)
                        images = model.encode_image(images)
                        texts = model.encode_text(texts)
                        val_total_loss += ContrastiveLoss(images, texts).item()

                avg_val_loss = val_total_loss / len(self.val_dataloader)
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
                    plt.savefig(os.path.join(self.output_path, self.plots_folder, f"loss_plot_epoch_{epoch + 1}.png"))
                    plt.close()        
                
                
                print(Fore.YELLOW + "======================== STATS =============================")
                print(Fore.YELLOW + f"Epoch {epoch + 1}/{self.epochs} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
                print(Fore.YELLOW + "============================================================" + Style.RESET_ALL)
                
                with open(os.path.join(self.output_path, self.text_logs_folder, "log_training.txt", "a", encoding='utf-8')) as f:
                    f.write("======================== STATS =============================\n")
                    f.write(f"Epoch {epoch + 1}/{self.epochs} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}\n")
                    f.write("============================================================\n")

                # Save model every epoch unless only saving min
                if (self.save_min_loss == False or (self.save_min_loss == True and min_flag == True)):
                    output_path = os.path.join(self.output_path, self.ft_checkpoints_folder, f"{epoch+1}.pt")
                    torch.save(model.state_dict(), output_path)      
                    print(Fore.GREEN + f"Checkpoint saved at: {output_path}" + Style.RESET_ALL)

        # Destroy process once all epochs have finished
        if self.distributed:
            dist.destroy_process_group()

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
            plt.savefig(os.path.join(self.output_path, self.plots_folder, f"gradient_norms_epoch_{epoch}_log.png"))
        else:
            plt.savefig(os.path.join(self.output_path, self.plots_folder, f"gradient_norms_epoch_{epoch}.png"))
        
        plt.close()