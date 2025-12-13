import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from pydub import AudioSegment
import os
import random
import time
import logging
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

import transformers
from torchvision.models import resnet18, ResNet18_Weights
from transformers import ViTModel, Wav2Vec2Model, WhisperModel, WhisperProcessor # Ensure WhisperProcessor is imported
from torch.utils.data import random_split
from torch.cuda.amp import autocast, GradScaler
 
PREPROCESSED_DATA_DIR = "/content/drive/MyDrive/preprocess"
ANNOTATIONS_CSV = "/content/drive/MyDrive/train_sent_emo.csv"

LOG_PARENT_DIR = "/content/drive/MyDrive/Temporal-Experiment-Logs"

AUDIO_SAMPLE_RATE = 16000
FACE_CROP_SIZE = (224, 224)
FIXED_FACE_BBOX = (150, 80, 200, 200)

# --- Dummy Model Placeholders (Revised for clarity and consistency) ---
class DummyVisionExtractor(nn.Module):
    def forward(self, cropped_frames_batch):
        N, T, C, H, W = cropped_frames_batch.shape
        # Returns a sequence of features (batch_size * num_frames, feature_dim)
        return torch.randn(N * T, 512, device=cropped_frames_batch.device)

class DummyAudioExtractor(nn.Module):
    def forward(self, audio_input_batch):
        # This dummy extractor needs to simulate the output of actual backbones.
        # For Wav2Vec2, input is raw audio (batch, samples), output is (batch, seq_len, hidden_dim)
        # For Whisper, input is mel_features (batch, mel_bands, seq_len_mel), output is (batch, seq_len, hidden_dim)

        if audio_input_batch.dim() == 2: # Raw audio (batch_size, num_samples) - Simulating Wav2Vec2
            # Simulate a downsampled sequence output
            seq_len = audio_input_batch.shape[1] // 160 # Wav2Vec2 downsampling factor approx 160
            if seq_len == 0: seq_len = 1 # Avoid zero length
            hidden_dim = 768
            # Return an object with last_hidden_state to match actual model behavior
            from transformers.modeling_outputs import BaseModelOutput # General output class
            return BaseModelOutput(last_hidden_state=torch.randn(audio_input_batch.shape[0], seq_len, hidden_dim, device=audio_input_batch.device))
        elif audio_input_batch.dim() == 3: # Mel features (batch_size, num_mel_bands, seq_len_mel) - Simulating Whisper
            # Simulate a downsampled sequence output from Whisper encoder
            seq_len = audio_input_batch.shape[2] // 2 # Whisper tiny downsampling factor approx 2
            if seq_len == 0: seq_len = 1
            hidden_dim = 384 # For whisper-tiny
            return BaseModelOutput(last_hidden_state=torch.randn(audio_input_batch.shape[0], seq_len, hidden_dim, device=audio_input_batch.device))
        else:
            raise ValueError(f"DummyAudioExtractor: Unsupported audio input dim: {audio_input_batch.dim()}")


class DummyMultimodalFusionModel(nn.Module): # This model isn't actually used if get_actual_backbones is called
    def __init__(self, vision_dim, audio_dim, aggregated_dim, num_emotions, num_sentiments):
        super().__init__()
        self.vision_extractor = DummyVisionExtractor()
        self.audio_extractor = DummyAudioExtractor()
        self.fusion_mlp = nn.Linear(vision_dim + audio_dim, aggregated_dim)
        self.emotion_head = nn.Linear(aggregated_dim, num_emotions)
        self.sentiment_head = nn.Linear(aggregated_dim, num_sentiments)

    def forward(self, cropped_frames_batch, audio_input_features_batch):
        # This forward pass should mimic MultimodalEmotionModelStage1 for consistency
        batch_size, num_frames, C, H, W = cropped_frames_batch.shape
        vision_feats_flat = self.vision_extractor(cropped_frames_batch.view(-1, C, H, W))
        vision_feats = vision_feats_flat.view(batch_size, num_frames, -1)
        agg_vision_feat = torch.mean(vision_feats, dim=1)

        audio_output = self.audio_extractor(audio_input_features_batch)
        if hasattr(audio_output, 'last_hidden_state'):
            audio_feats = torch.mean(audio_output.last_hidden_state, dim=1)
        elif isinstance(audio_output, torch.Tensor):
            if audio_output.dim() == 3: # (batch, seq, dim)
                audio_feats = torch.mean(audio_output, dim=1)
            elif audio_output.dim() == 2: # (batch, dim)
                audio_feats = audio_output
            else:
                raise TypeError(f"DummyAudioExtractor output tensor has unexpected dimensions: {audio_output.dim()}")
        else:
            raise TypeError(f"Unexpected audio_backbone output type: {type(audio_output)}")


        fused_feat_input = torch.cat((agg_vision_feat, audio_feats), dim=1)
        utterance_embedding = self.fusion_mlp(fused_feat_input)
        emotion_logits = self.emotion_head(utterance_embedding)
        sentiment_scores = self.sentiment_head(utterance_embedding)

        return utterance_embedding, emotion_logits, sentiment_scores

 
class UtteranceEmotionDataset(Dataset):
    def __init__(self, annotations_df, preprocessed_data_dir, emotion_map, sentiment_map, dataset_size_limit=None, audio_processor=None, audio_model_choice=""): # Pass audio_model_choice to __init__
        
        if dataset_size_limit:
            self.annotations = annotations_df.head(dataset_size_limit).copy()
        else:
            self.annotations = annotations_df.copy()

        self.preprocessed_data_dir = preprocessed_data_dir
        self.emotion_map = emotion_map
        self.sentiment_map = sentiment_map
        self.audio_processor = audio_processor 
        self.audio_model_choice = audio_model_choice # Store choice for __getitem__

        self.data_entries = [] 
        
        for idx, row in self.annotations.iterrows():
            dialogue_id = row['Dialogue_ID'] # Changed to Dialogue_ID
            utterance_id = row['Utterance_ID'] # Changed to Utterance_ID
            
            base_filename = f"dia{dialogue_id}_utt{utterance_id}" # Adjusted filename pattern
            masked_frames_path = os.path.join(self.preprocessed_data_dir, 'masked_faces', f"{base_filename}_frames.npy") # Removed sub-dir
            audio_path = os.path.join(self.preprocessed_data_dir, 'audio_chunks', f"{base_filename}_audio.npy") # Removed sub-dir
            
            if os.path.exists(masked_frames_path) and os.path.exists(audio_path):
                self.data_entries.append({
                    'dialogue_id': dialogue_id,
                    'utterance_id': utterance_id,
                    'speaker_name': row['Speaker'], 
                    'emotion_label': row['Emotion'],
                    'sentiment_label': row['Sentiment'],
                    'masked_frames_path': masked_frames_path,
                    'audio_path': audio_path
                })
            else:
                logging.warning(f"Preprocessed files not found for D:{dialogue_id}, U:{utterance_id}. Skipping.")
                print(f"Preprocessed files not found for D:{dialogue_id}, U:{utterance_id}. Skipping.")
        
        if not self.data_entries:
            raise ValueError("No valid preprocessed data entries found. Check paths and preprocessing step.")

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        item_data = self.data_entries[idx] 

        dialogue_id = item_data['dialogue_id']
        utterance_id = item_data['utterance_id']
        speaker_name = item_data['speaker_name']
        emotion_label = item_data['emotion_label']
        sentiment_label = item_data['sentiment_label']
        
        try:
            cropped_frames_np = np.load(item_data['masked_frames_path'])
            audio_data_np = np.load(item_data['audio_path'])
        except Exception as e:
            logging.error(f"Error loading NPY files for D:{dialogue_id}, U:{utterance_id}: {e}. Returning dummy data.")
            print(f"Error loading NPY files for D:{dialogue_id}, U:{utterance_id}: {e}. Returning dummy data.")
            dummy_frames = np.zeros((1, FACE_CROP_SIZE[0], FACE_CROP_SIZE[1], 3), dtype=np.uint8)
            dummy_raw_audio = np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32) # Dummy raw audio
            
            # Prepare dummy audio input based on audio_model_choice
            if self.audio_model_choice == "whisper_encoder" and self.audio_processor:
                # Whisper expects mel features of a specific size (e.g., 80 mel bands, 3000 seq length)
                dummy_audio_input_features = self.audio_processor(dummy_raw_audio, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt", padding="max_length", truncation=True).input_features.squeeze(0)
            elif self.audio_model_choice == "wav2vec2_base":
                # Wav2Vec2 expects raw audio samples, so return dummy_raw_audio
                dummy_audio_input_features = torch.from_numpy(dummy_raw_audio).float()
            else: # Fallback for other/dummy audio models
                dummy_audio_input_features = torch.from_numpy(dummy_raw_audio).float() # Default to raw audio tensor
            
            return {
                'cropped_frames': torch.from_numpy(dummy_frames).float().permute(0, 3, 1, 2) / 255.0,
                'audio_chunk': dummy_audio_input_features,
                'speaker_id': -1, 
                'emotion_label_id': torch.tensor(0, dtype=torch.long),
                'sentiment_label_id': torch.tensor(0, dtype=torch.long),
                'dialogue_id': dialogue_id,
                'utterance_id': utterance_id
            }
        
        cropped_frames_tensor = torch.from_numpy(cropped_frames_np).float().permute(0, 3, 1, 2) / 255.0 
        
        # --- Conditional Audio Input Preparation ---
        if self.audio_model_choice == "whisper_encoder" and self.audio_processor:
            # Whisper processor handles resampling, mel spectrogram conversion, padding/truncation
            audio_input_features = self.audio_processor(audio_data_np, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt", padding="max_length", truncation=True).input_features.squeeze(0)
        elif self.audio_model_choice == "wav2vec2_base":
            # Wav2Vec2 expects raw audio samples (as a tensor)
            audio_input_features = torch.from_numpy(audio_data_np).float()
        else:
            # Fallback for other audio models or dummy (e.g., raw audio as tensor)
            audio_input_features = torch.from_numpy(audio_data_np).float()
        
        # Map labels to numerical IDs
        emotion_id = self.emotion_map.get(emotion_label, -1) 
        sentiment_id = self.sentiment_map.get(sentiment_label, -1)
        
        if emotion_id == -1 or sentiment_id == -1:
             logging.warning(f"Invalid label for D:{dialogue_id}, U:{utterance_id}. Emotion: {emotion_label}, Sentiment: {sentiment_label}. Using dummy labels.")
             print(f"Invalid label for D:{dialogue_id}, U:{utterance_id}. Emotion: {emotion_label}, Sentiment: {sentiment_label}. Using dummy labels.")
             emotion_id = 0 
             sentiment_id = 0 

        speaker_person_id = hash(speaker_name) % 1000 

        return {
            'cropped_frames': cropped_frames_tensor,
            'audio_chunk': audio_input_features, 
            'speaker_id': speaker_person_id,
            'emotion_label_id': torch.tensor(emotion_id, dtype=torch.long),
            'sentiment_label_id': torch.tensor(sentiment_id, dtype=torch.long),
            'dialogue_id': dialogue_id,
            'utterance_id': utterance_id
        }

# --- Custom Collate Function  
def collate_fn_utterance_emotion(batch):
    batch = [item for item in batch if item['speaker_id'] != -1]
    if not batch:
        return None 

    max_frames = max(item['cropped_frames'].shape[0] for item in batch)
    
    # Determine padding strategy based on the shape of audio_chunk from __getitem__
    # If audio_chunk is raw audio (2D: samples), pad samples.
    # If audio_chunk is mel (3D: mel_bands, seq_len_mel), pad seq_len_mel.
    
    is_mel_input = (batch[0]['audio_chunk'].dim() == 2 and batch[0]['audio_chunk'].shape[0] > 1) # Heuristic: if dim 2 and first dim >1, likely mel bands
    # More robust check: check the audio_model_choice or add a flag to dataset item

    # For Wav2Vec2 (raw audio): (batch_size, max_audio_samples)
    # For Whisper (mel features): (batch_size, num_mel_bands, max_audio_seq_len)

    if is_mel_input: # If audio_chunk is (num_mel_bands, seq_len_mel)
        num_mel_bands = batch[0]['audio_chunk'].shape[0]
        max_audio_seq_len = max(item['audio_chunk'].shape[1] for item in batch)
    else: # If audio_chunk is raw samples (num_samples,)
        max_audio_len = max(item['audio_chunk'].shape[0] for item in batch) # This is max_samples

    
    padded_frames = []
    padded_audio_inputs = [] # Renamed for clarity

    for item in batch:
        num_frames = item['cropped_frames'].shape[0]
        padding_frames = torch.zeros(max_frames - num_frames, *item['cropped_frames'].shape[1:])
        padded_frames.append(torch.cat((item['cropped_frames'], padding_frames), dim=0))
        
        # Pad audio conditionally
        if is_mel_input:
            current_audio_seq_len = item['audio_chunk'].shape[1]
            padding_audio = torch.zeros(num_mel_bands, max_audio_seq_len - current_audio_seq_len)
            padded_audio_inputs.append(torch.cat((item['audio_chunk'], padding_audio), dim=1))
        else: # Raw audio samples
            current_audio_len = item['audio_chunk'].shape[0]
            padding_audio = torch.zeros(max_audio_len - current_audio_len)
            padded_audio_inputs.append(torch.cat((item['audio_chunk'], padding_audio), dim=0))
        
    return {
        'cropped_frames': torch.stack(padded_frames),
        'audio_chunk': torch.stack(padded_audio_inputs), # Renamed
        'speaker_ids': torch.tensor([item['speaker_id'] for item in batch], dtype=torch.long),
        'emotion_label_ids': torch.stack([item['emotion_label_id'] for item in batch]),
        'sentiment_label_ids': torch.stack([item['sentiment_label_id'] for item in batch]),
        'dialogue_ids': torch.tensor([item['dialogue_id'] for item in batch], dtype=torch.long),
        'utterance_ids': torch.tensor([item['utterance_id'] for item in batch], dtype=torch.long)
    }


# --- Stage 1 Multimodal Emotion Model Definition  ---
class MultimodalEmotionModelStage1(nn.Module):
    def __init__(self, vision_backbone, audio_backbone,
                 vision_feature_dim, audio_feature_dim,
                 fusion_output_dim, num_emotions, num_sentiments, audio_model_choice=""): # Pass audio_model_choice
        super().__init__()
        
        self.vision_backbone = vision_backbone
        self.audio_backbone = audio_backbone
        self.audio_model_choice = audio_model_choice # Store for forward pass
        
        for param in self.vision_backbone.parameters():
            param.requires_grad = False
        for param in self.audio_backbone.parameters():
            param.requires_grad = False
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(vision_feature_dim + audio_feature_dim, fusion_output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_output_dim, fusion_output_dim)
        )
        
        self.emotion_head = nn.Linear(fusion_output_dim, num_emotions)
        self.sentiment_head = nn.Linear(fusion_output_dim, num_sentiments) 

    def forward(self, video_frames_batch, audio_input_features_batch): # Renamed for clarity
        batch_size, num_frames, C, H, W = video_frames_batch.shape
        vision_feats_flat = self.vision_backbone(video_frames_batch.view(-1, C, H, W))
        if hasattr(vision_feats_flat, 'last_hidden_state'):
             vision_feats_flat = vision_feats_flat.last_hidden_state 
        vision_feats = vision_feats_flat.view(batch_size, num_frames, -1)
        
        agg_vision_feat = torch.mean(vision_feats, dim=1)

        # --- NEW AUDIO AGGREGATION LOGIC ---
        if self.audio_model_choice == "whisper_encoder":
            # Whisper encoder expects input_features as keyword arg
            audio_output = self.audio_backbone(input_features=audio_input_features_batch)
        elif self.audio_model_choice == "wav2vec2_base":
            # Wav2Vec2 expects raw audio samples, directly as positional arg
            audio_output = self.audio_backbone(audio_input_features_batch)
        else: # Dummy or other backbones might also take positional arg
            audio_output = self.audio_backbone(audio_input_features_batch)
        
        if hasattr(audio_output, 'last_hidden_state'):
            audio_feats = torch.mean(audio_output.last_hidden_state, dim=1) 
        elif isinstance(audio_output, torch.Tensor):
            if audio_output.dim() == 3: # (batch_size, seq_len, dim)
                audio_feats = torch.mean(audio_output, dim=1)
            elif audio_output.dim() == 2: # (batch_size, dim) - already aggregated
                audio_feats = audio_output
            else:
                raise TypeError(f"Audio backbone output tensor has unexpected dimensions: {audio_output.dim()}")
        else:
            raise TypeError(f"Unexpected audio_backbone output type: {type(audio_output)}")


        fused_feat_input = torch.cat((agg_vision_feat, audio_feats), dim=1)
        
        utterance_embedding = self.fusion_mlp(fused_feat_input) 
        
        emotion_logits = self.emotion_head(utterance_embedding)
        sentiment_scores = self.sentiment_head(utterance_embedding)
        
        return utterance_embedding, emotion_logits, sentiment_scores


from torch.cuda.amp import autocast, GradScaler
def validate_stage1_model(model, dataloader, emotion_loss_fn, sentiment_loss_fn, device):
    model.eval() # Set model to evaluation mode
    val_total_loss = 0
    val_correct_emotions = 0
    val_total_samples = 0
    
    with torch.no_grad(): # Disable gradient calculations for validation
        for i, batch in enumerate(dataloader):
            if batch is None:
                logging.warning(f"Skipping empty validation batch {i}.")
                continue
            
            cropped_frames = batch['cropped_frames'].to(device)
            audio_input_features = batch['audio_chunk'].to(device) 
            emotion_labels = batch['emotion_label_ids'].to(device)
            sentiment_labels = batch['sentiment_label_ids'].to(device)

            # --- NEW: autocast context manager for mixed precision during validation ---
            with autocast(): 
                _, emotion_logits, sentiment_scores = model(cropped_frames, audio_input_features) 
                
                emotion_loss = emotion_loss_fn(emotion_logits, emotion_labels)
                sentiment_loss = sentiment_loss_fn(sentiment_scores, sentiment_labels.float().unsqueeze(1)) 
                
                loss = emotion_loss + sentiment_loss 
            # --- END autocast context manager ---
            
            val_total_loss += loss.item()

            _, predicted_emotions = torch.max(emotion_logits, 1)
            val_correct_emotions += (predicted_emotions == emotion_labels).sum().item()
            val_total_samples += emotion_labels.size(0)
            
    val_avg_loss = val_total_loss / len(dataloader)
    val_emotion_accuracy = val_correct_emotions / val_total_samples
    
    return val_avg_loss, val_emotion_accuracy

# --- Training / Evaluation Helper Function Mixed Precision ---
def train_stage1_model(model, train_dataloader, val_dataloader, emotion_loss_fn, sentiment_loss_fn, optimizer, device, num_epochs, log_dir):
    # ADDED val_dataloader to parameters
    
    metrics_log_path = os.path.join(log_dir, 'logs', 'metrics.csv')
    with open(metrics_log_path, 'w') as f:
        f.write("epoch,time_s,avg_train_loss,train_emotion_accuracy,avg_val_loss,val_emotion_accuracy\n") # UPDATED HEADER
    
    logging.info(f"Starting Stage 1 fine-tuning for {len(train_dataloader.dataset)} utterances...")
    
    scaler = GradScaler()

    for epoch in range(num_epochs):
        start_time = time.time()
        
        # --- Training Loop ---
        model.train() # Set model to training mode
        train_total_loss = 0
        train_correct_emotions = 0
        train_total_samples = 0
        
        for i, batch in enumerate(train_dataloader):
            if batch is None:
                logging.warning(f"Skipping empty training batch {i}.")
                continue
                
            cropped_frames = batch['cropped_frames'].to(device)
            audio_input_features = batch['audio_chunk'].to(device) 
            emotion_labels = batch['emotion_label_ids'].to(device)
            sentiment_labels = batch['sentiment_label_ids'].to(device)

            optimizer.zero_grad()
            
            with autocast(): 
                utterance_embedding, emotion_logits, sentiment_scores = model(cropped_frames, audio_input_features) 
                
                emotion_loss = emotion_loss_fn(emotion_logits, emotion_labels)
                sentiment_loss = sentiment_loss_fn(sentiment_scores, sentiment_labels.float().unsqueeze(1)) 
                
                loss = emotion_loss + sentiment_loss 
            
            scaler.scale(loss).backward() 
            scaler.step(optimizer)        
            scaler.update()               
            
            train_total_loss += loss.item()

            _, predicted_emotions = torch.max(emotion_logits, 1)
            train_correct_emotions += (predicted_emotions == emotion_labels).sum().item()
            train_total_samples += emotion_labels.size(0)

            if i % 10 == 0: 
                logging.info(f"Epoch {epoch+1}, Train Batch {i+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
        
        train_avg_loss = train_total_loss / len(train_dataloader)
        train_emotion_accuracy = train_correct_emotions / train_total_samples
        
        # --- Validation Loop ---
        val_avg_loss, val_emotion_accuracy = validate_stage1_model(model, val_dataloader, emotion_loss_fn, sentiment_loss_fn, device) # NEW: Call validation
        
        end_time = time.time()
        
        # --- Log to file and console via standard logging - UPDATED MESSAGE ---
        logging.info(f"Epoch {epoch+1} finished. Time: {end_time - start_time:.2f}s")
        logging.info(f"  Train Loss: {train_avg_loss:.4f}, Train Acc: {train_emotion_accuracy:.4f}")
        logging.info(f"  Val Loss:   {val_avg_loss:.4f}, Val Acc:   {val_emotion_accuracy:.4f}\n")
        
        # --- Save epoch metrics to CSV - UPDATED CONTENT ---
        with open(metrics_log_path, 'a') as f:
            f.write(f"{epoch+1},{end_time - start_time:.2f},{train_avg_loss:.4f},{train_emotion_accuracy:.4f},{val_avg_loss:.4f},{val_emotion_accuracy:.4f}\n")
    
    logging.info("Stage 1 Fine-tuning Complete.")
    print("Stage 1 Fine-tuning Complete.")
# def train_stage1_model(model, train_dataloader, val_dataloader, emotion_loss_fn, sentiment_loss_fn, optimizer, device, num_epochs, log_dir):
#     model.train()
    
#     metrics_log_path = os.path.join(log_dir, 'logs', 'metrics.csv')
#     with open(metrics_log_path, 'w') as f:
#         f.write("epoch,time_s,avg_loss,emotion_accuracy\n") 

#     logging.info(f"Starting Stage 1 fine-tuning for {len(dataloader.dataset)} utterances...")
#     print(f"Starting Stage 1 fine-tuning for {len(dataloader.dataset)} utterances...")
             
#     # Initialize GradScaler for mixed precision
#     scaler = GradScaler() # <--- NEW: Initialize GradScaler

#     for epoch in range(num_epochs):
#         start_time = time.time()
#         total_loss = 0
#         correct_emotions = 0
#         total_samples = 0
        
#         for i, batch in enumerate(dataloader):
#             if batch is None:
#                 logging.warning(f"Skipping empty batch {i}.")
#                 print(f"Skipping empty batch {i}.")
#                 continue
                
#             cropped_frames = batch['cropped_frames'].to(device)
#             audio_input_features = batch['audio_chunk'].to(device) 
#             emotion_labels = batch['emotion_label_ids'].to(device)
#             sentiment_labels = batch['sentiment_label_ids'].to(device)

#             optimizer.zero_grad()
            
#             # --- NEW: autocast context manager for mixed precision ---
#             with autocast(): 
#                 utterance_embedding, emotion_logits, sentiment_scores = model(cropped_frames, audio_input_features) 
                
#                 emotion_loss = emotion_loss_fn(emotion_logits, emotion_labels)
#                 sentiment_loss = sentiment_loss_fn(sentiment_scores, sentiment_labels.float().unsqueeze(1)) 
                
#                 loss = emotion_loss + sentiment_loss 
#             # --- END autocast context manager ---
            
#             # --- NEW: Scaler operations for backward pass and optimizer step ---
#             scaler.scale(loss).backward() # Scale loss before backward()
#             scaler.step(optimizer)        # Optimizer step with scaled gradients
#             scaler.update()               # Update the scaler for the next iteration
#             # --- END Scaler operations ---
            
#             total_loss += loss.item() # Use original loss value for logging
#                                       # (scaler.scale() returns a scaled loss)

#             _, predicted_emotions = torch.max(emotion_logits, 1)
#             correct_emotions += (predicted_emotions == emotion_labels).sum().item()
#             total_samples += emotion_labels.size(0)

#             if i % 10 == 0: 
#                 logging.info(f"Epoch {epoch+1}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
#                 print("Epoch {epoch+1}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
#         avg_loss = total_loss / len(dataloader)
#         emotion_accuracy = correct_emotions / total_samples
#         end_time = time.time()
        
#         logging.info(f"Epoch {epoch+1} finished. Time: {end_time - start_time:.2f}s, Avg Loss: {avg_loss:.4f}, Emotion Accuracy: {emotion_accuracy:.4f}\n")
#         print(f"Epoch {epoch+1} finished. Time: {end_time - start_time:.2f}s, Avg Loss: {avg_loss:.4f}, Emotion Accuracy: {emotion_accuracy:.4f}\n")
#         with open(metrics_log_path, 'a') as f:
#             f.write(f"{epoch+1},{end_time - start_time:.2f},{avg_loss:.4f},{emotion_accuracy:.4f}\n")
    
#     logging.info("Stage 1 Fine-tuning Complete.")
#     print("Stage 1 Fine-tuning Complete.")

def get_actual_backbones(vision_model_name, audio_model_name, device):
    vision_dim = 0
    audio_dim = 0
    audio_processor = None 

    if vision_model_name == "resnet18":
        vision_backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        vision_backbone.fc = nn.Identity() 
        vision_dim = 512 
    elif vision_model_name == "vit_b_16":
        vision_backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        vision_backbone.pooler = nn.Identity() 
        vision_dim = 768 
    else: 
        vision_backbone = nn.Identity() 
        vision_dim = 512
    
    if audio_model_name == "wav2vec2_base":
        audio_backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        audio_dim = 768 
    elif audio_model_name == "whisper_encoder":
        whisper_model_name_or_path = "openai/whisper-tiny" 
        whisper_model = WhisperModel.from_pretrained(whisper_model_name_or_path) 
        audio_backbone = whisper_model.encoder
        audio_dim = whisper_model.config.d_model 
        audio_processor = WhisperProcessor.from_pretrained(whisper_model_name_or_path) 
    else: 
        audio_backbone = nn.Identity() 
        audio_dim = 768
    
    vision_backbone.to(device)
    audio_backbone.to(device)
    
    return vision_backbone, audio_backbone, vision_dim, audio_dim, audio_processor 
  
if __name__ == "__main__":
    # --- Configuration for Dataset Size and Training ---
    DATASET_SIZE_LIMIT = 534
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 20
    VALIDATION_SPLIT = 0.2 # 20% of data for validation

    VISION_MODEL_CHOICE = "resnet18"
    AUDIO_MODEL_CHOICE = "wav2vec2_base" # Using Wav2Vec2
    FUSION_OUTPUT_DIM = 512
 
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_{timestamp}_{VISION_MODEL_CHOICE}_{AUDIO_MODEL_CHOICE}"
    run_log_dir = os.path.join(LOG_PARENT_DIR, run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(os.path.join(run_log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(run_log_dir, 'embeddings'), exist_ok=True)
    os.makedirs(os.path.join(run_log_dir, 'logs'), exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(run_log_dir, 'logs', 'training_log.txt')),
                            logging.StreamHandler()
                        ])
    logging.info(f"Experiment results will be saved to: {run_log_dir}")


    # Load annotations and prepare label maps
    annotations_df = pd.read_csv(ANNOTATIONS_CSV)

    emotion_labels = annotations_df['Emotion'].unique().tolist()
    sentiment_labels = annotations_df['Sentiment'].unique().tolist()

    emotion_map = {label: i for i, label in enumerate(emotion_labels)}
    sentiment_map = {label: i for i, label in enumerate(sentiment_labels)}

    num_emotions = len(emotion_labels)
    num_sentiments = len(sentiment_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    vision_backbone_instance, audio_backbone_instance, \
    vision_feature_dim, audio_feature_dim, audio_processor_instance = get_actual_backbones(
        VISION_MODEL_CHOICE, AUDIO_MODEL_CHOICE, device
    )

    full_dataset = UtteranceEmotionDataset(annotations_df, PREPROCESSED_DATA_DIR, emotion_map, sentiment_map,
                                      dataset_size_limit=DATASET_SIZE_LIMIT,
                                      audio_processor=audio_processor_instance,
                                      audio_model_choice=AUDIO_MODEL_CHOICE)
    
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    logging.info(f"Full dataset size: {len(full_dataset)}")
    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")

 
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_utterance_emotion, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_utterance_emotion, num_workers=2) # Shuffle=False for validation
    
    logging.info(f"DataLoaders created. Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")


    stage1_model = MultimodalEmotionModelStage1(
        vision_backbone=vision_backbone_instance,
        audio_backbone=audio_backbone_instance,
        vision_feature_dim=vision_feature_dim,
        audio_feature_dim=audio_feature_dim,
        fusion_output_dim=FUSION_OUTPUT_DIM,
        num_emotions=num_emotions,
        num_sentiments=num_sentiments,
        audio_model_choice=AUDIO_MODEL_CHOICE
    )
    stage1_model.to(device)
    logging.info(f"Stage 1 Model initialized with {sum(p.numel() for p in stage1_model.parameters() if p.requires_grad)} trainable parameters.")


    emotion_loss_fn = nn.CrossEntropyLoss()
    sentiment_loss_fn = nn.MSELoss()

    trainable_params = filter(lambda p: p.requires_grad, stage1_model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    # --- Start Stage 1 Fine-tuning ---
    train_stage1_model(
        model=stage1_model,
        train_dataloader=train_dataloader, # Pass train_dataloader
        val_dataloader=val_dataloader,     # Pass val_dataloader
        emotion_loss_fn=emotion_loss_fn,
        sentiment_loss_fn=sentiment_loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        log_dir=run_log_dir
    )

    logging.info("\n--- Post-Training Actions: Saving Model and Embeddings ---")

    model_save_path = os.path.join(run_log_dir, 'models', 'stage1_multimodal_emotion_model.pth')
    torch.save(stage1_model.state_dict(), model_save_path)
    logging.info(f"Stage 1 Model saved to {model_save_path}")
    print(f"Stage 1 Model saved to {model_save_path}")

    stage1_model.eval()
    all_utterance_embeddings = {}

    logging.info("Generating and saving utterance embeddings for Stage 2 (using entire dataset)...")
    # NEW: Create a DataLoader for the *full* dataset if you want embeddings for all of it
    full_dataloader_for_embeddings = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_utterance_emotion, num_workers=2)

    with torch.no_grad():
        for i, batch in enumerate(full_dataloader_for_embeddings): # Iterate through full dataset
            if batch is None:
                continue
            cropped_frames = batch['cropped_frames'].to(device)
            audio_input_features = batch['audio_chunk'].to(device)

            utterance_embeddings, _, _ = stage1_model(cropped_frames, audio_input_features)

            for idx in range(len(batch['dialogue_ids'])):
                dialogue_id = batch['dialogue_ids'][idx].item()
                utterance_id = batch['utterance_ids'][idx].item()
                key = f"D{dialogue_id}_U{utterance_id}"
                all_utterance_embeddings[key] = utterance_embeddings[idx].cpu().numpy()

            if i % 100 == 0:
                logging.info(f"Processed {i+1} batches for embedding generation.")
                print(f"Processed {i+1} batches for embedding generation.")
    embeddings_save_path = os.path.join(run_log_dir, 'embeddings', 'utterance_embeddings_stage1.npy')
    np.save(embeddings_save_path, all_utterance_embeddings)
    logging.info(f"All utterance embeddings saved to {embeddings_save_path}")
