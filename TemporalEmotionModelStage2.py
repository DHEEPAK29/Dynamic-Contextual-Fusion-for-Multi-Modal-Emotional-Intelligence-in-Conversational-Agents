import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import random
import time
import logging
from datetime import datetime

import transformers
from torchvision.models import resnet18, ResNet18_Weights
from transformers import ViTModel, Wav2Vec2Model, WhisperModel, WhisperProcessor

from torch.utils.data import random_split
from torch.cuda.amp import autocast, GradScaler
 
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, mean_absolute_error, mean_squared_error 
import matplotlib.pyplot as plt # For confusion matrix plotting
import seaborn as sns # For confusion matrix plotting


# --- Configuration ---
ANNOTATIONS_CSV = "/content/drive/MyDrive/train_sent_emo.csv"
LOG_PARENT_DIR = "/content/drive/MyDrive/Temporal-Experiment-Logs-TMP"

STAGE1_RUN_DIR = "/content/drive/MyDrive/Temporal-Experiment-Logs/run_2025-12-01_01-27-35_resnet18_wav2vec2_base" # EXAMPLE! ADJUST THIS!
STAGE1_EMBEDDINGS_PATH = os.path.join(STAGE1_RUN_DIR, 'embeddings', 'utterance_embeddings_stage1.npy')
FUSION_OUTPUT_DIM = 512


# --- UtteranceEmotionDataset Class (from previous response) ---
# ... (All previous class definitions from DummyVisionExtractor to MultimodalEmotionModelStage1) ...
# These are kept for context if you run the whole script, but not directly used by Stage 2's main block.
class DummyVisionExtractor(nn.Module):
    def forward(self, cropped_frames_batch):
        N, T, C, H, W = cropped_frames_batch.shape
        return torch.randn(N * T, 512, device=cropped_frames_batch.device)

class DummyAudioExtractor(nn.Module):
    def forward(self, audio_input_batch):
        if audio_input_batch.dim() == 2:
            seq_len = audio_input_batch.shape[1] // 160
            if seq_len == 0: seq_len = 1
            hidden_dim = 768
            from transformers.modeling_outputs import BaseModelOutput
            return BaseModelOutput(last_hidden_state=torch.randn(audio_input_batch.shape[0], seq_len, hidden_dim, device=audio_input_batch.device))
        elif audio_input_batch.dim() == 3:
            seq_len = audio_input_batch.shape[2] // 2
            if seq_len == 0: seq_len = 1
            hidden_dim = 384
            from transformers.modeling_outputs import BaseModelOutput
            return BaseModelOutput(last_hidden_state=torch.randn(audio_input_batch.shape[0], seq_len, hidden_dim, device=audio_input_batch.device))
        else:
            raise ValueError(f"DummyAudioExtractor: Unsupported audio input dim: {audio_input_batch.dim()}")


class MultimodalEmotionModelStage1(nn.Module):
    def __init__(self, vision_backbone, audio_backbone,
                 vision_feature_dim, audio_feature_dim,
                 fusion_output_dim, num_emotions, num_sentiments, audio_model_choice=""):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.audio_backbone = audio_backbone
        self.audio_model_choice = audio_model_choice
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

    def forward(self, video_frames_batch, audio_input_features_batch):
        batch_size, num_frames, C, H, W = video_frames_batch.shape
        vision_feats_flat = self.vision_backbone(video_frames_batch.view(-1, C, H, W))
        if hasattr(vision_feats_flat, 'last_hidden_state'):
             vision_feats_flat = vision_feats_flat.last_hidden_state
        vision_feats = vision_feats_flat.view(batch_size, num_frames, -1)
        agg_vision_feat = torch.mean(vision_feats, dim=1)

        if self.audio_model_choice == "whisper_encoder":
            audio_output = self.audio_backbone(input_features=audio_input_features_batch)
        elif self.audio_model_choice == "wav2vec2_base":
            audio_output = self.audio_backbone(audio_input_features_batch)
        else:
            audio_output = self.audio_backbone(audio_input_features_batch)

        if hasattr(audio_output, 'last_hidden_state'):
            audio_feats = torch.mean(audio_output.last_hidden_state, dim=1)
        elif isinstance(audio_output, torch.Tensor):
            if audio_output.dim() == 3:
                audio_feats = torch.mean(audio_output, dim=1)
            elif audio_output.dim() == 2:
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


class UtteranceEmotionDataset(Dataset):
    def __init__(self, annotations_df, preprocessed_data_dir, emotion_map, sentiment_map, dataset_size_limit=None, audio_processor=None, audio_model_choice=""):
        if dataset_size_limit:
            self.annotations = annotations_df.head(dataset_size_limit).copy()
        else:
            self.annotations = annotations_df.copy()

        self.preprocessed_data_dir = preprocessed_data_dir
        self.emotion_map = emotion_map
        self.sentiment_map = sentiment_map
        self.audio_processor = audio_processor
        self.audio_model_choice = audio_model_choice

        self.data_entries = []
        for idx, row in self.annotations.iterrows():
            dialogue_id = row['Dialogue_ID']
            utterance_id = row['Utterance_ID']

            base_filename = f"dia{dialogue_id}_utt{utterance_id}"
            masked_frames_path = os.path.join(self.preprocessed_data_dir, 'masked_faces', f"{base_filename}_frames.npy")
            audio_path = os.path.join(self.preprocessed_data_dir, 'audio_chunks', f"{base_filename}_audio.npy")

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
            dummy_frames = np.zeros((1, FACE_CROP_SIZE[0], FACE_CROP_SIZE[1], 3), dtype=np.uint8)
            dummy_raw_audio = np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32)

            if self.audio_model_choice == "whisper_encoder" and self.audio_processor:
                dummy_audio_input_features = self.audio_processor(dummy_raw_audio, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt", padding="max_length", truncation=True).input_features.squeeze(0)
            elif self.audio_model_choice == "wav2vec2_base":
                dummy_audio_input_features = torch.from_numpy(dummy_raw_audio).float()
            else:
                dummy_audio_input_features = torch.from_numpy(dummy_raw_audio).float()

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

        if self.audio_model_choice == "whisper_encoder" and self.audio_processor:
            audio_input_features = self.audio_processor(audio_data_np, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt", padding="max_length", truncation=True).input_features.squeeze(0)
        elif self.audio_model_choice == "wav2vec2_base":
            audio_input_features = torch.from_numpy(audio_data_np).float()
        else:
            audio_input_features = torch.from_numpy(audio_data_np).float()

        emotion_id = self.emotion_map.get(emotion_label, -1)
        sentiment_id = self.sentiment_map.get(sentiment_label, -1)

        if emotion_id == -1 or sentiment_id == -1:
             logging.warning(f"Invalid label for D:{dialogue_id}, U:{utterance_id}. Emotion: {emotion_label}, Sentiment: {sentiment_label}. Using dummy labels.")
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

class AudioInputType:
    RAW_SAMPLES = 1
    MEL_SPECTROGRAMS = 2

def collate_fn_utterance_emotion(batch):
    batch = [item for item in batch if item['speaker_id'] != -1]
    if not batch:
        return None

    max_frames = max(item['cropped_frames'].shape[0] for item in batch)
    
    audio_input_dim = batch[0]['audio_chunk'].dim()

    if audio_input_dim == 2:
        audio_input_type = AudioInputType.MEL_SPECTROGRAMS
        num_mel_bands = batch[0]['audio_chunk'].shape[0]
        max_audio_seq_len = max(item['audio_chunk'].shape[1] for item in batch)
    elif audio_input_dim == 1:
        audio_input_type = AudioInputType.RAW_SAMPLES
        max_audio_len = max(item['audio_chunk'].shape[0] for item in batch)
    else:
        raise ValueError(f"Unexpected audio_chunk dimension in collate_fn: {audio_input_dim}")

    padded_frames = []
    padded_audio_inputs = []

    for item in batch:
        num_frames = item['cropped_frames'].shape[0]
        padding_frames = torch.zeros(max_frames - num_frames, *item['cropped_frames'].shape[1:])
        padded_frames.append(torch.cat((item['cropped_frames'], padding_frames), dim=0))

        if audio_input_type == AudioInputType.MEL_SPECTROGRAMS:
            current_audio_seq_len = item['audio_chunk'].shape[1]
            padding_audio = torch.zeros(num_mel_bands, max_audio_seq_len - current_audio_seq_len)
            padded_audio_inputs.append(torch.cat((item['audio_chunk'], padding_audio), dim=1))
        else:
            current_audio_len = item['audio_chunk'].shape[0]
            padding_audio = torch.zeros(max_audio_len - current_audio_len)
            padded_audio_inputs.append(torch.cat((item['audio_chunk'], padding_audio), dim=0))

    return {
        'cropped_frames': torch.stack(padded_frames),
        'audio_chunk': torch.stack(padded_audio_inputs),
        'speaker_ids': torch.tensor([item['speaker_id'] for item in batch], dtype=torch.long),
        'emotion_label_ids': torch.stack([item['emotion_label_id'] for item in batch]),
        'sentiment_label_ids': torch.stack([item['sentiment_label_id'] for item in batch]),
        'dialogue_ids': torch.tensor([item['dialogue_id'] for item in batch], dtype=torch.long),
        'utterance_ids': torch.tensor([item['utterance_id'] for item in batch], dtype=torch.long)
    }


class MultimodalEmotionModelStage1(nn.Module):
    def __init__(self, vision_backbone, audio_backbone, 
                 vision_feature_dim, audio_feature_dim, 
                 fusion_output_dim, num_emotions, num_sentiments, audio_model_choice=""):
        super().__init__()
        
        self.vision_backbone = vision_backbone
        self.audio_backbone = audio_backbone
        self.audio_model_choice = audio_model_choice 
        
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

    def forward(self, video_frames_batch, audio_input_features_batch): 
        batch_size, num_frames, C, H, W = video_frames_batch.shape
        vision_feats_flat = self.vision_backbone(video_frames_batch.view(-1, C, H, W))
        if hasattr(vision_feats_flat, 'last_hidden_state'):
             vision_feats_flat = vision_feats_flat.last_hidden_state 
        vision_feats = vision_feats_flat.view(batch_size, num_frames, -1)
        
        agg_vision_feat = torch.mean(vision_feats, dim=1) 

        if self.audio_model_choice == "whisper_encoder":
            audio_output = self.audio_backbone(input_features=audio_input_features_batch)
        elif self.audio_model_choice == "wav2vec2_base":
            audio_output = self.audio_backbone(audio_input_features_batch)
        else: # Default for dummy or other backbones that take positional arg
            audio_output = self.audio_backbone(audio_input_features_batch)
        
        if hasattr(audio_output, 'last_hidden_state'):
            audio_feats = torch.mean(audio_output.last_hidden_state, dim=1) 
        elif isinstance(audio_output, torch.Tensor):
            if audio_output.dim() == 3: 
                audio_feats = torch.mean(audio_output, dim=1)
            elif audio_output.dim() == 2: 
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

# --- Training / Evaluation Helper Function ---
def train_stage1_model(model, train_dataloader, val_dataloader, emotion_loss_fn, sentiment_loss_fn, optimizer, device, num_epochs, log_dir):
    
    metrics_log_path = os.path.join(log_dir, 'logs', 'metrics.csv')
    with open(metrics_log_path, 'w') as f:
        f.write("epoch,time_s,avg_train_loss,train_emotion_accuracy,avg_val_loss,val_emotion_accuracy\n") 
    
    logging.info(f"Starting Stage 1 fine-tuning for {len(train_dataloader.dataset)} utterances...")
    
    scaler = GradScaler()

    for epoch in range(num_epochs):
        start_time = time.time()
        
        # --- Training Loop ---
        model.train() 
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
        
        logging.info(f"Epoch {epoch+1} finished. Time: {end_time - start_time:.2f}s")
        logging.info(f"  Train Loss: {train_avg_loss:.4f}, Train Acc: {train_emotion_accuracy:.4f}")
        logging.info(f"  Val Loss:   {val_avg_loss:.4f}, Val Acc:   {val_emotion_accuracy:.4f}\n")
        
        with open(metrics_log_path, 'a') as f:
            f.write(f"{epoch+1},{end_time - start_time:.2f},{train_avg_loss:.4f},{train_emotion_accuracy:.4f},{val_avg_loss:.4f},{val_emotion_accuracy:.4f}\n")
    
    logging.info("Stage 1 Fine-tuning Complete.")


# --- NEW FUNCTION: validate_stage1_model ---
def validate_stage1_model(model, dataloader, emotion_loss_fn, sentiment_loss_fn, device):
    model.eval() 
    val_total_loss = 0
    val_correct_emotions = 0
    val_total_samples = 0
    
    with torch.no_grad(): 
        for i, batch in enumerate(dataloader):
            if batch is None:
                logging.warning(f"Skipping empty validation batch {i}.")
                continue
            
            cropped_frames = batch['cropped_frames'].to(device)
            audio_input_features = batch['audio_chunk'].to(device) 
            emotion_labels = batch['emotion_label_ids'].to(device)
            sentiment_labels = batch['sentiment_label_ids'].to(device) 

            with autocast(): 
                _, emotion_logits, sentiment_scores = model(cropped_frames, audio_input_features) 
                
                emotion_loss = emotion_loss_fn(emotion_logits, emotion_labels)
                sentiment_loss = sentiment_loss_fn(sentiment_scores, sentiment_labels.float().unsqueeze(1)) 
                
                loss = emotion_loss + sentiment_loss 
            
            val_total_loss += loss.item()

            _, predicted_emotions = torch.max(emotion_logits, 1)
            val_correct_emotions += (predicted_emotions == emotion_labels).sum().item()
            val_total_samples += emotion_labels.size(0)
            
    val_avg_loss = val_total_loss / len(dataloader)
    val_emotion_accuracy = val_correct_emotions / val_total_samples
    
    return val_avg_loss, val_emotion_accuracy


# --- Helper for Loading Actual Pre-trained Backbones ---
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


# --- Stage 2: Data Loading and Preparation ---

# Dataset for sequences of utterances (grouped by Dialogue_ID)
class DialogueSequenceDataset(Dataset):
    def __init__(self, annotations_df, utterance_embeddings_path, emotion_map, sentiment_map, dataset_size_limit=534):
        
        self.annotations = annotations_df.copy()
        if dataset_size_limit:
            self.annotations = self.annotations.head(dataset_size_limit)

        self.emotion_map = emotion_map
        self.sentiment_map = sentiment_map

        self.all_utterance_embeddings = np.load(utterance_embeddings_path, allow_pickle=True).item()

        self.dialogue_sequences = []
        
        valid_annotations = []
        for idx, row in self.annotations.iterrows():
            dialogue_id = row['Dialogue_ID']
            utterance_id = row['Utterance_ID']
            key = f"dia{dialogue_id}_utt{utterance_id}"
            if key in self.all_utterance_embeddings:
                valid_annotations.append(row)
            else:
                logging.warning(f"Embedding not found for {key}. Skipping this utterance.")
        
        valid_annotations_df = pd.DataFrame(valid_annotations)

        grouped_dialogues = valid_annotations_df.groupby('Dialogue_ID').apply(
            lambda x: x.sort_values('Utterance_ID').reset_index(drop=True)
        )

        for dialogue_id, dialogue_df in grouped_dialogues.groupby(level=0):
            embeddings_sequence = []
            emotion_labels_sequence = []
            sentiment_labels_sequence = []
            speaker_ids_sequence = []
            utterance_ids_sequence = []

            for idx, utt_row in dialogue_df.iterrows():
                key = f"D{utt_row['Dialogue_ID']}_U{utt_row['Utterance_ID']}"
                embeddings_sequence.append(self.all_utterance_embeddings[key])
                emotion_labels_sequence.append(self.emotion_map[utt_row['Emotion']])
                sentiment_labels_sequence.append(self.sentiment_map.get(utt_row['Sentiment'], 0) if isinstance(utt_row['Sentiment'], str) else utt_row['Sentiment'])
                speaker_ids_sequence.append(hash(utt_row['Speaker']) % 1000)
                utterance_ids_sequence.append(utt_row['Utterance_ID'])
            
            if embeddings_sequence:
                self.dialogue_sequences.append({
                    'dialogue_id': dialogue_id,
                    'embeddings_sequence': torch.from_numpy(np.array(embeddings_sequence)).float(),
                    'emotion_labels_sequence': torch.tensor(emotion_labels_sequence, dtype=torch.long),
                    'sentiment_labels_sequence': torch.tensor(sentiment_labels_sequence, dtype=torch.float).unsqueeze(1),
                    'speaker_ids_sequence': torch.tensor(speaker_ids_sequence, dtype=torch.long),
                    'utterance_ids_sequence': torch.tensor(utterance_ids_sequence, dtype=torch.long)
                })
        
        if not self.dialogue_sequences:
            raise ValueError("No valid dialogue sequences could be formed from embeddings and annotations.")

    def __len__(self):
        return len(self.dialogue_sequences)

    def __getitem__(self, idx):
        return self.dialogue_sequences[idx]

# Collate function for sequences of utterances
class AudioInputType:
    RAW_SAMPLES = 1
    MEL_SPECTROGRAMS = 2

def collate_fn_sequence(batch):
    max_seq_len = max(item['embeddings_sequence'].shape[0] for item in batch)
    embedding_dim = batch[0]['embeddings_sequence'].shape[1]

    padded_embeddings = []
    padded_emotion_labels = []
    padded_sentiment_labels = []
    padded_speaker_ids = []
    padded_utterance_ids = []

    attention_mask = torch.zeros(len(batch), max_seq_len, dtype=torch.bool) 

    for i, item in enumerate(batch):
        seq_len = item['embeddings_sequence'].shape[0]
        
        pad_embeddings = torch.zeros(max_seq_len - seq_len, embedding_dim)
        padded_embeddings.append(torch.cat((item['embeddings_sequence'], pad_embeddings), dim=0))

        pad_emotion = torch.full((max_seq_len - seq_len,), -1, dtype=torch.long) # -1 as ignore_index
        padded_emotion_labels.append(torch.cat((item['emotion_labels_sequence'], pad_emotion), dim=0))
        
        pad_sentiment = torch.zeros((max_seq_len - seq_len, 1), dtype=torch.float)
        padded_sentiment_labels.append(torch.cat((item['sentiment_labels_sequence'], pad_sentiment), dim=0))

        pad_speaker = torch.full((max_seq_len - seq_len,), 0, dtype=torch.long)
        padded_speaker_ids.append(torch.cat((item['speaker_ids_sequence'], pad_speaker), dim=0))

        pad_utt_id = torch.full((max_seq_len - seq_len,), -1, dtype=torch.long)
        padded_utterance_ids.append(torch.cat((item['utterance_ids_sequence'], pad_utt_id), dim=0))

        attention_mask[i, :seq_len] = True

    return {
        'dialogue_ids': torch.tensor([item['dialogue_id'] for item in batch], dtype=torch.long),
        'embeddings_sequence': torch.stack(padded_embeddings), # (batch_size, max_seq_len, embedding_dim)
        'emotion_labels_sequence': torch.stack(padded_emotion_labels), # (batch_size, max_seq_len)
        'sentiment_labels_sequence': torch.stack(padded_sentiment_labels), # (batch_size, max_seq_len, 1)
        'speaker_ids_sequence': torch.stack(padded_speaker_ids), # (batch_size, max_seq_len)
        'utterance_ids_sequence': torch.stack(padded_utterance_ids), # (batch_size, max_seq_len)
        'attention_mask': attention_mask # (batch_size, max_seq_len)
    }


# --- Stage 2: Temporal Emotion Model Definition (Flexible) --- 
class TemporalEmotionModelStage2(nn.Module):
    def __init__(self, model_type, embedding_dim, hidden_dim, num_layers, num_heads, num_emotions, num_sentiments, max_mem_slots, speaker_embedding_dim=64, num_speakers=1000):
        super().__init__()
        
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_emotions = num_emotions
        self.num_sentiments = num_sentiments
        
        # Speaker embeddings
        self.speaker_embedding = nn.Embedding(num_speakers, speaker_embedding_dim)

        # Input projection (to match hidden_dim if combined_input_dim != hidden_dim)
        combined_input_dim = embedding_dim + speaker_embedding_dim
        self.input_projection = nn.Linear(combined_input_dim, hidden_dim) if combined_input_dim != hidden_dim else nn.Identity()

        # --- Flexible Temporal Model Architecture ---
        if model_type == "RMT":
            # Simplified RMT-like behavior using TransformerEncoder and explicit memory vectors
            # A full RMT would involve more sophisticated read/write heads
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, batch_first=True)
            self.temporal_core = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Learnable memory
            self.memory_state = nn.Parameter(torch.randn(1, 1, hidden_dim)) # Shared memory state across batch initially
            self.mem_gate = nn.Linear(hidden_dim * 2, hidden_dim) # For updating memory
            self.mem_read_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True) # For reading memory
        
        elif model_type == "TransformerXL_like": # Simulated Transformer-XL
            # Use a standard TransformerEncoder, but the training loop handles state passing
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, batch_first=True)
            self.temporal_core = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            # No explicit internal memory, state passing is external
        
        elif model_type == "Longformer_like": # Simulated Longformer (standard Transformer but can handle longer sequences if padding is managed)
            # For actual Longformer, you'd load a LongformerModel
            # Here we use TransformerEncoder and assume input sequences are long (handled by collate)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, batch_first=True)
            self.temporal_core = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        elif model_type == "TCN":
            # Temporal Convolutional Network
            # Example TCN structure (dilation, kernel_size, num_channels)
            def _create_tcn_layer(in_channels, out_channels, kernel_size, dilation_size):
                return nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=(dilation_size * (kernel_size - 1)) // 2, dilation=dilation_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Conv1d(out_channels, out_channels, 1), # 1x1 conv for residual connection if needed
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
            
            tcn_layers = []
            in_channels = hidden_dim
            for i in range(num_layers):
                dilation_size = 2**i
                out_channels = hidden_dim
                tcn_layers.append(_create_tcn_layer(in_channels, out_channels, kernel_size=3, dilation_size=dilation_size))
                in_channels = out_channels # For next layer
            self.temporal_core = nn.Sequential(*tcn_layers)
            
        else:
            raise ValueError(f"Unknown model_type for Stage 2: {model_type}")

        # Prediction Heads
        self.emotion_head = nn.Linear(hidden_dim, num_emotions)
        self.sentiment_head = nn.Linear(hidden_dim, num_sentiments)


    def forward(self, embeddings_sequence, speaker_ids_sequence, attention_mask, previous_memory_state=None):
        # embeddings_sequence: (batch_size, seq_len, embedding_dim)
        # speaker_ids_sequence: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len) (True for real tokens, False for padding)

        batch_size, seq_len, _ = embeddings_sequence.shape

        # 1. Incorporate speaker embeddings
        speaker_embs = self.speaker_embedding(speaker_ids_sequence)
        combined_input = torch.cat((embeddings_sequence, speaker_embs), dim=-1)
        processed_input = self.input_projection(combined_input) # (batch_size, seq_len, hidden_dim)

        encoder_output = None
        current_memory_state = previous_memory_state # for RMT

        if self.model_type == "RMT":
            # --- Simplified RMT Memory Read/Update ---
            # Initial memory for the batch if not provided (start of dialogue or batch)
            if current_memory_state is None:
                current_memory_state = self.memory_state.expand(batch_size, -1, -1) # (batch_size, 1, hidden_dim)
            
            # Read Memory (Query the memory with the current input sequence)
            # Query is `processed_input`, Key/Value is `current_memory_state`
            read_output, _ = self.mem_read_attention(query=processed_input, key=current_memory_state, value=current_memory_state, key_padding_mask=None)
            # read_output is (batch_size, seq_len, hidden_dim)
            
            # Fuse input with read memory (e.g., residual connection)
            input_with_memory = processed_input + read_output 

            # Pass through Transformer Encoder
            src_key_padding_mask = ~attention_mask # (batch_size, seq_len)
            encoder_output = self.temporal_core(input_with_memory, src_key_padding_mask=src_key_padding_mask)

            # Update Memory (Simple gate based on last output token)
            # For each item in batch, update its memory_state based on its last output
            last_output_token = encoder_output[:, -1, :].unsqueeze(1) # (batch_size, 1, hidden_dim)
            combined_for_gate = torch.cat([current_memory_state, last_output_token], dim=-1) # (batch_size, 1, hidden_dim*2)
            gate_factor = torch.sigmoid(self.mem_gate(combined_for_gate)) # (batch_size, 1, hidden_dim)
            current_memory_state = current_memory_state * (1 - gate_factor) + last_output_token * gate_factor # (batch_size, 1, hidden_dim)
            
            # Pass memory_state for next batch/sequence if applicable (handled by training loop)
            # For now, we return it.

        elif self.model_type in ["TransformerXL_like", "Longformer_like"]:
            src_key_padding_mask = ~attention_mask
            encoder_output = self.temporal_core(processed_input, src_key_padding_mask=src_key_padding_mask)

        elif self.model_type == "TCN":
            # TCN expects (batch_size, channels, seq_len)
            processed_input_tcn = processed_input.permute(0, 2, 1) # (batch_size, hidden_dim, seq_len)
            encoder_output_tcn = self.temporal_core(processed_input_tcn)
            encoder_output = encoder_output_tcn.permute(0, 2, 1) # (batch_size, seq_len, hidden_dim)

        # 5. Predict Emotion and Sentiment for each token in the sequence
        emotion_logits_sequence = self.emotion_head(encoder_output)
        sentiment_scores_sequence = self.sentiment_head(encoder_output)

        return emotion_logits_sequence, sentiment_scores_sequence, current_memory_state # Return memory for RMT


# --- Helper for Concordance Correlation Coefficient (CCC) ---
def concordance_correlation_coefficient(y_true, y_pred):
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) == 0:
        return np.nan # Or 0, depending on how you want to handle empty sets

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    cov_true_pred = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * cov_true_pred) / (var_true + var_pred + (mean_true - mean_pred)**2 + 1e-8) # Add epsilon for stability
    return ccc


# --- Training / Evaluation Helper Functions for Stage 2 ---
def train_stage2_model(model, train_dataloader, val_dataloader, emotion_loss_fn, sentiment_loss_fn, optimizer, device, num_epochs, log_dir, emotion_ignore_index=-1, num_emotions=0, emotion_label_map=None):
    # Added num_emotions and emotion_label_map for confusion matrix
    model.train()
    
    # NEW: Updated CSV header for all metrics
    metrics_log_path = os.path.join(log_dir, 'logs', 'metrics_stage2.csv') 
    with open(metrics_log_path, 'w') as f:
        f.write("epoch,time_s,avg_train_loss,train_acc,train_f1,train_precision,train_recall,avg_val_loss,val_acc,val_f1,val_precision,val_recall,val_mae,val_rmse,val_ccc\n") 
    
    logging.info(f"Starting Stage 2 fine-tuning for {len(train_dataloader.dataset)} dialogue sequences...")
    
    scaler = GradScaler()

    for epoch in range(num_epochs):
        start_time = time.time()
        
        # --- Training Loop ---
        model.train() 
        train_total_loss = 0
        train_all_preds_emotion = []
        train_all_labels_emotion = []
        train_all_preds_sentiment = []
        train_all_labels_sentiment = []
        
        for i, batch in enumerate(train_dataloader):
            if batch is None:
                logging.warning(f"Skipping empty training batch {i}.")
                continue
                
            embeddings_sequence = batch['embeddings_sequence'].to(device)
            speaker_ids_sequence = batch['speaker_ids_sequence'].to(device)
            emotion_labels_sequence = batch['emotion_labels_sequence'].to(device)
            sentiment_labels_sequence = batch['sentiment_labels_sequence'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            
            with autocast(): 
                # RMT model returns memory state
                emotion_logits_sequence, sentiment_scores_sequence, current_memory_state = model(embeddings_sequence, speaker_ids_sequence, attention_mask)
                
                # Mask out padding from loss calculation
                active_elements = attention_mask.view(-1)
                
                # Flatten predictions and labels for loss functions
                active_emotion_logits = emotion_logits_sequence.view(-1, emotion_logits_sequence.shape[-1])[active_elements]
                active_emotion_labels = emotion_labels_sequence.view(-1)[active_elements]
                active_sentiment_scores = sentiment_scores_sequence.view(-1, sentiment_scores_sequence.shape[-1])[active_elements]
                active_sentiment_labels = sentiment_labels_sequence.view(-1, sentiment_labels_sequence.shape[-1])[active_elements]

                valid_emotion_indices = (active_emotion_labels != emotion_ignore_index)
                
                if valid_emotion_indices.sum() > 0: 
                    emotion_loss = emotion_loss_fn(active_emotion_logits[valid_emotion_indices], active_emotion_labels[valid_emotion_indices])
                else:
                    emotion_loss = torch.tensor(0.0, device=device)
                
                if active_sentiment_labels.shape[0] > 0:
                    sentiment_loss = sentiment_loss_fn(active_sentiment_scores, active_sentiment_labels)
                else:
                    sentiment_loss = torch.tensor(0.0, device=device)

                loss = emotion_loss + sentiment_loss 
            
            scaler.scale(loss).backward() 
            scaler.step(optimizer)        
            scaler.update()               
            
            train_total_loss += loss.item()

            # Collect predictions and labels for metrics
            if valid_emotion_indices.sum() > 0:
                _, predicted_emotions = torch.max(active_emotion_logits[valid_emotion_indices], 1)
                train_all_preds_emotion.extend(predicted_emotions.cpu().numpy())
                train_all_labels_emotion.extend(active_emotion_labels[valid_emotion_indices].cpu().numpy())
            
            if active_sentiment_labels.shape[0] > 0:
                train_all_preds_sentiment.extend(sentiment_scores_sequence.view(-1).cpu().numpy())
                train_all_labels_sentiment.extend(sentiment_labels_sequence.view(-1).cpu().numpy())

            if i % 10 == 0: 
                logging.info(f"Epoch {epoch+1}, Train Batch {i+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
        
        train_avg_loss = train_total_loss / len(train_dataloader)
        
        # Calculate full training metrics
        train_acc, train_f1, train_precision, train_recall, train_cm = calculate_emotion_metrics(
            train_all_labels_emotion, train_all_preds_emotion, num_emotions, emotion_label_map
        )
        train_mae, train_rmse, train_ccc = calculate_sentiment_metrics(
            train_all_labels_sentiment, train_all_preds_sentiment
        )
        
        # --- Validation Loop ---
        val_metrics = validate_stage2_model(model, val_dataloader, emotion_loss_fn, sentiment_loss_fn, device, emotion_ignore_index, num_emotions, emotion_label_map, log_dir, epoch+1) # Pass log_dir and epoch
        val_avg_loss, val_acc, val_f1, val_precision, val_recall, val_mae, val_rmse, val_ccc = val_metrics
        
        end_time = time.time()
        
        logging.info(f"Epoch {epoch+1} finished. Time: {end_time - start_time:.2f}s")
        logging.info(f"  Train Loss: {train_avg_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, P: {train_precision:.4f}, R: {train_recall:.4f}")
        logging.info(f"  Val Loss:   {val_avg_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, P: {val_precision:.4f}, R: {val_recall:.4f}")
        logging.info(f"  Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}, Val CCC: {val_ccc:.4f}\n")
        
        # Save epoch metrics to CSV
        with open(metrics_log_path, 'a') as f:
            f.write(f"{epoch+1},{end_time - start_time:.2f},{train_avg_loss:.4f},{train_acc:.4f},{train_f1:.4f},{train_precision:.4f},{train_recall:.4f},{val_avg_loss:.4f},{val_acc:.4f},{val_f1:.4f},{val_precision:.4f},{val_recall:.4f},{val_mae:.4f},{val_rmse:.4f},{val_ccc:.4f}\n")
    
    logging.info("Stage 2 Fine-tuning Complete.")


#  validate_stage2_model function  
def validate_stage2_model(model, dataloader, emotion_loss_fn, sentiment_loss_fn, device, emotion_ignore_index=-1, num_emotions=0, emotion_label_map=None, log_dir=None, epoch=0):
    model.eval() 
    val_total_loss = 0
    val_all_preds_emotion = []
    val_all_labels_emotion = []
    val_all_preds_sentiment = []
    val_all_labels_sentiment = []
    
    with torch.no_grad(): 
        for i, batch in enumerate(dataloader):
            if batch is None:
                logging.warning(f"Skipping empty validation batch {i}.")
                continue
            
            embeddings_sequence = batch['embeddings_sequence'].to(device)
            speaker_ids_sequence = batch['speaker_ids_sequence'].to(device)
            emotion_labels_sequence = batch['emotion_labels_sequence'].to(device)
            sentiment_labels_sequence = batch['sentiment_labels_sequence'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with autocast(): 
                emotion_logits_sequence, sentiment_scores_sequence, _ = model(embeddings_sequence, speaker_ids_sequence, attention_mask) # RMT returns memory state too
                
                active_elements = attention_mask.view(-1)
                
                active_emotion_logits = emotion_logits_sequence.view(-1, emotion_logits_sequence.shape[-1])[active_elements]
                active_emotion_labels = emotion_labels_sequence.view(-1)[active_elements]
                active_sentiment_scores = sentiment_scores_sequence.view(-1, sentiment_scores_sequence.shape[-1])[active_elements]
                active_sentiment_labels = sentiment_labels_sequence.view(-1, sentiment_labels_sequence.shape[-1])[active_elements]

                valid_emotion_indices = (active_emotion_labels != emotion_ignore_index)
                
                if valid_emotion_indices.sum() > 0:
                    emotion_loss = emotion_loss_fn(active_emotion_logits[valid_emotion_indices], active_emotion_labels[valid_emotion_indices])
                else:
                    emotion_loss = torch.tensor(0.0, device=device)
                
                if active_sentiment_labels.shape[0] > 0:
                    sentiment_loss = sentiment_loss_fn(active_sentiment_scores, active_sentiment_labels)
                else:
                    sentiment_loss = torch.tensor(0.0, device=device)

                loss = emotion_loss + sentiment_loss 
            
            val_total_loss += loss.item()

            # Collect predictions and labels for metrics
            if valid_emotion_indices.sum() > 0:
                _, predicted_emotions = torch.max(active_emotion_logits[valid_emotion_indices], 1)
                val_all_preds_emotion.extend(predicted_emotions.cpu().numpy())
                val_all_labels_emotion.extend(active_emotion_labels[valid_emotion_indices].cpu().numpy())
            
            if active_sentiment_labels.shape[0] > 0:
                val_all_preds_sentiment.extend(sentiment_scores_sequence.view(-1).cpu().numpy())
                val_all_labels_sentiment.extend(active_sentiment_labels.view(-1).cpu().numpy())
            
    val_avg_loss = val_total_loss / len(dataloader)
    
    # Calculate all validation metrics
    val_acc, val_f1, val_precision, val_recall, val_cm = calculate_emotion_metrics(
        val_all_labels_emotion, val_all_preds_emotion, num_emotions, emotion_label_map
    )
    val_mae, val_rmse, val_ccc = calculate_sentiment_metrics(
        val_all_labels_sentiment, val_all_preds_sentiment
    )
    
    # Plot and save confusion matrix
    if log_dir and val_cm is not None:
        plot_confusion_matrix(val_cm, emotion_label_map, os.path.join(log_dir, 'logs', f'confusion_matrix_epoch_{epoch}.png'))

    return val_avg_loss, val_acc, val_f1, val_precision, val_recall, val_mae, val_rmse, val_ccc


# --- Helper for Loading Actual Pre-trained Backbones 
import transformers
from torchvision.models import resnet18, ResNet18_Weights
from transformers import ViTModel, Wav2Vec2Model, WhisperModel, WhisperProcessor

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
    EMBEDDING_DIM_STAGE1 = FUSION_OUTPUT_DIM # This comes from Stage 1's fusion output
    STAGE2_BATCH_SIZE = 2
    STAGE2_LEARNING_RATE = 1e-4
    STAGE2_NUM_EPOCHS = 10 
    STAGE2_VALIDATION_SPLIT = 0.2

    # --- Temporal Model Hyperparameters ---
    # NEW: Define chosen model type
    TEMPORAL_MODEL_TYPE = "RMT" # Options: "RMT", "TransformerXL_like", "Longformer_like", "TCN"

    HIDDEN_DIM_STAGE2 = 512 
    NUM_TRANSFORMER_LAYERS = 2 
    NUM_TRANSFORMER_HEADS = 4 
    MAX_MEMORY_SLOTS = 10 
    SPEAKER_EMBEDDING_DIM = 64
    MAX_SPEAKERS = 1000 

    # --- Ensure consistent label maps and number of classes ---
    annotations_df = pd.read_csv(ANNOTATIONS_CSV)
    
    emotion_labels_list = annotations_df['Emotion'].unique().tolist() # Store as list for confusion matrix
    sentiment_labels = annotations_df['Sentiment'].unique().tolist()
    
    emotion_map = {label: i for i, label in enumerate(emotion_labels_list)}
    sentiment_map = {label: i for i, label in enumerate(sentiment_labels)}
    
    num_emotions = len(emotion_labels_list)
    num_sentiments = 1 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Setup Experiment Logging Directory for Stage 2 ---
    stage2_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stage2_run_name = f"stage2_run_{stage2_timestamp}_{TEMPORAL_MODEL_TYPE}" 
    stage2_log_dir = os.path.join(LOG_PARENT_DIR, stage2_run_name)
    os.makedirs(stage2_log_dir, exist_ok=True)
    os.makedirs(os.path.join(stage2_log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(stage2_log_dir, 'logs'), exist_ok=True)

    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(stage2_log_dir, 'logs', 'training_log_stage2.txt')),
                            logging.StreamHandler()
                        ])
    logging.info(f"Stage 2 experiment results will be saved to: {stage2_log_dir}")


    # --- Load Utterance Embeddings and create Stage 2 Dataset ---
    full_stage2_dataset = DialogueSequenceDataset(
        annotations_df=annotations_df,
        utterance_embeddings_path=STAGE1_EMBEDDINGS_PATH,
        emotion_map=emotion_map,
        sentiment_map=sentiment_map,
        dataset_size_limit=None 
    )

    val_size_stage2 = int(len(full_stage2_dataset) * STAGE2_VALIDATION_SPLIT)
    train_size_stage2 = len(full_stage2_dataset) - val_size_stage2
    train_dataset_stage2, val_dataset_stage2 = random_split(full_stage2_dataset, [train_size_stage2, val_size_stage2])
    
    logging.info(f"Full Stage 2 dataset size (dialogue sequences): {len(full_stage2_dataset)}")
    logging.info(f"Stage 2 Training dataset size: {len(train_dataset_stage2)}")
    logging.info(f"Stage 2 Validation dataset size: {len(val_dataset_stage2)}")

    train_dataloader_stage2 = DataLoader(
        train_dataset_stage2, 
        batch_size=STAGE2_BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn_sequence, 
        num_workers=2 
    )
    val_dataloader_stage2 = DataLoader(
        val_dataset_stage2, 
        batch_size=STAGE2_BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn_sequence, 
        num_workers=2 
    )
    logging.info(f"Stage 2 DataLoaders created. Train batches: {len(train_dataloader_stage2)}, Val batches: {len(val_dataloader_stage2)}")
 
    # --- Initialize Stage 2 Temporal Model ---
    stage2_model = TemporalEmotionModelStage2(
        model_type=TEMPORAL_MODEL_TYPE, # Pass chosen model type
        embedding_dim=EMBEDDING_DIM_STAGE1,
        hidden_dim=HIDDEN_DIM_STAGE2,
        num_layers=NUM_TRANSFORMER_LAYERS,
        num_heads=NUM_TRANSFORMER_HEADS,
        num_emotions=num_emotions,
        num_sentiments=num_sentiments,
        max_mem_slots=MAX_MEMORY_SLOTS,
        speaker_embedding_dim=SPEAKER_EMBEDDING_DIM,
        num_speakers=MAX_SPEAKERS
    )
    stage2_model.to(device)
    logging.info(f"Stage 2 Model initialized with {sum(p.numel() for p in stage2_model.parameters() if p.requires_grad)} trainable parameters.")

    # --- Define Loss Functions for Stage 2 ---
    emotion_loss_fn_stage2 = nn.CrossEntropyLoss(ignore_index=-1) 
    sentiment_loss_fn_stage2 = nn.MSELoss() 

    # --- Define Optimizer for Stage 2 ---
    optimizer_stage2 = torch.optim.Adam(stage2_model.parameters(), lr=STAGE2_LEARNING_RATE) 
    
    # --- Start Stage 2 Fine-tuning ---
    train_stage2_model(
        model=stage2_model, 
        train_dataloader=train_dataloader_stage2, 
        val_dataloader=val_dataloader_stage2,     
        emotion_loss_fn=emotion_loss_fn_stage2,
        sentiment_loss_fn=sentiment_loss_fn_stage2,
        optimizer=optimizer_stage2, 
        device=device, 
        num_epochs=STAGE2_NUM_EPOCHS,
        log_dir=stage2_log_dir,
        emotion_ignore_index=-1,
        num_emotions=num_emotions, # Pass num_emotions
        emotion_label_map=emotion_labels_list # Pass label names for confusion matrix
    )
 
    logging.info("\n--- Post-Training Actions: Saving Stage 2 Model ---")

    model_save_path_stage2 = os.path.join(stage2_log_dir, 'models', 'stage2_temporal_emotion_model.pth')
    torch.save(stage2_model.state_dict(), model_save_path_stage2)
    logging.info(f"Stage 2 Model saved to {model_save_path_stage2}")
 
