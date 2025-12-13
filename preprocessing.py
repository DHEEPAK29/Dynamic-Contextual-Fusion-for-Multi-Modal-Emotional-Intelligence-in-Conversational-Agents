import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
# import librosa # For audio processing
import os
import random
from tqdm import tqdm # For progress bar
from pydub import AudioSegment
 
VIDEO_DIR = "/content/drive/MyDrive/ar-mp4" #"mp4/videos"
ANNOTATIONS_CSV = "/content/drive/MyDrive/train_sent_emo.csv" #"annotations.csv"
PREPROCESSED_OUTPUT_DIR = "/content/drive/MyDrive/preprocess"
# subdirectories here for 'masked_faces' and 'audio_chunks'

VIDEO_FPS = 30 # Assuming standard video FPS
AUDIO_SAMPLE_RATE = 16000 # Hz
FACE_CROP_SIZE = (224, 224) # Standard input size for many vision models
 
class DummyVisionExtractor(nn.Module):
    def forward(self, cropped_frames_batch):
        return torch.randn(cropped_frames_batch.shape[0], cropped_frames_batch.shape[1], 512)

class DummyAudioExtractor(nn.Module):
    def forward(self, audio_segment_batch):
        return torch.randn(audio_segment_batch.shape[0], 1, 768)

class DummyMultimodalFusionModel(nn.Module):
    def __init__(self, vision_dim, audio_dim, aggregated_dim, num_emotions, num_sentiments):
        super().__init__()
        self.vision_extractor = DummyVisionExtractor()
        self.audio_extractor = DummyAudioExtractor()
        self.fusion_mlp = nn.Linear(vision_dim + audio_dim, aggregated_dim)
        self.emotion_head = nn.Linear(aggregated_dim, num_emotions)
        self.sentiment_head = nn.Linear(aggregated_dim, num_sentiments)

    def forward(self, cropped_frames_batch, audio_segment_batch):
        vision_feats = self.vision_extractor(cropped_frames_batch)
        audio_feats = self.audio_extractor(audio_segment_batch)
        agg_vision_feat = torch.mean(vision_feats, dim=1)
        agg_audio_feat = audio_feats.squeeze(1)
        fused_feat = torch.cat((agg_vision_feat, agg_audio_feat), dim=1)
        utterance_embedding = self.fusion_mlp(fused_feat)
        emotion_logits = self.emotion_head(utterance_embedding)
        sentiment_scores = self.sentiment_head(utterance_embedding)
        return utterance_embedding, emotion_logits, sentiment_scores

# --- Helper for Person Tracking & Segmentation (Conceptual with Masking) ---
def get_speaker_masked_face_and_id(video_path, speaker_name, face_detector):
    """
    Identifies the speaker, gets their face region, masks it, and assigns a person_id.

    Args:
        video_path (str): Path to the single-utterance video.
        speaker_name (str): Name of the speaker from annotations.
        face_detector: A loaded face detection model (e.g., OpenCV's Haar Cascade, MTCNN, RetinaFace).

    Returns:
        tuple: (list of masked_face_frames, speaker_person_id)
            - masked_face_frames: List of (H, W, C) numpy arrays, where background is blacked out.
            - speaker_person_id: Integer ID for the speaker.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return [], None

    masked_face_frames = []
    # In a real system, you'd track the speaker's face consistently.
    # For now, we'll assume the speaker is the most prominent face found, or the first.

    speaker_person_id = hash(speaker_name) % 1000 # Consistent ID for speaker_name

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Dummy Face Detection (REPLACE with actual face detection and tracking) ---
        # Actual implementation would use a robust face detector (e.g., MTCNN, RetinaFace)
        # and a tracker to maintain identity and smooth bounding boxes across frames.
        # This is the most complex part to make robust.

        # Example using a simple OpenCV Haar cascade (less robust than deep learning models)
        # You'd need to download 'haarcascade_frontalface_default.xml'
        # face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # faces = face_detector.detectMultiScale(gray_frame, 1.1, 4)

        # For a more robust demo, let's just create a dummy face region
        dummy_x, dummy_y, dummy_w, dummy_h = 50, 50, 150, 150 # Placeholder bbox

        if frame.shape[0] > dummy_y + dummy_h and frame.shape[1] > dummy_x + dummy_w:
            # Create a black image of the same size as the frame
            masked_frame = np.zeros_like(frame)

            # Extract the speaker's face region (from dummy bbox)
            face_region = frame[dummy_y:dummy_y+dummy_h, dummy_x:dummy_x+dummy_w]

            # Resize the face region
            resized_face = cv2.resize(face_region, FACE_CROP_SIZE)

            # Paste the resized face back into a black canvas for consistency
            # This creates a "masked" effect where only the face is visible on black background
            # For simplicity, we create a full black image then paste the resized face in center
            final_masked_img = np.zeros((FACE_CROP_SIZE[0], FACE_CROP_SIZE[1], 3), dtype=np.uint8)
            final_masked_img[0:FACE_CROP_SIZE[0], 0:FACE_CROP_SIZE[1]] = resized_face

            masked_face_frames.append(final_masked_img)
        else:
            # Fallback for frames where dummy bbox goes out of bounds or no face found
            # Could be a black frame, or resized full frame, depending on strategy
            masked_face_frames.append(np.zeros((FACE_CROP_SIZE[0], FACE_CROP_SIZE[1], 3), dtype=np.uint8))

    cap.release()
    return masked_face_frames, speaker_person_id
 
def preprocess_and_save_utterance(
    dialogue_id, utterance_id, speaker_name, video_path,
    preprocessed_output_dir, face_detector
):
    """
    Processes a single video (utterance), extracts masked face frames and audio,
    and saves them to disk.
    """
    base_filename = f"dia{dialogue_id}_utt{utterance_id}"
    masked_faces_dir = os.path.join(preprocessed_output_dir, 'masked_faces')#, str(dialogue_id))
    audio_chunks_dir = os.path.join(preprocessed_output_dir, 'audio_chunks')#, str(dialogue_id))

    # os.makedirs(masked_faces_dir, exist_ok=True)
    # os.makedirs(audio_chunks_dir, exist_ok=True)

    # --- Visual Preprocessing ---
    masked_face_frames, speaker_person_id = get_speaker_masked_face_and_id(
        video_path, speaker_name, face_detector
    )

    if not masked_face_frames:
        print(f"Skipping {base_filename}: No masked face frames extracted.")
        return None # Indicate failure

    # Save masked face frames (e.g., as individual JPEGs or a numpy array/tensor file)
    # For now, let's save as a single .npy file containing all frames
    masked_frames_filepath = os.path.join(masked_faces_dir, f"{base_filename}_frames.npy")
    np.save(masked_frames_filepath, np.array(masked_face_frames, dtype=np.uint8))

    # --- Audio Preprocessing ---
    audio_data = np.array([]) # Initialize as empty
    try:
        audio_segment = AudioSegment.from_file(video_path)

        # Convert to desired sample rate and mono
        audio_segment = audio_segment.set_frame_rate(AUDIO_SAMPLE_RATE).set_channels(1)

        # Convert to numpy array (pydub stores in milliseconds)
        # This converts to raw samples, then normalizes to float range [-1, 1]
        audio_data = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        audio_data /= (1 << (audio_segment.sample_width * 8 - 1)) # Normalize to float range

    except Exception as e:
        print(f"Error loading audio with pydub for {video_path}: {e}. Returning dummy audio.")
        audio_data = np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32) # 1 second of silence

    # audio_data, sr = librosa.load(video_path, sr=AUDIO_SAMPLE_RATE, mono=True)
    audio_filepath = os.path.join(audio_chunks_dir, f"{base_filename}_audio.npy")
    np.save(audio_filepath, audio_data)
    # # try:
    #     audio_data, sr = librosa.load(video_path, sr=AUDIO_SAMPLE_RATE, mono=True)
    #     audio_filepath = os.path.join(audio_chunks_dir, f"{base_filename}_audio.npy")
    #     np.save(audio_filepath, audio_data)
    # except Exception as e:
    #     print(f"Error processing audio for {base_filename}: {e}")
    #     return None

    return {
        'dialogue_id': dialogue_id,
        'utterance_id': utterance_id,
        'speaker_id': speaker_person_id,
        'masked_frames_path': masked_frames_filepath,
        'audio_path': audio_filepath
    }


# --- Dataset Class for Stage 1 Fine-tuning (modified to load preprocessed data) ---
class UtteranceEmotionDataset(Dataset):
    def __init__(self, annotations_df, preprocessed_data_dir, emotion_map, sentiment_map, dataset_size_limit=None):

        # Limit dataset size if specified
        if dataset_size_limit:
            self.annotations = annotations_df.head(dataset_size_limit).copy()
        else:
            self.annotations = annotations_df.copy()

        self.preprocessed_data_dir = preprocessed_data_dir
        self.emotion_map = emotion_map
        self.sentiment_map = sentiment_map

        # Map original annotations to the paths of preprocessed files
        self.data_paths = []
        for idx, row in self.annotations.iterrows():
            dialogue_id = row['Dialogue_ID']
            utterance_id = row['Utterance_ID']
            base_filename = f"dia{dialogue_id}_utt{utterance_id}"

            masked_frames_filepath = os.path.join(preprocessed_data_dir, 'masked_faces', str(dialogue_id), f"{base_filename}_frames.npy")
            audio_filepath = os.path.join(preprocessed_data_dir, 'audio_chunks', str(dialogue_id), f"{base_filename}_audio.npy")

            if os.path.exists(masked_frames_filepath) and os.path.exists(audio_filepath):
                self.data_paths.append({
                    'dialogue_id': dialogue_id,
                    'utterance_id': utterance_id,
                    'speaker_name': row['Speaker'], # Keep for person_id during __getitem__
                    'emotion_label': row['Emotion'],
                    'sentiment_label': row['Sentiment'],
                    'masked_frames_path': masked_frames_filepath,
                    'audio_path': audio_filepath
                })
            else:
                print(f"Warning: Preprocessed files not found for D:{dialogue_id}, U:{utterance_id}. Skipping.")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        item_data = self.data_paths[idx]

        # Load preprocessed data
        cropped_frames_np = np.load(item_data['masked_frames_path']) # (T, H, W, C)
        audio_chunk_np = np.load(item_data['audio_path']) # (num_samples,)

        # Convert to PyTorch tensors
        # Normalize frames (already uint8, so divide by 255.0)
        cropped_frames_tensor = torch.from_numpy(cropped_frames_np).float().permute(0, 3, 1, 2) / 255.0
        audio_chunk_tensor = torch.from_numpy(audio_chunk_np).float()

        # Map labels to numerical IDs
        emotion_id = self.emotion_map.get(item_data['emotion_label'], -1)
        sentiment_id = self.sentiment_map.get(item_data['sentiment_label'], -1)

        # Re-generate speaker_id based on name, or load if saved
        # This is for consistency, as speaker_id might not be saved with frames/audio
        speaker_person_id = hash(item_data['speaker_name']) % 1000

        return {
            'cropped_frames': cropped_frames_tensor,
            'audio_chunk': audio_chunk_tensor,
            'speaker_id': speaker_person_id,
            'emotion_label_id': torch.tensor(emotion_id, dtype=torch.long),
            'sentiment_label_id': torch.tensor(sentiment_id, dtype=torch.long),
            'dialogue_id': item_data['dialogue_id'],
            'utterance_id': item_data['utterance_id']
        }
 
def collate_fn_utterance_emotion(batch):
    max_frames = max(item['cropped_frames'].shape[0] for item in batch)
    max_audio_len = max(item['audio_chunk'].shape[0] for item in batch)

    padded_frames = []
    padded_audio_chunks = []

    for item in batch:
        num_frames = item['cropped_frames'].shape[0]
        padding_frames = torch.zeros(max_frames - num_frames, *item['cropped_frames'].shape[1:])
        padded_frames.append(torch.cat((item['cropped_frames'], padding_frames), dim=0))

        audio_len = item['audio_chunk'].shape[0]
        padding_audio = torch.zeros(max_audio_len - audio_len)
        padded_audio_chunks.append(torch.cat((item['audio_chunk'], padding_audio), dim=0))

    return {
        'cropped_frames': torch.stack(padded_frames),
        'audio_chunk': torch.stack(padded_audio_chunks),
        'speaker_ids': torch.tensor([item['speaker_id'] for item in batch], dtype=torch.long),
        'emotion_label_ids': torch.stack([item['emotion_label_id'] for item in batch]),
        'sentiment_label_ids': torch.stack([item['sentiment_label_id'] for item in batch]),
        'dialogue_ids': torch.tensor([item['dialogue_id'] for item in batch], dtype=torch.long),
        'utterance_ids': torch.tensor([item['utterance_id'] for item in batch], dtype=torch.long)
    }

 
if __name__ == "__main__":
    # --- Configuration for Preprocessing Run ---
    PREPROCESS_LIMIT = 533 # Select first 534 MP4s for preprocessing
    print("Preprocessing limit set to: ", PREPROCESS_LIMIT)
    # Initialize a dummy face detector (REPLACE with actual deep learning detector like MTCNN, MediaPipe, etc.)
    # For a real system, you'd load a robust face detection model here.
    # For this example, it's just a placeholder to keep the `face_detector` argument.
    dummy_face_detector = None

    os.makedirs(PREPROCESSED_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(PREPROCESSED_OUTPUT_DIR, 'masked_faces'), exist_ok=True)
    os.makedirs(os.path.join(PREPROCESSED_OUTPUT_DIR, 'audio_chunks'), exist_ok=True)

    # Load your annotations
    annotations_df = pd.read_csv(ANNOTATIONS_CSV)

    # Select first N videos for preprocessing
    videos_to_process = annotations_df.head(PREPROCESS_LIMIT)
    # print(videos_to_process)
    print(f"Starting preprocessing for the first {len(videos_to_process)} utterances...")

    processed_count = 0

    for idx, row in tqdm(videos_to_process.iterrows(), total=len(videos_to_process)):
        print(row, idx)
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']
        speaker_name = row['Speaker']

        # print(dialogue_id, utterance_id, speaker_name)

        # Construct video path (ensure your naming convention matches)
        video_filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_filename)

        if not os.path.exists(video_path):
            print(f"Skipping D:{dialogue_id}, U:{utterance_id}: Video file not found at {video_path}")
            continue

        result = preprocess_and_save_utterance(
            dialogue_id, utterance_id, speaker_name, video_path,
            PREPROCESSED_OUTPUT_DIR, dummy_face_detector # Pass the dummy detector
        )
        if result:
            processed_count += 1

    print(f"Preprocessing complete. Successfully processed and saved {processed_count} utterances.")
    print(f"Processed data saved to: {PREPROCESSED_OUTPUT_DIR}")

    # --- Example of how to load processed data for Stage 1 fine-tuning ---
    # print("\n--- Example: Loading preprocessed data for Stage 1 fine-tuning ---")
    # emotion_labels = annotations_df['Emotion'].unique().tolist()
    # sentiment_labels = annotations_df['Sentiment'].unique().tolist()

    # emotion_map = {label: i for i, label in enumerate(emotion_labels)}
    # sentiment_map = {label: i for i, label in enumerate(sentiment_labels)}

    # # Initialize Dataset with a limit, which will now load from saved files
    # # The limit ensures it only attempts to load the number of videos that were preprocessed
    # stage1_dataset = UtteranceEmotionDataset(annotations_df, PREPROCESSED_OUTPUT_DIR, emotion_map, sentiment_map, dataset_size_limit=PREPROCESS_LIMIT)
    # stage1_dataloader = DataLoader(stage1_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_utterance_emotion, num_workers=2)

    # print(f"DataLoader created for Stage 1 with {len(stage1_dataset)} preprocessed utterances.")
    # # You would then proceed with your Stage 1 model initialization and training loop here.
    # # For example, iterate one batch:
    # # if len(stage1_dataset) > 0:
    # #     sample_batch = next(iter(stage1_dataloader))
    # #     print(f"Sample batch loaded successfully. Cropped frames shape: {sample_batch['cropped_frames'].shape}")
    # #     print(f"Sample batch audio shape: {sample_batch['audio_chunk'].shape}")
