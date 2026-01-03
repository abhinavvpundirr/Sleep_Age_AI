import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from scipy import signal
from pathlib import Path

# Try to import required libraries
try:
    import mne
    import yasa
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: Install with: pip install mne yasa")


# ============================================================
# EXACT 24 FEATURES (in order from your training)
# ============================================================

FEATURE_NAMES = [
    'delta_abs',
    'delta_rel',
    'theta_abs',
    'theta_rel',
    'alpha_abs',
    'alpha_rel',
    'sigma_abs',
    'sigma_rel',
    'beta_abs',
    'beta_rel',
    'theta_alpha_ratio',
    'delta_beta_ratio',
    'W_pct',
    'N1_pct',
    'N2_pct',
    'N3_pct',
    'R_pct',
    'sleep_efficiency',
    'fragmentation',
    'spindle_density',
    'spindle_amplitude',
    'spindle_duration',
    'signal_std',
    'signal_skew'
]

N_FEATURES = 24  # Your model uses 24 features


# ============================================================
# STEP 1: Feature Extractor (EXACT COPY FROM YOUR TRAINING)
# ============================================================

class SleepFeatureExtractor:
    """
    Extracts 24 features from sleep EEG for brain age prediction.
    THIS IS IDENTICAL TO YOUR TRAINING CODE.
    """
    
    def __init__(self, sfreq=100):
        self.sfreq = sfreq
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'sigma': (12, 16),
            'beta': (16, 30)
        }
    
    def load_and_preprocess(self, edf_path):
        """Load EDF file and apply preprocessing."""
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch]
        if eeg_channels:
            raw.pick_channels(eeg_channels[:1])
        
        if raw.info['sfreq'] != self.sfreq:
            raw.resample(self.sfreq, verbose=False)
        
        raw.filter(0.3, 35, verbose=False)
        return raw
    
    def extract_bandpower(self, data):
        """Extract power in each frequency band using Welch method."""
        features = {}
        
        freqs, psd = signal.welch(data, self.sfreq, nperseg=self.sfreq * 4)
        total_power = np.trapz(psd, freqs) + 1e-10
        
        for band, (low, high) in self.freq_bands.items():
            mask = (freqs >= low) & (freqs <= high)
            power = np.trapz(psd[mask], freqs[mask])
            features[f'{band}_abs'] = power
            features[f'{band}_rel'] = power / total_power
        
        features['theta_alpha_ratio'] = features['theta_abs'] / (features['alpha_abs'] + 1e-10)
        features['delta_beta_ratio'] = features['delta_abs'] / (features['beta_abs'] + 1e-10)
        
        return features
    
    def extract_sleep_features(self, data):
        """Extract sleep stage features using YASA."""
        features = {}
        
        try:
            sls = yasa.SleepStaging(
                data.reshape(1, -1),
                sf=self.sfreq,
                ch_names=['EEG']
            )
            stages = sls.predict()
            
            total = len(stages)
            for stage in ['W', 'N1', 'N2', 'N3', 'R']:
                features[f'{stage}_pct'] = np.sum(stages == stage) / total
            
            features['sleep_efficiency'] = 1 - features['W_pct']
            
            transitions = np.sum(stages[:-1] != stages[1:])
            features['fragmentation'] = transitions / total
            
        except Exception:
            for stage in ['W', 'N1', 'N2', 'N3', 'R']:
                features[f'{stage}_pct'] = 0.2
            features['sleep_efficiency'] = 0.85
            features['fragmentation'] = 0.1
        
        return features
    
    def extract_spindle_features(self, data):
        """Detect and characterize sleep spindles."""
        features = {}
        
        try:
            sp = yasa.spindles_detect(data, self.sfreq, verbose=False)
            
            if sp is not None and len(sp.summary()) > 0:
                summary = sp.summary()
                duration_minutes = len(data) / self.sfreq / 60
                features['spindle_density'] = len(summary) / duration_minutes
                features['spindle_amplitude'] = summary['Amplitude'].mean()
                features['spindle_duration'] = summary['Duration'].mean()
            else:
                features['spindle_density'] = 0
                features['spindle_amplitude'] = 0
                features['spindle_duration'] = 0
                
        except Exception:
            features['spindle_density'] = 0
            features['spindle_amplitude'] = 0
            features['spindle_duration'] = 0
        
        return features
    
    def extract_all_features(self, edf_path):
        """Extract ALL 24 features from an EDF file."""
        raw = self.load_and_preprocess(edf_path)
        
        # CONVERT TO MICROVOLTS (YASA expects microvolts)
        data = raw.get_data().flatten() * 1e6
        
        # Take middle 2 hours
        samples_2h = 2 * 60 * 60 * self.sfreq
        if len(data) > samples_2h:
            start = (len(data) - samples_2h) // 2
            data = data[start:start + samples_2h]
        
        features = {}
        features.update(self.extract_bandpower(data))
        features.update(self.extract_sleep_features(data))
        features.update(self.extract_spindle_features(data))
        
        features['signal_std'] = np.std(data)
        features['signal_skew'] = float(np.mean(((data - np.mean(data)) / (np.std(data) + 1e-10)) ** 3))
        
        return features


# ============================================================
# STEP 2: Neural Network (24 features -> 64 -> 32 -> 16 -> 1)
# ============================================================

class BrainAgeModel(nn.Module):
    """
    Neural network for brain age prediction.
    Architecture: 24 features -> 64 -> 32 -> 16 -> 1 (age)
    """
    def __init__(self, n_features=24):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


# ============================================================
# STEP 3: Load Model and Scaler
# ============================================================

def load_model_and_scaler():
    """Load the trained model and feature scaler."""
    model = BrainAgeModel(n_features=N_FEATURES)
    
    try:
        checkpoint = torch.load(
            'brain_age_model.pth', 
            map_location='cpu',
            weights_only=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler_mean = checkpoint['scaler_mean']
        scaler_std = checkpoint['scaler_std']
        feature_names = checkpoint.get('feature_names', FEATURE_NAMES)
        model.eval()
        print("[OK] Model loaded successfully!")
        return model, scaler_mean, scaler_std, feature_names
    except FileNotFoundError:
        print("[WARNING] Model not found. Using untrained model.")
        return model, None, None, FEATURE_NAMES
    except Exception as e:
        print(f"[WARNING] Error loading model: {e}")
        return model, None, None, FEATURE_NAMES

model, scaler_mean, scaler_std, feature_names = load_model_and_scaler()
extractor = SleepFeatureExtractor()


# ============================================================
# STEP 4: Prediction Function
# ============================================================

def predict_brain_age(edf_file, actual_age):
    """
    Main prediction function:
    1. Extract 24 features from EDF
    2. Normalize features
    3. Predict brain age
    4. Calculate brain age gap
    """
    
    if edf_file is None:
        return "[ERROR] Please upload an EDF file", "", ""
    
    if not MNE_AVAILABLE:
        return "[ERROR] Required libraries not installed. Run: pip install mne yasa", "", ""
    
    try:
        # Step 1: Extract features
        features = extractor.extract_all_features(edf_file.name)
        
        # Step 2: Convert to array in correct order
        X = np.array([[features[name] for name in FEATURE_NAMES]])
        
        # Step 3: Normalize
        if scaler_mean is not None and scaler_std is not None:
            X_scaled = (X - scaler_mean) / (scaler_std + 1e-10)
        else:
            X_scaled = X
        
        # Step 4: Predict
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            predicted_age = model(X_tensor).item()
        
        # Step 5: Calculate brain age gap
        brain_age_gap = predicted_age - actual_age
        
        # Format outputs
        summary = format_summary(predicted_age, actual_age, brain_age_gap)
        features_text = format_features(features)
        interpretation = format_interpretation(brain_age_gap)
        
        return summary, features_text, interpretation
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"[ERROR] {str(e)}\n\nDetails:\n```\n{error_details}\n```", "", ""


def format_summary(predicted, actual, gap):
    """Format the main results."""
    text = "## Brain Age Prediction Results\n\n"
    text += "| Metric | Value |\n"
    text += "|--------|-------|\n"
    text += f"| **Predicted Brain Age** | **{predicted:.1f} years** |\n"
    text += f"| Actual (Chronological) Age | {actual} years |\n"
    
    gap_sign = "+" if gap > 0 else ""
    text += f"| **Brain Age Gap** | **{gap_sign}{gap:.1f} years** |\n"
    
    return text


def format_features(features):
    """Format extracted features."""
    text = "## Extracted Features (24 total)\n\n"
    
    text += "### Power Spectrum (12 features)\n"
    text += "| Band | Frequency | Absolute | Relative |\n"
    text += "|------|-----------|----------|----------|\n"
    bands_info = {
        'delta': '0.5-4 Hz',
        'theta': '4-8 Hz', 
        'alpha': '8-12 Hz',
        'sigma': '12-16 Hz',
        'beta': '16-30 Hz'
    }
    for band, freq in bands_info.items():
        abs_val = features.get(f'{band}_abs', 0)
        rel_val = features.get(f'{band}_rel', 0)
        text += f"| {band.capitalize()} | {freq} | {abs_val:.2f} | {rel_val:.1%} |\n"
    
    text += f"\n**Theta/Alpha Ratio:** {features.get('theta_alpha_ratio', 0):.2f}\n"
    text += f"\n**Delta/Beta Ratio:** {features.get('delta_beta_ratio', 0):.2f}\n"
    
    text += "\n### Sleep Architecture (7 features)\n"
    text += "| Stage | Percentage |\n"
    text += "|-------|------------|\n"
    stage_names = {'W': 'Wake', 'N1': 'N1', 'N2': 'N2', 'N3': 'N3 (Deep)', 'R': 'REM'}
    for stage, name in stage_names.items():
        pct = features.get(f'{stage}_pct', 0)
        text += f"| {name} | {pct:.1%} |\n"
    
    text += f"\n**Sleep Efficiency:** {features.get('sleep_efficiency', 0):.1%}\n"
    text += f"\n**Fragmentation:** {features.get('fragmentation', 0):.3f}\n"
    
    text += "\n### Spindle Features (3 features)\n"
    text += f"- **Density:** {features.get('spindle_density', 0):.2f} /min\n"
    text += f"- **Amplitude:** {features.get('spindle_amplitude', 0):.2f} uV\n"
    text += f"- **Duration:** {features.get('spindle_duration', 0):.2f} sec\n"
    
    text += "\n### Signal Statistics (2 features)\n"
    text += f"- **Std Dev:** {features.get('signal_std', 0):.2f}\n"
    text += f"- **Skewness:** {features.get('signal_skew', 0):.3f}\n"
    
    return text


def format_interpretation(gap):
    """Interpret the brain age gap."""
    text = "## Interpretation\n\n"
    
    if gap < -5:
        verdict = "Excellent - Brain appears significantly younger"
        detail = "This suggests healthy brain aging."
    elif gap < -2:
        verdict = "Good - Brain slightly younger than expected"
        detail = "Your brain appears to be aging well."
    elif gap <= 2:
        verdict = "Normal - Brain age matches chronological age"
        detail = "This is a typical result."
    elif gap <= 5:
        verdict = "Slightly Elevated"
        detail = "Consider lifestyle factors: sleep, exercise, diet."
    else:
        verdict = "Elevated - Worth monitoring"
        detail = "Consider discussing with a healthcare provider."
    
    text += f"### {verdict}\n\n"
    text += f"{detail}\n\n"
    text += "---\n"
    text += "*Note: This is a research tool, not clinical diagnosis.*"
    
    return text


# ============================================================
# STEP 5: Gradio Interface
# ============================================================

def create_app():
    with gr.Blocks(
        title="Brain Age Predictor",
        theme=gr.themes.Soft(primary_hue="blue")
    ) as app:
        
        gr.Markdown("""
        # Brain Age Predictor
        
        Upload an EDF file containing sleep EEG to predict biological brain age.
        
        ---
        
        ### Required EDF File Format
        
        | Requirement | Details |
        |-------------|---------|
        | EEG Channel | At least one channel with 'EEG' in the name |
        | Examples | 'EEG Fpz-Cz', 'EEG Pz-Oz', 'EEG C3-A2' |
        | Sampling Rate | Any (will be resampled to 100 Hz) |
        | Duration | Minimum 2 hours recommended |
        
        ### Download Test Files
        
        Get free EDF files from Sleep-EDF database:
        - https://physionet.org/content/sleep-edfx/1.0.0/
        - Download files ending in '-PSG.edf'
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                
                file_input = gr.File(
                    label="Upload EDF File (.edf)",
                    file_types=[".edf"],
                    type="filepath"
                )
                
                age_input = gr.Number(
                    label="Actual Age (years)",
                    value=35,
                    minimum=18,
                    maximum=100
                )
                
                predict_btn = gr.Button(
                    "Predict Brain Age",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                ---
                ### Model Info
                
                - **Features:** 24
                - **Architecture:** 24 -> 64 -> 32 -> 16 -> 1
                - **Output:** Predicted age (years)
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("### Results")
                summary_output = gr.Markdown()
                
                with gr.Row():
                    features_output = gr.Markdown()
                    interpretation_output = gr.Markdown()
        
        predict_btn.click(
            fn=predict_brain_age,
            inputs=[file_input, age_input],
            outputs=[summary_output, features_output, interpretation_output]
        )
        
        gr.Markdown("""
        ---
        *For research demonstration only. Not for clinical diagnosis.*
        """)
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)