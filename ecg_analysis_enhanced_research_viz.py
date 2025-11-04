

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.stats import entropy
import os
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
SAMPLING_RATE = 360  # Hz

# DS1/DS2 INTER-PATIENT SPLIT
DS1_TRAIN = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 
             201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2_TEST = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 
            213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# AAMI EC57 Standard
AAMI_CLASSES = {
    'N': ['N', 'L', 'R', 'e', 'j'],
    'S': ['A', 'a', 'J', 'S'],
    'V': ['V', 'E'],
    'F': ['F'],
    'Q': ['/', 'f', 'Q']
}

OUTPUT_DIR = 'ecg_research_visualizations'

# ============================================================================
# COPY OF YOUR FFT FEATURE EXTRACTION (UNCHANGED)
# ============================================================================
def extract_frequency_band_power(fft_magnitude, freqs, band_low, band_high):
    """Extract power in a specific frequency band"""
    band_mask = (freqs >= band_low) & (freqs <= band_high)
    if np.sum(band_mask) == 0:
        return 0.0
    band_power = np.sum(fft_magnitude[band_mask] ** 2)
    return band_power

def extract_enhanced_fft_features(beat_segment, sampling_rate=360):
    """Your enhanced FFT feature extraction"""
    segment_norm = (beat_segment - np.mean(beat_segment)) / (np.std(beat_segment) + 1e-8)
    window = np.hanning(len(segment_norm))
    windowed = segment_norm * window
    N = len(windowed)
    yf = rfft(windowed)
    xf = rfftfreq(N, 1/sampling_rate)
    magnitude = np.abs(yf)
    magnitude = magnitude / (N/2)
    
    features = []
    
    # Frequency bands
    p_wave_power = extract_frequency_band_power(magnitude, xf, 5, 15)
    qrs_power = extract_frequency_band_power(magnitude, xf, 10, 40)
    t_wave_power = extract_frequency_band_power(magnitude, xf, 1, 7)
    low_freq_power = extract_frequency_band_power(magnitude, xf, 0.5, 5)
    mid_freq_power = extract_frequency_band_power(magnitude, xf, 5, 25)
    high_freq_power = extract_frequency_band_power(magnitude, xf, 25, 60)
    
    features.extend([p_wave_power, qrs_power, t_wave_power, 
                    low_freq_power, mid_freq_power, high_freq_power])
    
    # Power ratios
    total_power = np.sum(magnitude ** 2) + 1e-10
    features.extend([
        qrs_power / total_power,
        low_freq_power / (mid_freq_power + 1e-10),
        high_freq_power / total_power,
    ])
    
    # Dominant frequency
    relevant_range = (xf >= 1) & (xf <= 60)
    if np.sum(relevant_range) > 0:
        relevant_magnitude = magnitude[relevant_range]
        relevant_freqs = xf[relevant_range]
        if len(relevant_magnitude) > 0:
            dom_idx = np.argmax(relevant_magnitude)
            dominant_freq = relevant_freqs[dom_idx]
            dominant_power = relevant_magnitude[dom_idx]
            harmonic_2_power = 0
            harmonic_3_power = 0
            harm2_mask = (xf >= dominant_freq * 1.8) & (xf <= dominant_freq * 2.2)
            if np.sum(harm2_mask) > 0:
                harmonic_2_power = np.max(magnitude[harm2_mask])
            harm3_mask = (xf >= dominant_freq * 2.8) & (xf <= dominant_freq * 3.2)
            if np.sum(harm3_mask) > 0:
                harmonic_3_power = np.max(magnitude[harm3_mask])
        else:
            dominant_freq = 0
            dominant_power = 0
            harmonic_2_power = 0
            harmonic_3_power = 0
    else:
        dominant_freq = 0
        dominant_power = 0
        harmonic_2_power = 0
        harmonic_3_power = 0
    
    features.extend([dominant_freq, dominant_power, harmonic_2_power, 
                    harmonic_3_power, harmonic_2_power / (dominant_power + 1e-10)])
    
    # Spectral statistics
    if np.sum(magnitude) > 0:
        spectral_centroid = np.sum(xf * magnitude) / np.sum(magnitude)
        spectral_spread = np.sqrt(np.sum(((xf - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude))
    else:
        spectral_centroid = 0
        spectral_spread = 0
    
    magnitude_normalized = magnitude / (np.sum(magnitude) + 1e-10)
    spectral_entropy = entropy(magnitude_normalized + 1e-10)
    
    cumsum = np.cumsum(magnitude ** 2)
    if cumsum[-1] > 0:
        threshold = 0.85 * cumsum[-1]
        rolloff_idx = np.where(cumsum >= threshold)[0]
        spectral_rolloff = xf[rolloff_idx[0]] if len(rolloff_idx) > 0 else xf[-1]
    else:
        spectral_rolloff = 0
    
    features.extend([spectral_centroid, spectral_spread, spectral_entropy, spectral_rolloff])
    
    # FFT coefficients
    n_coeffs = 20
    max_freq_idx = min(len(magnitude), int(60 * N / sampling_rate))
    if max_freq_idx > n_coeffs:
        indices = np.linspace(0, max_freq_idx-1, n_coeffs, dtype=int)
        sampled_coeffs = magnitude[indices]
    else:
        sampled_coeffs = magnitude[:n_coeffs]
        if len(sampled_coeffs) < n_coeffs:
            sampled_coeffs = np.pad(sampled_coeffs, (0, n_coeffs - len(sampled_coeffs)), 'constant')
    
    features.extend(sampled_coeffs.tolist())
    return np.array(features), xf, magnitude

def map_to_aami_class(beat_type):
    """Map beat annotation to AAMI class"""
    for aami_class, beat_types in AAMI_CLASSES.items():
        if beat_type in beat_types:
            return aami_class
    return 'Q'

# ============================================================================
# SIGNAL PROCESSING (YOUR CODE)
# ============================================================================
def assess_signal_quality(ecg_signal, sampling_rate=360):
    """Signal quality assessment"""
    if np.max(ecg_signal) == np.min(ecg_signal):
        return 0.0, False
    signal_range = np.max(ecg_signal) - np.min(ecg_signal)
    if signal_range < 50:
        return 0.2, False
    signal_power = np.mean(ecg_signal ** 2)
    noise_estimate = np.std(np.diff(ecg_signal))
    snr = 10 * np.log10(signal_power / (noise_estimate ** 2 + 1e-10)) if noise_estimate != 0 else 100
    quality_score = 0.0
    if snr > 15:
        quality_score += 0.5
    elif snr > 10:
        quality_score += 0.3
    if signal_range > 200:
        quality_score += 0.3
    elif signal_range > 100:
        quality_score += 0.2
    if np.std(ecg_signal) > 50:
        quality_score += 0.2
    is_acceptable = quality_score > 0.5
    return quality_score, is_acceptable

def preprocess_signal(ecg_signal, sampling_rate=360):
    """Enhanced preprocessing"""
    sos_hp = signal.butter(4, 0.67, btype='high', fs=sampling_rate, output='sos')
    signal_hp = signal.sosfilt(sos_hp, ecg_signal)
    b_notch, a_notch = signal.iirnotch(60, 30, sampling_rate)
    signal_notch = signal.filtfilt(b_notch, a_notch, signal_hp)
    noise_level = np.std(np.diff(signal_notch))
    if noise_level > 10:
        signal_notch = gaussian_filter1d(signal_notch, sigma=1)
    return signal_notch

def detect_r_peaks_enhanced(ecg_signal, sampling_rate=360, invert=False):
    """Enhanced Pan-Tompkins"""
    if invert:
        ecg_signal = -ecg_signal
    sos = signal.butter(4, [5, 18], btype='band', fs=sampling_rate, output='sos')
    filtered = signal.sosfilt(sos, ecg_signal)
    derivative = np.diff(filtered)
    derivative = np.append(derivative, 0)
    squared = derivative ** 2
    window_size = int(0.15 * sampling_rate)
    moving_avg = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    signal_peaks, _ = signal.find_peaks(moving_avg)
    if len(signal_peaks) > 0:
        peak_values = moving_avg[signal_peaks]
        threshold = np.percentile(peak_values, 10)
        threshold = max(threshold, 0.03 * np.max(moving_avg))
    else:
        threshold = 0.08 * np.max(moving_avg)
    min_distance = int(0.25 * sampling_rate)
    peaks, _ = signal.find_peaks(moving_avg, height=threshold, distance=min_distance, prominence=threshold * 0.1)
    refined_peaks = []
    search_window = int(0.08 * sampling_rate)
    for peak in peaks:
        start = max(0, peak - search_window)
        end = min(len(filtered), peak + search_window)
        local_segment = filtered[start:end]
        if len(local_segment) > 0:
            local_max_idx = np.argmax(np.abs(local_segment))
            refined_peak = start + local_max_idx
            refined_peaks.append(refined_peak)
    return np.array(refined_peaks)

def detect_r_peaks_multi_strategy(ecg_signal, sampling_rate=360):
    """Multi-strategy detection"""
    peaks_normal = detect_r_peaks_enhanced(ecg_signal, sampling_rate, invert=False)
    peaks_inverted = detect_r_peaks_enhanced(ecg_signal, sampling_rate, invert=True)
    peaks_absolute = detect_r_peaks_enhanced(np.abs(ecg_signal), sampling_rate, invert=False)
    duration_sec = len(ecg_signal) / sampling_rate
    expected_min = duration_sec * 0.67
    expected_max = duration_sec * 3.0
    candidates = [
        (len(peaks_normal), peaks_normal),
        (len(peaks_inverted), peaks_inverted),
        (len(peaks_absolute), peaks_absolute)
    ]
    valid = [(count, peaks) for count, peaks in candidates if expected_min <= count <= expected_max]
    if not valid:
        valid = candidates
    best = max(valid, key=lambda x: x[0])
    return best[1]

# ============================================================================
# NEW COMPREHENSIVE VISUALIZATION FUNCTIONS
# ============================================================================

def create_comprehensive_patient_visualizations(patient_id, patient_dir, signal_raw, signal_processed,
                                                detected_peaks, annotations, beat_features_df):
    """
    Create a comprehensive set of research visualizations for each patient
    """
    print(f"  Creating comprehensive visualizations...")
    
    # 1. MULTI-PANEL ECG ANALYSIS
    create_multipanel_ecg_analysis(patient_id, patient_dir, signal_raw, signal_processed, 
                                   detected_peaks, annotations)
    
    # 2. FFT SPECTRUM ANALYSIS
    create_fft_spectrum_analysis(patient_id, patient_dir, signal_processed, detected_peaks)
    
    # 3. BEAT MORPHOLOGY CLUSTERING
    create_beat_morphology_clustering(patient_id, patient_dir, signal_processed, 
                                     detected_peaks, annotations)
    
    # 4. HRV ANALYSIS
    create_hrv_analysis(patient_id, patient_dir, detected_peaks)
    
    # 5. RR INTERVAL ANALYSIS
    create_rr_interval_analysis(patient_id, patient_dir, detected_peaks)
    
    # 6. FREQUENCY BAND ANALYSIS
    create_frequency_band_analysis(patient_id, patient_dir, beat_features_df)
    
    # 7. FEATURE HEATMAP
    create_feature_heatmap(patient_id, patient_dir, beat_features_df)
    
    # 8. SIGNAL QUALITY TIMELINE
    create_signal_quality_timeline(patient_id, patient_dir, signal_raw, signal_processed)
    
    # 9. 3D FEATURE SPACE
    create_3d_feature_space(patient_id, patient_dir, beat_features_df)
    
    # 10. BEAT TYPE DISTRIBUTION
    create_beat_type_distribution(patient_id, patient_dir, annotations, beat_features_df)

def create_multipanel_ecg_analysis(patient_id, patient_dir, signal_raw, signal_processed, 
                                   detected_peaks, annotations):
    """
    6-panel comprehensive ECG signal analysis
    """
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    time_axis = np.arange(len(signal_raw)) / SAMPLING_RATE
    plot_duration = min(30, len(signal_raw) / SAMPLING_RATE)  # 30 seconds
    plot_samples = int(plot_duration * SAMPLING_RATE)
    
    # Panel 1: Raw ECG Signal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_axis[:plot_samples], signal_raw[:plot_samples], 'k-', linewidth=0.6, alpha=0.8)
    ax1.set_title(f'Patient {patient_id}: Raw ECG Signal', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (mV)')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Filtered Signal with R-peaks
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_axis[:plot_samples], signal_processed[:plot_samples], 'b-', linewidth=0.6)
    det_mask = detected_peaks < plot_samples
    ax2.scatter(detected_peaks[det_mask] / SAMPLING_RATE, 
               signal_processed[detected_peaks[det_mask]], 
               color='red', s=80, marker='o', edgecolors='darkred', linewidths=2, 
               label='R-peaks', zorder=5)
    ax2.set_title('Preprocessed Signal + R-peak Detection', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Annotated Beat Types
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time_axis[:plot_samples], signal_processed[:plot_samples], 'k-', linewidth=0.5, alpha=0.5)
    
    # Color code by AAMI class
    colors_aami = {'N': 'green', 'V': 'red', 'S': 'blue', 'F': 'orange', 'Q': 'gray'}
    ann_mask = annotations['Sample'] < plot_samples
    for _, row in annotations[ann_mask].iterrows():
        aami_class = map_to_aami_class(row['Type'])
        color = colors_aami.get(aami_class, 'gray')
        ax3.axvline(x=row['Sample']/SAMPLING_RATE, color=color, alpha=0.6, 
                   linewidth=2, linestyle='--', label=aami_class if aami_class not in ax3.get_legend_handles_labels()[1] else "")
    
    ax3.set_title('Beat Classification (AAMI Classes)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Heart Rate Over Time
    ax4 = fig.add_subplot(gs[1, 1])
    if len(detected_peaks) > 1:
        rr_intervals = np.diff(detected_peaks) / SAMPLING_RATE
        heart_rates = 60 / rr_intervals
        hr_times = detected_peaks[1:] / SAMPLING_RATE
        mask = hr_times < plot_duration
        ax4.plot(hr_times[mask], heart_rates[mask], 'r-', linewidth=1.5)
        ax4.axhline(y=np.mean(heart_rates), color='b', linestyle='--', linewidth=2, label=f'Mean: {np.mean(heart_rates):.1f} bpm')
        ax4.fill_between(hr_times[mask], 
                        np.mean(heart_rates) - np.std(heart_rates), 
                        np.mean(heart_rates) + np.std(heart_rates), 
                        alpha=0.2, color='blue')
        ax4.set_ylim([max(30, np.mean(heart_rates) - 50), min(200, np.mean(heart_rates) + 50)])
    ax4.set_title('Instantaneous Heart Rate', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Heart Rate (bpm)')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Signal Derivatives (for QRS detection visualization)
    ax5 = fig.add_subplot(gs[2, 0])
    derivative = np.diff(signal_processed)
    derivative = np.append(derivative, 0)
    ax5.plot(time_axis[:plot_samples], derivative[:plot_samples], 'purple', linewidth=0.6)
    ax5.set_title('First Derivative (QRS Detection Feature)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Derivative Amplitude')
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: R-R Interval Trend
    ax6 = fig.add_subplot(gs[2, 1])
    if len(detected_peaks) > 1:
        rr_intervals_ms = np.diff(detected_peaks) * 1000 / SAMPLING_RATE
        ax6.plot(range(len(rr_intervals_ms)), rr_intervals_ms, 'o-', linewidth=1, markersize=4)
        ax6.axhline(y=np.median(rr_intervals_ms), color='r', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(rr_intervals_ms):.1f} ms')
        ax6.set_title('R-R Interval Sequence', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Beat Number')
        ax6.set_ylabel('RR Interval (ms)')
        ax6.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Patient {patient_id}: Comprehensive ECG Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(os.path.join(patient_dir, '01_multipanel_ecg_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

def create_fft_spectrum_analysis(patient_id, patient_dir, signal_processed, detected_peaks):
    """
    Comprehensive FFT spectrum analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Full signal FFT
    N = len(signal_processed)
    yf = rfft(signal_processed)
    xf = rfftfreq(N, 1/SAMPLING_RATE)
    magnitude = np.abs(yf)
    
    # Panel 1: Full spectrum
    ax1 = axes[0, 0]
    ax1.plot(xf, magnitude, 'b-', linewidth=0.8)
    ax1.set_title('Full Signal FFT Spectrum', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude')
    ax1.set_xlim([0, 100])
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Physiological bands highlighted
    ax2 = axes[0, 1]
    ax2.plot(xf, magnitude, 'k-', linewidth=0.6, alpha=0.5)
    # Highlight bands
    ax2.axvspan(1, 7, alpha=0.3, color='blue', label='T-wave (1-7 Hz)')
    ax2.axvspan(5, 15, alpha=0.3, color='green', label='P-wave (5-15 Hz)')
    ax2.axvspan(10, 40, alpha=0.3, color='red', label='QRS (10-40 Hz)')
    ax2.set_title('Physiological Frequency Bands', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_xlim([0, 60])
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Power Spectral Density
    ax3 = axes[1, 0]
    freqs, psd = signal.welch(signal_processed, SAMPLING_RATE, nperseg=1024)
    ax3.semilogy(freqs, psd, 'r-', linewidth=1)
    ax3.set_title('Power Spectral Density (Welch Method)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('PSD (log scale)')
    ax3.set_xlim([0, 60])
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Average beat spectrum
    ax4 = axes[1, 1]
    if len(detected_peaks) > 10:
        beat_spectra = []
        BEAT_WINDOW = 90
        for peak in detected_peaks[:min(50, len(detected_peaks))]:
            start = max(0, peak - BEAT_WINDOW)
            end = min(len(signal_processed), peak + BEAT_WINDOW)
            if end - start == 2 * BEAT_WINDOW:
                beat_segment = signal_processed[start:end]
                _, _, beat_magnitude = extract_enhanced_fft_features(beat_segment, SAMPLING_RATE)
                beat_spectra.append(beat_magnitude)
        
        if beat_spectra:
            min_len = min(len(spec) for spec in beat_spectra)
            beat_spectra_trimmed = [spec[:min_len] for spec in beat_spectra]
            avg_spectrum = np.mean(beat_spectra_trimmed, axis=0)
            std_spectrum = np.std(beat_spectra_trimmed, axis=0)
            freq_range = np.linspace(0, SAMPLING_RATE/2, len(avg_spectrum))
            
            ax4.plot(freq_range, avg_spectrum, 'b-', linewidth=2, label='Mean')
            ax4.fill_between(freq_range, avg_spectrum - std_spectrum, 
                           avg_spectrum + std_spectrum, alpha=0.3, color='blue')
            ax4.set_title('Average Beat Spectrum ± StdDev', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Frequency (Hz)')
            ax4.set_ylabel('Magnitude')
            ax4.set_xlim([0, 60])
            ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Patient {patient_id}: FFT Spectrum Analysis', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(patient_dir, '02_fft_spectrum_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

def create_beat_morphology_clustering(patient_id, patient_dir, signal_processed, 
                                     detected_peaks, annotations):
    """
    Visualize beat morphologies grouped by AAMI class
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    BEAT_WINDOW = 90
    aami_classes = ['N', 'V', 'S', 'F', 'Q']
    colors = {'N': 'green', 'V': 'red', 'S': 'blue', 'F': 'orange', 'Q': 'gray'}
    
    ann_dict = {row['Sample']: row['Type'] for _, row in annotations.iterrows()}
    
    for idx, aami_class in enumerate(aami_classes):
        ax = axes[idx]
        beats_in_class = []
        
        for peak in detected_peaks:
            if peak in ann_dict:
                beat_type = ann_dict[peak]
                if map_to_aami_class(beat_type) == aami_class:
                    start = max(0, peak - BEAT_WINDOW)
                    end = min(len(signal_processed), peak + BEAT_WINDOW)
                    if end - start == 2 * BEAT_WINDOW:
                        beat_segment = signal_processed[start:end]
                        beats_in_class.append(beat_segment)
                    if len(beats_in_class) >= 20:  # Max 20 per class
                        break
        
        if beats_in_class:
            time_axis_beat = np.arange(-BEAT_WINDOW, BEAT_WINDOW) / SAMPLING_RATE * 1000
            
            # Plot all beats with transparency
            for beat in beats_in_class:
                ax.plot(time_axis_beat, beat, color=colors[aami_class], 
                       alpha=0.3, linewidth=0.8)
            
            # Plot average beat
            avg_beat = np.mean(beats_in_class, axis=0)
            ax.plot(time_axis_beat, avg_beat, color=colors[aami_class], 
                   linewidth=3, label=f'Average (n={len(beats_in_class)})')
            
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
            ax.set_title(f'AAMI Class {aami_class}', fontsize=11, fontweight='bold', 
                        color=colors[aami_class])
            ax.set_xlabel('Time from R-peak (ms)')
            ax.set_ylabel('Amplitude')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No {aami_class} beats found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'AAMI Class {aami_class}', fontsize=11, fontweight='bold')
    
    # Use last subplot for legend
    axes[-1].axis('off')
    legend_text = "AAMI Classification:\nN: Normal\nV: Ventricular\nS: Supraventricular\nF: Fusion\nQ: Unclassifiable"
    axes[-1].text(0.1, 0.5, legend_text, fontsize=11, verticalalignment='center')
    
    plt.suptitle(f'Patient {patient_id}: Beat Morphology by AAMI Class', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(patient_dir, '03_beat_morphology_clustering.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

def create_hrv_analysis(patient_id, patient_dir, detected_peaks):
    """
    Heart Rate Variability analysis (time and frequency domain)
    """
    if len(detected_peaks) < 10:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate RR intervals
    rr_intervals = np.diff(detected_peaks) / SAMPLING_RATE * 1000  # in ms
    
    # Panel 1: RR Interval Histogram
    ax1 = axes[0, 0]
    ax1.hist(rr_intervals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(rr_intervals), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(rr_intervals):.1f} ms')
    ax1.axvline(np.median(rr_intervals), color='green', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(rr_intervals):.1f} ms')
    ax1.set_title('RR Interval Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('RR Interval (ms)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Poincaré Plot (RRn vs RRn+1)
    ax2 = axes[0, 1]
    if len(rr_intervals) > 1:
        ax2.scatter(rr_intervals[:-1], rr_intervals[1:], alpha=0.5, s=20, color='purple')
        # Add identity line
        min_rr, max_rr = np.min(rr_intervals), np.max(rr_intervals)
        ax2.plot([min_rr, max_rr], [min_rr, max_rr], 'r--', linewidth=2, label='Identity')
        
        # Calculate SD1 and SD2
        sd1 = np.std(np.subtract(rr_intervals[:-1], rr_intervals[1:]) / np.sqrt(2))
        sd2 = np.std(np.add(rr_intervals[:-1], rr_intervals[1:]) / np.sqrt(2))
        
        ax2.set_title(f'Poincaré Plot (SD1={sd1:.1f}, SD2={sd2:.1f})', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('RR(n) [ms]')
        ax2.set_ylabel('RR(n+1) [ms]')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Panel 3: HRV Frequency Domain (if enough data)
    ax3 = axes[1, 0]
    if len(rr_intervals) > 50:
        # Resample to regular intervals (4 Hz)
        rr_interpolated = signal.resample(rr_intervals, len(rr_intervals) * 4)
        
        # FFT of RR intervals
        N = len(rr_interpolated)
        yf = rfft(rr_interpolated)
        xf = rfftfreq(N, 1/4)  # 4 Hz sampling
        magnitude = np.abs(yf)
        
        ax3.plot(xf, magnitude, 'b-', linewidth=1.5)
        
        # Mark VLF, LF, HF bands
        ax3.axvspan(0.003, 0.04, alpha=0.2, color='yellow', label='VLF')
        ax3.axvspan(0.04, 0.15, alpha=0.2, color='green', label='LF')
        ax3.axvspan(0.15, 0.4, alpha=0.2, color='blue', label='HF')
        
        ax3.set_title('HRV Frequency Domain', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Power')
        ax3.set_xlim([0, 0.5])
        ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Time-domain HRV metrics
    ax4 = axes[1, 1]
    metrics = {
        'SDNN': np.std(rr_intervals),
        'RMSSD': np.sqrt(np.mean(np.diff(rr_intervals) ** 2)),
        'NN50': np.sum(np.abs(np.diff(rr_intervals)) > 50),
        'pNN50': (np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals)) * 100,
        'Mean RR': np.mean(rr_intervals),
        'Median RR': np.median(rr_intervals)
    }
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    ax4.barh(metric_names, metric_values, color='steelblue', edgecolor='black', alpha=0.7)
    ax4.set_title('Time-Domain HRV Metrics', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Value')
    ax4.grid(True, alpha=0.3, axis='x')
    
    for i, (name, value) in enumerate(metrics.items()):
        ax4.text(value, i, f' {value:.1f}', va='center', fontsize=9, fontweight='bold')
    
    plt.suptitle(f'Patient {patient_id}: Heart Rate Variability Analysis', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(patient_dir, '04_hrv_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

def create_rr_interval_analysis(patient_id, patient_dir, detected_peaks):
    """
    Detailed RR interval analysis
    """
    if len(detected_peaks) < 5:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    rr_intervals = np.diff(detected_peaks) / SAMPLING_RATE * 1000  # ms
    
    # Panel 1: Tachogram (RR over time)
    ax1 = axes[0, 0]
    beat_numbers = np.arange(1, len(rr_intervals) + 1)
    ax1.plot(beat_numbers, rr_intervals, 'b-', linewidth=1, alpha=0.7)
    ax1.scatter(beat_numbers, rr_intervals, s=10, color='red', alpha=0.5)
    ax1.set_title('Tachogram (RR Intervals Over Time)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Beat Number')
    ax1.set_ylabel('RR Interval (ms)')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Successive differences
    ax2 = axes[0, 1]
    successive_diff = np.diff(rr_intervals)
    ax2.plot(successive_diff, 'g-', linewidth=1)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_title('Successive RR Differences', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Beat Number')
    ax2.set_ylabel('ΔRR (ms)')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Autocorrelation
    ax3 = axes[1, 0]
    if len(rr_intervals) > 20:
        autocorr = np.correlate(rr_intervals - np.mean(rr_intervals), 
                               rr_intervals - np.mean(rr_intervals), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        lags = range(min(100, len(autocorr)))
        ax3.plot(lags, autocorr[:len(lags)], color='purple', linewidth=2)
        ax3.axhline(0, color='red', linestyle='--', linewidth=1)
        ax3.set_title('RR Interval Autocorrelation', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Lag (beats)')
        ax3.set_ylabel('Autocorrelation')
        ax3.grid(True, alpha=0.3)
    
    # Panel 4: Box plot and statistics
    ax4 = axes[1, 1]
    box = ax4.boxplot([rr_intervals], vert=True, patch_artist=True, widths=0.5)
    for patch in box['boxes']:
        patch.set_facecolor('lightblue')
    
    stats_text = f"""RR Interval Statistics:
    
Mean:    {np.mean(rr_intervals):.1f} ms
Median:  {np.median(rr_intervals):.1f} ms
StdDev:  {np.std(rr_intervals):.1f} ms
Min:     {np.min(rr_intervals):.1f} ms
Max:     {np.max(rr_intervals):.1f} ms
Range:   {np.max(rr_intervals) - np.min(rr_intervals):.1f} ms
CV:      {(np.std(rr_intervals) / np.mean(rr_intervals) * 100):.1f}%
    """
    
    ax4.text(1.5, np.median(rr_intervals), stats_text, fontsize=10, 
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('RR Interval Box Plot', fontsize=12, fontweight='bold')
    ax4.set_ylabel('RR Interval (ms)')
    ax4.set_xticks([])
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Patient {patient_id}: RR Interval Analysis', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(patient_dir, '05_rr_interval_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

def create_frequency_band_analysis(patient_id, patient_dir, beat_features_df):
    """
    Analyze frequency band power distribution
    """
    if len(beat_features_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract frequency band features (first 6 features)
    band_names = ['P-wave\n(5-15 Hz)', 'QRS\n(10-40 Hz)', 'T-wave\n(1-7 Hz)', 
                  'Low Freq\n(0.5-5 Hz)', 'Mid Freq\n(5-25 Hz)', 'High Freq\n(25-60 Hz)']
    
    feature_cols = [col for col in beat_features_df.columns if col.startswith('FFT_Feature_')]
    
    if len(feature_cols) >= 6:
        # Panel 1: Average band powers by AAMI class
        ax1 = axes[0, 0]
        aami_classes = beat_features_df['AAMI_Class'].unique()
        colors_aami = {'N': 'green', 'V': 'red', 'S': 'blue', 'F': 'orange', 'Q': 'gray'}
        
        x = np.arange(len(band_names))
        width = 0.15
        
        for i, aami_class in enumerate(sorted(aami_classes)):
            class_data = beat_features_df[beat_features_df['AAMI_Class'] == aami_class]
            band_powers = [class_data[feature_cols[j]].mean() for j in range(6)]
            offset = (i - len(aami_classes)/2) * width
            ax1.bar(x + offset, band_powers, width, label=aami_class, 
                   color=colors_aami.get(aami_class, 'gray'), alpha=0.8)
        
        ax1.set_title('Average Frequency Band Power by AAMI Class', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Frequency Band')
        ax1.set_ylabel('Power')
        ax1.set_xticks(x)
        ax1.set_xticklabels(band_names, fontsize=9)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Panel 2: Distribution of QRS power (most important)
        ax2 = axes[0, 1]
        for aami_class in sorted(aami_classes):
            class_data = beat_features_df[beat_features_df['AAMI_Class'] == aami_class]
            qrs_power = class_data[feature_cols[1]].values
            ax2.hist(qrs_power, bins=30, alpha=0.5, label=aami_class, 
                    color=colors_aami.get(aami_class, 'gray'), edgecolor='black')
        
        ax2.set_title('QRS Power Distribution by Beat Type', fontsize=12, fontweight='bold')
        ax2.set_xlabel('QRS Power (10-40 Hz)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Power ratio analysis
        ax3 = axes[1, 0]
        # QRS/Total ratio is feature 6
        if len(feature_cols) >= 7:
            for aami_class in sorted(aami_classes):
                class_data = beat_features_df[beat_features_df['AAMI_Class'] == aami_class]
                qrs_ratio = class_data[feature_cols[6]].values
                ax3.hist(qrs_ratio, bins=30, alpha=0.5, label=aami_class, 
                        color=colors_aami.get(aami_class, 'gray'), edgecolor='black')
            
            ax3.set_title('QRS/Total Power Ratio Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('QRS Power Ratio')
            ax3.set_ylabel('Frequency')
            ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Band power correlation matrix
        ax4 = axes[1, 1]
        band_data = beat_features_df[feature_cols[:6]].values
        corr_matrix = np.corrcoef(band_data.T)
        
        im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks(range(6))
        ax4.set_yticks(range(6))
        ax4.set_xticklabels(band_names, rotation=45, ha='right', fontsize=8)
        ax4.set_yticklabels(band_names, fontsize=8)
        ax4.set_title('Frequency Band Correlation Matrix', fontsize=12, fontweight='bold')
        
        # Add correlation values
        for i in range(6):
            for j in range(6):
                text = ax4.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax4, label='Correlation')
    
    plt.suptitle(f'Patient {patient_id}: Frequency Band Analysis', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(patient_dir, '06_frequency_band_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

def create_feature_heatmap(patient_id, patient_dir, beat_features_df):
    """
    Create heatmap of features across beats
    """
    if len(beat_features_df) < 10:
        return
    
    feature_cols = [col for col in beat_features_df.columns if col.startswith('FFT_Feature_')]
    
    if len(feature_cols) > 0:
        # Sample beats if too many
        sample_size = min(200, len(beat_features_df))
        sampled_df = beat_features_df.sample(n=sample_size, random_state=42)
        
        # Get feature matrix
        feature_matrix = sampled_df[feature_cols[:30]].values.T  # First 30 features
        
        # Normalize for visualization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        im = ax.imshow(feature_matrix_scaled, aspect='auto', cmap='viridis', interpolation='nearest')
        
        ax.set_title(f'Patient {patient_id}: Beat Feature Heatmap (Standardized)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Beat Number (sampled)')
        ax.set_ylabel('Feature Index')
        ax.set_yticks(range(0, min(30, len(feature_cols)), 2))
        ax.set_yticklabels(range(0, min(30, len(feature_cols)), 2))
        
        plt.colorbar(im, ax=ax, label='Standardized Value')
        plt.tight_layout()
        plt.savefig(os.path.join(patient_dir, '07_feature_heatmap.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()

def create_signal_quality_timeline(patient_id, patient_dir, signal_raw, signal_processed):
    """
    Visualize signal quality metrics over time
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Split signal into segments
    segment_duration = 10 * SAMPLING_RATE  # 10 second segments
    n_segments = len(signal_raw) // segment_duration
    
    if n_segments < 2:
        return
    
    quality_scores = []
    snr_values = []
    amplitude_ranges = []
    times = []
    
    for i in range(n_segments):
        start = i * segment_duration
        end = (i + 1) * segment_duration
        segment = signal_raw[start:end]
        
        quality_score, _ = assess_signal_quality(segment, SAMPLING_RATE)
        quality_scores.append(quality_score)
        
        # Calculate SNR
        signal_power = np.mean(segment ** 2)
        noise_estimate = np.std(np.diff(segment))
        snr = 10 * np.log10(signal_power / (noise_estimate ** 2 + 1e-10))
        snr_values.append(snr)
        
        # Amplitude range
        amp_range = np.max(segment) - np.min(segment)
        amplitude_ranges.append(amp_range)
        
        times.append(i * 10)  # Time in seconds
    
    # Panel 1: Quality score timeline
    ax1 = axes[0]
    ax1.plot(times, quality_scores, 'b-o', linewidth=2, markersize=4)
    ax1.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Acceptable Threshold')
    ax1.fill_between(times, 0, quality_scores, alpha=0.3, color='blue')
    ax1.set_title('Signal Quality Score Over Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Quality Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Panel 2: SNR timeline
    ax2 = axes[1]
    ax2.plot(times, snr_values, 'g-o', linewidth=2, markersize=4)
    ax2.axhline(15, color='orange', linestyle='--', linewidth=2, label='Good SNR Threshold')
    ax2.fill_between(times, 0, snr_values, alpha=0.3, color='green')
    ax2.set_title('Signal-to-Noise Ratio Over Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('SNR (dB)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Amplitude range timeline
    ax3 = axes[2]
    ax3.plot(times, amplitude_ranges, color='purple', marker='o', linewidth=2, markersize=4)
    ax3.fill_between(times, 0, amplitude_ranges, alpha=0.3, color='purple')
    ax3.set_title('Signal Amplitude Range Over Time', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Amplitude Range')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'Patient {patient_id}: Signal Quality Timeline', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(patient_dir, '08_signal_quality_timeline.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

def create_3d_feature_space(patient_id, patient_dir, beat_features_df):
    """
    3D visualization of feature space using PCA
    """
    if len(beat_features_df) < 20:
        return
    
    feature_cols = [col for col in beat_features_df.columns if col.startswith('FFT_Feature_')]
    
    if len(feature_cols) < 3:
        return
    
    # Prepare data
    X = beat_features_df[feature_cols].values
    y = beat_features_df['AAMI_Class'].values
    
    # Remove any NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors_aami = {'N': 'green', 'V': 'red', 'S': 'blue', 'F': 'orange', 'Q': 'gray'}
    
    for aami_class in np.unique(y):
        mask = y == aami_class
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                  c=colors_aami.get(aami_class, 'gray'), label=aami_class,
                  s=30, alpha=0.6, edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11, fontweight='bold')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Patient {patient_id}: 3D Feature Space (PCA)\nTotal Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(patient_dir, '09_3d_feature_space.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

def create_beat_type_distribution(patient_id, patient_dir, annotations, beat_features_df):
    """
    Comprehensive beat type distribution analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Original annotation distribution
    ax1 = axes[0, 0]
    beat_type_counts = annotations['Type'].value_counts()
    colors = plt.cm.tab20(np.linspace(0, 1, len(beat_type_counts)))
    ax1.bar(range(len(beat_type_counts)), beat_type_counts.values, color=colors, edgecolor='black')
    ax1.set_xticks(range(len(beat_type_counts)))
    ax1.set_xticklabels(beat_type_counts.index, rotation=45, ha='right')
    ax1.set_title('Original Beat Type Distribution', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: AAMI class distribution
    ax2 = axes[0, 1]
    aami_counts = beat_features_df['AAMI_Class'].value_counts()
    colors_aami = {'N': 'green', 'V': 'red', 'S': 'blue', 'F': 'orange', 'Q': 'gray'}
    bar_colors = [colors_aami.get(cls, 'gray') for cls in aami_counts.index]
    
    bars = ax2.bar(range(len(aami_counts)), aami_counts.values, color=bar_colors, 
                   edgecolor='black', linewidth=2)
    ax2.set_xticks(range(len(aami_counts)))
    ax2.set_xticklabels(aami_counts.index)
    ax2.set_title('AAMI Class Distribution', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add percentages on bars
    total = sum(aami_counts.values)
    for i, (bar, count) in enumerate(zip(bars, aami_counts.values)):
        height = bar.get_height()
        pct = (count / total) * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # Panel 3: Pie chart of AAMI classes
    ax3 = axes[1, 0]
    ax3.pie(aami_counts.values, labels=aami_counts.index, autopct='%1.1f%%',
           colors=[colors_aami.get(cls, 'gray') for cls in aami_counts.index],
           startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax3.set_title('AAMI Class Proportion', fontsize=12, fontweight='bold')
    
    # Panel 4: Beat distribution over time
    ax4 = axes[1, 1]
    if 'Time_seconds' in beat_features_df.columns:
        for aami_class in sorted(beat_features_df['AAMI_Class'].unique()):
            class_data = beat_features_df[beat_features_df['AAMI_Class'] == aami_class]
            times = class_data['Time_seconds'].values
            ax4.scatter(times, [aami_class] * len(times), 
                       color=colors_aami.get(aami_class, 'gray'),
                       alpha=0.6, s=20, label=aami_class)
        
        ax4.set_title('Beat Distribution Over Time', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('AAMI Class')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Patient {patient_id}: Beat Type Distribution Analysis', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(patient_dir, '10_beat_type_distribution.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# MODIFIED PROCESS PATIENT FUNCTION
# ============================================================================
def process_patient_with_research_viz(patient_id, output_dir):
    """Process patient with comprehensive research visualizations"""
    print(f"\n{'='*60}")
    print(f"Processing Patient {patient_id}")
    print(f"{'='*60}")
    
    patient_dir = os.path.join(output_dir, f'patient_{patient_id}')
    os.makedirs(patient_dir, exist_ok=True)
    
    csv_file = f'{patient_id}.csv'
    ann_file = f'{patient_id}annotations.txt'
    
    if not os.path.exists(csv_file) or not os.path.exists(ann_file):
        print(f"ERROR: Files not found. Skipping.")
        return None
    
    try:
        # Load data
        ecg_data = pd.read_csv(csv_file)
        ecg_data.columns = ecg_data.columns.str.strip()
        
        annotations = pd.read_csv(ann_file, delim_whitespace=True, skiprows=1,
                                 names=['Time', 'Sample', 'Type', 'Sub', 'Chan', 'Num', 'Aux'])
        
        if 'MLII' in ecg_data.columns:
            signal_data = ecg_data['MLII'].values
        else:
            signal_data = ecg_data.iloc[:, 1].values
        
        # Signal quality
        quality_score, is_acceptable = assess_signal_quality(signal_data, SAMPLING_RATE)
        print(f"Signal Quality Score: {quality_score:.2f} - {'ACCEPTABLE' if is_acceptable else 'POOR'}")
        
        # Preprocess
        signal_processed = preprocess_signal(signal_data, SAMPLING_RATE)
        
        # R-peak detection
        print("Detecting R-peaks...")
        detected_peaks = detect_r_peaks_multi_strategy(signal_processed, SAMPLING_RATE)
        print(f"  Detected {len(detected_peaks)} peaks")
        
        # Extract features
        print("Extracting FFT features...")
        BEAT_WINDOW = 90
        
        beat_features = []
        for idx, row in annotations.iterrows():
            r_peak = row['Sample']
            beat_type = row['Type']
            aami_class = map_to_aami_class(beat_type)
            
            start = max(0, r_peak - BEAT_WINDOW)
            end = min(len(signal_data), r_peak + BEAT_WINDOW)
            segment = signal_processed[start:end]
            
            if len(segment) == 2 * BEAT_WINDOW:
                fft_feats, _, _ = extract_enhanced_fft_features(segment, SAMPLING_RATE)
                
                beat_features.append({
                    'Patient_ID': patient_id,
                    'Beat_Index': idx,
                    'R_Peak_Sample': r_peak,
                    'Time_seconds': r_peak / SAMPLING_RATE,
                    'Beat_Type': beat_type,
                    'AAMI_Class': aami_class,
                    'Features': fft_feats.tolist(),
                    'Quality_Score': quality_score
                })
        
        beat_df = pd.DataFrame(beat_features)
        
        # Expand features
        features_array = np.array(beat_df['Features'].tolist())
        feature_cols = [f'FFT_Feature_{i}' for i in range(features_array.shape[1])]
        features_df = pd.DataFrame(features_array, columns=feature_cols)
        beat_df = pd.concat([beat_df.drop('Features', axis=1), features_df], axis=1)
        
        # Save features
        beat_csv = os.path.join(patient_dir, 'beat_features.csv')
        beat_df.to_csv(beat_csv, index=False)
        
        # CREATE ALL RESEARCH VISUALIZATIONS
        print("  Generating research visualizations...")
        create_comprehensive_patient_visualizations(
            patient_id, patient_dir, signal_data, signal_processed,
            detected_peaks, annotations, beat_df
        )
        
        print(f"[OK] Patient {patient_id} complete with {len(beat_features)} beats")
        print(f"     Generated 10 comprehensive research visualizations")
        
        return {
            'Patient_ID': patient_id,
            'Total_beats': len(beat_features),
            'N_count': len(beat_df[beat_df['AAMI_Class'] == 'N']),
            'V_count': len(beat_df[beat_df['AAMI_Class'] == 'V']),
            'S_count': len(beat_df[beat_df['AAMI_Class'] == 'S']),
            'F_count': len(beat_df[beat_df['AAMI_Class'] == 'F']),
            'Q_count': len(beat_df[beat_df['AAMI_Class'] == 'Q']),
        }
        
    except Exception as e:
        print(f"ERROR processing patient {patient_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*60)
    print("ECG ANALYSIS - COMPREHENSIVE RESEARCH VISUALIZATIONS")
    print("="*60)
    print("\n10 Visualizations Per Patient:")
    print("  1. Multi-panel ECG Analysis (6 panels)")
    print("  2. FFT Spectrum Analysis")
    print("  3. Beat Morphology Clustering")
    print("  4. Heart Rate Variability (HRV)")
    print("  5. RR Interval Analysis")
    print("  6. Frequency Band Analysis")
    print("  7. Feature Heatmap")
    print("  8. Signal Quality Timeline")
    print("  9. 3D Feature Space (PCA)")
    print("  10. Beat Type Distribution")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process selected patients (you can modify this list)
    patients_to_process = [100, 101, 103, 106, 108, 119, 200, 207, 208]  # Example patients
    
    all_summaries = []
    for patient_id in patients_to_process:
        summary = process_patient_with_research_viz(patient_id, OUTPUT_DIR)
        if summary is not None:
            all_summaries.append(summary)
    
    # Save summary
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(os.path.join(OUTPUT_DIR, 'processing_summary.csv'), index=False)
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE!")
        print("="*60)
        print(f"Processed: {len(all_summaries)} patients")
        print(f"Output directory: {OUTPUT_DIR}/")
        print(f"\nEach patient has 10 detailed research visualizations")
        print("="*60)
