ECG Analysis System Documentation
Overview
This is a comprehensive ECG (electrocardiogram) signal analysis system that processes heart rhythm data from the MIT-BIH Arrhythmia Database. The system performs two main tasks: it creates detailed research visualizations for each patient, and it uses machine learning to classify different types of heartbeats.
The code analyzes ECG signals to detect R-peaks (the main spike in each heartbeat), extracts frequency features using Fast Fourier Transform (FFT), and then classifies heartbeats into five categories defined by the AAMI (Association for the Advancement of Medical Instrumentation) standard.
What You Need
Python Libraries
The code requires these Python packages:
•	numpy: For numerical computations
•	pandas: For data handling and CSV operations
•	matplotlib: For creating all the visualizations
•	scipy: For signal processing and FFT analysis
•	scikit-learn: For machine learning classification
•	imbalanced-learn: For handling imbalanced class distribution
•	seaborn: For enhanced visualizations
Input Data Files
For each patient, you need two files:
1.	CSV file (e.g., 101.csv): Contains the raw ECG signal data with a column called 'MLII' for the signal values
2.	Annotation file (e.g., 101annotations.txt): Contains beat annotations marking where each heartbeat occurs and what type it is
Configuration
The code has several important configuration settings at the top:
Sampling Rate
The sampling rate is set to 360 Hz, which is the standard sampling rate for MIT-BIH database recordings. This means the ECG signal has 360 data points per second.
Patient Datasets
The code divides patients into two groups:
•	DS1_TRAIN: 22 patients used for training the machine learning model
•	DS2_TEST: 22 patients used for testing the model
This follows the standard inter-patient evaluation protocol where you train on one set of patients and test on completely different patients. This is important because it tests whether the system can generalize to new patients it has never seen before.
AAMI Classification
Heartbeats are grouped into five categories:
•	N (Normal): Includes normal beats (N), left bundle branch block (L), right bundle branch block (R), atrial escape (e), and nodal escape (j)
•	S (Supraventricular): Includes atrial premature (A), aberrated atrial premature (a), nodal premature (J), and supraventricular premature (S)
•	V (Ventricular): Includes ventricular premature (V) and ventricular escape (E)
•	F (Fusion): Includes fusion of ventricular and normal (F)
•	Q (Unclassifiable): Includes paced beats (/), fusion of paced and normal (f), and unclassifiable (Q)
How the Code Works
Step 1: Signal Processing
For each patient, the system performs the following operations:
1.1 Data Loading
The system loads the ECG signal from the CSV file and the beat annotations from the text file. It looks for a column named 'MLII' in the CSV file, which is the standard lead used in the MIT-BIH database. If this column is not found, it uses the second column by default.
1.2 Signal Quality Assessment
Before processing, the code checks if the signal is good enough to analyze. It calculates:
•	Signal range: The difference between maximum and minimum values
•	Signal-to-noise ratio (SNR): A measure of how clean the signal is
•	Signal standard deviation: A measure of signal variability
A quality score between 0 and 1 is assigned. Signals with a score above 0.5 are considered acceptable.
1.3 Signal Preprocessing
The raw ECG signal goes through several filtering steps:
High-pass filter: Removes baseline wander (slow drifts in the signal) using a 4th order Butterworth filter with a cutoff frequency of 0.67 Hz.
Notch filter: Removes 60 Hz power line interference that comes from electrical equipment.
Gaussian smoothing: If the signal is still noisy after filtering, a Gaussian filter is applied to smooth it out.
1.4 R-Peak Detection
R-peaks are the main spikes in each heartbeat. The code uses an enhanced Pan-Tompkins algorithm:
Bandpass filtering: The signal is filtered to emphasize the QRS complex (the main spike) using frequencies between 5 and 18 Hz.
Differentiation: The derivative of the signal is calculated to emphasize rapid changes.
Squaring: The derivative is squared to make all values positive and emphasize larger changes.
Moving average: A moving window integrates the squared signal to smooth it.
Peak detection: Peaks in the integrated signal are found using adaptive thresholds.
Refinement: Each detected peak is refined by searching for the maximum in the original filtered signal.
The system tries three different strategies (normal signal, inverted signal, and absolute value) and picks the one that gives the most reasonable number of peaks for the duration of the recording.
Step 2: Feature Extraction
For each detected heartbeat, the system extracts a comprehensive set of features:
2.1 Beat Segmentation
Around each R-peak, the code extracts a window of 180 samples (90 samples before and 90 samples after the peak). At 360 Hz sampling rate, this gives a window of 500 milliseconds, which captures the entire heartbeat.
2.2 FFT Features
The system performs Fast Fourier Transform on each beat segment to analyze its frequency content. The FFT converts the time-domain signal into the frequency domain, showing which frequencies are present and how strong they are.
Frequency band powers: The system calculates the power (energy) in specific frequency bands that correspond to different parts of the heartbeat:
•	P-wave band (5-15 Hz): Corresponds to atrial contraction
•	QRS band (10-40 Hz): Corresponds to ventricular contraction (the main spike)
•	T-wave band (1-7 Hz): Corresponds to ventricular recovery
•	Low frequency (0.5-5 Hz): General slow components
•	Mid frequency (5-25 Hz): General medium components
•	High frequency (25-60 Hz): General fast components
Power ratios: The code calculates ratios between different bands, which are useful for distinguishing beat types. For example, the ratio of QRS power to total power tells you how prominent the main spike is.
Dominant frequency: The frequency with the highest power in the spectrum. This gives you the main oscillation rate of the beat.
Harmonic analysis: The code looks for harmonics (multiples) of the dominant frequency, which can indicate certain types of abnormal beats.
Spectral statistics:
•	Spectral centroid: The "centre of mass" of the spectrum, indicating where most of the energy is concentrated
•	Spectral spread: How spread out the frequencies are
•	Spectral entropy: A measure of how uniform the frequency distribution is (high entropy means many frequencies, low entropy means concentrated in few frequencies)
•	Spectral Rolloff: The frequency below which 85% of the energy is contained
FFT coefficients: The code samples 20 evenly spaced points from the FFT magnitude spectrum to capture the overall shape of the frequency response.
2.3 RR Interval Features
RR intervals are the time gaps between consecutive heartbeats. These are crucial for detecting rhythm abnormalities:
Time-domain features:
•	Mean RR interval: Average time between beats
•	Standard deviation: Variability in beat timing
•	Last 4 intervals: Recent rhythm pattern
Frequency-domain HRV (Heart Rate Variability):
The code performs FFT on the RR interval sequence to analyze heart rate variability in the frequency domain:
•	VLF (Very Low Frequency, 0.003-0.04 Hz): Long-term regulation
•	LF (Low Frequency, 0.04-0.15 Hz): Related to sympathetic nervous system
•	HF (High Frequency, 0.15-0.4 Hz): Related to parasympathetic nervous system (breathing)
•	LF/HF ratio: Indicator of sympatho-vagal balance
•	Total HRV power: Overall variability
Step 3: Visualization Generation
For each patient, the system creates 10 comprehensive visualizations:
Visualization 1: Multi-Panel ECG Analysis
This creates a 6-panel figure showing:
Panel 1 - Raw ECG Signal: Shows the first 30 seconds of the original, unprocessed ECG signal.
Panel 2 - Preprocessed Signal with R-peaks: Shows the filtered signal with detected R-peaks marked as red dots.
Panel 3 - Beat Classification: Shows the signal with colored vertical lines indicating different beat types (green for normal, red for ventricular, blue for supraventricular, etc.).
Panel 4 - Heart Rate Over Time: Calculates instantaneous heart rate from consecutive beats and plots it over time. Shows mean heart rate and variability bands.
Panel 5 - First Derivative: Shows the derivative of the signal, which is used in the peak detection algorithm.
Panel 6 - RR Interval Sequence: Plots the time between consecutive beats as a sequence, showing rhythm regularity.
Visualization 2: FFT Spectrum Analysis
This creates a 4-panel figure analyzing frequency content:
Panel 1 - Full Signal Spectrum: FFT of the entire ECG signal, showing frequency content up to 100 Hz.
Panel 2 - Physiological Bands: Same spectrum with colored overlays highlighting the P-wave, QRS, and T-wave frequency bands.
Panel 3 - Power Spectral Density: Uses Welch's method to estimate power spectral density, which is more robust than a simple FFT. Plotted on a logarithmic scale.
Panel 4 - Average Beat Spectrum: Computes the FFT for up to 50 individual beats and averages them, showing the typical frequency content of a single heartbeat with standard deviation bands.
Visualization 3: Beat Morphology Clustering
This shows the actual shape of heartbeats grouped by AAMI class:
For each of the 5 AAMI classes, the code:
•	Extracts up to 20 beats of that type
•	Overlays them with transparency so you can see the variation
•	Plots the average beat in bold
•	Aligns all beats so the R-peak is at time zero
This lets you visually compare the shape differences between normal and abnormal beats.
Visualization 4: Heart Rate Variability (HRV) Analysis
This creates a 4-panel figure dedicated to HRV analysis:
Panel 1 - RR Interval Histogram: Shows the distribution of RR intervals with mean and median marked.
Panel 2 - Poincare Plot: Plots each RR interval against the next one (RRn vs RRn+1). The shape of this plot reveals important rhythm characteristics. It also calculates SD1 and SD2, which are standard HRV metrics.
Panel 3 - HRV Frequency Domain: If there are enough beats, performs FFT on the RR interval sequence and highlights VLF, LF, and HF bands.
Panel 4 - Time-Domain HRV Metrics: Bar chart showing important time-domain metrics:
•	SDNN: Standard deviation of all RR intervals
•	RMSSD: Root mean square of successive differences
•	NN50: Count of successive RR intervals differing by more than 50 ms
•	pNN50: Percentage of NN50
Visualization 5: RR Interval Analysis
This provides detailed analysis of beat-to-beat timing:
Panel 1 - Tachogram: Line plot of RR intervals over time, showing rhythm stability.
Panel 2 - Successive Differences: Plots the change in RR interval from one beat to the next, highlighting sudden rhythm changes.
Panel 3 - Autocorrelation: Shows how the RR interval pattern correlates with itself at different time lags, revealing periodic patterns.
Panel 4 - Box Plot and Statistics: Box plot showing the distribution of RR intervals with a text box containing detailed statistics.
Visualization 6: Frequency Band Analysis
This analyzes how different frequency bands vary across beat types:
Panel 1 - Average Band Powers by Class: Grouped bar chart comparing the 6 frequency bands across all AAMI classes.
Panel 2 - QRS Power Distribution: Histogram of QRS band power for each beat type, since QRS is the most important component.
Panel 3 - QRS/Total Power Ratio: Histogram showing how much of the total energy is in the QRS band for different beat types.
Panel 4 - Band Correlation Matrix: Heatmap showing how the 6 frequency bands correlate with each other, with correlation coefficients displayed.
Visualization 7: Feature Heatmap
This creates a heatmap showing all features across beats:
•	Samples up to 200 beats (to keep the visualization readable)
•	Shows the first 30 features as rows
•	Each column is a beat
•	Values are standardized (mean 0, std 1) for better visualization
•	Color indicates feature strength
This gives a bird's-eye view of how features vary across the recording.
Visualization 8: Signal Quality Timeline
This divides the recording into 10-second segments and analyzes quality over time:
Panel 1 - Quality Score: Shows the computed quality score for each segment. Scores above 0.5 (red line) are acceptable.
Panel 2 - Signal-to-Noise Ratio: Shows SNR in decibels for each segment. Higher values mean cleaner signal.
Panel 3 - Amplitude Range: Shows the signal range for each segment. Consistent range indicates stable recording.
This helps identify portions of the recording that may be problematic.
Visualization 9: 3D Feature Space
This uses Principal Component Analysis (PCA) to reduce all features to 3 dimensions:
•	PCA finds the 3 directions in feature space that capture the most variance
•	Each beat is plotted as a point in this 3D space
•	Points are colored by AAMI class
•	The plot shows how well different beat types cluster in feature space
•	Axis labels show how much variance each principal component explains
This gives insight into whether the features are good at separating different beat types.
Visualization 10: Beat Type Distribution
This creates a 4-panel summary of beat types:
Panel 1 - Original Beat Type Distribution: Bar chart of all original beat annotation types.
Panel 2 - AAMI Class Distribution: Bar chart of the 5 AAMI classes with count and percentage on each bar.
Panel 3 - AAMI Class Pie Chart: Pie chart showing proportions of each class.
Panel 4 - Beat Distribution Over Time: Scatter plot showing when each beat type occurs throughout the recording, revealing temporal patterns.
Step 4: Feature Storage
After processing each patient, the code saves a CSV file containing:
•	Patient ID
•	Beat index
•	R-peak sample location
•	Time in seconds
•	Original beat type
•	AAMI class
•	All extracted features (typically 50+ features per beat)
•	Signal quality score
This CSV file is the input for the machine learning phase.
Step 5: Machine Learning Classification
After all patients are processed, the code trains a classifier:
5.1 Data Preparation
The system:
•	Loads all beat feature CSV files from all patients
•	Combines them into one large dataset
•	Filters out classes with fewer than 20 samples
•	Separates DS1 (training) and DS2 (testing) patients
5.2 Feature Scaling
Features are standardized to have mean 0 and standard deviation 1. This is important because features have different scales (some are small decimals, others are large numbers), and machine learning algorithms work better with standardized data.
5.3 Handling Class Imbalance
ECG datasets are highly imbalanced. Normal beats might be 90% of the data, while abnormal beats are only 10%. To handle this:
SMOTE (Synthetic Minority Over-sampling Technique): Creates synthetic examples of minority classes by interpolating between existing samples. This balances the training data.
Class weights: Assigns higher weights to minority classes during training, so the classifier pays more attention to them.
5.4 Model Training
The code uses a Support Vector Machine (SVM) with RBF (Radial Basis Function) kernel:
•	SVM is effective for high-dimensional data (we have 50+ features)
•	RBF kernel can model complex, non-linear decision boundaries
•	C parameter is set to 10.0 (controls trade-off between correct classification and margin width)
•	Gamma is set to 'scale' (automatically determines kernel width)
5.5 Evaluation
The trained model is tested on DS2 patients (completely unseen data):
Overall accuracy: Percentage of correctly classified beats across all classes.
Per-class metrics:
•	Precision: Of all beats predicted as class X, how many actually were class X?
•	Recall: Of all actual class X beats, how many were correctly identified?
•	F1-score: Harmonic mean of precision and recall
Macro average: Unweighted average across all classes (treats all classes equally).
Confusion matrix: Shows which classes are confused with which. Each row is the true class, each column is the predicted class. Diagonal elements are correct predictions.
5.6 Results Visualization
The code creates:
Confusion Matrix Plot: Heatmap showing the confusion matrix with counts. Darker colors indicate more beats. Ideal matrix has high values on the diagonal and low values elsewhere.
Per-Class Accuracy Plot: Bar chart showing accuracy for each AAMI class. Bars are colored according to class (green for N, red for V, blue for S, orange for F, gray for Q). Percentages are displayed on each bar.
Output Files
The system creates a structured output directory:
ecg_research_visualizations/
├── patient_101/
│   ├── beat_features.csv
│   ├── 01_multipanel_ecg_analysis.png
│   ├── 02_fft_spectrum_analysis.png
│   ├── 03_beat_morphology_clustering.png
│   ├── 04_hrv_analysis.png
│   ├── 05_rr_interval_analysis.png
│   ├── 06_frequency_band_analysis.png
│   ├── 07_feature_heatmap.png
│   ├── 08_signal_quality_timeline.png
│   ├── 09_3d_feature_space.png
│   └── 10_beat_type_distribution.png
├── patient_106/
│   └── (same files)
├── ...
├── processing_summary.csv
└── ml_classification_results/
    ├── classification_report.csv
    ├── confusion_matrix.csv
    ├── confusion_matrix.png
    └── per_class_accuracy.png
File Descriptions
beat_features.csv: Contains all extracted features for every beat in the patient's recording. Each row is one beat.
Visualization PNGs: Ten high-resolution (150 DPI) images showing different aspects of the ECG analysis.
processing_summary.csv: Summary of all processed patients with beat counts for each AAMI class.
classification_report.csv: Detailed metrics for each class (precision, recall, F1-score, support).
confusion_matrix.csv: Full confusion matrix in CSV format for further analysis.
confusion_matrix.png: Visual representation of the confusion matrix.
per_class_accuracy.png: Bar chart of accuracy for each class.
Key Functions
Signal Processing Functions
assess_signal_quality: Evaluates whether the ECG signal is good enough for analysis. Returns a quality score and a boolean indicating if the signal is acceptable.
preprocess_signal: Applies filtering to remove baseline wander, power line interference, and noise. Returns the cleaned signal.
detect_r_peaks_enhanced: Implements the Pan-Tompkins algorithm with refinements. Takes a signal and returns the sample indices where R-peaks occur.
detect_r_peaks_multi_strategy: Tries multiple detection strategies (normal, inverted, absolute value) and picks the best one. This makes the system more robust to different ECG morphologies.
Feature Extraction Functions
extract_enhanced_fft_features: Takes a beat segment and extracts comprehensive frequency-domain features. Returns a feature vector, frequency axis, and magnitude spectrum.
extract_frequency_band_power: Helper function that calculates the power in a specific frequency range from an FFT spectrum.
extract_rr_features: Extracts features related to the rhythm by analyzing RR intervals. Returns both time-domain and frequency-domain HRV features.
extract_rr_fft_features: Performs FFT on the RR interval sequence to get frequency-domain HRV metrics.
map_to_aami_class: Converts original beat annotations to AAMI standard classes.
Visualization Functions
Each visualization function follows the same pattern:
•	Takes patient ID, directory, and relevant data as input
•	Creates a matplotlib figure with appropriate size
•	Generates the visualization
•	Saves it as a high-resolution PNG
•	Closes the figure to free memory
create_comprehensive_patient_visualizations: Main function that calls all 10 visualization functions in sequence.
Processing Functions
process_patient_with_research_viz: Main processing function for each patient. It:
•	Loads data
•	Checks quality
•	Detects peaks
•	Extracts features
•	Saves features to CSV
•	Generates all visualizations
•	Returns a summary dictionary
classify_beats_ml: Performs machine learning classification:
•	Loads features from all patients
•	Splits into train/test based on DS1/DS2
•	Standardizes features
•	Applies SMOTE
•	Trains SVM
•	Evaluates on test set
•	Generates result visualizations
•	Saves all results
Usage
To run the code:
1.	Place all patient CSV files and annotation files in the same directory as the script
2.	Run the script: python script_name.py
3.	The system will process all 44 patients (this takes several minutes)
4.	Check the ecg_research_visualizations/ folder for results
The console output will show:
•	Progress for each patient
•	Number of peaks detected
•	Number of beats processed
•	Quality assessment
•	Machine learning training progress
•	Final accuracy metrics
Technical Details
Computational Complexity
•	Signal preprocessing: O(n) where n is signal length
•	R-peak detection: O(n log n) due to filtering and peak finding
•	Feature extraction per beat: O(m log m) where m is beat window size (180 samples)
•	Machine learning training: O(n_samples^2 × n_features) for SVM
Memory Usage
The system processes patients one at a time, so memory usage is reasonable. The largest memory consumer is typically the combined feature matrix during machine learning, which holds features for all beats from all patients.
Parameter Tuning
Key parameters that can be adjusted:
BEAT_WINDOW = 90: Number of samples before/after R-peak. Increasing this captures more of the T-wave but includes more noise.
C = 10.0: SVM regularization. Higher values fit training data more closely but may overfit.
SMOTE k_neighbors = 5: Number of neighbors for generating synthetic samples. Lower values create samples closer to existing ones.
min_samples = 20: Minimum class size. Classes with fewer samples are excluded from training.
Known Limitations
Inter-patient variability: The system may not perform well on patients with very different ECG morphology than the training set.
Rare beat types: Classes like F (Fusion) are very rare and hard to learn.
Signal quality dependency: Poor quality signals may lead to missed beats or false detections.
Computational time: Processing all patients takes significant time (several minutes to tens of minutes depending on hardware).
Clinical Relevance
The five AAMI classes have different clinical significance:
N (Normal): No treatment needed. These are expected in healthy hearts.
S (Supraventricular): May indicate atrial fibrillation or other atrial arrhythmias. Can lead to stroke if persistent.
V (Ventricular): More serious. Frequent ventricular beats can indicate heart disease and risk of sudden cardiac death.
F (Fusion): Rare but interesting. Shows competition between normal conduction and ventricular focus.
Q (Unclassifiable): May indicate paced rhythms (artificial pacemaker) or artifacts.
Accurate automated classification helps cardiologists by:
•	Screening large amounts of ECG data
•	Flagging potentially dangerous rhythms
•	Quantifying arrhythmia burden
•	Monitoring patient status over time
Research Applications
This system can be used for:
Algorithm development: Testing new feature extraction or classification methods
Comparative studies: Comparing FFT-based features to other approaches (wavelets, deep learning, etc.)
Clinical validation: Assessing whether automated systems match cardiologist annotations
Patient stratification: Identifying which patients have high arrhythmia burden
Database analysis: Understanding the distribution of beat types in the MIT-BIH database
Future Improvements
Possible enhancements:
Deep learning: Replace feature extraction and SVM with a convolutional neural network that learns features automatically
Real-time processing: Optimize code for streaming ECG data
Multi-lead analysis: Use all 12 leads instead of just MLII
Temporal features: Model the sequence of beats, not just individual beats
Patient-specific adaptation: Fine-tune the classifier for individual patients
Explainability: Add methods to explain why a beat was classified a certain way
Conclusion
This is a complete ECG analysis pipeline that goes from raw signal to classified beats with comprehensive visualizations. It follows established signal processing methods (Pan-Tompkins), uses physiologically meaningful features (frequency bands corresponding to ECG waves), and employs robust machine learning (SVM with proper validation).
The system is particularly strong in:
•	Comprehensive visualization for research and debugging
•	Thorough feature extraction capturing both morphology and rhythm
•	Proper evaluation using inter-patient testing
•	Handling of class imbalance
It provides both visual insights (through the 10 visualizations per patient) and quantitative results (through the machine learning classification), making it suitable for both research and educational purposes.

<img width="451" height="702" alt="image" src="https://github.com/user-attachments/assets/59e3548f-708a-43ee-831f-cf78233e0e8c" />
