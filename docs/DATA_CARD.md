# Data Card: UMAFall Dataset

## Dataset Description

### Overview
The UMAFall (Universidad de Málaga Fall) dataset contains wearable sensor data for fall detection and activity recognition research.

### Collection Process
- **Participants**: 19 healthy subjects (age 20-35)
- **Protocol**: Controlled laboratory environment
- **Equipment**: Wearable IMU sensors (accelerometer + gyroscope)
- **Placement**: Waist-mounted sensor

### Data Characteristics
- **Total Samples**: ~500,000 sensor readings
- **Activities**: 
  - 12 ADL: Walking, Running, Sitting, Standing, etc.
  - 3 Falls: Forward, Backward, Lateral
- **Class Distribution**:
  - ADL: 85%
  - Falls: 15%
- **Sampling Rate**: 50 Hz
- **Duration**: Variable (5-30 seconds per trial)

## Data Processing

### Preprocessing Steps
1. **Windowing**: 2.56s windows with 50% overlap
2. **Normalization**: Z-score normalization per channel
3. **Filtering**: Optional low-pass filter at 20 Hz
4. **Augmentation**: Time warping, noise addition, rotation

### Label Assignment
- **Window Labels**: Majority voting within window
- **Dual Labels**: 
  - Activity class (0-12)
  - Fall binary flag (0/1)

### Data Splits
- **Strategy**: Cross-subject validation
- **K-Fold**: 5 folds with 60/20/20 train/val/test
- **LOSO**: Leave-One-Subject-Out for robustness

## Quality Assurance

### Validation Checks
1. **Schema Validation**: Verify expected columns
2. **Range Checks**: Sensor values within physical limits
3. **Completeness**: No missing values
4. **Leakage Detection**: No subject overlap between splits

### Known Issues
1. **Imbalanced Classes**: Falls underrepresented
2. **Limited Diversity**: Young, healthy subjects only
3. **Controlled Environment**: May not reflect real-world conditions

## Usage Guidelines

### Recommended Uses
- Algorithm development for fall detection
- Multi-task learning research
- Benchmark comparisons

### Not Recommended For
- Direct deployment without additional validation
- Medical diagnosis without clinical validation
- Safety-critical systems without redundancy

## Privacy & Ethics

### Privacy Measures
- No personally identifiable information
- Anonymized subject IDs
- Motion data only (no video/audio)

### Ethical Considerations
- Informed consent obtained from all participants
- IRB approval for data collection
- Public dataset with open access

## Updates & Maintenance

- **Version**: 1.0
- **Last Updated**: 2023
- **Maintainer**: Universidad de Málaga
- **License**: Creative Commons Attribution 4.0