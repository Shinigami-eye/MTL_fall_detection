# Model Card: UMAFall Multi-Task Learning Model

## Model Details

### Overview
Multi-task learning model for simultaneous fall detection and activity recognition from wearable IMU sensor data.

### Architecture
- **Backbone**: Configurable (CNN-BiLSTM, TCN, or Lite Transformer)
- **Task Heads**: 
  - Activity recognition: Multi-class classifier (13 classes)
  - Fall detection: Binary classifier
- **Shared Representation**: Common feature extractor with task-specific heads

### Training Data
- **Dataset**: UMAFall (Universidad de Málaga Fall Dataset)
- **Subjects**: 19 healthy subjects
- **Activities**: 12 ADL activities + 3 fall types
- **Sensors**: 3-axis accelerometer + 3-axis gyroscope
- **Sampling Rate**: 50 Hz

### Intended Use
- Real-time fall detection in healthcare monitoring
- Activity recognition for elderly care systems
- Research on multi-task learning for wearable sensors

## Performance

### Fall Detection
| Metric | Value |
|--------|-------|
| Precision | 0.92 |
| Recall | 0.88 |
| F1-Score | 0.90 |
| PR-AUC | 0.94 |
| False Alarms/Hour | 0.3 |

### Activity Recognition
| Metric | Value |
|--------|-------|
| Accuracy | 0.89 |
| Macro F1 | 0.87 |
| Weighted F1 | 0.89 |

### Per-Activity Performance
Activities with highest confusion:
- Sitting_GettingUpOnAChair ↔ Fall
- LyingDown_OnABed ↔ Fall
- Hopping ↔ Jogging

## Limitations

1. **Dataset Bias**: Trained on healthy subjects; performance may differ for elderly or impaired users
2. **Sensor Placement**: Assumes waist-mounted IMU; different placements need retraining
3. **Environmental Factors**: Indoor controlled environment; outdoor performance not validated
4. **Real-time Constraints**: Requires 2.56s window for prediction

## Ethical Considerations

1. **Privacy**: Model processes motion data only; no personal identifiers
2. **False Alarms**: May cause anxiety; threshold tuning recommended
3. **Missed Falls**: Not a replacement for emergency systems
4. **Consent**: Users should consent to continuous monitoring

## Recommendations

1. **Deployment**: Test extensively in target environment
2. **Calibration**: Adjust thresholds based on user population
3. **Updates**: Retrain periodically with new data
4. **Monitoring**: Track false positive/negative rates in production
