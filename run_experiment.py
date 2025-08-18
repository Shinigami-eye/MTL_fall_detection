"""
Main experiment runner for multi-task fall detection
Implements the experiment grid from the research plan
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import yaml
import json
from datetime import datetime
import random
import numpy as np
from typing import Dict, Optional

from data.dataset import create_dataloaders, DataConfig, IMUDataset
from models.mtl_model import MultiTaskFallDetector, ModelConfig
from models.losses import MultiTaskLoss
from training.trainer import MultiTaskTrainer
from training.evaluator import MultiTaskEvaluator


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_experiment_config(experiment_name: str) -> Dict:
    """
    Create configuration for specific experiments from the research plan
    
    MVS (Minimal Viable Set) experiments:
    - E1: Single-task fall detector baseline
    - E2: Single-task activity classifier baseline  
    - E3: Multi-task with static weights
    - E4: Multi-task with uncertainty weighting
    - E6: Multi-task with focal loss
    """
    
    base_config = {
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data': {
            'batch_size': 64,
            'num_workers': 4,
            'window_size': 2.56,
            'sampling_rate': 100,
            'overlap': 0.5
        },
        'model': {
            'input_channels': 8,
            'window_length': 256,
            'cnn_channels': [32, 64, 128],
            'lstm_hidden': 256,
            'lstm_layers': 2,
            'lstm_dropout': 0.3,
            'num_activities': 16,
            'shared_dim': 512,
            'dropout': 0.3
        },
        'training': {
            'num_epochs': 100,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'scheduler': 'cosine',
            'monitor_gradients': True,
            'gradient_conflict_threshold': -0.3
        }
    }
    
    # Experiment-specific configurations
    experiments = {
        'E1_ST_Fall': {
            'description': 'Single-task fall detector baseline',
            'loss': {
                'type': 'single_fall',
                'focal_loss': False,
                'weighting_strategy': 'static',
                'activity_weight': 0.0,
                'fall_weight': 1.0
            },
            'success_criteria': {
                'pr_auc': 0.80,
                'precision_at_95_recall': 0.70
            }
        },
        'E2_ST_ADL': {
            'description': 'Single-task activity classifier baseline',
            'loss': {
                'type': 'single_activity',
                'weighting_strategy': 'static',
                'activity_weight': 1.0,
                'fall_weight': 0.0
            },
            'success_criteria': {
                'macro_f1': 0.75
            }
        },
        'E3_MTL_Static': {
            'description': 'Multi-task with static weights (0.3:0.7)',
            'loss': {
                'type': 'multi_task',
                'weighting_strategy': 'static',
                'activity_weight': 0.3,
                'fall_weight': 0.7,
                'focal_loss': False
            },
            'success_criteria': {
                'precision_at_95_recall': 0.85,
                'risky_fp_reduction': 0.25,
                'activity_f1_drop': 0.05
            }
        },
        'E4_MTL_Uncertainty': {
            'description': 'Multi-task with uncertainty weighting',
            'loss': {
                'type': 'multi_task',
                'weighting_strategy': 'uncertainty',
                'focal_loss': False
            },
            'success_criteria': {
                'better_than_E3': True,
                'stable_weights': True
            }
        },
        'E5_MTL_GradNorm': {
            'description': 'Multi-task with GradNorm balancing',
            'loss': {
                'type': 'multi_task',
                'weighting_strategy': 'gradnorm',
                'gradnorm_alpha': 1.5,
                'focal_loss': False
            },
            'success_criteria': {
                'better_generalization': True,
                'balanced_gradients': True
            }
        },
        'E6_MTL_Focal': {
            'description': 'Multi-task with focal loss for fall detection',
            'loss': {
                'type': 'multi_task',
                'weighting_strategy': 'static',
                'activity_weight': 0.3,
                'fall_weight': 0.7,
                'focal_loss': True,
                'focal_alpha': 0.25,
                'focal_gamma': 2.0
            },
            'success_criteria': {
                'lower_alarm_rate': True,
                'precision_gain': 0.05
            }
        }
    }
    
    if experiment_name not in experiments:
        raise ValueError(f"Unknown experiment: {experiment_name}. "
                        f"Available: {list(experiments.keys())}")
    
    # Merge experiment-specific config with base config
    config = base_config.copy()
    config['experiment'] = experiments[experiment_name]
    config['experiment']['name'] = experiment_name
    
    return config


class ExperimentRunner:
    """Run and track experiments"""
    
    def __init__(self, config: Dict, output_dir: Optional[Path] = None):
        self.config = config
        self.experiment_name = config['experiment']['name']
        
        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(f'experiments/{self.experiment_name}_{timestamp}')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Set device
        self.device = torch.device(config['device'])
        print(f"Using device: {self.device}")
        
        # Set seed for reproducibility
        set_seed(config['seed'])
    
    def create_model(self) -> nn.Module:
        """Create model based on experiment type"""
        model_config = ModelConfig(**self.config['model'])
        
        # For single-task experiments, we still use the multi-task model
        # but only train one head
        model = MultiTaskFallDetector(model_config)
        
        return model.to(self.device)
    
    def create_loss_function(self) -> MultiTaskLoss:
        """Create loss function based on experiment configuration"""
        loss_config = self.config['experiment']['loss']
        
        return MultiTaskLoss(
            num_activities=self.config['model']['num_activities'],
            weighting_strategy=loss_config.get('weighting_strategy', 'static'),
            activity_weight=loss_config.get('activity_weight', 0.3),
            fall_weight=loss_config.get('fall_weight', 0.7),
            focal_loss=loss_config.get('focal_loss', False),
            focal_alpha=loss_config.get('focal_alpha', 0.25),
            focal_gamma=loss_config.get('focal_gamma', 2.0)
        )
    
    def run(self, data_path: str = 'data/sisfall'):
        """Run the complete experiment"""
        
        print(f"\n{'='*50}")
        print(f"Running Experiment: {self.experiment_name}")
        print(f"Description: {self.config['experiment']['description']}")
        print(f"{'='*50}\n")
        
        # Create data loaders
        data_config = DataConfig(**self.config['data'])
        train_loader, val_loader, test_loader = create_dataloaders(
            data_path=data_path,
            batch_size=self.config['data']['batch_size'],
            config=data_config,
            num_workers=self.config['data']['num_workers']
        )
        
        print(f"Data loaded: {len(train_loader)} train batches, "
              f"{len(val_loader)} val batches, {len(test_loader)} test batches")
        
        # Create model
        model = self.create_model()
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created: {num_params:,} trainable parameters")
        
        # Create loss function
        loss_fn = self.create_loss_function()
        
        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Create scheduler
        if self.config['training']['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['training']['num_epochs']
            )
        else:
            scheduler = None
        
        # Create trainer
        trainer = MultiTaskTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=self.device,
            experiment_dir=self.output_dir,
            monitor_gradients=self.config['training']['monitor_gradients'],
            gradient_conflict_threshold=self.config['training']['gradient_conflict_threshold']
        )
        
        # Train model
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.config['training']['num_epochs'],
            scheduler=scheduler
        )
        
        # Load best model for evaluation
        checkpoint = torch.load(self.output_dir / 'checkpoints' / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model from epoch {checkpoint['epoch']}")
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        evaluator = MultiTaskEvaluator(
            model=model,
            device=self.device,
            activity_classes=IMUDataset.ACTIVITY_CLASSES,
            output_dir=self.output_dir / 'evaluation'
        )
        
        test_metrics = evaluator.evaluate(test_loader)
        
        # Save test results
        with open(self.output_dir / 'test_results.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            clean_metrics = self._clean_metrics_for_json(test_metrics)
            json.dump(clean_metrics, f, indent=2)
        
        # Check success criteria
        self._check_success_criteria(test_metrics)
        
        # Print summary
        self._print_summary(test_metrics)
        
        return test_metrics
    
    def _clean_metrics_for_json(self, metrics: Dict) -> Dict:
        """Convert numpy types to Python types for JSON serialization"""
        clean = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                clean[key] = self._clean_metrics_for_json(value)
            elif isinstance(value, (np.ndarray, np.generic)):
                clean[key] = value.tolist()
            elif isinstance(value, (tuple, list)):
                # Skip curves for JSON
                if key not in ['pr_curve', 'roc_curve', 'confusion_matrix']:
                    clean[key] = value
            else:
                clean[key] = value
        return clean
    
    def _check_success_criteria(self, metrics: Dict):
        """Check if experiment meets success criteria"""
        criteria = self.config['experiment']['success_criteria']
        
        print("\n" + "="*50)
        print("SUCCESS CRITERIA CHECK:")
        print("="*50)
        
        success = True
        
        for criterion, threshold in criteria.items():
            if criterion == 'pr_auc':
                value = metrics['fall']['pr_auc']
                passed = value >= threshold
                print(f"PR-AUC: {value:.3f} (target: ≥{threshold}) {'✓' if passed else '✗'}")
                success &= passed
                
            elif criterion == 'precision_at_95_recall':
                value = metrics['fall']['precision_at_95_recall']
                passed = value >= threshold
                print(f"Precision@0.95Recall: {value:.3f} (target: ≥{threshold}) {'✓' if passed else '✗'}")
                success &= passed
                
            elif criterion == 'macro_f1':
                value = metrics['activity']['macro_f1']
                passed = value >= threshold
                print(f"Activity Macro F1: {value:.3f} (target: ≥{threshold}) {'✓' if passed else '✗'}")
                success &= passed
                
            elif criterion == 'risky_fp_reduction':
                value = metrics['cross_task']['fp_reduction_ratio']
                passed = value >= threshold
                print(f"Risky FP Reduction: {value:.3f} (target: ≥{threshold}) {'✓' if passed else '✗'}")
                success &= passed
        
        print("\n" + ("EXPERIMENT PASSED! ✓" if success else "EXPERIMENT FAILED ✗"))
        print("="*50)
    
    def _print_summary(self, metrics: Dict):
        """Print experiment summary"""
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        
        print("\nActivity Recognition:")
        print(f"  Accuracy: {metrics['activity']['accuracy']:.3f}")
        print(f"  Macro F1: {metrics['activity']['macro_f1']:.3f}")
        
        print("\nFall Detection:")
        print(f"  Precision: {metrics['fall']['precision']:.3f}")
        print(f"  Recall: {metrics['fall']['recall']:.3f}")
        print(f"  F1: {metrics['fall']['f1']:.3f}")
        print(f"  PR-AUC: {metrics['fall']['pr_auc']:.3f}")
        print(f"  Precision@0.95Recall: {metrics['fall']['precision_at_95_recall']:.3f}")
        print(f"  Alarm Rate: {metrics['fall']['alarm_rate_per_hour']:.1f} per hour")
        
        print("\nCross-Task Analysis:")
        print(f"  Risky Activities FP Rate: {metrics['cross_task']['risky_activities_fp_rate']:.3f}")
        print(f"  Non-Risky Activities FP Rate: {metrics['cross_task']['non_risky_activities_fp_rate']:.3f}")
        print(f"  FP Reduction Ratio: {metrics['cross_task']['fp_reduction_ratio']:.3f}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Multi-Task Fall Detection Experiments')
    parser.add_argument(
        '--experiment',
        type=str,
        default='E3_MTL_Static',
        choices=['E1_ST_Fall', 'E2_ST_ADL', 'E3_MTL_Static', 
                'E4_MTL_Uncertainty', 'E5_MTL_GradNorm', 'E6_MTL_Focal'],
        help='Experiment to run'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/sisfall',
        help='Path to dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (auto-generated if not specified)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Create experiment configuration
    config = create_experiment_config(args.experiment)
    
    # Override with command line arguments
    config['device'] = args.device
    config['seed'] = args.seed
    
    # Run experiment
    runner = ExperimentRunner(config, args.output_dir)
    results = runner.run(args.data_path)
    
    print("\n✅ Experiment completed successfully!")
    print(f"Results saved to: {runner.output_dir}")


if __name__ == '__main__':
    main()
