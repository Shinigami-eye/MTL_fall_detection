# File: scripts/hyperopt.py (NEW)
import optuna

def objective(trial):
    """Optuna objective for hyperparameter search."""
    
    # Suggest hyperparameters
    config = {
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
        'n_layers': trial.suggest_int('n_layers', 1, 4),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
    }
    
    # Train model with suggested hyperparameters
    model = create_model_with_config(config)
    trainer = MTLTrainer(model, config)
    val_metric = trainer.train_and_evaluate()
    
    return val_metric

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)