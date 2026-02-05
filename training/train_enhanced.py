#!/usr/bin/env python3
"""
ChessWorldAI - Enhanced Training Script
Fine-tune YOLO model on chess piece dataset with optimized settings
"""

import torch
from pathlib import Path
from ultralytics import YOLO
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train YOLO for chess pieces")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--model', default='models/pieces.pt', help='Base model')
    parser.add_argument('--data', default='datasets/chess_pieces/data.yaml')
    parser.add_argument('--name', default='pieces_enhanced')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--freeze', type=int, default=0, help='Freeze first N layers')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ChessWorldAI Enhanced Training")
    print("="*60)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected, training will be slow!")
        args.device = 'cpu'
        
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Base model not found: {model_path}")
        print("Using yolov8n.pt as base...")
        model = YOLO('yolov8n.pt')
    else:
        model = YOLO(str(model_path))
        print(f"Loaded: {model_path}")
        
    # Check dataset
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Dataset config not found: {data_path}")
        sys.exit(1)
    print(f"Dataset: {data_path}")
    
    # Training configuration
    train_args = {
        'data': str(data_path),
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': args.workers,
        'patience': args.patience,
        'name': args.name,
        'exist_ok': True,
        'verbose': True,
        
        # Optimization
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # Augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 5,
        'translate': 0.1,
        'scale': 0.3,
        'shear': 2,
        'perspective': 0.0001,
        'flipud': 0.0,
        'fliplr': 0.0,
        'mosaic': 0.5,
        'mixup': 0.1,
        
        # Loss
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # Other
        'close_mosaic': 10,
        'amp': True,
        'cache': True,
        'rect': False,
        'single_cls': False,
        'fraction': 1.0,
        
        # Saving
        'save': True,
        'save_period': 10,
        'plots': True,
    }
    
    if args.resume:
        train_args['resume'] = True
        
    if args.freeze > 0:
        train_args['freeze'] = args.freeze
        print(f"Freezing first {args.freeze} layers")
        
    print("\nTraining Configuration:")
    for key in ['epochs', 'batch', 'imgsz', 'device', 'patience', 'lr0']:
        print(f"  {key}: {train_args.get(key)}")
        
    print("\n" + "-"*60)
    print("Starting training...")
    print("-"*60 + "\n")
    
    try:
        results = model.train(**train_args)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        # Best model location
        best_path = Path(f"runs/detect/{args.name}/weights/best.pt")
        if best_path.exists():
            output_path = Path(f"models/{args.name}.pt")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy(best_path, output_path)
            print(f"Best model saved: {output_path}")
            
        # Show metrics
        if hasattr(results, 'results_dict'):
            print("\nFinal Metrics:")
            for k, v in results.results_dict.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}")
                    
    except KeyboardInterrupt:
        print("\nTraining interrupted!")
        print("Partial results may be in runs/detect/" + args.name)
        
    except Exception as e:
        print(f"\nTraining error: {e}")
        raise


def validate_model(model_path: str, data_path: str):
    """Validate a trained model"""
    model = YOLO(model_path)
    results = model.val(data=data_path)
    return results


if __name__ == '__main__':
    main()
