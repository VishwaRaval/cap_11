#!/usr/bin/env python3
"""
Fish Detection Project - Master Script
Complete pipeline for ensemble, labeling, and deployment
"""

import argparse
from pathlib import Path
import sys
import json

# Import our modules
from ensemble_predictor import FishEnsemble, test_ensemble_combinations
from dataset_labeler import FishLabeler, label_dataset_with_top_models, create_live_demo
from edge_deployer import EdgeDeployer, deploy_top_models, create_deployment_guide


def setup_argparser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Fish Detection Project - Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ensemble on test dataset
  python main.py ensemble --dataset /path/to/test

  # Label dataset with top 5 models
  python main.py label --dataset /path/to/images --output labeled_outputs

  # Deploy models to edge formats
  python main.py deploy --models runs/detect/*/weights/best.pt

  # Live demo
  python main.py live --model runs/detect/extreme_stable_v1/weights/best.pt

  # Run everything
  python main.py all --dataset /path/to/test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Ensemble command
    ensemble_parser = subparsers.add_parser('ensemble', help='Run ensemble predictions')
    ensemble_parser.add_argument('--models', nargs='+', required=True,
                                help='Paths to model .pt files')
    ensemble_parser.add_argument('--weights', nargs='+', type=float, default=None,
                                help='Weights for each model (must sum to 1.0)')
    
    # Support both --data (YAML) and --dataset/--labels (directories)
    data_group = ensemble_parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--data',
                           help='Path to data.yaml file (contains paths to images and labels)')
    data_group.add_argument('--dataset',
                           help='Path to test dataset images directory')
    
    ensemble_parser.add_argument('--labels',
                                help='Path to test dataset labels directory (required if using --dataset)')
    ensemble_parser.add_argument('--method', choices=['wbf', 'voting'], default='wbf',
                                help='Ensemble method')
    ensemble_parser.add_argument('--conf', type=float, default=0.25,
                                help='Confidence threshold')
    ensemble_parser.add_argument('--iou', type=float, default=0.45,
                                help='IoU threshold')
    ensemble_parser.add_argument('--output', default='ensemble_results.json',
                                help='Output file for results')
    
    # Label command
    label_parser = subparsers.add_parser('label', help='Label dataset with models')
    label_parser.add_argument('--models', nargs='+', required=True,
                             help='Paths to model .pt files')
    label_parser.add_argument('--dataset', required=True,
                             help='Path to dataset images directory')
    label_parser.add_argument('--output', default='labeled_outputs',
                             help='Output directory for labeled data')
    label_parser.add_argument('--conf', type=float, default=0.25,
                             help='Confidence threshold')
    label_parser.add_argument('--no-visuals', action='store_true',
                             help='Skip creating visual outputs (only create label files)')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy models to edge formats')
    deploy_parser.add_argument('--models', nargs='+', required=True,
                              help='Paths to model .pt files')
    deploy_parser.add_argument('--output', default='edge_deployments',
                              help='Output directory for deployed models')
    deploy_parser.add_argument('--formats', nargs='+', 
                              choices=['onnx', 'tflite', 'coreml', 'torchscript', 'all'],
                              default=['all'],
                              help='Export formats')
    
    # Live demo command
    live_parser = subparsers.add_parser('live', help='Run live inference demo')
    live_parser.add_argument('--model', required=True,
                            help='Path to model .pt file')
    live_parser.add_argument('--source', default='0',
                            help='Video source (0 for webcam, or path to video file)')
    live_parser.add_argument('--conf', type=float, default=0.25,
                            help='Confidence threshold')
    
    # All command
    all_parser = subparsers.add_parser('all', help='Run complete pipeline')
    all_parser.add_argument('--models', nargs='+',
                           help='Paths to model .pt files (if not provided, uses top 5)')
    all_parser.add_argument('--dataset', required=True,
                           help='Path to test dataset images directory')
    all_parser.add_argument('--labels', required=True,
                           help='Path to test dataset labels directory')
    all_parser.add_argument('--output', default='project_outputs',
                           help='Base output directory')
    
    return parser


def run_ensemble(args):
    """Run ensemble predictions"""
    print(f"\n{'='*80}")
    print("ENSEMBLE PREDICTION")
    print(f"{'='*80}\n")
    
    # Parse data paths from YAML or direct arguments
    if args.data:
        import yaml
        print(f"Loading dataset info from: {args.data}")
        with open(args.data, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Extract paths from YAML
        # Support both absolute paths and paths relative to YAML location
        yaml_dir = Path(args.data).parent
        
        # Try to find test/val split
        if 'test' in data_config:
            dataset_path = data_config['test']
        elif 'val' in data_config:
            dataset_path = data_config['val']
        else:
            raise ValueError("data.yaml must contain 'test' or 'val' key")
        
        # Handle relative paths
        if not Path(dataset_path).is_absolute():
            dataset_path = yaml_dir / dataset_path
        
        # Assume standard YOLO structure: images/ and labels/ directories
        if Path(dataset_path).is_dir():
            # If it's a directory, look for images and labels subdirs
            if (Path(dataset_path) / 'images').exists():
                image_dir = str(Path(dataset_path) / 'images')
                label_dir = str(Path(dataset_path) / 'labels')
            else:
                # Assume it's the images directory
                image_dir = str(dataset_path)
                # Infer labels directory
                label_dir = str(Path(dataset_path).parent / 'labels')
        else:
            raise ValueError(f"Dataset path not found: {dataset_path}")
        
        print(f"  Images: {image_dir}")
        print(f"  Labels: {label_dir}")
    else:
        if not args.labels:
            raise ValueError("--labels is required when using --dataset")
        image_dir = args.dataset
        label_dir = args.labels
    
    # Create ensemble
    ensemble = FishEnsemble(
        model_paths=args.models,
        weights=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Evaluate
    metrics = ensemble.evaluate_dataset(
        image_dir=image_dir,
        labels_dir=label_dir,
        method=args.method
    )
    
    # Save results
    results = {
        'models': args.models,
        'weights': args.weights if args.weights else ensemble.weights,
        'method': args.method,
        'conf_threshold': args.conf,
        'iou_threshold': args.iou,
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'num_samples': int(metrics['num_samples'])
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1 Score:  {metrics['f1']*100:.2f}%")
    print(f"\nResults saved to: {args.output}")
    
    # Check if we hit 70% target
    if metrics['accuracy'] >= 0.70:
        print(f"\n🎉 SUCCESS! Achieved 70% accuracy target!")
    else:
        shortfall = (0.70 - metrics['accuracy']) * 100
        print(f"\n📊 Current: {metrics['accuracy']*100:.2f}% (need {shortfall:.2f}% more for 70% target)")


def run_labeling(args):
    """Run dataset labeling"""
    print(f"\n{'='*80}")
    print("DATASET LABELING")
    print(f"{'='*80}\n")
    
    output_base = Path(args.output)
    all_stats = {}
    
    for model_path in args.models:
        model_name = Path(model_path).parent.parent.name
        print(f"\nProcessing with: {model_name}")
        
        labeler = FishLabeler(model_path, model_name)
        
        output_dir = output_base / model_name
        stats = labeler.process_dataset(
            image_dir=args.dataset,
            output_base_dir=str(output_dir),
            conf_threshold=args.conf,
            create_labels=True,
            create_visuals=not args.no_visuals
        )
        
        all_stats[model_name] = stats
    
    # Save combined statistics
    with open(output_base / 'all_models_stats.json', 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n{'='*80}")
    print("LABELING COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {output_base}/")
    print(f"  - <model_name>/labels/  : YOLO format label files")
    if not args.no_visuals:
        print(f"  - <model_name>/visuals/ : Labeled images")
    print(f"  - all_models_stats.json : Combined statistics")


def run_deployment(args):
    """Run model deployment"""
    print(f"\n{'='*80}")
    print("EDGE DEPLOYMENT")
    print(f"{'='*80}\n")
    
    output_base = Path(args.output)
    all_results = []
    
    for model_path in args.models:
        deployer = EdgeDeployer(model_path)
        
        if 'all' in args.formats:
            results = deployer.export_all(str(output_base))
        else:
            # Export specific formats
            results = {
                'model_name': deployer.model_name,
                'exports': []
            }
            
            if 'onnx' in args.formats:
                results['exports'].append(deployer.export_onnx(str(output_base)))
            if 'tflite' in args.formats:
                for quant in [False, True]:  # FP32 and INT8
                    results['exports'].append(deployer.export_tflite(str(output_base), int8=quant))
            if 'coreml' in args.formats:
                results['exports'].append(deployer.export_coreml(str(output_base)))
            if 'torchscript' in args.formats:
                results['exports'].append(deployer.export_torchscript(str(output_base)))
        
        all_results.append(results)
    
    # Save combined results
    with open(output_base / 'all_exports.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create deployment guide
    create_deployment_guide()
    
    print(f"\n{'='*80}")
    print("DEPLOYMENT COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {output_base}/")
    print(f"📖 See DEPLOYMENT_GUIDE.md for usage instructions")


def run_live_demo(args):
    """Run live inference demo"""
    print(f"\n{'='*80}")
    print("LIVE DEMO")
    print(f"{'='*80}\n")
    
    import cv2
    
    labeler = FishLabeler(args.model)
    
    # Determine source
    if args.source.isdigit():
        source = int(args.source)
        print(f"Using webcam: {source}")
    else:
        source = args.source
        print(f"Using video file: {source}")
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        return
    
    print("\nControls:")
    print("  q - Quit")
    print("  s - Save screenshot")
    print("\nStarting demo...")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = labeler.model.predict(
            source=frame,
            conf=args.conf,
            verbose=False
        )[0]
        
        # Draw detections
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            
            color = labeler.CLASS_COLORS.get(cls, (255, 255, 255))
            
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            
            label = f"{labeler.CLASS_NAMES[cls]} {conf:.2f}"
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add info
        cv2.putText(frame, f"Model: {labeler.model_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"Detections: {len(results.boxes)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow('Fish Detection - Live Demo', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'screenshot_{timestamp}.jpg'
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames")


def run_all(args):
    """Run complete pipeline"""
    print(f"\n{'='*80}")
    print("RUNNING COMPLETE PIPELINE")
    print(f"{'='*80}\n")
    
    # Use default top 5 models if not provided
    if args.models is None:
        args.models = [
            'runs/detect/extreme_stable_v1/weights/best.pt',
            'runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt',
            'runs/detect/large_precision_v1_scratch/weights/best.pt',
            'runs/detect/extreme_stable_v2_native/weights/best.pt',
            'runs/detect/moderate_balanced_v1/weights/best.pt',
        ]
    
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 1. Ensemble
    print("\n" + "="*80)
    print("STEP 1/3: ENSEMBLE PREDICTIONS")
    print("="*80)
    
    ensemble_args = argparse.Namespace(
        models=args.models,
        weights=None,
        dataset=args.dataset,
        labels=args.labels,
        method='wbf',
        conf=0.25,
        iou=0.45,
        output=str(output_base / 'ensemble_results.json')
    )
    run_ensemble(ensemble_args)
    
    # 2. Labeling
    print("\n" + "="*80)
    print("STEP 2/3: DATASET LABELING")
    print("="*80)
    
    label_args = argparse.Namespace(
        models=args.models,
        dataset=args.dataset,
        output=str(output_base / 'labeled_outputs'),
        conf=0.25,
        no_visuals=False
    )
    run_labeling(label_args)
    
    # 3. Deployment
    print("\n" + "="*80)
    print("STEP 3/3: EDGE DEPLOYMENT")
    print("="*80)
    
    deploy_args = argparse.Namespace(
        models=args.models,
        output=str(output_base / 'edge_deployments'),
        formats=['all']
    )
    run_deployment(deploy_args)
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE PIPELINE FINISHED")
    print(f"{'='*80}")
    print(f"\nAll outputs saved to: {output_base}/")
    print(f"  - ensemble_results.json    : Ensemble evaluation metrics")
    print(f"  - labeled_outputs/         : Labeled datasets for each model")
    print(f"  - edge_deployments/        : Exported models for edge devices")
    print(f"  - DEPLOYMENT_GUIDE.md      : Deployment instructions")


def main():
    """Main entry point"""
    parser = setup_argparser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate function
    if args.command == 'ensemble':
        run_ensemble(args)
    elif args.command == 'label':
        run_labeling(args)
    elif args.command == 'deploy':
        run_deployment(args)
    elif args.command == 'live':
        run_live_demo(args)
    elif args.command == 'all':
        run_all(args)


if __name__ == "__main__":
    main()
