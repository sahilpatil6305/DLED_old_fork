"""
Evaluation script for DLED (Dual-Level Evidence Detector).

Loads a trained `dled` checkpoint and reports spatial / frequency /
fused accuracy on the requested test datasets.
"""
import os
import argparse
import datetime
import json
import random
import shlex
import sys
from pathlib import Path
import time
import yaml
from tqdm import tqdm
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from sklearn import metrics as sklearn_metrics

from detectors import DETECTOR
from dataset import DeepfakeFrequencyDataset
from logger import create_logger
from trainer.trainer import move_batch_to_device
from experiment_tracking import create_run, file_sha256


def get_args():
    parser = argparse.ArgumentParser(description='DLED evaluation script.')
    parser.add_argument('--detector_path', type=str,
                        default='./training/config/detector/dled.yaml',
                        help='Path to detector YAML config.')
    parser.add_argument('--train_config', type=str,
                        default='./training/config/train_config.yaml',
                        help='Path to the training YAML config.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained checkpoint (.pth) to evaluate.')
    parser.add_argument('--train_dataset', nargs='+', required=True,
                        help='Train dataset name (used only to recompute the uncertainty threshold).')
    parser.add_argument('--test_dataset', nargs='+', required=True)
    parser.add_argument('--max_samples_per_class', type=int, default=None)
    parser.add_argument('--max_calibration_samples_per_class', type=int, default=None,
                        help='Optional cap for the training-only uncertainty calibration loader.')
    parser.add_argument('--workers', type=int, default=None,
                        help='Override the configured data-loader worker count.')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device_choice', choices=['old', 'new'], default='old')
    parser.add_argument('--training_mode', type=str, default='multi',
                        choices=['single', 'multi'])
    parser.add_argument('--training_data_ratio', type=float, default=1.0)
    parser.add_argument('--task_target', type=str, default='test')
    parser.add_argument('--detect_novel', action='store_true', default=False)
    parser.add_argument('--avu_reg', action='store_true', default=True)
    parser.add_argument('--kl_reg', action='store_true', default=False)
    parser.add_argument('--ddp', action='store_true', default=False)
    parser.add_argument('--dataset_json_folder', type=str, default=None,
                        help='Override the JSON directory for training and evaluation datasets.')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Legacy compatibility option; logs go to the tracked run directory.')
    parser.add_argument('--metrics_output', type=str, default=None,
                        help='Optional metrics filename; output stays inside the tracked run.')
    parser.add_argument('--no_balance_test_data', action='store_true',
                        help='Evaluate every referenced test frame instead of downsampling to equal class counts.')
    parser.add_argument('--experiment_type', default='evaluation',
                        help='Subdirectory under results/ for this isolated run.')
    parser.add_argument('--output_root', default='results',
                        help='Parent directory for automatically isolated run outputs.')
    return parser.parse_args()


def init_seed(config):
    seed = config.get('manualSeed') or 123456
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config['cuda'] and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_train_loader(config):
    train_config = config.copy()
    train_config['max_samples_per_class'] = config.get(
        'calibration_max_samples_per_class', config.get('max_samples_per_class'))
    train_set = DeepfakeFrequencyDataset(config=train_config, mode='train')
    return torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config['train_batchSize'],
        shuffle=True,
        num_workers=int(config['workers']),
        collate_fn=train_set.collate_fn,
    )


def get_test_loaders(config):
    loaders = {}
    for test_name in config['test_dataset']:
        cfg = config.copy()
        cfg['test_dataset'] = test_name
        test_set = DeepfakeFrequencyDataset(config=cfg, mode='test')
        loaders[test_name] = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=config['test_batchSize'],
            shuffle=False,
            num_workers=int(config['workers']),
            collate_fn=test_set.collate_fn,
            drop_last=(test_name == 'DeepFakeDetection'),
        )
    return loaders


@torch.no_grad()
def calculate_uthre(train_data_loader, num_classes, model, threshold_ratio=0.05):
    un_dict = {i: {'spatial': [], 'frequency': [], 'fused': []} for i in range(num_classes)}
    thre_dict = {'spatial': {}, 'frequency': {}, 'fused': {}}

    for data_dict in tqdm(train_data_loader, total=len(train_data_loader), desc='Calibrating'):
        data_dict = move_batch_to_device(data_dict, model.device)
        predictions = model.model_full_fp(data_dict)
        for label_spe in range(num_classes):
            label_mask = data_dict['label'] == label_spe
            un_dict[label_spe]['spatial'].extend(
                predictions['spatial_vacuity'][label_mask].detach().cpu().tolist())
            un_dict[label_spe]['frequency'].extend(
                predictions['frequency_vacuity'][label_mask].detach().cpu().tolist())
            un_dict[label_spe]['fused'].extend(
                predictions['vacuity'][label_mask].detach().cpu().tolist())

    for uncer_key in un_dict.keys():
        for branch in thre_dict.keys():
            un_dict[uncer_key][branch].sort(reverse=True)
            thre_index = int(len(un_dict[uncer_key][branch]) * threshold_ratio)
            thre_dict[branch][uncer_key] = float(un_dict[uncer_key][branch][thre_index])
    print('Retrieved uncertainty dict {} !'.format(thre_dict))
    return thre_dict


def main():
    args = get_args()
    run = create_run(
        args.experiment_type,
        dataset_combination=','.join(args.train_dataset + args.test_dataset),
        command=shlex.join([sys.executable] + sys.argv),
        output_root=args.output_root,
        config_id=f"{','.join(args.train_dataset)}_to_{','.join(args.test_dataset)}",
        metadata={'run_kind': 'evaluation', 'checkpoint': args.model_path},
    )
    try:
        with run.capture_console():
            print(f"RUN_DIR={run.path}")
            _execute(args, run)
    except BaseException as exc:
        run.finish('FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    run.finish('COMPLETED', completed_epochs=0)


def _execute(args, run):
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.train_config, 'r') as f:
        train_cfg = yaml.safe_load(f)
    if 'label_dict' in config:
        train_cfg['label_dict'] = config['label_dict']
    config.update(train_cfg)
    config['log_dir'] = str(run.path / 'logs')
    if args.workers is not None:
        config['workers'] = args.workers
        config.setdefault('DATALOADER', {})['NUM_WORKERS'] = args.workers
    if not torch.cuda.is_available():
        config.setdefault('TRAINER', {})['PREC'] = 'fp32'

    if args.device_choice == 'old':
        local_rank = args.local_rank
    else:
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    config['local_rank'] = local_rank

    config.update({
        'train_dataset': args.train_dataset,
        'test_dataset': args.test_dataset,
        'max_samples_per_class': args.max_samples_per_class,
        'calibration_max_samples_per_class': (
            args.max_calibration_samples_per_class
            if args.max_calibration_samples_per_class is not None
            else args.max_samples_per_class
        ),
        'training_data_ratio': args.training_data_ratio,
        'detect_novel': args.detect_novel,
        'num_classes': args.num_classes,
        'training_mode': args.training_mode,
        'device_choice': args.device_choice,
        'ddp': args.ddp,
        'avu_reg': args.avu_reg,
        'kl_reg': args.kl_reg,
        'balance_test_data': not args.no_balance_test_data,
    })
    if args.dataset_json_folder:
        config['dataset_json_folder'] = args.dataset_json_folder
    config['log_dir'] = str(run.path / 'logs')
    config.setdefault('backbone_config', {})
    config['backbone_config']['num_classes'] = args.num_classes

    with open(run.path / 'config.yaml', 'w', encoding='utf-8') as config_file:
        yaml.safe_dump(config, config_file, sort_keys=True)
    run.update(seed=config.get('manualSeed'), dataset_json_folder=config.get('dataset_json_folder'),
               device='cuda' if torch.cuda.is_available() else 'cpu',
               manifest_hashes={str(path): file_sha256(path) for path in
                                sorted(Path(config['dataset_json_folder']).glob('*.json'))})

    timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = f'_{args.task_target}' if args.task_target else ''
    logger_path = os.path.join(config['log_dir'],
                               config['model_name'] + task_str + '_' + timenow)
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'evaluation.log'))
    logger.info(f'Save log to {logger_path}')

    init_seed(config)
    if config['cudnn']:
        cudnn.benchmark = True
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')

    train_data_loader = get_train_loader(config)
    test_data_loaders = get_test_loaders(config)

    model_class = DETECTOR['dled']
    model = model_class(config)
    device = model.device
    model.load_model(args.model_path)
    model.eval()

    num_classes = config['num_classes']
    u_threshold = calculate_uthre(train_data_loader, num_classes, model)

    for test_name, loader in test_data_loaders.items():
        image_total = 0
        spatial_correct = frequency_correct = fused_correct = 0
        spatial_bin_correct = frequency_bin_correct = fused_bin_correct = 0
        labels_all = []
        sample_paths = []
        predictions_all = {b: [] for b in ('spatial', 'frequency', 'fused')}
        scores_all = {b: [] for b in ('spatial', 'frequency', 'fused')}
        inference_seconds = 0.0
        cat_correct = {b: {x: 0 for x in range(num_classes + 1)}
                       for b in ('spatial', 'frequency', 'fused')}
        cat_total = {b: {x: 0 for x in range(num_classes + 1)}
                     for b in ('spatial', 'frequency', 'fused')}

        for data_dict in tqdm(loader, desc=f'Testing[{test_name}]'):
            image_total += len(data_dict['label'])
            label_np = data_dict['label'].numpy()
            start = len(sample_paths)
            sample_paths.extend(loader.dataset.image_list[start:start + len(label_np)])
            data_dict_cuda = move_batch_to_device(data_dict, device)
            with torch.no_grad():
                inference_start = time.perf_counter()
                predictions = model.model_inference(data_dict_cuda, u_threshold)
                inference_seconds += time.perf_counter() - inference_start

            preds = {b: predictions['prob'][b].cpu().numpy() for b in ('spatial', 'frequency', 'fused')}
            labels_all.extend(label_np.tolist())
            for branch in ('spatial', 'frequency', 'fused'):
                predictions_all[branch].extend(preds[branch].tolist())
                alpha = predictions['pred_alpha'][branch].float()
                probabilities = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)
                fake_probability = probabilities[:, 1] if num_classes == 2 else probabilities[:, 1:].sum(dim=1)
                scores_all[branch].extend(fake_probability.cpu().numpy().tolist())
            spatial_correct += np.sum(preds['spatial'] == label_np)
            frequency_correct += np.sum(preds['frequency'] == label_np)
            fused_correct += np.sum(preds['fused'] == label_np)

            binary_label = (label_np != 0).astype(int)
            spatial_bin_correct += np.sum((preds['spatial'] != 0).astype(int) == binary_label)
            frequency_bin_correct += np.sum((preds['frequency'] != 0).astype(int) == binary_label)
            fused_bin_correct += np.sum((preds['fused'] != 0).astype(int) == binary_label)

            for class_idx in range(num_classes + 1):
                class_mask = label_np == class_idx
                if not np.any(class_mask):
                    continue
                for b in ('spatial', 'frequency', 'fused'):
                    cat_correct[b][class_idx] += np.sum(preds[b][class_mask] == class_idx)
                    cat_total[b][class_idx] += int(class_mask.sum())

        logger.info(f'==== {test_name} ====')
        for b in ('spatial', 'frequency', 'fused'):
            for class_idx in range(num_classes + 1):
                if cat_total[b][class_idx]:
                    acc = cat_correct[b][class_idx] / cat_total[b][class_idx]
                    logger.info(f'Class {class_idx} {b} accuracy: {acc:.5f}')
        logger.info(f'Spatial   binary: {spatial_bin_correct / image_total:.5f}, '
                    f'multi: {spatial_correct / image_total:.5f}')
        logger.info(f'Frequency binary: {frequency_bin_correct / image_total:.5f}, '
                    f'multi: {frequency_correct / image_total:.5f}')
        logger.info(f'Fused     binary: {fused_bin_correct / image_total:.5f}, '
                    f'multi: {fused_correct / image_total:.5f}')

        binary_labels = (np.asarray(labels_all) != 0).astype(int)
        report = {
            'dataset': test_name,
            'class_mapping': {'real': 0, 'fake': 1},
            'sample_count': int(image_total),
            'uncertainty_thresholds': u_threshold,
            'per_class_sample_counts': {
                'real': int(np.sum(binary_labels == 0)),
                'fake': int(np.sum(binary_labels == 1)),
            },
            'inference_seconds': inference_seconds,
            'inference_images_per_second': image_total / inference_seconds if inference_seconds else None,
            'branches': {},
        }
        prediction_arrays = {
            'image_path': np.asarray(sample_paths, dtype=str),
            'binary_labels': binary_labels,
        }
        for branch in ('spatial', 'frequency', 'fused'):
            branch_preds = np.asarray(predictions_all[branch])
            binary_preds = (branch_preds != 0).astype(int)
            scores = np.asarray(scores_all[branch], dtype=float)
            auc = (float(sklearn_metrics.roc_auc_score(binary_labels, scores))
                   if len(np.unique(binary_labels)) == 2 else None)
            precision, recall, f1, _ = sklearn_metrics.precision_recall_fscore_support(
                binary_labels, binary_preds, average='binary', zero_division=0
            )
            report['branches'][branch] = {
                'multiclass_accuracy': float(np.mean(branch_preds == np.asarray(labels_all))),
                'binary_accuracy': float(sklearn_metrics.accuracy_score(binary_labels, binary_preds)),
                'precision_fake': float(precision),
                'recall_fake': float(recall),
                'f1_fake': float(f1),
                'roc_auc_fake_score': auc,
                'confusion_matrix_real_fake': sklearn_metrics.confusion_matrix(
                    binary_labels, binary_preds, labels=[0, 1]
                ).tolist(),
            }
            prediction_arrays[f'{branch}_class_prediction'] = branch_preds
            prediction_arrays[f'{branch}_fake_probability'] = scores
        logger.info(f"{test_name} binary metrics: {json.dumps(report, sort_keys=True)}")
        if args.metrics_output or run:
            metrics_name = os.path.basename(args.metrics_output) if args.metrics_output else f'{test_name}_metrics.json'
            metrics_path = str(run.path / 'predictions' / metrics_name)
            if len(test_data_loaders) > 1:
                stem, ext = os.path.splitext(metrics_path)
                metrics_path = f'{stem}_{test_name}{ext or ".json"}'
            os.makedirs(os.path.dirname(metrics_path) or '.', exist_ok=True)
            with open(metrics_path, 'w', encoding='utf-8') as file:
                json.dump(report, file, indent=2, allow_nan=False)
                file.write('\n')
            prediction_path = os.path.splitext(metrics_path)[0] + '_predictions.npz'
            np.savez_compressed(prediction_path, **prediction_arrays)
            logger.info(f"Metrics saved to {metrics_path}; predictions saved to {prediction_path}")
            run.update(test_accuracy=report['branches']['fused']['binary_accuracy'],
                       test_roc_auc=report['branches']['fused']['roc_auc_fake_score'],
                       metrics_path=metrics_path, predictions_path=prediction_path,
                       evaluation_dataset=test_name)


if __name__ == '__main__':
    main()
