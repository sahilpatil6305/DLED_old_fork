# description: trainer
import os
import sys
import random
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import json
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from metrics.utils import get_test_metrics
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

FFpp_pool=['FaceForensics++','FF-DF','FF-F2F','FF-FS','FF-NT']


def move_batch_to_device(data_dict, device):
    """Move tensor batch values to one device while preserving metadata/None."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in data_dict.items()
    }


def _as_metric_scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu().item())
    return float(np.asarray(value, dtype=float).mean())


class Trainer(object):
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        metric_scoring='auc',
        time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        swa_model=None,
        train_loader=None,
        test_loaders=None
        ):
        # check if all the necessary components are implemented
        if config is None or model is None or logger is None:
            raise ValueError("config, model, logger, and tensorboard writer must be implemented")

        self.logger = logger
        self.logger.info("=== Initializing Validation Trainer ===")
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.threshold_ratio = 0.05
        print("uncertainty thrshold ratio: ", self.threshold_ratio)
        
        if torch.cuda.is_available():
            if self.config['device_choice'] == 'new':
                local_rank = int(os.environ.get(
                    "LOCAL_RANK", self.config.get('local_rank', 0)
                ))
                self.device = torch.device("cuda", local_rank)
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.logger.info(f"Using device: {self.device}")
        
        self.metric_scoring = metric_scoring
        # maintain the best metric of all epochs
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        self.speed_up()  # move model to GPU

        # get current time
        self.timenow = time_now
        # create directory path
        if 'task_target' not in config:
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + task_str + '_' + self.timenow
            )
        self.logger.info(f"Logging to: {self.log_dir}")

        # Record testing statistics
        self.uncertainty_dict = {}
        self.spatial_acc_dict, self.frequency_acc_dict, self.fused_acc_dict = {}, {}, {}
        self.spatial_binary_acc_dict, self.frequency_binary_acc_dict, self.fused_binary_acc_dict = {}, {}, {}
        self.acc_real_dict = {}
        self.acc_fake_dict = {}

        for dataset_name in self.config['test_dataset']:
            self.spatial_acc_dict[dataset_name] = []
            self.spatial_binary_acc_dict[dataset_name] = []
            self.frequency_acc_dict[dataset_name] = []
            self.frequency_binary_acc_dict[dataset_name] = []
            self.fused_acc_dict[dataset_name] = []
            self.fused_binary_acc_dict[dataset_name] = []

            self.acc_real_dict[dataset_name] = []
            self.acc_fake_dict[dataset_name] = []
        
        os.makedirs(self.log_dir, exist_ok=True)

    def log_class_distribution(self, data_loader, phase):
        """Log the distribution of classes in a dataset"""
        # Access labels directly from the dataset instead of loading data
        if hasattr(data_loader.dataset, 'data_dict') and 'label' in data_loader.dataset.data_dict:
            labels = data_loader.dataset.data_dict['label']
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            else:
                labels = np.array(labels)
        else:
            self.logger.warning(f"Could not efficiently access labels for {phase} dataset, skipping distribution analysis")
            return
            
        unique_labels, counts = np.unique(labels, return_counts=True)
        dist_str = f"\n{phase} dataset class distribution:"
        total = len(labels)
        
        # Log absolute counts and percentages for each class
        for label, count in zip(unique_labels, counts):
            class_name = "Real" if label == 0 else f"Fake_{label}"
            percentage = (count / total) * 100
            dist_str += f"\n  {class_name}: {count} ({percentage:.1f}%)"
        
        self.logger.info(dist_str)

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            # update directory path
            writer_path = os.path.join(
                self.log_dir,
                phase,
                dataset_key,
                metric_key,
                "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
            # update writers dictionary
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]

    def speed_up(self):
        self.model.to(self.device)
        self.model.device = self.device
        if self.config['ddp'] == True:
            num_gpus = torch.cuda.device_count()
            print(f'avai gpus: {num_gpus}')
            # local_rank=[i for i in range(0,num_gpus)]
            self.model = DDP(self.model, device_ids=[self.config['local_rank']],find_unused_parameters=True, output_device=self.config['local_rank'])
            #self.optimizer =  nn.DataParallel(self.optimizer, device_ids=[int(os.environ['LOCAL_RANK'])])

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key,ckpt_info=None):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ddp'] == True:
            torch.save(self.model.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save({'R': self.model.R,
                            'c': self.model.c,
                            'state_dict': self.model.state_dict(),}, save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")

    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key, epoch=None, is_best=False):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        epoch_tag = f"{epoch + 1:03d}" if epoch is not None else "final"
        prediction_arrays = {}
        report = {}
        for name, value in metric_one_dataset.items():
            if isinstance(value, np.ndarray):
                prediction_arrays[name] = value
            elif isinstance(value, np.generic):
                report[name] = value.item()
            elif isinstance(value, dict):
                report[name] = {
                    key: (item.item() if isinstance(item, np.generic) else item)
                    for key, item in value.items()
                }
            else:
                report[name] = value

        metrics_path = os.path.join(save_dir, f"metrics_epoch_{epoch_tag}.json")
        with open(metrics_path, 'w', encoding='utf-8') as file:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write('\n')
        if prediction_arrays:
            predictions_path = os.path.join(save_dir, f"predictions_epoch_{epoch_tag}.npz")
            np.savez_compressed(predictions_path, **prediction_arrays)
        if is_best:
            file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
            with open(file_path, 'wb') as file:
                pickle.dump(metric_one_dataset, file)
            self.logger.info(f"Best metrics saved to {file_path}")
        self.logger.info(f"Epoch metrics saved to {metrics_path}")

    def save_latest_checkpoint(self, epoch):
        """Save model weights after each completed epoch for recovery."""
        if not self.config.get('save_model', False):
            return
        save_dir = self.config.get('save_path', self.log_dir)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'ckpt_latest.pth')
        model = self.model.module if isinstance(self.model, DDP) else self.model
        torch.save(model.state_dict(), path)
        with open(os.path.join(save_dir, 'latest_epoch.json'), 'w', encoding='utf-8') as file:
            json.dump({'completed_epoch': epoch + 1, 'checkpoint': path}, file, indent=2)
            file.write('\n')
        self.logger.info(f"Latest checkpoint saved to {path} (completed epoch {epoch + 1})")

    def train_step(self, data_dict, epoch, total_epoch):
        if hasattr(self, 'step_count'):
            self.step_count += 1
        else:
            self.step_count = 0

        losses = dict()
        fused_total_loss = 0
        predictions = None

        if type(self.model) is DDP:
            self.model.module.model_zero_grad()
        else:
            self.model.model_zero_grad()

        fused_prediction = self.model(data_dict, 'fusion')
        fused_prediction.update({'epoch': epoch, 'total_epoch':total_epoch, 'training':True})
        if type(self.model) is DDP:
            fused_losses = self.model.module.get_losses(data_dict, fused_prediction, mode='train')
        else:
            fused_losses = self.model.get_losses(data_dict, fused_prediction, mode='train')

        if type(self.model) is DDP:
            self.model.module.model_backward(fused_losses['overall'])
        else:
            self.model.model_backward(fused_losses['overall'])

        # Gradient clipping threshold ramps up over the first ~1000 steps.
        base_clip = 1.0
        clip_value = base_clip * min(5, 1 + self.step_count / 1000)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_value)

        log_interval = 10 if self.step_count < 100 else 100
        if self.step_count % log_interval == 0:
            if type(self.model) is DDP:
                current_lr = self.model.module.optim.param_groups[0]['lr']
            else:
                current_lr = self.model.optim.param_groups[0]['lr']
            self.logger.info(f"Fusion Step {self.step_count} - lr: {current_lr:.6f}, clip: {clip_value:.5f}")
            
            # Log prediction distribution on training data
            if isinstance(fused_prediction, dict) and 'logits' in fused_prediction:
                logits = fused_prediction['logits']
                probs = torch.softmax(logits, dim=1)
                mean_probs = probs.mean(dim=0)
                self.logger.info(f"Mean class probabilities: {[f'{p:.5f}' for p in mean_probs.tolist()]}")

        if type(self.model) is DDP:
            self.model.module.model_update()
        else:
            self.model.model_update()
    
        for k, v in fused_losses.items():
            if 'loss' in k:
                fused_total_loss += v
                losses.update({"fused_" + k: v})
        losses.update({'overall': fused_total_loss})
        predictions = fused_prediction
        return losses, predictions

    def train_epoch(
        self,
        epoch,
        train_data_loader,
        test_data_loaders=None,
        test_epoch_freq = 2
        ):

        self.logger.info("===> Epoch[{}] start!".format(epoch))
        step_cnt = epoch * len(train_data_loader)

        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)
        epoch_loss_values = defaultdict(list)
        epoch_metric_values = defaultdict(list)
    
        for iteration, data_dict in tqdm(enumerate(train_data_loader),total=len(train_data_loader)):
            self.setTrain()
            data_dict = move_batch_to_device(data_dict, self.device)

            losses, predictions=self.train_step(data_dict, epoch, self.config['nEpochs'])

            # compute training metric for each batch data
            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            # store data by recorder
            ## store metric
            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
                epoch_metric_values[name].append(_as_metric_scalar(value))
            ## store loss
            for name, value in losses.items():
                train_recorder_loss[name].update(value)
                epoch_loss_values[name].append(_as_metric_scalar(value))

            # run tensorboard to visualize the training process
            if iteration % 300 == 0 and self.config['local_rank']==0:
                # info for loss
                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg == None:
                        loss_str += f"training-loss, {k}: not calculated"
                        continue
                    loss_str += f"training-loss, {k}: {v_avg}    "
                    # tensorboard-1. loss
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss/{k}', v_avg, global_step=step_cnt)
                self.logger.info(loss_str)
                # info for metric
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg == None:
                        metric_str += f"training-metric, {k}: not calculated    "
                        continue
                    metric_str += f"training-metric, {k}: {v_avg}    "
                    # tensorboard-2. metric
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
                self.logger.info(metric_str)

                # clear recorder.
                # Note we only consider the current 300 samples for computing batch-level loss/metric
                for name, recorder in train_recorder_loss.items():  # clear loss recorder
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():  # clear metric recorder
                    recorder.clear()

            # run test
            test_best_metric = None
            step_cnt += 1

        epoch_loss = {name: float(np.mean(values)) for name, values in epoch_loss_values.items() if values}
        epoch_metrics = {name: float(np.mean(values)) for name, values in epoch_metric_values.items() if values}
        self.logger.info(f"Epoch[{epoch + 1}] training loss means: {epoch_loss}")
        self.logger.info(f"Epoch[{epoch + 1}] training metric means: {epoch_metrics}")

        # adjust learning rate
        if type(self.model) is DDP:
            self.model.module.update_lr()
        else:
            self.model.update_lr()

        self.save_latest_checkpoint(epoch)

        if epoch == 0 or (epoch+1)%test_epoch_freq==0:
            u_threshold = self.calculate_uthre(train_data_loader)
            self.uncertainty_dict[epoch] = u_threshold
            if test_data_loaders is not None and (not self.config['ddp'] ):
                self.logger.info("===> Test start!")
                test_best_metric = self.test_epoch(
                    epoch,
                    iteration,
                    test_data_loaders,
                    step_cnt,
                    u_threshold
                )
            elif test_data_loaders is not None and (self.config['ddp'] and dist.get_rank() == 0):
                self.logger.info("===> Test start!")
                test_best_metric = self.test_epoch(
                    epoch,
                    iteration,
                    test_data_loaders,
                    step_cnt,
                    u_threshold
                )
        return test_best_metric

    def get_respect_acc(self,prob,label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = (pred == label)
        zero_num = len(label) - np.count_nonzero(label)
        if len(judge[zero_num:])!=0:
            acc_fake = np.count_nonzero(judge[zero_num:]) / len(judge[zero_num:])
        else:
            acc_fake = 0
        if len(judge[:zero_num])!=0:
            acc_real = np.count_nonzero(judge[:zero_num]) / len(judge[:zero_num])
        else:
            acc_real = 0
        return acc_real,acc_fake

    def test_one_dataset(self, data_loader, u_threshold):
        """Test the model on one dataset."""
        self.setEval()
        image_total = 0
        spatial_correct_total, frequency_correct_total, fused_correct_total = 0, 0, 0
        spatial_binary_correct_total, frequency_binary_correct_total, fused_binary_correct_total = 0, 0, 0
        spatial_predictions_nps, frequency_predictions_nps, fused_predictions_nps = [], [], []
        label_nps = []
        spatial_pred_cls_nps, frequency_pred_cls_nps, fused_pred_cls_nps = [], [], []
        feature_nps = []
        spatial_binary_y_pred, frequency_binary_y_pred, fused_binary_y_pred = [], [], []
        binary_y_true = []
        spatial_all_probs, frequency_all_probs, fused_all_probs = [], [], []

        all_confidences_spatial, all_confidences_frequency, all_confidences_fused = [], [], []

        spatial_category_correct_dict = {x:0 for x in range(self.config['num_classes']+1)}
        spatial_category_correct_total = {x:0 for x in range(self.config['num_classes']+1)}

        frequency_category_correct_dict = {x:0 for x in range(self.config['num_classes']+1)}
        frequency_category_correct_total = {x:0 for x in range(self.config['num_classes']+1)}

        fused_category_correct_dict = {x:0 for x in range(self.config['num_classes']+1)}
        fused_category_correct_total = {x:0 for x in range(self.config['num_classes']+1)}
        
        # Track losses
        losses_one_dataset_recorder = defaultdict(AverageMeter)
        
        # Get number of classes from config
        num_classes = self.config['num_classes']
            
        # Log class distribution before testing
        self.log_class_distribution(data_loader, "test")
        
        with torch.no_grad():
            for data_dict in tqdm(data_loader, desc='Testing'):
                image_total += len(data_dict['label'])
                
                # Move data to device
                data_dict_cuda = {}
                for k, v in data_dict.items():
                    if isinstance(v, torch.Tensor):
                        data_dict_cuda[k] = v.to(self.device)
                    else:
                        data_dict_cuda[k] = v
                
                try:
                    # Forward pass
                    predictions, predictions_np, label_np, feature_np = self.inference(data_dict_cuda, u_threshold)

                    # Calculate confidences using softmax probabilities
                    confidences_spatial = np.max(predictions_np['spatial'], axis=1)
                    all_confidences_spatial.extend(confidences_spatial.tolist())
                    confidences_frequency = np.max(predictions_np['frequency'], axis=1)
                    all_confidences_frequency.extend(confidences_frequency.tolist())
                    confidences_fused = np.max(predictions_np['fused'], axis=1)
                    all_confidences_fused.extend(confidences_fused.tolist())
                    
                    # Calculate losses only for known classes
                    known_mask = label_np < num_classes
                    if np.any(known_mask):
                        # Handle data dict masking
                        known_dict = {k: v[known_mask] if isinstance(v, torch.Tensor) and v.shape[0] == len(known_mask) else v 
                                    for k, v in data_dict_cuda.items()}
                        
                        # Handle predictions masking - special case for attention maps
                        known_pred = {}
                        for k, v in predictions.items():
                            # key: prob, pred_alpha, image_feat
                            if k == 'image_feat':
                                # Don't mask attention maps since they have different dimensions
                                known_pred[k] = v
                            elif k == 'pred_alpha':
                                if predictions[k]['spatial'].shape[0] == len(known_mask):
                                    known_pred['spatial_pred_alpha'] = predictions[k]['spatial'][known_mask]
                                else:
                                    known_pred['spatial_pred_alpha'] = predictions[k]['spatial']
                                if predictions[k]['frequency'].shape[0] == len(known_mask):
                                    known_pred['frequency_pred_alpha'] = predictions[k]['frequency'][known_mask]
                                else:
                                    known_pred['frequency_pred_alpha'] = predictions[k]['frequency']
                                if predictions[k]['fused'].shape[0] == len(known_mask):
                                    known_pred['pred_alpha'] = predictions[k]['fused'][known_mask]
                                else:
                                    known_pred['pred_alpha'] = predictions[k]['fused']
                            else:
                                known_pred[k] = {}
                                for domain in ['spatial', 'frequency', 'fused']:
                                    sub_dict = predictions[k][domain]
                                    if sub_dict.shape[0] == len(known_mask):
                                        known_pred[k][domain] = predictions[k][domain][known_mask]
                                    else:
                                        known_pred[k][domain] = predictions[k][domain]
                        known_pred.update({'training':False})

                        if type(self.model) is DDP:
                            losses = self.model.module.get_losses(known_dict, known_pred)
                        else:
                            losses = self.model.get_losses(known_dict, known_pred)
                        
                        # Update loss meters
                        for k, v in losses.items():
                            if torch.isnan(v) or torch.isinf(v):
                                continue
                            losses_one_dataset_recorder[k].update(v.item(), known_dict['label'].size(0))
                    
                    # Store predictions and features
                    spatial_predictions_nps.append(predictions_np['spatial'])
                    frequency_predictions_nps.append(predictions_np['frequency'])
                    fused_predictions_nps.append(predictions_np['fused'])
                    feature_nps.append(feature_np)

                    spatial_pred_classes = predictions['prob']['spatial'].cpu().detach().numpy()
                    spatial_correct_total += np.sum(spatial_pred_classes == label_np)
                    frequency_pred_classes = predictions['prob']['frequency'].cpu().detach().numpy()
                    frequency_correct_total += np.sum(frequency_pred_classes == label_np)
                    fused_pred_classes = predictions['prob']['fused'].cpu().detach().numpy()
                    fused_correct_total += np.sum(fused_pred_classes == label_np)
                    
                    # Calculate binary accuracy (real vs fake) using proper class mapping
                    spatial_binary_pred = (spatial_pred_classes != 0).astype(int)  # Assuming 0 is real class
                    frequency_binary_pred = (frequency_pred_classes != 0).astype(int)  # Assuming 0 is real class
                    fused_binary_pred = (fused_pred_classes != 0).astype(int)  # Assuming 0 is real class
                    
                    for class_idx in range(num_classes+1):
                        class_mask = label_np == class_idx
                        if np.any(class_mask):
                            spatial_category_correct_dict[class_idx] += np.sum(spatial_pred_classes[class_mask] == class_idx)
                            spatial_category_correct_total[class_idx] += len(spatial_pred_classes[class_mask])

                            frequency_category_correct_dict[class_idx] += np.sum(frequency_pred_classes[class_mask] == class_idx)
                            frequency_category_correct_total[class_idx] += len(frequency_pred_classes[class_mask])

                            fused_category_correct_dict[class_idx] += np.sum(fused_pred_classes[class_mask] == class_idx)
                            fused_category_correct_total[class_idx] += len(fused_pred_classes[class_mask])

                    binary_label = (label_np != 0).astype(int)
                    spatial_binary_correct_total += np.sum(spatial_binary_pred == binary_label)
                    frequency_binary_correct_total += np.sum(frequency_binary_pred == binary_label)
                    fused_binary_correct_total += np.sum(fused_binary_pred == binary_label)

                    spatial_binary_y_pred.extend(spatial_binary_pred.tolist())
                    frequency_binary_y_pred.extend(frequency_binary_pred.tolist())
                    fused_binary_y_pred.extend(fused_binary_pred.tolist())
                    binary_y_true.extend(binary_label.tolist())

                    label_nps.extend(label_np.tolist())
                    spatial_pred_cls_nps.extend(spatial_pred_classes.tolist())
                    frequency_pred_cls_nps.extend(frequency_pred_classes.tolist())
                    fused_pred_cls_nps.extend(fused_pred_classes.tolist())
                    
                    # Calculate fake probability properly
                    if predictions_np['spatial'].shape[1] == 2:  # Binary classification
                        spatial_fake_prob = predictions_np['spatial'][:, 1]  # Use fake class probability directly
                    else:  # Multi-class
                        spatial_fake_prob = np.sum(predictions_np['spatial'][:, 1:], axis=1)  # Sum all fake class probabilities
                    spatial_all_probs.extend(spatial_fake_prob.tolist())

                    if predictions_np['frequency'].shape[1] == 2:  # Binary classification
                        frequency_fake_prob = predictions_np['frequency'][:, 1]  # Use fake class probability directly
                    else:  # Multi-class
                        frequency_fake_prob = np.sum(predictions_np['frequency'][:, 1:], axis=1)  # Sum all fake class probabilities
                    frequency_all_probs.extend(frequency_fake_prob.tolist())

                    if predictions_np['fused'].shape[1] == 2:  # Binary classification
                        fake_prob = predictions_np['fused'][:, 1]  # Use fake class probability directly
                    else:  # Multi-class
                        fake_prob = np.sum(predictions_np['fused'][:, 1:], axis=1)  # Sum all fake class probabilities
                    fused_all_probs.extend(fake_prob.tolist())
                    
                except Exception as e:
                    self.logger.error(f"Error in batch processing: {str(e)}")
                    raise
        
        # Calculate metrics
        spatial_acc = spatial_correct_total/image_total
        spatial_binary_acc = spatial_binary_correct_total/image_total
        frequency_acc = frequency_correct_total/image_total
        frequency_binary_acc = frequency_binary_correct_total/image_total
        fused_acc = fused_correct_total/image_total
        fused_binary_acc = fused_binary_correct_total/image_total
        
        # Calculate AUC and other metrics
        y_true = np.array(binary_y_true)
        spatial_y_pred = np.array(spatial_binary_y_pred)
        spatial_y_prob = np.array(spatial_all_probs)
        spatial_confidences = np.array(all_confidences_spatial)

        frequency_y_pred = np.array(frequency_binary_y_pred)
        frequency_y_prob = np.array(frequency_all_probs)
        frequency_confidences = np.array(all_confidences_frequency)

        fused_y_pred = np.array(fused_binary_y_pred)
        fused_y_prob = np.array(fused_all_probs)
        fused_confidences = np.array(all_confidences_fused)
        fused_precision, fused_recall, fused_f1, _ = metrics.precision_recall_fscore_support(
            y_true, fused_y_pred, average='binary', zero_division=0
        ) if len(binary_y_true) else (0.0, 0.0, 0.0, None)
        fused_confusion = metrics.confusion_matrix(y_true, fused_y_pred, labels=[0, 1]).tolist()
        class_counts = np.bincount(y_true, minlength=2).tolist()

        label_np = np.array(label_nps)
        spatial_pred_classes = np.array(spatial_pred_cls_nps)
        frequency_pred_classes = np.array(frequency_pred_cls_nps)
        fused_pred_classes = np.array(fused_pred_cls_nps)

        try:
            spatial_auc = roc_auc_score(y_true, spatial_y_prob)
            spatial_ap = average_precision_score(y_true, spatial_y_prob)
            fpr, tpr, thresholds = roc_curve(y_true, spatial_y_prob)
            fnr = 1 - tpr
            spatial_eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

            frequency_auc = roc_auc_score(y_true, frequency_y_prob)
            frequency_ap = average_precision_score(y_true, frequency_y_prob)
            fpr, tpr, thresholds = roc_curve(y_true, frequency_y_prob)
            fnr = 1 - tpr
            frequency_eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

            fused_auc = roc_auc_score(y_true, fused_y_prob)
            fused_ap = average_precision_score(y_true, fused_y_prob)
            fpr, tpr, thresholds = roc_curve(y_true, fused_y_prob)
            fnr = 1 - tpr
            fused_eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
            
            # Calculate per-class metrics
            for class_idx in range(num_classes+1):
                class_mask = label_np == class_idx
                if spatial_category_correct_total[class_idx] != 0:
                    class_acc = spatial_category_correct_dict[class_idx] / spatial_category_correct_total[class_idx]
                    self.logger.info(f"Class {class_idx} spatial accuracy: {class_acc:.5f}")
                if frequency_category_correct_total[class_idx] != 0:
                    class_acc = frequency_category_correct_dict[class_idx] / frequency_category_correct_total[class_idx]
                    self.logger.info(f"Class {class_idx} frequency accuracy: {class_acc:.5f}")
                if fused_category_correct_total[class_idx] != 0:
                    class_acc = fused_category_correct_dict[class_idx] / fused_category_correct_total[class_idx]
                    self.logger.info(f"Class {class_idx} fused accuracy: {class_acc:.5f}")
            
            # Log detailed prediction statistics
            spatial_mean_conf = np.mean(confidences_spatial)
            spatial_std_conf = np.std(confidences_spatial)
            self.logger.info(f"Sptial Prediction confidence - mean: {spatial_mean_conf:.5f}, std: {spatial_std_conf:.5f}")
            self.logger.info(f"Sptial Prediction confidence range - min: {np.min(confidences_spatial):.5f}, max: {np.max(confidences_spatial):.5f}")
            
            frequency_mean_conf = np.mean(confidences_frequency)
            frequency_std_conf = np.std(confidences_frequency)
            self.logger.info(f"Sptial Prediction confidence - mean: {frequency_mean_conf:.5f}, std: {frequency_std_conf:.5f}")
            self.logger.info(f"Sptial Prediction confidence range - min: {np.min(confidences_frequency):.5f}, max: {np.max(confidences_frequency):.5f}")
            
            fused_mean_conf = np.mean(confidences_fused)
            fused_std_conf = np.std(confidences_fused)
            self.logger.info(f"Sptial Prediction confidence - mean: {fused_mean_conf:.5f}, std: {fused_std_conf:.5f}")
            self.logger.info(f"Sptial Prediction confidence range - min: {np.min(confidences_fused):.5f}, max: {np.max(confidences_fused):.5f}")
            
            # Log probability distribution
            spatial_mean_probs = np.mean(predictions_np['spatial'], axis=0)
            self.logger.info(f"Spatial Mean probabilities per class: {[f'{p:.5f}' for p in spatial_mean_probs]}")

            frequency_mean_probs = np.mean(predictions_np['frequency'], axis=0)
            self.logger.info(f"Frequency Mean probabilities per class: {[f'{p:.5f}' for p in frequency_mean_probs]}")
            
            fused_mean_probs = np.mean(predictions_np['fused'], axis=0)
            self.logger.info(f"Fused Mean probabilities per class: {[f'{p:.5f}' for p in fused_mean_probs]}")

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            spatial_auc, spatial_eer, spatial_ap = -1, -1, -1
            spatial_mean_conf, spatial_std_conf = -1, -1

            frequency_auc, frequency_eer, frequency_ap = -1, -1, -1
            frequency_mean_conf, frequency_std_conf = -1, -1

            fused_auc, fused_eer, fused_ap = -1, -1, -1
            fused_mean_conf, fused_std_conf = -1, -1

        metric_one_dataset = {
            'spatial_acc': spatial_acc,
            'spatial_binary_acc': spatial_binary_acc,
            'frequency_acc': frequency_acc,
            'frequency_binary_acc': frequency_binary_acc,
            'fused_acc': fused_acc,
            'fused_binary_acc': fused_binary_acc,

            'spatial_auc': spatial_auc,
            'spatial_eer': spatial_eer,
            'spatial_ap': spatial_ap,
            'frequency_auc': frequency_auc,
            'frequency_eer': frequency_eer,
            'frequency_ap': frequency_ap,
            'fused_auc': fused_auc,
            'fused_eer': fused_eer,
            'fused_ap': fused_ap,
            'fused_precision': float(fused_precision),
            'fused_recall': float(fused_recall),
            'fused_f1': float(fused_f1),
            'binary_confusion_matrix_real_fake': fused_confusion,
            'binary_class_counts_real_fake': class_counts,

            'spatial_pred': spatial_y_pred,
            'frequency_pred': frequency_y_pred,
            'fused_pred': fused_y_pred,
            'spatial_fake_probability': spatial_y_prob,
            'frequency_fake_probability': frequency_y_prob,
            'fused_fake_probability': fused_y_prob,
            'label': y_true,
            'losses': {k: v.avg for k, v in losses_one_dataset_recorder.items()},
            'spatial_mean_confidence': spatial_mean_conf,
            'spatial_std_confidence': spatial_std_conf,
            'frequency_mean_confidence': frequency_mean_conf,
            'frequency_std_confidence': frequency_std_conf,
            'fused_mean_confidence': fused_mean_conf,
            'fused_std_confidence': fused_std_conf
        }
        
        fused_predictions = np.concatenate(fused_predictions_nps, axis=0) if fused_predictions_nps else np.empty((0,))
        features = np.concatenate(feature_nps, axis=0) if feature_nps else np.empty((0,))
        return losses_one_dataset_recorder, fused_predictions, np.array(label_nps), features, metric_one_dataset

    def test_epoch(self, epoch, iteration, test_data_loaders, step, u_threshold):
        # set model to eval mode
        self.setEval()

        # define test recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric
        avg_metric = {'fused_acc': 0, 'fused_binary_acc': 0, 'fused_auc': 0, 'fused_eer': 0, 'fused_ap': 0, 'dataset_dict':{}}
        # testing for all test data
        keys = list(test_data_loaders.keys())
        for key_idx in range(len(keys)):
            test_key = keys[key_idx]

            # compute loss for each dataset
            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps, metric_one_dataset = self.test_one_dataset(test_data_loaders[test_key], u_threshold)
            losses_all_datasets[test_key] = losses_one_dataset_recorder
            self.spatial_acc_dict[test_key].append(metric_one_dataset['spatial_acc'])
            self.spatial_binary_acc_dict[test_key].append(metric_one_dataset['spatial_binary_acc'])
            self.frequency_acc_dict[test_key].append(metric_one_dataset['frequency_acc'])
            self.frequency_binary_acc_dict[test_key].append(metric_one_dataset['frequency_binary_acc'])
            self.fused_acc_dict[test_key].append(metric_one_dataset['fused_acc'])
            self.fused_binary_acc_dict[test_key].append(metric_one_dataset['fused_binary_acc'])

            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name]+=value
            avg_metric['dataset_dict'][test_key] = metric_one_dataset["fused_"+self.metric_scoring]
            if type(self.model) is AveragedModel:
                metric_str = f"Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    metric_str += f"testing-metric, {k}: {v}    "
                self.logger.info(metric_str)
                continue
            self.save_best(epoch,iteration,step,losses_one_dataset_recorder,test_key,metric_one_dataset)

        if len(keys)>0 and self.config.get('save_avg',False):
            # calculate avg value
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= len(keys)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric)

        self.logger.info('===> Test Done!')
        return self.best_metrics_all_time  # return all types of mean metrics for determining the best ckpt

    def save_best(self, epoch, iteration, step, losses_one_dataset_recorder, key, metric_one_dataset):
        """Save the best model and metrics."""
        save_dir = os.path.join(self.log_dir, 'test', key)
        os.makedirs(save_dir, exist_ok=True)
        primary_metric = f"fused_{self.metric_scoring}"
        primary_value = metric_one_dataset.get(primary_metric, -1)
        primary_best = self.best_metrics_all_time[key][primary_metric]
        primary_is_valid = isinstance(primary_value, (int, float, np.number)) and np.isfinite(primary_value) and primary_value >= 0
        is_best = primary_is_valid and (
            primary_value < primary_best if self.metric_scoring == 'eer' else primary_value > primary_best
        )

        # Log dataset metrics
        metrics_str = f"dataset: {key}    step: {step}    "
        
        # Log losses
        if losses_one_dataset_recorder:
            for loss_name, loss_meter in losses_one_dataset_recorder.items():
                metrics_str += f"testing-loss, {loss_name}: {loss_meter.avg}    "
            self.logger.info(metrics_str)

        # Log metrics
        metrics_str = f"dataset: {key}    step: {step}    "
        acc_classes = []
        for metric_name, metric_value in metric_one_dataset.items():
            if metric_name not in ['spatial_pred', 'frequency_pred', 'fused_pred', 'label', 'losses']:
                if isinstance(metric_value, (int, float)) and metric_value != -1:  # Only process numeric metrics
                    if not np.isfinite(metric_value):
                        continue
                    metrics_str += f"testing-metric, {metric_name}: {metric_value}    "
                    
                    # Initialize if not present
                    if metric_name not in self.best_metrics_all_time[key]:
                        self.best_metrics_all_time[key][metric_name] = float('-inf') if self.metric_scoring != 'eer' else float('inf')
                    
                    # Update best metrics
                    current_best = self.best_metrics_all_time[key][metric_name]
                    is_better = False
                    if self.metric_scoring == 'eer':
                        is_better = metric_value < current_best
                    else:
                        is_better = metric_value > current_best
                        
                    if is_better:
                        self.best_metrics_all_time[key][metric_name] = metric_value
                        metrics_str += f"(new best)    "
        
        # Add class-specific accuracies
        if 'fused_pred' in metric_one_dataset and 'label' in metric_one_dataset:
            y_true = metric_one_dataset['label']

            spatial_y_pred = metric_one_dataset['spatial_pred']
            spatial_acc_real = np.mean((spatial_y_pred == 0) & (y_true == 0)) if np.any(y_true == 0) else 0
            spatial_acc_fake = np.mean((spatial_y_pred == 1) & (y_true == 1)) if np.any(y_true == 1) else 0
            metrics_str += f"spatial_acc_real:{spatial_acc_real:.5f}; spatial_acc_fake:{spatial_acc_fake:.5f}"
            
            frequency_y_pred = metric_one_dataset['frequency_pred']
            frequency_acc_real = np.mean((frequency_y_pred == 0) & (y_true == 0)) if np.any(y_true == 0) else 0
            frequency_acc_fake = np.mean((frequency_y_pred == 1) & (y_true == 1)) if np.any(y_true == 1) else 0
            metrics_str += f"frequency_acc_real:{frequency_acc_real:.5f}; frequency_acc_fake:{frequency_acc_fake:.5f}"
            
            fused_y_pred = metric_one_dataset['fused_pred']
            fused_acc_real = np.mean((fused_y_pred == 0) & (y_true == 0)) if np.any(y_true == 0) else 0
            fused_acc_fake = np.mean((fused_y_pred == 1) & (y_true == 1)) if np.any(y_true == 1) else 0
            metrics_str += f"fused_acc_real:{fused_acc_real:.5f}; fused_acc_fake:{fused_acc_fake:.5f}"          

        self.logger.info(metrics_str)

        # Preserve each validation epoch and update the best snapshot only when
        # the configured fused metric improves.
        self.save_metrics('test', metric_one_dataset, key, epoch=epoch, is_best=is_best)
        if is_best:
            ckpt_name = 'ckpt_best.pth'
            save_path = os.path.join(save_dir, ckpt_name)
            model = self.model.module if isinstance(self.model, DDP) else self.model
            torch.save(model.state_dict(), save_path)
            self.logger.info(
                f"Best checkpoint saved to {save_path}, {primary_metric}={primary_value} "
                f"at epoch {epoch + 1} (iteration {iteration})"
            )


    @torch.no_grad()
    def calculate_uthre(self,train_data_loader):
        self.logger.info("===> Calculate Uncertainty Threshold !")
        un_dict = {'overall': {'spatial': [], 'frequency': [], 'fused': []}}
        thre_dict={'spatial':{}, 'frequency':{}, 'fused':{}}
        label_value_list = list(range(self.config['backbone_config']['num_classes']))

        for count, label_spe in enumerate(label_value_list):
            un_dict[label_spe] = {'spatial': [],
                                  'frequency': [],
                                  'fused': []}
   
        for _, data_dict in tqdm(enumerate(train_data_loader),total=len(train_data_loader)):
            data_dict = move_batch_to_device(data_dict, self.device)
            if type(self.model) is DDP:
                predictions = self.model.module.model_full_fp(data_dict)
            else:
                predictions = self.model.model_full_fp(data_dict)
            un_dict['overall']['spatial'].extend(predictions['spatial_vacuity'])
            un_dict['overall']['frequency'].extend(predictions['frequency_vacuity'])
            un_dict['overall']['fused'].extend(predictions['vacuity'])
            
            for label_spe in label_value_list:
                label_mask = data_dict['label'] == label_spe
                un_dict[label_spe]['spatial'].extend(predictions['spatial_vacuity'][label_mask])
                un_dict[label_spe]['frequency'].extend(predictions['frequency_vacuity'][label_mask])
                un_dict[label_spe]['fused'].extend(predictions['vacuity'][label_mask])

        for uncer_key in un_dict.keys():
            for branch in thre_dict.keys():
                un_dict[uncer_key][branch].sort(reverse=True)
                thre_index =int(len(un_dict[uncer_key][branch])*self.threshold_ratio)
                Re_thre = un_dict[uncer_key][branch][thre_index].cpu().item()
                thre = Re_thre
                thre_dict[branch][uncer_key] = thre
        self.logger.info("Retrieved uncertainty dict {} !".format(thre_dict))
        return thre_dict

    def inference(self, data_dict, u_threshold):
        """Run inference on a batch of data.
        
        Args:
            data_dict: Dictionary containing input data (already on correct device)
            
        Returns:
            Tuple containing:
            - predictions: Model predictions
            - predictions_np: Numpy array of predictions
            - label_np: Numpy array of labels
            - feature_np: Numpy array of features
        """
        # Get predictions
        with torch.no_grad():
            if type(self.model) is DDP:
                predictions = self.model.module.model_inference(data_dict, u_threshold)
            else:
                predictions = self.model.model_inference(data_dict, u_threshold)
            predictions_np = {}
            spatial_probs = F.softmax(predictions['pred_alpha']['spatial'], dim=1)  # [B, num_classes]
            frequency_probs = F.softmax(predictions['pred_alpha']['frequency'], dim=1)  # [B, num_classes]
            fused_probs = F.softmax(predictions['pred_alpha']['fused'], dim=1)  # [B, num_classes]

            # Get max probability and entropy for each prediction
            # max_probs, pred_classes = torch.max(probs, dim=1)  # [B], [B]
            # entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # [B]
            
            # Convert to numpy for metric calculation
            spatial_predictions_np = spatial_probs.cpu().detach().numpy()  # [B, num_classes]
            frequency_predictions_np = frequency_probs.cpu().detach().numpy()  # [B, num_classes]
            fused_predictions_np = fused_probs.cpu().detach().numpy()  # [B, num_classes]

            label_np = data_dict['label'].cpu().detach().numpy()  # [B]
            feature_np = predictions['image_feat'][0].cpu().detach().numpy()  # [B, C, H, W]
            
            # Keep original probabilities for confidence metrics
            predictions['pred_alpha']['spatial'] = spatial_probs
            predictions['pred_alpha']['frequency'] = frequency_probs
            predictions['pred_alpha']['fused'] = fused_probs
            
            predictions_np['spatial'] = spatial_predictions_np
            predictions_np['frequency'] = frequency_predictions_np
            predictions_np['fused'] = fused_predictions_np

            return predictions, predictions_np, label_np, feature_np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
