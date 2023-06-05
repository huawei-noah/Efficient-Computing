import logging
import os
import time
from datetime import datetime

import numpy as np

from timm.data import ImageDataset
from timm.data.mixup import Mixup, mixup_target
from timm.utils import CheckpointSaver, unwrap_model


class ImageNetInstanceSample(ImageDataset):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """

    def __init__(self, root, name, class_map, load_bytes, is_sample=False, k=4096, **kwargs):
        super().__init__(root, parser=name, class_map=class_map, load_bytes=load_bytes, **kwargs)
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 1000
            num_samples = len(self.parser)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                _, target = self.parser[i]
                label[i] = target

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = super().__getitem__(index)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


class MultiSmoothingMixup(Mixup):
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, smoothings=(0.1,), num_classes=1000):
        super(MultiSmoothingMixup, self).__init__(mixup_alpha, cutmix_alpha, cutmix_minmax, prob, switch_prob,
                                                  mode, correct_lam, 0, num_classes)
        self.smoothings = smoothings

    def __call__(self, x, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        targets = []
        for smoothing in self.smoothings:
            targets.append(mixup_target(target, self.num_classes, lam, smoothing, x.device))
        return x, targets


class CheckpointSaverWithLogger(CheckpointSaver):
    def __init__(
            self,
            logger,
            model,
            optimizer,
            args=None,
            model_ema=None,
            amp_scaler=None,
            checkpoint_prefix='checkpoint',
            recovery_prefix='recovery',
            checkpoint_dir='',
            recovery_dir='',
            decreasing=False,
            max_history=10,
            unwrap_fn=unwrap_model):
        super(CheckpointSaverWithLogger, self).__init__(model, optimizer, args, model_ema, amp_scaler,
                                                        checkpoint_prefix, recovery_prefix, checkpoint_dir,
                                                        recovery_dir, decreasing, max_history, unwrap_fn)
        self.logger = logger

    def save_checkpoint(self, epoch, metric=None):
        assert epoch >= 0
        tmp_save_path = os.path.join(self.checkpoint_dir, 'tmp' + self.extension)
        last_save_path = os.path.join(self.checkpoint_dir, 'last' + self.extension)
        self._save(tmp_save_path, epoch, metric)
        if os.path.exists(last_save_path):
            os.unlink(last_save_path)  # required for Windows support.
        os.rename(tmp_save_path, last_save_path)
        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if (len(self.checkpoint_files) < self.max_history
                or metric is None or self.cmp(metric, worst_file[1])):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)
            filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            os.link(last_save_path, save_path)
            self.checkpoint_files.append((save_path, metric))
            self.checkpoint_files = sorted(
                self.checkpoint_files, key=lambda x: x[1],
                reverse=not self.decreasing)  # sort in descending order if a lower metric is not better

            checkpoints_str = "Current checkpoints:\n"
            for c in self.checkpoint_files:
                checkpoints_str += ' {}\n'.format(c)
            self.logger.info(checkpoints_str)

            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                best_save_path = os.path.join(self.checkpoint_dir, 'model_best' + self.extension)
                if os.path.exists(best_save_path):
                    os.unlink(best_save_path)
                os.link(last_save_path, best_save_path)

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index < 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            try:
                self.logger.debug("Cleaning checkpoint: {}".format(d))
                os.remove(d[0])
            except Exception as e:
                self.logger.error("Exception '{}' while deleting checkpoint".format(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]

    def save_recovery(self, epoch, batch_idx=0):
        assert epoch >= 0
        filename = '-'.join([self.recovery_prefix, str(epoch), str(batch_idx)]) + self.extension
        save_path = os.path.join(self.recovery_dir, filename)
        self._save(save_path, epoch)
        if os.path.exists(self.last_recovery_file):
            try:
                self.logger.debug("Cleaning recovery: {}".format(self.last_recovery_file))
                os.remove(self.last_recovery_file)
            except Exception as e:
                self.logger.error("Exception '{}' while removing {}".format(e, self.last_recovery_file))
        self.last_recovery_file = self.curr_recovery_file
        self.curr_recovery_file = save_path


def setup_default_logging(logger, default_level=logging.INFO, log_path=''):
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(name)15s: %(message)s")
    console_handler.setFormatter(console_formatter)
    # console_handler.setFormatter(FormatterNoInfo())
    logger.addHandler(console_handler)
    logger.setLevel(default_level)
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_formatter = logging.Formatter("%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)


class TimePredictor:
    def __init__(self, steps, most_recent=30, drop_first=True):
        self.init_time = time.time()
        self.steps = steps
        self.most_recent = most_recent
        self.drop_first = drop_first  # drop iter 0

        self.time_list = []
        self.temp_time = self.init_time

    def update(self):
        time_interval = time.time() - self.temp_time
        self.time_list.append(time_interval)

        if self.drop_first and len(self.time_list) > 1:
            self.time_list = self.time_list[1:]
            self.drop_first = False

        self.time_list = self.time_list[-self.most_recent:]
        self.temp_time = time.time()

    def get_pred_text(self):
        single_step_time = np.mean(self.time_list)
        end_timestamp = self.init_time + single_step_time * self.steps
        return datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d %H:%M:%S')


def process_feat(distiller, source_feat):
    if getattr(distiller, 'pre_act_feat', False):
        feat = source_feat[0]
    else:
        feat = source_feat[1]
    return feat
