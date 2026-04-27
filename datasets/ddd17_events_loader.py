import glob
from pathlib import Path
from os.path import join, exists, dirname, basename
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as f
import torchvision.transforms as transforms

from datasets.extract_data_tools.example_loader_ddd17 import load_files_in_directory, extract_events_from_memmap
import datasets.data_util as data_util
import albumentations as A
from PIL import Image
from utils.labels import shiftUpId, shiftDownId
from utils.ddd17_pseudolabel_utils import build_pseudolabel_path, load_pseudolabel_mask
from utils.fcclip_confidence_utils import load_confidence_map
from utils.fcclip_top2_soft_utils import load_top2_soft_artifact, top2_soft_artifact_path


def resolve_pseudolabel_root(pseudolabels_path, split):
    pseudolabel_root = Path(pseudolabels_path)
    if (pseudolabel_root / split).is_dir():
        return pseudolabel_root
    if pseudolabel_root.name == split and pseudolabel_root.is_dir():
        return pseudolabel_root.parent
    raise FileNotFoundError(f"could not resolve DDD17 pseudolabel root for split {split}: {pseudolabels_path}")


def normalize_pseudolabel_mask(mask, num_classes, ignore_label=255):
    mask = np.asarray(mask, dtype=np.int64)
    normalized = np.full(mask.shape, ignore_label, dtype=np.int64)
    valid = (mask >= 0) & (mask < num_classes)
    normalized[valid] = mask[valid]
    normalized[mask == ignore_label] = ignore_label
    return normalized


def normalize_confidence_map(confidence):
    confidence = np.asarray(confidence, dtype=np.float32)
    if confidence.ndim != 2:
        raise ValueError(f'confidence map must be 2D, got shape {confidence.shape}')
    if np.isnan(confidence).any():
        raise ValueError('confidence map contains NaN values')
    return np.clip(confidence, 0.0, 1.0)


def normalize_top2_class_map(class_map, num_classes):
    class_map = np.asarray(class_map, dtype=np.int64)
    if class_map.ndim != 2:
        raise ValueError(f'top-2 class map must be 2D, got shape {class_map.shape}')
    if np.any(class_map < 0) or np.any(class_map >= num_classes):
        raise ValueError('top-2 class map contains out-of-range values')
    return class_map


def normalize_top2_probability_map(probability_map):
    probability_map = np.asarray(probability_map, dtype=np.float32)
    if probability_map.ndim != 2:
        raise ValueError(f'top-2 probability map must be 2D, got shape {probability_map.shape}')
    if np.isnan(probability_map).any():
        raise ValueError('top-2 probability map contains NaN values')
    return np.clip(probability_map, 0.0, 1.0)


def get_split(dirs, split):
    return {
        "train": [dirs[0], dirs[2], dirs[3], dirs[5], dirs[6]],
        "test": [dirs[4]],
        "valid": [dirs[1]]
    }[split]


def unzip_segmentation_masks(dirs):
    for d in dirs:
        assert exists(join(d, "segmentation_masks.zip"))
        if not exists(join(d, "segmentation_masks")):
            print("Unzipping segmentation mask in %s" % d)
            os.system("unzip %s -d %s" % (join(d, "segmentation_masks"), d))


class DDD17Events(Dataset):
    def __init__(self, root, split="train", event_representation='voxel_grid',
                 nr_events_data=5, delta_t_per_data=50, nr_bins_per_data=5, require_paired_data=False,
                 separate_pol=False, normalize_event=False, augmentation=False, fixed_duration=False,
                 nr_events_per_data=32000, resize=True, random_crop=False, pseudolabels_path="",
                 pseudolabel_ignore_label=255, semseg_num_classes=6, confidence_path="", top2_soft_path="",
                 max_samples=None):
        data_dirs = sorted(glob.glob(join(root, "dir*")))
        assert len(data_dirs) > 0
        assert split in ["train", "valid", "test"]

        self.split = split
        self.augmentation = augmentation
        self.fixed_duration = fixed_duration
        self.nr_events_per_data = nr_events_per_data

        self.nr_events_data = nr_events_data
        self.delta_t_per_data = delta_t_per_data
        if self.fixed_duration:
            self.t_interval = nr_events_data * delta_t_per_data
        else:
            self.t_interval = -1
            self.nr_events = self.nr_events_data * self.nr_events_per_data
        assert self.t_interval in [10, 50, 250, -1]
        self.nr_temporal_bins = nr_bins_per_data
        self.require_paired_data = require_paired_data
        self.event_representation = event_representation
        self.shape = [260, 346]
        self.resize = resize
        self.shape_resize = [260, 352]
        self.random_crop = random_crop
        self.shape_crop = [120, 216]
        self.separate_pol = separate_pol
        self.normalize_event = normalize_event
        self.pseudolabels_path = pseudolabels_path
        self.pseudolabel_ignore_label = pseudolabel_ignore_label
        self.semseg_num_classes = semseg_num_classes
        self.confidence_path = confidence_path
        self.top2_soft_path = top2_soft_path
        if self.confidence_path and self.top2_soft_path:
            raise ValueError('DDD17Events does not support confidence_path together with top2_soft_path')
        self.dirs = get_split(data_dirs, split)
        # unzip_segmentation_masks(self.dirs)

        self.files = []
        for d in self.dirs:
            self.files += glob.glob(join(d, "segmentation_masks", "*.png"))
        self.files = sorted(self.files)
        if max_samples is not None:
            self.files = self.files[:int(max_samples)]

        print("[DDD17Segmentation]: Found %s segmentation masks for split %s" % (len(self.files), split))

        # load events and image_idx -> event index mapping
        self.img_timestamp_event_idx = {}
        self.event_data = {}

        print("[DDD17Segmentation]: Loading real events.")
        self.event_dirs = self.dirs

        for d in self.event_dirs:
            img_timestamp_event_idx, t_events, xyp_events, _ = load_files_in_directory(d, self.t_interval)
            self.img_timestamp_event_idx[d] = img_timestamp_event_idx
            self.event_data[d] = [t_events, xyp_events]

        if self.augmentation:
            self.transform_a = A.ReplayCompose([
                A.HorizontalFlip(p=0.5)
            ])
            self.transform_a_random_crop = A.ReplayCompose([
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True)])
        self.transform_a_center_crop = A.ReplayCompose([
            A.CenterCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True),
        ])

    def __len__(self):
        return len(self.files)

    def apply_augmentation(self, transform_a, events, label, pseudolabel=None, confidence=None, top2_soft=None):
        label = shiftUpId(label)
        A_data = transform_a(image=events[0, :, :].numpy(), mask=label)
        label = A_data['mask']
        label = shiftDownId(label)
        if self.random_crop and self.split == 'train':
            events_tensor = torch.zeros((events.shape[0], self.shape_crop[0], self.shape_crop[1]))
        else:
            events_tensor = events
        for k in range(events.shape[0]):
            events_tensor[k, :, :] = torch.from_numpy(
                A.ReplayCompose.replay(A_data['replay'], image=events[k, :, :].numpy())['image'])
        if confidence is not None:
            confidence = A.ReplayCompose.replay(A_data['replay'],
                                                image=confidence.astype(np.float32))['image'].astype(np.float32)
        if top2_soft is not None:
            top2_soft = {
                'top1_id': A.ReplayCompose.replay(
                    A_data['replay'], image=events[0, :, :].numpy(), mask=top2_soft['top1_id']
                )['mask'],
                'top2_id': A.ReplayCompose.replay(
                    A_data['replay'], image=events[0, :, :].numpy(), mask=top2_soft['top2_id']
                )['mask'],
                'p1': A.ReplayCompose.replay(
                    A_data['replay'], image=top2_soft['p1'].astype(np.float32)
                )['image'].astype(np.float32),
                'p2': A.ReplayCompose.replay(
                    A_data['replay'], image=top2_soft['p2'].astype(np.float32)
                )['image'].astype(np.float32),
            }
        if pseudolabel is not None and confidence is not None:
            pseudolabel = A.ReplayCompose.replay(A_data['replay'], image=events[0, :, :].numpy(), mask=pseudolabel)['mask']
            return events_tensor, label, pseudolabel, confidence
        if pseudolabel is not None and top2_soft is not None:
            pseudolabel = A.ReplayCompose.replay(A_data['replay'], image=events[0, :, :].numpy(), mask=pseudolabel)['mask']
            return events_tensor, label, pseudolabel, top2_soft
        if pseudolabel is not None:
            pseudolabel = A.ReplayCompose.replay(A_data['replay'], image=events[0, :, :].numpy(), mask=pseudolabel)['mask']
            return events_tensor, label, pseudolabel
        if confidence is not None:
            return events_tensor, label, confidence
        return events_tensor, label

    def __getitem__(self, idx):
        segmentation_mask_file = self.files[idx]
        segmentation_mask = cv2.imread(segmentation_mask_file, 0)
        label_original = np.array(segmentation_mask)
        if self.resize:
            segmentation_mask = cv2.resize(segmentation_mask, (self.shape_resize[1], self.shape_resize[0] - 60),
                                           interpolation=cv2.INTER_NEAREST)
        label = np.array(segmentation_mask)

        pseudolabel = None
        confidence = None
        top2_soft = None
        if self.split == 'train' and self.require_paired_data and self.pseudolabels_path:
            pseudolabel_root = resolve_pseudolabel_root(self.pseudolabels_path, self.split)
            pseudolabel_path = build_pseudolabel_path(segmentation_mask_file, pseudolabel_root)
            pseudolabel = load_pseudolabel_mask(pseudolabel_path)
            pseudolabel = normalize_pseudolabel_mask(pseudolabel,
                                                     num_classes=self.semseg_num_classes,
                                                     ignore_label=self.pseudolabel_ignore_label)
            if self.resize:
                pseudolabel = cv2.resize(pseudolabel.astype(np.float32),
                                         (self.shape_resize[1], self.shape_resize[0] - 60),
                                         interpolation=cv2.INTER_NEAREST).astype(np.int64)
            if self.top2_soft_path:
                top2_soft_root = resolve_pseudolabel_root(self.top2_soft_path, self.split)
                top2_soft_mask_path = build_pseudolabel_path(segmentation_mask_file, top2_soft_root)
                top2_soft_artifact = load_top2_soft_artifact(top2_soft_artifact_path(top2_soft_mask_path))
                top2_soft = {
                    'top1_id': normalize_top2_class_map(top2_soft_artifact.top1_id, self.semseg_num_classes),
                    'top2_id': normalize_top2_class_map(top2_soft_artifact.top2_id, self.semseg_num_classes),
                    'p1': normalize_top2_probability_map(top2_soft_artifact.p1),
                    'p2': normalize_top2_probability_map(top2_soft_artifact.p2),
                }
                if self.resize:
                    top2_soft['top1_id'] = cv2.resize(
                        top2_soft['top1_id'].astype(np.float32),
                        (self.shape_resize[1], self.shape_resize[0] - 60),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.int64)
                    top2_soft['top2_id'] = cv2.resize(
                        top2_soft['top2_id'].astype(np.float32),
                        (self.shape_resize[1], self.shape_resize[0] - 60),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.int64)
                    top2_soft['p1'] = cv2.resize(
                        top2_soft['p1'].astype(np.float32),
                        (self.shape_resize[1], self.shape_resize[0] - 60),
                        interpolation=cv2.INTER_LINEAR,
                    ).astype(np.float32)
                    top2_soft['p2'] = cv2.resize(
                        top2_soft['p2'].astype(np.float32),
                        (self.shape_resize[1], self.shape_resize[0] - 60),
                        interpolation=cv2.INTER_LINEAR,
                    ).astype(np.float32)
            elif self.confidence_path:
                confidence_root = resolve_pseudolabel_root(self.confidence_path, self.split)
                confidence_path = build_pseudolabel_path(segmentation_mask_file, confidence_root)
                confidence = normalize_confidence_map(load_confidence_map(confidence_path))
                if self.resize:
                    confidence = cv2.resize(confidence.astype(np.float32),
                                            (self.shape_resize[1], self.shape_resize[0] - 60),
                                            interpolation=cv2.INTER_LINEAR).astype(np.float32)

        directory = dirname(dirname(segmentation_mask_file))

        img_idx = int(basename(segmentation_mask_file).split("_")[-1].split(".")[0]) - 1

        img_timestamp_event_idx = self.img_timestamp_event_idx[directory]
        t_events, xyp_events = self.event_data[directory]

        # events has form x, y, t_ns, p (in [0,1])
        events = extract_events_from_memmap(t_events, xyp_events, img_idx, img_timestamp_event_idx, self.fixed_duration,
                                            self.nr_events)
        t_ns = events[:, 2]
        delta_t_ns = int((t_ns[-1] - t_ns[0]) / self.nr_events_data)
        nr_events_loaded = events.shape[0]
        nr_events_temp = nr_events_loaded // self.nr_events_data

        id_end = 0
        event_tensor = None
        for i in range(self.nr_events_data):
            id_start = id_end
            if self.fixed_duration:
                id_end = np.searchsorted(t_ns, t_ns[0] + (i + 1) * delta_t_ns)
            else:
                id_end += nr_events_temp

            if id_end > nr_events_loaded:
                id_end = nr_events_loaded

            event_representation = data_util.generate_input_representation(events[id_start:id_end],
                                                                           self.event_representation,
                                                                           self.shape,
                                                                           nr_temporal_bins=self.nr_temporal_bins,
                                                                           separate_pol=self.separate_pol)

            event_representation = torch.from_numpy(event_representation)

            if self.normalize_event:
                event_representation = data_util.normalize_voxel_grid(event_representation)

            if self.resize:
                event_representation_resize = f.interpolate(event_representation.unsqueeze(0),
                                                            size=(self.shape_resize[0], self.shape_resize[1]),
                                                            mode='bilinear', align_corners=True)
                event_representation = event_representation_resize.squeeze(0)

            if event_tensor is None:
                event_tensor = event_representation
            else:
                event_tensor = torch.cat([event_tensor, event_representation], dim=0)

        event_tensor = event_tensor[:, :-60, :]  # remove 60 bottom rows

        if self.random_crop and self.split == 'train':
            event_tensor = event_tensor[:, -self.shape_crop[0]:, :]
            label = label[-self.shape_crop[0]:, :]
            if pseudolabel is not None:
                pseudolabel = pseudolabel[-self.shape_crop[0]:, :]
            if top2_soft is not None:
                top2_soft['top1_id'] = top2_soft['top1_id'][-self.shape_crop[0]:, :]
                top2_soft['top2_id'] = top2_soft['top2_id'][-self.shape_crop[0]:, :]
                top2_soft['p1'] = top2_soft['p1'][-self.shape_crop[0]:, :]
                top2_soft['p2'] = top2_soft['p2'][-self.shape_crop[0]:, :]
            if confidence is not None:
                confidence = confidence[-self.shape_crop[0]:, :]
            if self.augmentation:
                if pseudolabel is not None and top2_soft is not None:
                    event_tensor, label, pseudolabel, top2_soft = self.apply_augmentation(
                        self.transform_a_random_crop,
                        event_tensor,
                        label,
                        pseudolabel,
                        top2_soft=top2_soft,
                    )
                elif pseudolabel is not None and confidence is not None:
                    event_tensor, label, pseudolabel, confidence = self.apply_augmentation(
                        self.transform_a_random_crop,
                        event_tensor,
                        label,
                        pseudolabel,
                        confidence,
                    )
                elif pseudolabel is not None:
                    event_tensor, label, pseudolabel = self.apply_augmentation(
                        self.transform_a_random_crop,
                        event_tensor,
                        label,
                        pseudolabel,
                    )
                else:
                    event_tensor, label = self.apply_augmentation(self.transform_a_random_crop, event_tensor, label)

        else:
            if self.augmentation:
                if pseudolabel is not None and top2_soft is not None:
                    event_tensor, label, pseudolabel, top2_soft = self.apply_augmentation(
                        self.transform_a,
                        event_tensor,
                        label,
                        pseudolabel,
                        top2_soft=top2_soft,
                    )
                elif pseudolabel is not None and confidence is not None:
                    event_tensor, label, pseudolabel, confidence = self.apply_augmentation(
                        self.transform_a,
                        event_tensor,
                        label,
                        pseudolabel,
                        confidence,
                    )
                elif pseudolabel is not None:
                    event_tensor, label, pseudolabel = self.apply_augmentation(
                        self.transform_a,
                        event_tensor,
                        label,
                        pseudolabel,
                    )
                else:
                    event_tensor, label = self.apply_augmentation(self.transform_a, event_tensor, label)

        label_tensor = torch.from_numpy(label).long()

        if self.split == 'train' and self.require_paired_data and pseudolabel is not None:
            pseudolabel_tensor = torch.from_numpy(np.asarray(pseudolabel, dtype=np.int64)).long()
            if top2_soft is not None:
                top1_id_tensor = torch.from_numpy(np.asarray(top2_soft['top1_id'], dtype=np.int64)).long()
                top2_id_tensor = torch.from_numpy(np.asarray(top2_soft['top2_id'], dtype=np.int64)).long()
                p1_tensor = torch.from_numpy(np.asarray(top2_soft['p1'], dtype=np.float32)).float()
                p2_tensor = torch.from_numpy(np.asarray(top2_soft['p2'], dtype=np.float32)).float()
                return (
                    event_tensor,
                    pseudolabel_tensor,
                    top1_id_tensor,
                    top2_id_tensor,
                    p1_tensor,
                    p2_tensor,
                    label_tensor,
                )
            if confidence is not None:
                confidence_tensor = torch.from_numpy(np.asarray(confidence, dtype=np.float32)).float()
                return event_tensor, pseudolabel_tensor, confidence_tensor, label_tensor
            return event_tensor, pseudolabel_tensor, label_tensor

        if self.split == 'valid' and self.require_paired_data:
            segmentation_mask_filepath_list = str(segmentation_mask_file).split('/')
            segmentation_mask_filename = segmentation_mask_filepath_list[-1]
            dir_name = segmentation_mask_filepath_list[-3]
            filename_id = segmentation_mask_filename.split('_')[-1]
            img_filename = '_'.join(['img', filename_id])
            img_filepath_list = segmentation_mask_filepath_list
            img_filepath_list[-2] = 'imgs'
            img_filepath_list[-1] = img_filename
            img_file = '/'.join(img_filepath_list)
            if not os.path.exists(img_file):
                img_filename = filename_id.zfill(14)
                img_filepath_list[-1] = img_filename
                img_file = '/'.join(img_filepath_list)
            img = Image.open(img_file)

            if self.resize:
                img = img.resize((self.shape_resize[1], self.shape_resize[0]))
            img_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
            ])
            img_tensor = img_transform(img)
            img_tensor = img_tensor[:, :-60, :]

            label_original_tensor = torch.from_numpy(label_original).long()
            return event_tensor, img_tensor, label_tensor, label_original_tensor
        return event_tensor, label_tensor
