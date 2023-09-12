import copy
import json
import math
import random
import sys

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

from utils.data import load_data_3dpw_multiperson, load_data_somof
from utils.utils import path_to_data

sys.path.append("/PoseForecasters/")
import utils_skelda


def collate_batch(batch):
    joints_list = []
    masks_list = []
    preds_list = []
    num_people_list = []
    for joints, masks, preds in batch:
        # Make sure first dimension is # people for single person case
        if len(joints.shape) == 3:
            joints = joints.unsqueeze(0)
            masks = masks.unsqueeze(0)
            preds = preds.unsqueeze(0)

        joints_list.append(joints)
        masks_list.append(masks)
        preds_list.append(preds)
        num_people_list.append(torch.zeros(joints.shape[0]))

    joints = pad_sequence(joints_list, batch_first=True)
    masks = pad_sequence(masks_list, batch_first=True)
    preds = pad_sequence(preds_list, batch_first=True)
    padding_mask = pad_sequence(
        num_people_list, batch_first=True, padding_value=1
    ).bool()

    return joints, masks, padding_mask, preds


def batch_process_joints(
    joints, masks, padding_mask, config, training=False, multiperson=True
):
    joints = joints.to(config["DEVICE"])
    masks = masks.to(config["DEVICE"])

    if multiperson and len(joints.shape) == 4:
        joints = joints.unsqueeze(1)
        masks = masks.unsqueeze(1)

    in_F = config["TRAIN"]["input_track_size"]

    if multiperson:
        # if (
        #     config["DATA"]["joints"] == "somof"
        #     or config["DATA"]["joints"] == "posetrack"
        # ):
        #     in_joints_pelvis = joints[:, :, (in_F - 1) : in_F, 6:7, :].clone()
        # elif config["DATA"]["joints"] == "cmu":
        #     in_joints_pelvis = joints[:, :, (in_F - 1) : in_F, 12:13, :].clone()
        in_joints_pelvis = joints[:, :, (in_F - 1) : in_F, 0:1, :].clone()
    else:
        # if (
        #     config["DATA"]["joints"] == "somof"
        #     or config["DATA"]["joints"] == "posetrack"
        # ):
        #     in_joints_pelvis = joints[:, (in_F - 1) : in_F, 6:7, :].clone()
        # else:
        #     in_joints_pelvis = torch.zeros_like(in_joints_pelvis)
        in_joints_pelvis = joints[:, (in_F - 1) : in_F, 0:1, :].clone()

    joints -= in_joints_pelvis

    if multiperson:
        B, N, F, J, K = joints.shape
        joints = joints.transpose(1, 2).reshape(B, F, N * J, K)
        in_joints_pelvis = in_joints_pelvis.reshape(B, 1, N, K)
        masks = masks.transpose(1, 2).reshape(B, F, N * J)

    # If training, can do augmentations
    if training:
        if config["DATA"]["aug_rotate"]:
            joints = getRandomRotatePoseTransform(config)(joints)
        if config["DATA"]["aug_scale"]:
            joints = getRandomScaleTransform()(joints)
        if "aug_permute" in config["DATA"] and config["DATA"]["aug_permute"]:
            joints, masks, padding_mask = getRandomPermuteOrder(
                joints, masks, padding_mask
            )

    in_F, out_F = (
        config["TRAIN"]["input_track_size"],
        config["TRAIN"]["output_track_size"],
    )
    in_joints = joints[:, :in_F].float()
    out_joints = joints[:, in_F : in_F + out_F].float()
    in_masks = masks[:, :in_F].float()
    out_masks = masks[:, in_F : in_F + out_F].float()

    if training:
        if config["DATA"]["aug_noise"]:
            noise = torch.normal(mean=torch.zeros_like(in_joints), std=0.050)
            noise = torch.clip(noise, -0.200, 0.200)
            in_joints = in_joints + noise

    return (
        in_joints,
        in_masks,
        out_joints,
        out_masks,
        in_joints_pelvis.float(),
        padding_mask.float(),
    )


def getRandomScaleTransform(r1=0.8, r2=1.2):
    def do_scale(x):
        # scale = (r1 - r2) * torch.rand(1) + r2
        scale = (r1 - r2) * torch.rand(x.shape[0]).reshape(-1, 1, 1, 1) + r2
        return x * scale.to(x.device)

    return transforms.Lambda(lambda x: do_scale(x))


def getRandomPermuteOrder(joints, masks, padding_mask):
    """
    Randomly permutes persons across the input token dimension. This helps
    expose all learned embeddings to a variety of poses.
    """

    def do_permute(joints, masks, padding_mask):
        B, N = padding_mask.shape
        B, F, NJ, K = joints.shape
        J = NJ // N

        perm = torch.argsort(torch.rand(B, N), dim=-1).reshape(B, N)
        idx = torch.arange(B).unsqueeze(-1)

        joints = joints.view(B, F, N, J, K).transpose(1, 2)[idx, perm]
        joints = joints.transpose(1, 2).reshape(B, F, N * J, K)

        masks = masks.view(B, F, N, J).transpose(1, 2)[idx, perm]
        masks = masks.transpose(1, 2).reshape(B, F, N * J)

        padding_mask = padding_mask[idx, perm]

        return joints, masks, padding_mask

    return do_permute(joints, masks, padding_mask)


def getRandomRotatePoseTransform(config):
    """
    Performs a random rotation about the origin (0, 0, 0)
    """

    def do_rotate(pose_seq):
        """
        pose_seq: torch.Tensor of size (B, S, J, 3) where S is sequence length, J is number
            of joints, and the last dimension is the coordinate
        """
        B, F, J, K = pose_seq.shape

        angles = torch.deg2rad(torch.rand(B) * 360)

        # rotation_matrix = torch.zeros(B, 3, 3).to(pose_seq.device)
        # rotation_matrix[:, 1, 1] = 1
        # rotation_matrix[:, 0, 0] = torch.cos(angles)
        # rotation_matrix[:, 0, 2] = torch.sin(angles)
        # rotation_matrix[:, 2, 0] = -torch.sin(angles)
        # rotation_matrix[:, 2, 2] = torch.cos(angles)

        rotation_matrix = torch.zeros(B, 3, 3).to(pose_seq.device)
        rotation_matrix[:, 2, 2] = 1
        rotation_matrix[:, 0, 0] = torch.cos(angles)
        rotation_matrix[:, 0, 1] = torch.sin(angles)
        rotation_matrix[:, 1, 0] = -torch.sin(angles)
        rotation_matrix[:, 1, 1] = torch.cos(angles)

        rot_pose = torch.bmm(pose_seq.reshape(B, -1, 3).float(), rotation_matrix)
        rot_pose = rot_pose.reshape(pose_seq.shape)
        return rot_pose

    return transforms.Lambda(lambda x: do_rotate(x))


class MultiPersonPoseDataset(torch.utils.data.Dataset):
    SOMOF_JOINTS = [1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21]
    COCO_TO_SOMOF = [6, 12, 7, 13, 8, 14, 0, 3, 9, 4, 10, 5, 11]

    def __init__(
        self,
        name,
        split="train",
        track_size=30,
        track_cutoff=16,
        segmented=True,
        add_flips=False,
        frequency=1,
    ):
        """
        name: The name of the dataset (e.g. "somof")
        split: one of ['train', 'valid', 'test']
        add_flips: whether to add flipped sequences to data as well (data augmentation)
        mode: one of ['inference', 'eval'] In eval mode, will not do train data augmentation.
        frequency: How often to take a frame (i.e. distance between frames). For example, if
                   frequency=2, will take every other frame.
        """
        self.name = name
        self.split = split
        self.track_size = track_size
        self.track_cutoff = track_cutoff
        self.segmented = segmented
        self.frequency = frequency
        self.add_flips = add_flips

        self.initialize()

    def load_data(self):
        raise NotImplementedError("Dataset load_data() method is not implemented.")

    def initialize(self):
        self.load_data()

        if self.segmented:
            tracks = []
            for scene in self.datalist:
                for seg, j in enumerate(
                    range(
                        0,
                        len(scene[0][0]) - self.track_size * self.frequency + 1,
                        self.track_size,
                    )
                ):
                    people = []
                    for person in scene:
                        start_idx = j
                        end_idx = start_idx + self.track_size * self.frequency
                        J_3D_real, J_3D_mask, J_3D_pred = (
                            person[0][start_idx : end_idx : self.frequency],
                            person[1][start_idx : end_idx : self.frequency],
                            person[2][start_idx : end_idx : self.frequency],
                        )
                        people.append((J_3D_real, J_3D_mask, J_3D_pred))
                    tracks.append(people)
            self.datalist = tracks

        # If we're on a train split, do some additional data augmentation as well
        if self.add_flips:
            print("doing some flips for " + self.name + ", " + self.split + " split")
            # for each sequence, we can also add a "flipped" sequence
            flipped_datalist = []
            for seq in self.datalist:
                flipped_seq = []
                for J_3D_real, J_3D_mask, J_3D_pred in seq:
                    J_3D_flipped = torch.flip(J_3D_real, dims=(0,)).clone()
                    J_3D_pred_flipped = torch.flip(J_3D_pred, dims=(0,)).clone()
                    J_3D_mask_flipped = torch.flip(J_3D_mask, dims=(0,)).clone()
                    flipped_seq.append(
                        (J_3D_flipped, J_3D_mask_flipped, J_3D_pred_flipped)
                    )
                flipped_datalist.append(flipped_seq)

            self.datalist += flipped_datalist

        if not self.segmented:
            # create a mapping from idx to which track/frame to look at
            # prevents extra computation at dataset time
            frame_count = 0
            self.track_map = []
            for i, scene in enumerate(self.datalist):
                track_frames = len(scene[0][0]) - self.track_size * self.frequency + 1
                for k in range(0, track_frames):
                    self.track_map.append((i, k))
                frame_count += track_frames

    def __len__(self):
        if self.segmented:
            return len(self.datalist)
        else:
            return sum(
                [
                    len(scene[0][0]) - self.track_size * self.frequency + 1
                    for scene in self.datalist
                ]
            )

    def __getitem__(self, idx):
        if self.segmented:
            scene = self.datalist[idx]

        else:
            # We want to count the idx-th valid frame in our list of tracks
            # A valid frame is any frame that has at least (track_size-1)
            # frames ahead of it (i.e. it can be used as the start frame)
            track_idx, frame_idx = self.track_map[idx]

            scene = []
            for person in self.datalist[track_idx]:
                J_3D_real = person[0][
                    frame_idx : frame_idx
                    + self.track_size * self.frequency : self.frequency
                ]
                J_3D_pred = person[2][
                    frame_idx : frame_idx
                    + self.track_size * self.frequency : self.frequency
                ]
                J_3D_mask = person[1][
                    frame_idx : frame_idx
                    + self.track_size * self.frequency : self.frequency
                ]
                scene.append((J_3D_real, J_3D_mask, J_3D_pred))

        J_3D_real = torch.stack([s[0] for s in scene])
        J_3D_mask = torch.stack([s[1] for s in scene])
        J_3D_pred = torch.stack([s[2] for s in scene])

        return J_3D_real, J_3D_mask, J_3D_pred


class SoMoFDataset(MultiPersonPoseDataset):
    SOMOF_JOINTS = [1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21]
    COCO_TO_SOMOF = [6, 12, 7, 13, 8, 14, 0, 3, 9, 4, 10, 5, 11]

    def __init__(self, **args):
        super(SoMoFDataset, self).__init__("somof", frequency=1, **args)

    def load_data(self):
        data_in, data_out, _, _ = load_data_somof(split=self.split)

        data = np.concatenate((data_in, data_out), axis=2)
        data = torch.from_numpy(data)
        data = data.reshape((*data.shape[:-1], 13, 3))  # (N, 30, 2, 13, 3)

        self.num_kps = 13
        self.datalist = [
            [(person, torch.ones(person.shape[:-1])) for person in track]
            for track in data
        ]


class ThreeDPWDataset(MultiPersonPoseDataset):
    def __init__(self, **args):
        super(ThreeDPWDataset, self).__init__("3dpw", frequency=2, **args)

    def load_data(self):
        self.data = load_data_3dpw_multiperson(split=self.split)

        self.datalist = []
        for scene in self.data:
            people = [
                (
                    torch.from_numpy(joints)[:, self.SOMOF_JOINTS],
                    torch.from_numpy(mask)[:, self.SOMOF_JOINTS],
                )
                for joints, mask in scene
            ]
            self.datalist.append(people)


class SkeldaDataset(MultiPersonPoseDataset):
    def __init__(self, **args):

        # self.datamode = "pred-pred"
        self.datamode = "pred-gt"
        # self.datamode = "gt-gt"

        self.vis = False
        # self.vis = True

        self.config = {
            "item_step": 1,
            # "item_step": 2,
            "select_joints": [
                "hip_middle",
                "hip_right",
                "knee_right",
                "ankle_right",
                "hip_left",
                "knee_left",
                "ankle_left",
                "nose",
                "shoulder_left",
                "elbow_left",
                "wrist_left",
                "shoulder_right",
                "elbow_right",
                "wrist_right",
                "shoulder_middle",
            ],
        }

        # self.datasets_train = [
        #     "/datasets/preprocessed/mocap/train_forecast_samples_10fps.json",
        #     "/datasets/preprocessed/amass/bmlmovi_train_forecast_samples_10fps.json",
        #     "/datasets/preprocessed/amass/bmlrub_train_forecast_samples_10fps.json",
        #     "/datasets/preprocessed/amass/kit_train_forecast_samples_10fps.json",
        # ]
        # self.datasets_train = [
        #     "/datasets/preprocessed/human36m/train_forecast_kppspose.json",
        #     # "/datasets/preprocessed/mocap/train_forecast_samples.json",
        # ]
        self.datasets_train = [
            "/datasets/preprocessed/human36m/train_forecast_kppspose_10fps.json",
        ]

        # self.dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_kppspose.json"
        # self.dataset_eval_test = "/datasets/preprocessed/mocap/{}_forecast_samples.json"
        self.dataset_eval_test = (
            "/datasets/preprocessed/human36m/{}_forecast_kppspose_10fps.json"
        )
        # self.dataset_eval_test = "/datasets/preprocessed/mocap/{}_forecast_samples_10fps.json"

        super(SkeldaDataset, self).__init__("h36m", frequency=1, **args)

    def filter_dataset(self, dataset, config):

        if "select_joints" in config:
            # Optionally select only specific joints
            joints_ids = [dataset["joints"].index(j) for j in config["select_joints"]]
            dataset["joints"] = [dataset["joints"][i] for i in joints_ids]

            for scene in dataset["sequences"]:
                for item in scene:
                    item["bodies3D"] = [
                        [item["bodies3D"][k][i] for i in joints_ids]
                        for k in range(len(item["bodies3D"]))
                    ]

            if "prediction_joints" in dataset:
                joints_ids = [
                    dataset["prediction_joints"].index(j)
                    for j in config["select_joints"]
                ]
                dataset["prediction_joints"] = [
                    dataset["prediction_joints"][i] for i in joints_ids
                ]

                for scene in dataset["sequences"]:
                    for item in scene:
                        item["predictions"] = [
                            [item["predictions"][k][i] for i in joints_ids]
                            for k in range(len(item["predictions"]))
                        ]

    def load_data(self):

        split = self.split
        if self.split in ["val", "valid"]:
            split = "eval"

        if "mocap" in self.dataset_eval_test and split == "eval":
            split = "test"

        if split in ["eval", "test"]:
            path = self.dataset_eval_test.format(split)
            dataset = utils_skelda.load_json(path)

            cfg = copy.deepcopy(self.config)
            if "mocap" in path:
                cfg["select_joints"][cfg["select_joints"].index("nose")] = "head_upper"

            self.filter_dataset(dataset, cfg)
            dataset = dataset["sequences"]

        else:
            dataset = []
            for dp in self.datasets_train:
                ds = utils_skelda.load_json(dp)

                cfg = copy.deepcopy(self.config)
                if "mocap" in dp:
                    cfg["select_joints"][
                        cfg["select_joints"].index("nose")
                    ] = "head_upper"

                self.filter_dataset(ds, cfg)
                dataset.extend(ds["sequences"])

        print("Using {} sequences".format(len(dataset)))

        self.datalist = []
        for scene in dataset:
            if self.vis and scene[0]["action"] != 14:
                continue

            poses = [np.array(item["bodies3D"][0])[:, 0:3] / 1000 for item in scene]
            masks = [np.ones(poses[0].shape[0]) for _ in scene]

            if "predictions" in scene[0]:
                poses_pred = [
                    np.array(item["predictions"][0])[:, 0:3] / 1000 for item in scene
                ]
            else:
                poses_pred = poses

            if len(poses) < self.track_size:
                continue

            # Take every other frame
            poses = poses[:: self.config["item_step"]]
            masks = masks[:: self.config["item_step"]]
            poses_pred = poses_pred[:: self.config["item_step"]]

            if self.datamode == "gt-gt":
                poses_pred = poses

            elif self.datamode == "pred-pred":
                poses = poses_pred

            elif self.datamode == "pred-gt":
                tmp = poses
                poses = poses_pred
                poses_pred = tmp

            if self.vis:
                sid = 20
                poses = poses[sid:]
                masks = masks[sid:]
                poses_pred = poses_pred[sid:]

            people = [
                (
                    torch.from_numpy(np.array(poses)),
                    torch.from_numpy(np.array(masks)),
                    torch.from_numpy(np.array(poses_pred)),
                )
            ]

            self.datalist.append(people)


def create_dataset(dataset_name, logger, **args):
    logger.info("Loading dataset " + dataset_name)

    if dataset_name == "3dpw":
        # dataset = ThreeDPWDataset(**args)
        dataset = SkeldaDataset(**args)
    elif dataset_name == "somof":
        # dataset = SoMoFDataset(**args)
        dataset = SkeldaDataset(**args)
    elif dataset_name == "skelda":
        dataset = SkeldaDataset(**args)
    else:
        raise ValueError(f"Dataset with name '{dataset_name}' not found.")

    logger.info(f"Loaded {len(dataset)} annotations for " + dataset_name)
    return dataset


def get_datasets(datasets_list, config, logger):
    """
    Returns a list of torch dataset objects
    datasets_list: [String]
    """
    in_F, out_F = (
        config["TRAIN"]["input_track_size"],
        config["TRAIN"]["output_track_size"],
    )
    datasets = []
    for dataset_name in datasets_list:
        datasets.append(
            create_dataset(
                dataset_name,
                logger,
                split="train",
                track_size=(in_F + out_F),
                track_cutoff=in_F,
                segmented=config["DATA"]["segmented"],
                add_flips=config["DATA"]["add_flips"],
            )
        )
    return datasets
