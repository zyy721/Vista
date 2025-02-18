import os

import torch

from .common import BaseDataset


def balance_with_actions(samples, increase_factor=5, exceptions=None):
    if exceptions is None:
        exceptions = [2, 3]
    sample_to_add = list()
    if increase_factor > 1:
        for each_sample in samples:
            if each_sample["cmd"] not in exceptions:
                for _ in range(increase_factor - 1):
                    sample_to_add.append(each_sample)
    return samples + sample_to_add


def resample_complete_samples(samples, increase_factor=5):
    sample_to_add = list()
    if increase_factor > 1:
        for each_sample in samples:
            if (each_sample["speed"] and each_sample["angle"] and each_sample["z"] > 0
                    and 0 < each_sample["goal"][0] < 1600 and 0 < each_sample["goal"][1] < 900):
                for _ in range(increase_factor - 1):
                    sample_to_add.append(each_sample)
    return samples + sample_to_add


import torch.utils.data as data
from torchvision import transforms
import pickle as pkl
import tqdm
import random, json
from PIL import Image


# class NuScenesDatasetMultiview(BaseDataset):
class NuScenesDatasetMultiview(data.Dataset):
    def __init__(self, data_root="data/nuscenes", anno_file="annos/nuScenes.json",
                 target_height=320, target_width=576, num_frames=25):
        # if not os.path.exists(data_root):
        #     raise ValueError("Cannot find dataset {}".format(data_root))
        # if not os.path.exists(anno_file):
        #     raise ValueError("Cannot find annotation {}".format(anno_file))
        # super().__init__(data_root, anno_file, target_height, target_width, num_frames)
        # print("nuScenes loaded:", len(self))
        # self.samples = balance_with_actions(self.samples, increase_factor=5)
        # print("nuScenes balanced:", len(self))
        # self.samples = resample_complete_samples(self.samples, increase_factor=2)
        # print("nuScenes resampled:", len(self))
        # self.action_mod = 0

        data_root = "data/sample_nusc_video_all_cam_train.pkl"
        self.data_root = data_root

        assert target_height % 64 == 0 and target_width % 64 == 0, "Resize to integer multiple of 64"
        self.img_preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0)
        ])

        # if isinstance(anno_file, list):
        #     self.samples = list()
        #     for each_file in anno_file:
        #         with open(each_file, "r") as anno_json:
        #             self.samples += json.load(anno_json)
        # else:
        #     with open(anno_file, "r") as anno_json:
        #         self.samples = json.load(anno_json)

        self.target_height = target_height
        self.target_width = target_width
        self.num_frames = num_frames

        self.videos = self._make_dataset_video(data_root)
        self.view_order = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]

    def __len__(self):
        # return len(self.videos)
        return len(self.videos['CAM_FRONT'])

        
    def _make_dataset_video(self, info_path):
        with open(info_path, 'rb') as f:
            all_cam_video_info = pkl.load(f)
        
        all_cam_output_videos = {}

        for cam_name, video_info in all_cam_video_info.items():
            output_videos = []
            for video_name, frames in tqdm.tqdm(video_info.items(), desc="Making Nuscenes dataset"):

                output_videos.append(frames)

            all_cam_output_videos[cam_name] = output_videos

        return all_cam_output_videos

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        ori_w, ori_h = image.size
        if ori_w / ori_h > self.target_width / self.target_height:
            tmp_w = int(self.target_width / self.target_height * ori_h)
            left = (ori_w - tmp_w) // 2
            right = (ori_w + tmp_w) // 2
            image = image.crop((left, 0, right, ori_h))
        elif ori_w / ori_h < self.target_width / self.target_height:
            tmp_h = int(self.target_height / self.target_width * ori_w)
            top = (ori_h - tmp_h) // 2
            bottom = (ori_h + tmp_h) // 2
            image = image.crop((0, top, ori_w, bottom))
        image = image.resize((self.target_width, self.target_height), resample=Image.LANCZOS)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = self.img_preprocessor(image)
        return image

    # def get_image_path(self, sample_dict, current_index):
    #     return os.path.join(self.data_root, sample_dict["frames"][current_index])

    def build_data_dict(self, image_seq):
        # log_cond_aug = self.log_cond_aug_dist.sample()
        # cond_aug = torch.exp(log_cond_aug)
        cond_aug = torch.tensor([0.0])
        # data_dict = {
        #     "img_seq": torch.stack(image_seq),
        #     "motion_bucket_id": torch.tensor([127]),
        #     "fps_id": torch.tensor([9]),
        #     "cond_frames_without_noise": image_seq[0],
        #     "cond_frames": image_seq[0] + cond_aug * torch.randn_like(image_seq[0]),
        #     "cond_aug": cond_aug
        # }

        data_dict = {
            "img_seq": image_seq,
            "motion_bucket_id": torch.tensor([127]),
            "fps_id": torch.tensor([9]),
            "cond_frames_without_noise": image_seq[:, 0],
            "cond_frames": image_seq[:, 0] + cond_aug * torch.randn_like(image_seq[:, 0]),
            "cond_aug": cond_aug
        }

        # if self.action_mod == 0:
        #     data_dict["trajectory"] = torch.tensor(sample_dict["traj"][2:])
        # elif self.action_mod == 1:
        #     data_dict["command"] = torch.tensor(sample_dict["cmd"])
        # elif self.action_mod == 2:
        #     # scene might be empty
        #     if sample_dict["speed"]:
        #         data_dict["speed"] = torch.tensor(sample_dict["speed"][1:])
        #     # scene might be empty
        #     if sample_dict["angle"]:
        #         data_dict["angle"] = torch.tensor(sample_dict["angle"][1:]) / 780
        # elif self.action_mod == 3:
        #     # point might be invalid
        #     if sample_dict["z"] > 0 and 0 < sample_dict["goal"][0] < 1600 and 0 < sample_dict["goal"][1] < 900:
        #         data_dict["goal"] = torch.tensor([
        #             sample_dict["goal"][0] / 1600,
        #             sample_dict["goal"][1] / 900
        #         ])
        # else:
        #     raise ValueError
        return data_dict

    def __getitem__(self, index):
        # sample_dict = self.samples[index]
        # self.action_mod = (self.action_mod + index) % 4

        # image_seq = list()
        # for i in range(self.num_frames):
        #     current_index = i
        #     img_path = self.get_image_path(sample_dict, current_index)
        #     image = self.preprocess_image(img_path)
        #     image_seq.append(image)
        # return self.build_data_dict(image_seq, sample_dict)

        images_list = []
        init_frame = random.randint(0,len(self.videos['CAM_FRONT'][index])-self.num_frames)
        for cam_name, video in self.videos.items():
            video = video[index]
            video = video[init_frame:init_frame+self.num_frames]
            cur_cam_img_list = []
            for img_path in video:
                img_path = img_path[1:]
                image = self.preprocess_image(img_path)
                cur_cam_img_list.append(image)
            cur_cam_img = torch.stack(cur_cam_img_list,dim=0)
            images_list.append(cur_cam_img)

        image_seq = torch.stack(images_list)
        return self.build_data_dict(image_seq)