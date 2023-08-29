import os
from threading import Thread, Semaphore
import subprocess
import numpy as np
from numpy.lib.function_base import disp
import torch
import random
import decord
import openpyxl
from PIL import Image
from torchvision import transforms
from random_erasing import RandomErasing
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import video_transforms as video_transforms 
import volume_transforms as volume_transforms

class TMHVideoClsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3,args=None):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        # import pandas as pd
        # cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        # self.dataset_samples = list(cleaned.values[:, 0])
        # self.label_array = list(cleaned.values[:, 1])
        self.dataset_samples, self.label_array = self.process_labelFile(self.anno_path)

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args 
            scale_t = 1

            sample = self.dataset_samples[index]
            # data_path = os.path.join(self.data_path, sample[0])
            buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t) # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(int(label))
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)
            
            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample)
            buffer = self.data_transform(buffer)
            label = int(self.label_array[index])
            return buffer, label, {}

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.loadvideo_decord(sample)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                 / (self.test_num_crop - 1)
            temporal_step = max(1.0 * (buffer.shape[0] - self.clip_len) \
                                / (self.test_num_segment - 1), 0)
            temporal_start = int(chunk_nb * temporal_step)
            spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                       spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                       :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.data_transform(buffer)
            return buffer, int(self.test_label_array[index]), sample.split("/")[-1].split(".")[0], \
                   chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))
    
    def process_labelFile(self, anno_path):
        tmp = [x.strip().split(' ') for x in open(anno_path)]
        return [item[0] for item in tmp], [item[1] for item in tmp]

    def _aug_frame(
        self,
        buffer,
        args,
    ):

        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C 
        
        # T H W C 
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True ,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer


    def loadvideo_decord(self, fname, sample_rate_scale=1):
        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=10, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        if self.mode == 'test':
            all_index = [x for x in range(0, len(vr), self.frame_sample_rate)]
            while len(all_index) < self.clip_len:
                all_index.append(all_index[-1])
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = len(vr) // self.num_segment
        # frame_rate = vr.get_avg_fps()
        # seg_len = (sample[2] - sample[1]) * frame_rate // self.num_segment
        # offset = sample[1] * frame_rate

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(0, seg_len, num=int(seg_len // self.frame_sample_rate))
                index = np.concatenate((index, np.ones(self.clip_len - int(seg_len // self.frame_sample_rate)) * (seg_len)))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i*seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        vr.seek(0)       
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer
    
    # def process_labelFile(self, path):
    #     # Load the workbook
    #     workbook = openpyxl.load_workbook(path)
    #     # Get the first worksheet (you can also select a specific sheet by name)
    #     sheet = workbook.active

    #     # Iterate through rows in the sheet
    #     rows = []
    #     for row in sheet.iter_rows():
    #         # Read the values of each cell in the row and store them in a list
    #         row_values = [cell.value for cell in row]
    #         # Check if the row is empty and skip it if it is
    #         if all(value is None or (isinstance(value, str) and value.strip() == '') for value in row_values):
    #             continue
    #         rows.append(row_values)
    #     rows = rows[1:]

    #     processedAnn = []
    #     for index1 in range(0, len(rows), 3):
    #         elementRow = {}
    #         for index2 in range(3):
    #             timeAnnos = []
    #             for index3,element in enumerate(rows[index1+index2]):
    #                 if element == None:
    #                     elementRow[self.classNames[index2 + 1]] = timeAnnos
    #                     break
    #                 if index3 == 0:
    #                     elementRow["video_name"] = element
    #                     continue
    #                 elif index3 == 1:
    #                     continue
    #                 timeAnnos.append(element)
    #         if elementRow['video_name'].find('_sides') != -1 and \
    #            elementRow['video_name'].find('20210522125912274_1') == -1 and \
    #            elementRow['video_name'].find('20210522121255861_1') == -1 :
    #             elementRow['video_name'] = elementRow['video_name'].replace('_sides', '')
    #             processedAnn.extend(self.find_overlaps(elementRow))
    #     # Set the random seed so the results are reproducible
    #     random.seed(42)
    #     # Shuffle the list
    #     random.shuffle(processedAnn)
    #     # Determine the index where to split the list
    #     split_index = int(0.8 * len(processedAnn))
    #     if self.mode == 'train':
    #         return processedAnn[:split_index]
    #     elif self.mode == 'test':
    #         return processedAnn[split_index:]
    #     else:  
    #         return processedAnn[split_index:]
        
    # def find_overlaps(self, data):
    #     # Convert all the timestamps into seconds and create ranges
    #     ranges = {c: list(zip(data[c][::2], data[c][1::2])) for c in data if c != 'video_name'}
    #     # Convert timestamp ranges into seconds
    #     ranges_in_seconds = {c: [(self.timestamp_to_seconds(start), self.timestamp_to_seconds(end)) for start, end in ranges[c]] for c in ranges}
    #     # Flatten the list of all labeled intervals
    #     labeled_intervals = [(data['video_name'],start, end, label) for label in ranges_in_seconds for start, end in ranges_in_seconds[label]]
    #     # Sort by start time
    #     labeled_intervals.sort()
    #     # Merge intervals
    #     merged = self.extract_overlaps(labeled_intervals)
    #     return merged
    
    # @staticmethod
    # def extract_overlaps(intervals):
    #     # Expand the intervals into individual events and sort them
    #     events = sorted((name, t, startOrEnd, label) for name, start, end, label in intervals for t, startOrEnd in [(start, -1), (end, 1)])

    #     interval = []
    #     labels = set()

    #     for i in range(1, len(events)):
    #         name, t1, startOrEnd1, label1 = events[i - 1]
    #         _,t2, _, _ = events[i]

    #         if startOrEnd1 == -1:
    #             labels.add(label1)
    #         elif startOrEnd1 == 1 and label1 in labels:
    #             labels.remove(label1)

    #         if t1 != t2 and labels:
    #             interval.append((name, t1, t2, ','.join(sorted(labels))))

    #         newIntervals = []
    #         prevEnd = 0

    #         for name, start, end, label in interval:
    #             if start > prevEnd:
    #                 newIntervals.append((name, prevEnd, start, 'normal'))
    #             newIntervals.append((name, start, end, label))
    #             prevEnd = end
        
    #     return newIntervals
    
    # @staticmethod
    # def timestamp_to_seconds(timestamp):
    #     minutes, seconds = map(int, timestamp.split(':'))
    #     return minutes * 60 + seconds

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

if __name__ == '__main__':
    from config.config import get_args
    opts, ds_init = get_args()
    dataset = TMHVideoClsDataset("/home/comp/cszsliu/tmh_wardvideo/labels-all.xlsx", \
                              "/home/comp/cszsliu/tmh_wardvideo/",mode='validation', args=opts)
    batch = dataset[1]

    # Suppose the buffer shape is (num_frames, height, width, channels)
    num_frames = batch[0].shape[0]

    import matplotlib.pyplot as plt

    for i in range(num_frames):
        plt.imshow(batch[0][i])
        plt.savefig(f'/home/comp/cszsliu/project/VideoMAE/figs/test_images/Frame{i}.png')
    print('hello')
