# Copyright (c) RuopengGao. All Rights Reserved.
# About:
import os
import json

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader
from data.seq_dataset import SeqDataset
from utils.nested_tensor import tensor_list_to_nested_tensor
from models.utils import get_model
from utils.box_ops import box_cxcywh_to_xyxy
from collections import deque
from structures.instances import Instances
from structures.ordered_set import OrderedSet

from utils.utils import yaml_to_dict, is_distributed, distributed_rank, distributed_world_size
from models import build_model
from models.utils import load_checkpoint
import cv2
import torchvision.transforms.functional as F
import numpy as np
import argparse
from configs.utils import update_config, load_super_config
from tqdm import tqdm



def parse_options():
    parser = argparse.ArgumentParser(description="Just pass a config path")
    parser.add_argument("--config_path", type=str)

    args = parser.parse_args()
    return args




def submit(config: dict):
    """
    Submit a model for a specific dataset.
    :param config:
    :param logger:
    :return:
    """
    if config["INFERENCE_CONFIG_PATH"] is None:
        model_config = config
    else:
        model_config = yaml_to_dict(path=config["INFERENCE_CONFIG_PATH"])
    model = build_model(config=model_config)
    load_checkpoint(model, path=config["INFERENCE_MODEL"])

    if is_distributed():
        model = DDP(model, device_ids=[distributed_rank()])

    # if config["INFERENCE_GROUP"] is not None:
    #     submit_outputs_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"], config["INFERENCE_GROUP"],
    #                                       config["INFERENCE_SPLIT"],
    #                                       f'{config["INFERENCE_MODEL"].split("/")[-1][:-4]}')
    # else:
    #     submit_outputs_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"], "default",
    #                                       config["INFERENCE_SPLIT"],
    #                                       f'{config["INFERENCE_MODEL"].split("/")[-1][:-4]}')

    # 需要调度整个 submit 流程
    submit_one_epoch(
        config=config,
        model=model,
        dataset=config["INFERENCE_DATASET"],
        outputs_dir=config["OUTPUTS_DIR"],
        only_detr=config["INFERENCE_ONLY_DETR"]
    )

    

    return


@torch.no_grad()
def submit_one_epoch(config: dict, model: nn.Module,
                     dataset: str,
                     outputs_dir: str, only_detr: bool = False):
    model.eval()

    
    submit_one_seq_video(
                    model=model, dataset=dataset, 
                    video_path=config["VIDEO_PATH"],
                    only_detr=only_detr, max_temporal_length=config["MAX_TEMPORAL_LENGTH"],
                    outputs_dir=outputs_dir,
                    det_thresh=config["DET_THRESH"],
                    newborn_thresh=config["DET_THRESH"] if "NEWBORN_THRESH" not in config else config["NEWBORN_THRESH"],
                    area_thresh=config["AREA_THRESH"], id_thresh=config["ID_THRESH"],
                    image_max_size=config["INFERENCE_MAX_SIZE"] if "INFERENCE_MAX_SIZE" in config else 1333,
                    inference_ensemble=config["INFERENCE_ENSEMBLE"] if "INFERENCE_ENSEMBLE" in config else 0,
                )
    

    if is_distributed():
        torch.distributed.barrier()

    return


def process_image(image, height: int = 800, width: int = 1333):
        image_height = height
        image_width = width
        ori_image = image.copy()
        h, w = image.shape[:2]
        scale = image_height / min(h, w)
        if max(h, w) * scale > image_width:
            scale = image_width / max(h, w)
        target_h = int(h * scale)
        target_w = int(w * scale)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = cv2.resize(image, (target_w, target_h))
        image = F.normalize(F.to_tensor(image), mean, std)
        # image = image.unsqueeze(0)
        return image, ori_image



@torch.no_grad()
def submit_one_seq_video(
            model: nn.Module, video_path: str, outputs_dir: str,
            dataset:str,
            only_detr: bool, max_temporal_length: int = 0,
            det_thresh: float = 0.5, newborn_thresh: float = 0.5, area_thresh: float = 100, id_thresh: float = 0.1,
            image_max_size: int = 1333,
            fake_submit: bool = False,
            inference_ensemble: int = 0,
            device:str="cuda"
        ):
    os.makedirs(os.path.join(outputs_dir, "tracker"), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video_res = outputs_dir + "/MOTIP_out_file.mp4"
    out = cv2.VideoWriter(out_video_res, fourcc, fps, (width, height))
    frame_skip_interval = 1
    stopframe=1000
    frame_count = 0
    while cap.isOpened():
        ret, ori_frame = cap.read()
        if not ret:
            break
        
        # Skip frames based on the interval
        if frame_count % frame_skip_interval != 0:
            frame_count += 1
            continue
        
        # stop after frame xxx
        if frame_count > stopframe:
            continue
        # H, W, _ = ori_frame.shape
        image, ori_image = process_image(image=ori_frame)
        H, W, _ = image.shape
        ori_h, ori_w = ori_image.shape[1], ori_image.shape[2]
        frame = tensor_list_to_nested_tensor([image]).to(device)
        detr_outputs = model(frames=frame)
        detr_logits = detr_outputs["pred_logits"]
        detr_scores = torch.max(detr_logits, dim=-1).values.sigmoid()
        detr_det_idxs = detr_scores > det_thresh        # filter by the detection threshold
        detr_det_logits = detr_logits[detr_det_idxs]
        detr_det_labels = torch.max(detr_det_logits, dim=-1).indices
        detr_det_boxes = detr_outputs["pred_boxes"][detr_det_idxs]
        detr_det_outputs = detr_outputs["outputs"][detr_det_idxs]   # detr output embeddings
        area_legal_idxs = (detr_det_boxes[:, 2] * ori_w * detr_det_boxes[:, 3] * ori_h) > area_thresh   # filter by area
        detr_det_outputs = detr_det_outputs[area_legal_idxs]
        detr_det_boxes = detr_det_boxes[area_legal_idxs]
        detr_det_logits = detr_det_logits[area_legal_idxs]
        detr_det_labels = detr_det_labels[area_legal_idxs]

        # De-normalize to target image size:
        print(detr_det_boxes)
        box_results = detr_det_boxes.cpu() * torch.tensor([ori_w, ori_h, ori_w, ori_h])
        print(box_results)
        box_results = box_cxcywh_to_xyxy(boxes=box_results)
        print(box_results)
        trajectory_history = deque(maxlen=max_temporal_length)
        if only_detr is False:
            if len(box_results) > get_model(model).num_id_vocabulary:
                print(f"[Carefully!] we only support {get_model(model).num_id_vocabulary} ids, "
                      f"but get {len(box_results)} detections in seq {video_path} {frame_count}th frame.")

        print(trajectory_history)
        # Decoding the current objects' IDs
        if only_detr is False:
            assert max_temporal_length - 1 > 0, f"MOTIP need at least T=1 trajectory history, " \
                                                f"but get T={max_temporal_length - 1} history in Eval setting."
            current_tracks = Instances(image_size=(0, 0))
            current_tracks.boxes = detr_det_boxes
            current_tracks.outputs = detr_det_outputs
            print(detr_det_boxes)
            current_tracks.ids = torch.tensor([get_model(model).num_id_vocabulary] * len(current_tracks),
                                              dtype=torch.long, device=current_tracks.outputs.device)
            current_tracks.confs = detr_det_logits.sigmoid()
            trajectory_history.append(current_tracks)
            print(trajectory_history)
            if len(trajectory_history) == 1:    # first frame, do not need decoding:
                newborn_filter = (trajectory_history[0].confs > newborn_thresh).reshape(-1, )   # filter by newborn
                trajectory_history[0] = trajectory_history[0][newborn_filter]
                box_results = box_results[newborn_filter.cpu()]
                ids = torch.tensor([current_id + _ for _ in range(len(trajectory_history[-1]))],
                                   dtype=torch.long, device=current_tracks.outputs.device)
                print(ids)
                trajectory_history[-1].ids = ids
                for _ in ids:
                    ids_to_results[_.item()] = current_id
                    current_id += 1
                id_results = []
                for _ in ids:
                    id_results.append(ids_to_results[_.item()])
                    id_deque.add(_.item())
                id_results = torch.tensor(id_results, dtype=torch.long)
            else:
                ids, trajectory_history, ids_to_results, current_id, id_deque, boxes_keep = get_model(model).inference(
                    trajectory_history=trajectory_history,
                    num_id_vocabulary=get_model(model).num_id_vocabulary,
                    ids_to_results=ids_to_results,
                    current_id=current_id,
                    id_deque=id_deque,
                    id_thresh=id_thresh,
                    newborn_thresh=newborn_thresh,
                    inference_ensemble=inference_ensemble,
                )   # already update the trajectory history/ids_to_results/current_id/id_deque in this function
                id_results = []
                for _ in ids:
                    id_results.append(ids_to_results[_])
                id_results = torch.tensor(id_results, dtype=torch.long)
                if boxes_keep is not None:
                    box_results = box_results[boxes_keep.cpu()]
        else:   # only detr, ID is just +1 for each detection.
            id_results = torch.tensor([current_id + _ for _ in range(len(box_results))], dtype=torch.long)
            current_id += len(id_results)

        # Output to tracker file:
        if fake_submit is False:
            # Write the outputs to the tracker file:
            result_file_path = os.path.join(outputs_dir,  f"{video_path.split('.')[0]}.txt")
            with open(result_file_path, "a") as file:
                assert len(id_results) == len(box_results), f"Boxes and IDs should in the same length, " \
                                                            f"but get len(IDs)={len(id_results)} and " \
                                                            f"len(Boxes)={len(box_results)}"
                object_id_colors = {}                                                            
                for obj_id, box in zip(id_results, box_results):
                    obj_id = int(obj_id.item())
                    obj_color = [np.random.randint(0,255) for _ in range(3)]
                    if not obj_id in object_id_colors:
                        object_id_colors[obj_id] = obj_color
                    x1, y1, x2, y2 = box.tolist()
                    print(box.tolist())
                    if dataset in ["DanceTrack", "MOT17", "SportsMOT", "MOT17_SPLIT", "MOT15", "MOT15_V2"]:
                        result_line = f"{frame_count}," \
                                      f"{obj_id}," \
                                      f"{x1},{y1},{x2 - x1},{y2 - y1},1,-1,-1,-1\n"
                    else:
                        raise NotImplementedError(f"Do not know the outputs format of dataset '{dataset}'.")
                    file.write(result_line)
                    x = x1
                    y = y1
                    w = x2-x1 
                    h = y2-y1 
                    cv2.rectangle(image, (x, y), (x + w, y + h), object_id_colors[obj_id], 2)
                    cv2.putText(image, f'id={obj_id}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, object_id_colors[obj_id], 2)
                
        out.write(image)
        print(frame_count* "=>")
    cap.release()                
    if fake_submit:
        print(f"[Fake] Finish >> Submit seq {outputs_dir}. ")
    else:
        print(f"Finish >> Submit seq {outputs_dir}. ")
    return


if __name__ == "__main__":
    args = parse_options()
    config_path = args.config_path
    cfg = yaml_to_dict(config_path)     # configs from .yaml file, path is set by runtime options.

    
    cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
    submit(config=cfg)
    
    