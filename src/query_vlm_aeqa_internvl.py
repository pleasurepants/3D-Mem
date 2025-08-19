import logging
from typing import Tuple, Optional, Union

from src.eval_utils_gpt_aeqa_internvl import explore_step
from src.tsdf_planner_hdbscan import TSDFPlanner, SnapShot, Frontier
from src.scene_aeqa import Scene
import json
import os

def save_snapshot_objects_with_names(
    question_id: str,
    snapshot_objects: dict,
    object_id_to_name: dict,
    json_path: str,
):
    """
    将当前step_dict['snapshot_objects']中object id转换为名称，并保存到json，支持增量更新。
    json_path就是最终的json文件路径。
    """
    # 保证路径上的目录存在
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # 加载历史数据（如果文件已存在）
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            all_data = json.load(f)
    else:
        all_data = {}

    # 构建本次数据（把object id转为名称）
    obj_name_dict = {}
    for snap_img, obj_ids in snapshot_objects.items():
        obj_name_dict[snap_img] = [
            object_id_to_name.get(str(obj_id), object_id_to_name.get(int(obj_id), str(obj_id)))
            for obj_id in obj_ids
        ]

    # 写入/更新该question_id的数据
    all_data[question_id] = obj_name_dict

    # 保存回json文件
    with open(json_path, 'w') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"保存成功：{json_path}")



def query_vlm_for_response(
    question: str,
    scene: Scene,
    tsdf_planner: TSDFPlanner,
    rgb_egocentric_views: list,
    cfg,
    verbose: bool = False,
    chosen_frontier_path: str = None,
    step_idx: int = 0,
    question_id: Optional[str] = None,
    lifelong_json_path: Optional[str] = None,
    # lifelong_context: Optional[str] = None,
) -> Optional[Tuple[Union[SnapShot, Frontier], str, int]]:
    # prepare input for vlm
    step_dict = {}

    # prepare snapshots
    object_id_to_name = {
        obj_id: obj["class_name"] for obj_id, obj in scene.objects.items()
    }
    step_dict["obj_map"] = object_id_to_name

    step_dict["snapshot_objects"] = {}
    step_dict["snapshot_imgs"] = {}
    for rgb_id, snapshot in scene.snapshots.items():
        step_dict["snapshot_objects"][rgb_id] = snapshot.cluster
        step_dict["snapshot_imgs"][rgb_id] = scene.all_observations[rgb_id]

    # prepare frontier
    step_dict["frontier_imgs"] = [
        frontier.feature for frontier in tsdf_planner.frontiers
    ]
    step_dict["frontier_imgs_0"] = [
        frontier.feature for frontier in tsdf_planner.frontiers_layer0
    ]
    step_dict["frontier_imgs_1"] = [
        frontier.feature for frontier in tsdf_planner.frontiers_layer1
    ]

    step_dict["layer0_to_layer1"] = tsdf_planner.layer0_to_layer1  
    step_dict["layer1_to_layer0"] = tsdf_planner.layer1_to_layer0


    # prepare egocentric views
    if cfg.egocentric_views:
        step_dict["egocentric_views"] = rgb_egocentric_views
        step_dict["use_egocentric_views"] = True

    # prepare question
    step_dict["question"] = question



    # save snapshot objects with names for lifelong
    save_snapshot_objects_with_names(
        question_id=question_id,
        snapshot_objects=step_dict["snapshot_objects"],
        object_id_to_name=object_id_to_name,
        json_path=lifelong_json_path,
    )





    # query vlm
    outputs, snapshot_id_mapping, reason, n_filtered_snapshots = explore_step(
        step_dict, cfg, verbose=verbose, chosen_frontier_path=chosen_frontier_path, step_idx=step_idx
    )
    if outputs is None:
        logging.error(f"explore_step failed and returned None")
        return None
    logging.info(f"Response: [{outputs}]\nReason: [{reason}]")

    # parse returned results
    try:
        target_type, target_index = outputs.split(" ")[0], outputs.split(" ")[1]
        logging.info(f"Prediction: {target_type}, {target_index}")
    except:
        logging.info(f"Wrong output format, failed!")
        return None

    if target_type not in ["snapshot", "frontier"]:
        logging.info(f"Wrong target type: {target_type}, failed!")
        return None

    if target_type == "snapshot":
        if int(target_index) < 0 or int(target_index) >= len(snapshot_id_mapping):
            logging.info(
                f"Target index can not match real objects: {target_index}, failed!"
            )
            return None
        target_index = snapshot_id_mapping[int(target_index)]
        logging.info(f"The index of target snapshot {target_index}")

        # get the target snapshot
        if target_index < 0 or target_index >= len(scene.snapshots):
            logging.info(
                f"Predicted snapshot target index out of range: {target_index}, failed!"
            )
            return None

        pred_target_snapshot = list(scene.snapshots.values())[target_index]
        logging.info(
            "Pred_target_class: "
            + str(
                " ".join(
                    [
                        object_id_to_name[obj_id]
                        for obj_id in pred_target_snapshot.cluster
                    ]
                )
            )
        )
        logging.info(f"Next choice Snapshot of {pred_target_snapshot.image}")

        return pred_target_snapshot, reason, n_filtered_snapshots
    else:  # target_type == "frontier"
        target_index = int(target_index)
        if target_index < 0 or target_index >= len(tsdf_planner.frontiers):
            logging.info(
                f"Predicted frontier target index out of range: {target_index}, failed!"
            )
            return None
        target_point = tsdf_planner.frontiers[target_index].position
        logging.info(f"Next choice: Frontier at {target_point}")
        pred_target_frontier = tsdf_planner.frontiers[target_index]

        return pred_target_frontier, reason, n_filtered_snapshots
    

    
