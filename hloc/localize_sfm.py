import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import cv2

import numpy as np
import pycolmap
from tqdm import tqdm

from . import logger
from .utils.io import get_keypoints, get_matches
from .utils.parsers import parse_image_lists, parse_retrieval

from kapture import PoseTransform
import quaternion

def do_covisibility_clustering(
    frame_ids: List[int], reconstruction: pycolmap.Reconstruction
):
    clusters = []
    visited = set()
    for frame_id in frame_ids:
        # Check if already labeled
        if frame_id in visited:
            continue

        # New component
        clusters.append([])
        queue = {frame_id}
        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue
            visited.add(exploration_frame)
            clusters[-1].append(exploration_frame)

            observed = reconstruction.images[exploration_frame].points2D
            connected_frames = {
                obs.image_id
                for p2D in observed
                if p2D.has_point3D()
                for obs in reconstruction.points3D[p2D.point3D_id].track.elements
            }
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


class QueryLocalizer:
    def __init__(self, reconstruction, config=None):
        self.reconstruction = reconstruction
        self.config = config or {}
        self.kf = cv2.KalmanFilter(6, 3)
        self.first = True
        self.warmup = 5000
        self.curr_iter = 0
        self._init_kalman_filter()

        self.log_file = open("/mnt/ssd2/victor/hloc_out/aqtr/slice7/run_5/cam0/aux_log.txt", "w")
        
        self.pred_file = open("/mnt/ssd2/victor/hloc_out/aqtr/slice7/run_5/cam0/pred_log.txt", "w")
        self.est_file = open("/mnt/ssd2/victor/hloc_out/aqtr/slice7/run_5/cam0/est_log.txt", "w")
        self.col_file = open("/mnt/ssd2/victor/hloc_out/aqtr/slice7/run_5/cam0/col_log.txt", "w")

        self.est_list = []
        self.pred_list = []

        self.aux_list = []

        self.cnt = 1

        self.last_pose = None
        self.errors_list = []

        self.min_points = 5000


    def _init_kalman_filter(self):
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        

        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) 
        # I4 = np.eye(3, dtype=np.float32) / 4
        # I2 = np.eye(3, dtype=np.float32) / 2
        # I1 = np.eye(3, dtype=np.float32)

        # self.kf.processNoiseCov[:3, :3] = I4
        # self.kf.processNoiseCov[3:, 3:] = I1
        # self.kf.processNoiseCov[3:, :3] = I2
        # self.kf.processNoiseCov[:3, 3:] = I2

        self.kf.processNoiseCov *= 2 # 0.4
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.0001

        self.kf.measurementNoiseCov[0,0] = 0.44
        self.kf.measurementNoiseCov[1,1] = 0.19
        self.kf.measurementNoiseCov[2,2] = 1.0

        
        self.kf.statePre = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32) # Just an assumption. We do not know exactly the starting point...
        self.kf.statePost = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32) # Just an assumption. We do not know exactly the first measurement values...
        
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 100.0 # 1.0
        self.kf.errorCovPre = np.eye(6, dtype=np.float32) * 100.0 # 1.0

    
    def _compute_error_last_pose(self, pose_est):
        if self.last_pose is None:
            return 0
        return np.linalg.norm(pose_est.inverse().t - self.last_pose.inverse().t)


    def estimate(self, points2D, points3D, query_camera):
        if len(points2D) < self.min_points:
            self.min_points = len(points2D)
            # print(f"Min points: {self.min_points}")

        if len(points2D) < 15 and not self.first:
            print(f"Few points at: {self.cnt}")
            # print(quaternion.as_float_array(self.last_pose.r))
            pose_est = self.kf.predict()
            ret = {
                "num_inliers": 0,
                "inliers": 0,
                "cam_from_world": pycolmap.Rigid3d(
                    rotation=quaternion.as_float_array(self.last_pose.r), # rotation of last pose as 'placeholder'
                    translation=np.array(pose_est[:3]),
                ),
            }
            return ret

        ret = pycolmap.absolute_pose_estimation(
            points2D,
            points3D,
            query_camera,
            estimation_options=self.config.get("estimation", {}),
            refinement_options=self.config.get("refinement", {}),
        )

        if self.first:
            # if it is the first image initialize the Kalman Filter with the first pose and return the Colmap pose
            self.first = False
            self.kf.statePre[:3] = ret["cam_from_world"].translation
            self.kf.statePost[:3] = ret["cam_from_world"].translation
            self.last_pose = PoseTransform(t=ret["cam_from_world"].translation, r=ret["cam_from_world"].rotation.quat)
            return ret
        
        # print(f"Current pose: {ret['cam_from_world'].rotation}")
        error_last_pose = self._compute_error_last_pose(PoseTransform(t=ret["cam_from_world"].translation, r=ret["cam_from_world"].rotation.quat))

        self.errors_list.append(error_last_pose)

        self.last_pose = PoseTransform(t=ret["cam_from_world"].translation, r=ret["cam_from_world"].rotation.quat)

        prediction = np.zeros((3, 1), dtype=np.float32)
        
        t, r = np.zeros((3, 1)), np.zeros((3, 1))

        # if error_last_pose > 10: # This should be in meters?
        #     # If the error is too big, we do not update the Kalman Filter and go with the Kalman Filter prediction
        #     prediction = self.kf.predict()
        #     t, r = ret["cam_from_world"].translation, ret["cam_from_world"].rotation
        #     pose_est = prediction[:3]
        # else:
        t = ret["cam_from_world"].translation
        r = ret["cam_from_world"].rotation

        prediction = self.kf.predict()
        estimated = self.kf.correct(np.array(t, dtype=np.float32))
        
        pose_est = np.reshape(estimated[:3], t.shape)
       
        self.pred_file.write(f"{prediction[0]} {prediction[1]} {prediction[2]}\n")
        self.est_file.write(f"{pose_est[0]} {pose_est[1]} {pose_est[2]}\n")
        self.col_file.write(f"{t[0]} {t[1]} {t[2]}\n")
        
        if self.curr_iter < self.warmup:
            # if it is not the first image but we are still in the warmup phase, just return the Colmap pose
            self.curr_iter += 1
            self.cnt += 1
            return ret
        

        ret = {
            "num_inliers": ret["num_inliers"],
            "inliers": ret["inliers"],
            "cam_from_world": pycolmap.Rigid3d(
                rotation=r,
                translation=np.array(pose_est),
            ),
        }
        self.cnt += 1
        return ret


    def localize(self, points2D_all, points2D_idxs, points3D_id, query_camera):
        points2D = points2D_all[points2D_idxs]
        points3D = [self.reconstruction.points3D[j].xyz for j in points3D_id]

        ret = self.estimate(points2D, points3D, query_camera)
        return ret
    

def pose_from_cluster(
    localizer: QueryLocalizer,
    qname: str,
    query_camera: pycolmap.Camera,
    db_ids: List[int],
    features_path: Path,
    matches_path: Path,
    **kwargs,
):
    kpq = get_keypoints(features_path, qname)
    kpq += 0.5  # COLMAP coordinates

    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches = 0
    for i, db_id in enumerate(db_ids):
        image = localizer.reconstruction.images[db_id]
        if image.num_points3D == 0:
            logger.debug(f"No 3D points found for {image.name}.")
            continue
        points3D_ids = np.array(
            [p.point3D_id if p.has_point3D() else -1 for p in image.points2D]
        )

        matches, _ = get_matches(matches_path, qname, image.name)
        matches = matches[points3D_ids[matches[:, 1]] != -1]
        num_matches += len(matches)
        for idx, m in matches:
            id_3D = points3D_ids[m]
            kp_idx_to_3D_to_db[idx][id_3D].append(i)
            # avoid duplicate observations
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)

    idxs = list(kp_idx_to_3D.keys())
    mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
    mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
    ret = localizer.localize(kpq, mkp_idxs, mp3d_ids, query_camera, **kwargs)
    if ret is not None:
        ret["camera"] = query_camera

    # mostly for logging and post-processing
    mkp_to_3D_to_db = [
        (j, kp_idx_to_3D_to_db[i][j]) for i in idxs for j in kp_idx_to_3D[i]
    ]
    log = {
        "db": db_ids,
        "PnP_ret": ret,
        "keypoints_query": kpq[mkp_idxs],
        "points3D_ids": mp3d_ids,
        "points3D_xyz": None,  # we don't log xyz anymore because of file size
        "num_matches": num_matches,
        "keypoint_index_to_db": (mkp_idxs, mkp_to_3D_to_db),
    }
    return ret, log


def main(
    reference_sfm: Union[Path, pycolmap.Reconstruction],
    queries: Path,
    retrieval: Path,
    features: Path,
    matches: Path,
    results: Path,
    ransac_thresh: int = 12,
    covisibility_clustering: bool = False,
    prepend_camera_name: bool = False,
    config: Dict = None,
):
    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    queries = parse_image_lists(queries, with_intrinsics=True)
    retrieval_dict = parse_retrieval(retrieval)

    logger.info("Reading the 3D model...")
    if not isinstance(reference_sfm, pycolmap.Reconstruction):
        reference_sfm = pycolmap.Reconstruction(reference_sfm)
    db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}

    config = {"estimation": {"ransac": {"max_error": ransac_thresh}}, **(config or {})}
    localizer = QueryLocalizer(reference_sfm, config)

    cam_from_world = {}
    logs = {
        "features": features,
        "matches": matches,
        "retrieval": retrieval,
        "loc": {},
    }

    logger.info("Starting localization...")
    for qname, qcam in tqdm(queries):
        if qname not in retrieval_dict:
            logger.warning(f"No images retrieved for query image {qname}. Skipping...")
            continue
        db_names = retrieval_dict[qname]
        db_ids = []
        for n in db_names:
            if n not in db_name_to_id:
                logger.warning(f"Image {n} was retrieved but not in database")
                continue
            db_ids.append(db_name_to_id[n])

        if covisibility_clustering:
            clusters = do_covisibility_clustering(db_ids, reference_sfm)
            best_inliers = 0
            best_cluster = None
            logs_clusters = []
            for i, cluster_ids in enumerate(clusters):
                ret, log = pose_from_cluster(
                    localizer, qname, qcam, cluster_ids, features, matches
                )
                if ret is not None and ret["num_inliers"] > best_inliers:
                    best_cluster = i
                    best_inliers = ret["num_inliers"]
                logs_clusters.append(log)
            if best_cluster is not None:
                ret = logs_clusters[best_cluster]["PnP_ret"]
                cam_from_world[qname] = ret["cam_from_world"]
            logs["loc"][qname] = {
                "db": db_ids,
                "best_cluster": best_cluster,
                "log_clusters": logs_clusters,
                "covisibility_clustering": covisibility_clustering,
            }
        else:
            ret, log = pose_from_cluster(
                localizer, qname, qcam, db_ids, features, matches
            )
            if ret is not None:
                cam_from_world[qname] = ret["cam_from_world"]
            else:
                closest = reference_sfm.images[db_ids[0]]
                cam_from_world[qname] = closest.cam_from_world
            log["covisibility_clustering"] = covisibility_clustering
            logs["loc"][qname] = log

    # for i, e in enumerate(localizer.errors_list):
    #     print(f"{i+1} --- {e}")
    
    logger.info(f"Localized {len(cam_from_world)} / {len(queries)} images.")
    logger.info(f"Writing poses to {results}...")
    with open(results, "w") as f:
        for query, t in cam_from_world.items():
            qvec = " ".join(map(str, t.rotation.quat[[3, 0, 1, 2]]))
            tvec = " ".join(map(str, t.translation))
            name = query.split("/")[-1]
            if prepend_camera_name:
                name = query.split("/")[-2] + "/" + name
            f.write(f"{name} {qvec} {tvec}\n")

    logs_path = f"{results}_logs.pkl"
    logger.info(f"Writing logs to {logs_path}...")
    # TODO: Resolve pickling issue with pycolmap objects.
    with open(logs_path, "wb") as f:
        pickle.dump(logs, f)
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_sfm", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--ransac_thresh", type=float, default=12.0)
    parser.add_argument("--covisibility_clustering", action="store_true")
    parser.add_argument("--prepend_camera_name", action="store_true")
    args = parser.parse_args()
    main(**args.__dict__)
