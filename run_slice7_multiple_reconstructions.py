from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
)

import torch
# torch.cuda.set_device(1)


# images = Path("/mnt/ssd2/victor/kapture/Extended-CMU-Seasons/slice7/mapping/sensors/records_data/cam0/")
# outputs = Path("/mnt/ssd2/victor/kapture/Extended-CMU-Seasons/slice7/hloc/mapping/cam0/")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    images = Path(args.image_dir)
    outputs = Path(args.output_dir)

    sfm_pairs = outputs / "pairs-netvlad.txt"
    
    sfm_dir = outputs / "sfm_superpoint+superglue"

    sfm_dirs = [outputs/"sfm_superpoint+superglue", outputs/"r2d2+NN", outputs/"d2net-ss+NN", outputs/"sift+NN"]

    global_feats = "netvlad"
    retrieval_conf = extract_features.confs["netvlad"]

    retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)

    local_feats = ['superpoint_max', 'r2d2', 'd2net-ss', 'sift']
    matchers = ['superglue', 'NN-ratio', 'NN-ratio', 'NN-ratio']

    # local_feats = ['d2net-ss', 'sift']
    # matchers = ['NN-ratio', 'NN-ratio']

    for i, local_feat in enumerate(local_feats):
        feature_conf = extract_features.confs[local_feat]
        matcher_conf = match_features.confs[matchers[i]]

        feature_path = extract_features.main(feature_conf, images, outputs)
        match_path = match_features.main(
            matcher_conf, sfm_pairs, feature_conf["output"], outputs
        )

        model = reconstruction.main(sfm_dirs[i], images, sfm_pairs, feature_path, match_path)