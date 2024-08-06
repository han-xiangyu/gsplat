
import os
import random
import json
import numpy as np
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import utils.general_utils as utils
import torch

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def extract_camera_position_in_world_space(camera):
    w2cR = camera.R
    w2cT = camera.T

    w2c = np.eye(4)
    w2c[:3, :3] = w2cR
    w2c[:3, 3] = w2cT

    c2w = np.linalg.inv(w2c)
    c2wT = c2w[:3, 3]

    return c2wT

def kmeans_v1(n_clusters, train_cameras_positions, test_cameras_positions):
    # This will generate uneven size of clusters. 
    # And the uneven sizes are bad especially for the test set. 

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(train_cameras_positions)
    train_cameras_cluster = kmeans.labels_
    test_cameras_cluster = kmeans.predict(test_cameras_positions)
    return train_cameras_cluster, test_cameras_cluster, kmeans

def assign_cameras_to_clusters_v2(n_clusters, centers, cameras_positions):
    cluster_size = len(cameras_positions) // n_clusters
    remaining_cameras = len(cameras_positions) % n_clusters

    repeated_centers_p1 = np.repeat(centers[:remaining_cameras], cluster_size + 1, axis=0)
    repeated_centers_p2 = np.repeat(centers[remaining_cameras:], cluster_size, axis=0)
    repeated_centers = np.concatenate([repeated_centers_p1, repeated_centers_p2], axis=0)

    assert repeated_centers.shape[0] == len(cameras_positions), "The number of cameras in the repeated centers is not equal to the number of cameras in the training set."
    assert repeated_centers.shape[1] == cameras_positions[0].shape[0], "The number of features in the repeated centers is not equal to the number of features in the training set."

    distance_matrix = cdist(np.array(cameras_positions), repeated_centers)
    cameras_cluster = linear_sum_assignment(distance_matrix)[1]

    # if idx < remaining_cameras * (cluster_size + 1) -> idx // (cluster_size + 1)
    # else -> remaining_cameras + (idx - remaining_cameras * (cluster_size + 1)) // cluster_size
    cameras_cluster = np.array(
        [idx // (cluster_size + 1) 
            if idx < remaining_cameras * (cluster_size + 1) else 
         remaining_cameras + (idx - remaining_cameras * (cluster_size + 1)) // cluster_size 
            for idx in cameras_cluster.tolist()]
    )

    average_distance_matrix = []
    for center_id in range(len(centers)):
        if center_id < remaining_cameras:
            column_id = center_id * (cluster_size + 1)
        else:
            column_id = remaining_cameras * (cluster_size + 1) + (center_id - remaining_cameras) * cluster_size
        distance_matrix_column = distance_matrix[:, column_id]
        average_distance_matrix.append([])
        for cluster in range(n_clusters):
            average_distance_matrix[-1].append(np.mean(distance_matrix_column[cameras_cluster == cluster]))
    
    utils.LOG_FILE.write("Average distance matrix: " + str(np.array(average_distance_matrix)) + "\n")

    return cameras_cluster

def kmeans_v2(n_clusters, train_cameras_positions, test_cameras_positions):
    # This will generate even size of clusters.

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(train_cameras_positions)
    centers = kmeans.cluster_centers_ # shape = (n_clusters, n_features)

    train_cameras_cluster = assign_cameras_to_clusters_v2(n_clusters, centers, train_cameras_positions)
    test_cameras_cluster = assign_cameras_to_clusters_v2(n_clusters, centers, test_cameras_positions)

    return train_cameras_cluster, test_cameras_cluster, kmeans

def aerial_image_placement_k_means(args, train_cameras, test_cameras):

    train_cameras = train_cameras

    train_cameras_positions = []
    for id, camera in enumerate(train_cameras):
        train_cameras_positions.append(extract_camera_position_in_world_space(camera))

    # get the test camera positions
    test_cameras_positions = []
    for id, camera in enumerate(test_cameras):
        test_cameras_positions.append(extract_camera_position_in_world_space(camera))

    num_clusters = utils.DEFAULT_GROUP.size()

    train_cameras_cluster, test_cameras_cluster, kmeans = \
        kmeans_v2(num_clusters, train_cameras_positions, test_cameras_positions)
        # kmeans_v1(num_clusters, train_cameras_positions, test_cameras_positions)


    # output cluster centers
    utils.LOG_FILE.write("Train Cluster centers: " + str(kmeans.cluster_centers_) + "\n")
    # output the number of cameras in each cluster
    for i in range(num_clusters):
        utils.LOG_FILE.write("Cluster " + str(i) + " has " + str(np.sum(train_cameras_cluster == i)) + " train cameras\n")
        utils.LOG_FILE.write("Cluster " + str(i) + " has " + str(np.sum(test_cameras_cluster == i)) + " test cameras\n")

    train_cameras_cluster = train_cameras_cluster.tolist()
    test_cameras_cluster = test_cameras_cluster.tolist()

    # output the cameras in each cluster
    # for cluster_id in range(num_clusters):
    #     utils.LOG_FILE.write("Cluster " + str(cluster_id) + ". Cluster Center: " + str(kmeans.cluster_centers_[cluster_id]) +"\n")
    #     for i, label in enumerate(train_cameras_cluster):
    #         if label == cluster_id:
    #             utils.LOG_FILE.write("Position: " + str(train_cameras_positions[i]) + ". train_camera_id: "+train_cameras[i].image_path +"\n")
    #     for i, label in enumerate(test_cameras_cluster):
    #         if label == cluster_id:
    #             utils.LOG_FILE.write("Position: " + str(test_cameras_positions[i]) + ". test_camera_id: "+test_cameras[i].image_path +"\n")

    # # Generate ply files for these clusters for visualizing the correctness of the clustering
    # if utils.DEFAULT_GROUP.rank() == 1:
    #     # save the cluster centers in .ply file
        
    #     from plyfile import PlyData, PlyElement

    #     centers = kmeans.cluster_centers_
    #     centers = centers.tolist()

    #     def save_ply_file(file_path, points3d):
    #         dtype = [
    #             ("x", "f4"),
    #             ("y", "f4"),
    #             ("z", "f4"),
    #         ]
    #         elements = np.empty(len(points3d), dtype=dtype)
    #         elements[:] = list(map(tuple, points3d))

    #         vertex = PlyElement.describe(elements, 'vertex')
    #         PlyData([vertex]).write(file_path)
        
    #     save_ply_file(os.path.join(args.model_path, "cluster_centers.ply"), centers)
    #     for cluster_id in range(num_clusters):

    #         train_points_3d = []
    #         for i, label in enumerate(train_cameras_cluster):
    #             if label == cluster_id:
    #                 train_points_3d.append(train_cameras_positions[i])
    #         save_ply_file(os.path.join(args.model_path, "train_cluster_" + str(cluster_id) + ".ply"), train_points_3d)

    #         test_points_3d = []
    #         for i, label in enumerate(test_cameras_cluster):
    #             if label == cluster_id:
    #                 test_points_3d.append(test_cameras_positions[i])
    #         save_ply_file(os.path.join(args.model_path, "test_cluster_" + str(cluster_id) + ".ply"), test_points_3d)

    #     torch.distributed.barrier()
    # else:
    #     torch.distributed.barrier()

    return train_cameras_cluster, test_cameras_cluster



