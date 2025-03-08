import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import g2o

class SLAM:
    def __init__(self):
        self.__init_graph()
        self.descriptor_list = []
        self.histograms = []
        self.initial_poses = []
        self.odometry_edges = []
        self.loop_edges = []
        self.kmeans = None

    def __init_graph(self):
        self.optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        self.optimizer.set_algorithm(solver)

    def build_vocabulary(self, descriptors, num_clusters=50):
        all_desc = np.vstack(descriptors)
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        self.kmeans.fit(all_desc)
        print("Vocabulary built with {} clusters.".format(num_clusters))

    def compute_bow_histogram(self, descriptors):
        if self.kmeans is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        words = self.kmeans.predict(descriptors)
        histogram = np.zeros(self.kmeans.n_clusters)
        for w in words:
            histogram[w] += 1
        return histogram

    def detect_loop_closure(self, hist, threshold=0.3):
        if len(self.histograms) == 0:
            return None
        neigh = NearestNeighbors(n_neighbors=1, metric='cosine')
        neigh.fit(self.histograms)
        distance, index = neigh.kneighbors([hist])
        if distance[0][0] < threshold:
            return index[0][0]
        return None

    def add_odometry_edge(self, from_idx, to_idx, relative_transform):
        self.odometry_edges.append((from_idx, to_idx, relative_transform))

    def add_loop_closure_edge(self, from_idx, to_idx, relative_transform):
        self.loop_edges.append((from_idx, to_idx, relative_transform))

    def optimize_pose_graph(self, num_iterations=10):
        for i, pose in enumerate(self.initial_poses):
            v = g2o.VertexSE3()
            v.set_id(i)
            v.set_estimate(g2o.Isometry3d(pose))
            if i == 0:
                v.set_fixed(True)
            self.optimizer.add_vertex(v)

        all_edges = self.odometry_edges + self.loop_edges
        for edge in all_edges:
            frm, to, rel_transform = edge
            e = g2o.EdgeSE3()
            e.set_vertex(0, self.optimizer.vertex(frm))
            e.set_vertex(1, self.optimizer.vertex(to))
            e.set_measurement(g2o.Isometry3d(rel_transform))
            e.set_information(np.identity(6))
            self.optimizer.add_edge(e)

        self.optimizer.initialize_optimization()
        self.optimizer.optimize(num_iterations)
        optimized = [self.optimizer.vertex(i).estimate().matrix() for i in range(len(self.initial_poses))]
        return optimized

    def process_frame(self, descriptors, initial_pose):
        self.descriptor_list.append(descriptors)
        self.initial_poses.append(initial_pose)
        loop_idx = None
        if self.kmeans is not None:
            hist = self.compute_bow_histogram(descriptors)
            loop_idx = self.detect_loop_closure(hist, threshold=0.3)
            self.histograms.append(hist)
        else:
            self.histograms.append(np.zeros(1))
        return loop_idx

if __name__ == "__main__":
    slam = SLAM()
    num_frames = 5
    simulated_descriptors = []
    simulated_poses = []
    
    for i in range(num_frames):
        desc = np.random.rand(100, 32).astype(np.float32)
        simulated_descriptors.append(desc)
        pose = np.eye(4)
        pose[0, 3] = i * 1.0
        simulated_poses.append(pose)
    
    slam.build_vocabulary(simulated_descriptors[:3], num_clusters=50)
    
    for i in range(num_frames):
        loop_idx = slam.process_frame(simulated_descriptors[i], simulated_poses[i])
        if i > 0:
            rel = np.eye(4)
            rel[0, 3] = 1.0
            slam.add_odometry_edge(i - 1, i, rel)
        if loop_idx is not None and abs(i - loop_idx) > 1:
            slam.add_loop_closure_edge(i, loop_idx, np.eye(4))
            print(f"Loop closure detected between frame {i} and frame {loop_idx}")

    optimized_poses = slam.optimize_pose_graph(num_iterations=10)
    for i, pose in enumerate(optimized_poses):
        print(f"Optimized pose for frame {i}:\n{pose}\n")
