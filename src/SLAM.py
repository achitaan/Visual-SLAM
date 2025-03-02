import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import g2o

class SLAM:
    def __init__(self):
        # Initialize the graph/optimizer.
        self.__init_graph()
        # Containers for BoW and pose graph data.
        self.descriptor_list = []   # List of numpy arrays (each image's descriptors)
        self.histograms = []        # BoW histograms for each frame
        self.initial_poses = []     # List of 4x4 numpy arrays (initial pose estimates)
        self.odometry_edges = []    # List of tuples: (from_idx, to_idx, relative_transform)
        self.loop_edges = []        # List of tuples for loop closures: (from_idx, to_idx, relative_transform)
        self.kmeans = None          # Will hold the vocabulary (KMeans model)

    def __init_graph(self):
        """Initialize the g2o optimizer with a Levenberg solver using Eigen-based linear solver."""
        self.optimizer = g2o.SparseOptimizer()
        # Using BlockSolverSE3 with LinearSolverEigenSE3 (you can change this based on your build)
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        self.optimizer.set_algorithm(solver)

    def build_vocabulary(self, descriptors, num_clusters=50):
        """
        Build the visual vocabulary using KMeans clustering.
        'descriptors' should be a list of numpy arrays (one per image).
        """
        # Stack all descriptors into one array
        all_desc = np.vstack(descriptors)
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        self.kmeans.fit(all_desc)
        print("Vocabulary built with {} clusters.".format(num_clusters))

    def compute_bow_histogram(self, descriptors):
        """
        Compute a Bag-of-Words histogram for a set of descriptors.
        Returns a 1D numpy array histogram.
        """
        if self.kmeans is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        # Predict cluster indices (visual word assignment)
        words = self.kmeans.predict(descriptors)
        histogram = np.zeros(self.kmeans.n_clusters)
        for w in words:
            histogram[w] += 1
        return histogram

    def detect_loop_closure(self, hist, threshold=0.3):
        """
        Compare the current histogram with previous ones to detect a loop closure.
        Returns the index of a matching frame if similarity (cosine distance) is high.
        """
        if len(self.histograms) == 0:
            return None
        neigh = NearestNeighbors(n_neighbors=1, metric='cosine')
        neigh.fit(self.histograms)
        distance, index = neigh.kneighbors([hist])
        if distance[0][0] < threshold:
            return index[0][0]
        return None

    def add_odometry_edge(self, from_idx, to_idx, relative_transform):
        """
        Add an odometry edge between consecutive frames.
        'relative_transform' is a 4x4 numpy array representing the relative motion.
        """
        self.odometry_edges.append((from_idx, to_idx, relative_transform))

    def add_loop_closure_edge(self, from_idx, to_idx, relative_transform):
        """
        Add a loop closure edge between non-consecutive frames.
        """
        self.loop_edges.append((from_idx, to_idx, relative_transform))

    def optimize_pose_graph(self, num_iterations=10):
        """
        Build the pose graph from the initial poses and edges (both odometry and loop closures),
        then run optimization using g2o.
        Returns a list of optimized 4x4 pose matrices.
        """
        # Add vertices for each pose.
        for i, pose in enumerate(self.initial_poses):
            v = g2o.VertexSE3()
            v.set_id(i)
            v.set_estimate(g2o.Isometry3d(pose))
            if i == 0:
                v.set_fixed(True)  # Fix the first pose as a reference.
            self.optimizer.add_vertex(v)
        # Combine all edges.
        all_edges = self.odometry_edges + self.loop_edges
        for edge in all_edges:
            frm, to, rel_transform = edge
            e = g2o.EdgeSE3()
            e.set_vertex(0, self.optimizer.vertex(frm))
            e.set_vertex(1, self.optimizer.vertex(to))
            e.set_measurement(g2o.Isometry3d(rel_transform))
            e.set_information(np.identity(6))
            self.optimizer.add_edge(e)
        # Run the optimization.
        self.optimizer.initialize_optimization()
        self.optimizer.optimize(num_iterations)
        # Retrieve optimized poses.
        optimized = [self.optimizer.vertex(i).estimate().matrix() for i in range(len(self.initial_poses))]
        return optimized

    def process_frame(self, descriptors, initial_pose):
        """
        Process one frame:
          - Append descriptors (for vocabulary).
          - Compute its BoW histogram (if vocabulary is built).
          - Detect loop closure (if possible).
          - Append the initial pose.
        Returns: loop closure index if detected (or None).
        """
        self.descriptor_list.append(descriptors)
        self.initial_poses.append(initial_pose)
        loop_idx = None
        if self.kmeans is not None:
            hist = self.compute_bow_histogram(descriptors)
            loop_idx = self.detect_loop_closure(hist, threshold=0.3)
            self.histograms.append(hist)
        else:
            # If vocabulary is not yet built, we still store an empty placeholder.
            self.histograms.append(np.zeros(1))
        return loop_idx

# Example usage:
if __name__ == "__main__":
    # In a real application, you would load descriptors from your feature extractor.
    # Here we simulate descriptors and poses for a sequence of frames.
    slam = SLAM()
    
    # Suppose we have 5 frames.
    num_frames = 5
    simulated_descriptors = []
    simulated_poses = []
    for i in range(num_frames):
        # Simulate descriptors as random float32 arrays (each with 100 keypoints and 32 dimensions)
        desc = np.random.rand(100, 32).astype(np.float32)
        simulated_descriptors.append(desc)
        # Simulate an initial pose as a translation along X (1 meter per frame)
        pose = np.eye(4)
        pose[0, 3] = i * 1.0
        simulated_poses.append(pose)
    
    # Build the vocabulary using descriptors from the first 3 frames.
    slam.build_vocabulary(simulated_descriptors[:3], num_clusters=50)
    
    # Process each frame.
    for i in range(num_frames):
        loop_idx = slam.process_frame(simulated_descriptors[i], simulated_poses[i])
        # For sequential frames, add an odometry edge.
        if i > 0:
            # Simulate relative transform as identity with translation of 1 meter in X.
            rel = np.eye(4)
            rel[0, 3] = 1.0
            slam.add_odometry_edge(i - 1, i, rel)
        # If a loop closure is detected and the frame is not too recent, add a loop closure edge.
        if loop_idx is not None and abs(i - loop_idx) > 1:
            # Here we assume an identity relative transform (in practice compute it from feature matching).
            slam.add_loop_closure_edge(i, loop_idx, np.eye(4))
            print(f"Loop closure detected between frame {i} and frame {loop_idx}")

    # Optimize the pose graph.
    optimized_poses = slam.optimize_pose_graph(num_iterations=10)
    for i, pose in enumerate(optimized_poses):
        print(f"Optimized pose for frame {i}:\n{pose}\n")
