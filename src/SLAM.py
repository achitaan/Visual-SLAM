import g2o


class SLAM:
    def __init__(self):
        pass

    def __init_graph(self):
        self.optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        self.optimizer.set_algorithm(solver)