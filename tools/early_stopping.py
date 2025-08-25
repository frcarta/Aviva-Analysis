class EarlyStopping:
    def __init__(self, patience, delta=0, relative=False):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
        self.relative = relative

    def check_early_stop(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
            self.no_improvement_count = 0
            return

        if not self.relative:
            improvement = loss < self.best_loss - self.delta
        else:
            improvement = (self.best_loss - loss) / self.best_loss > self.delta

        if improvement:
            self.best_loss = loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
