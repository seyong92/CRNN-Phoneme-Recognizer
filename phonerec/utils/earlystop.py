class EarlyStop:
    def __init__(self, patience=7, goal='minimize', delta=0):
        self.patience = patience
        self.goal = goal
        self.delta = delta

        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, score):
        if self.goal == 'minimize':
            score = -score
        elif self.goal == 'maximize':
            score = score

        if self.best_score is None:
            self.best_score = score
            return 'best'
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return 'stop'
            return 'patience'
        else:
            self.best_score = score
            self.counter = 0
            return 'best'
