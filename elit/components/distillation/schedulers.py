# Adopted from https://github.com/airaria/TextBrewer
# Apache License Version 2.0

# x is between 0 and 1


def linear_growth_weight_scheduler(x):
    return x


def linear_decay_weight_scheduler(x):
    return 1 - x


class LinearTeacherAnnealingScheduler(object):
    def __init__(self, num_training_steps: int) -> None:
        super().__init__()
        self._num_training_steps = num_training_steps
        self._current_training_steps = 0

    def step(self):
        self._current_training_steps += 1

    def __float__(self):
        return self._current_training_steps / self._num_training_steps


