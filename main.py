from config import algorithm
from varying_main import varying_dynamics
from varying_awe import varying_dynamics_awe


if __name__ == '__main__':

    if algorithm != 'fedawe':
        varying_dynamics()
    else:
        varying_dynamics_awe()

