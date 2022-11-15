import numpy as np

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    np.random.seed(round(label*1.333+2.222))
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    random_factor = np.random.randint(0,255,size=[1,1]).reshape(1,).tolist()[0]
    # color = np.random.randint(0,255,size=[1,3]).reshape(3,).tolist()
    # color = [(c*random_factor)%255 for c in color]

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]

    return tuple(color)