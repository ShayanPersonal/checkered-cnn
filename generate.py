import random

def generate_sequence(submap_count, method="naive", sampler_count=2):
    if method == "naive":
        return [0] * submap_count
    elif method == "lattice":
        # Gets log2(submap_count)
        index = submap_count.bit_length() - 1
        return lattice_generator[index]
    elif method == "random":
        return [random.randint(0, sampler_count-1) for _ in range(submap_count)]
    else:
        raise ValueError("The method for generating samplers should be either naive, lattice, or random.")

# Defines on what submaps to apply the standard checkered sampler (0) or the complementary sampler (1)
# in order to generate a low-discrepancy lattice sampling, up to 10 subsampling steps.
lattice_generator = [
        [0],
        [0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0, 0, 0, 1, 1],
        [0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0],
        [0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0],
        [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,
        0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0],

        [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,
        0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],

        [0]*20 + [1]*20 + [0]*20 + [1]*19 + [0]*20 + [1]*20 + [0]*19 + [1]*20 + [0]*20 + [1]*19 + [0]*20 + [1]*20 + [0]*19,

        # Alternative 0s and 1s
        [i % 2 for i in range(512)],
    ]