import subprocess
import random

import fire
import numpy as np
from PIL import Image

from generate import generate_sequence

color_vals = {"black": (0, 0, 0), "red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1),  
           "teal": (0, 1, 1), "pink": (1, 0, 1), "yellow": (0.8, 0.8, 0), "grey": (0.5, 0.5, 0.5)}
color_names = list(color_vals)

def run(im_size=32, steps=3, method="lattice", png_resize=None, color=False):
    im = np.ones((im_size, im_size, 3))
    feature_map = [im]

    if method == "traditional":
        im = np.ones((im_size, im_size, 3))
        stride = 2**steps
        im[::stride, ::stride] = [0, 0, 0]
        im = Image.fromarray(np.uint8(im*255))
        if png_resize:
            im = im.resize([png_resize, png_resize])
        im.save("visualize_output/traditional_{}_{}.png".format(im_size, steps), "PNG")

    elif method in ("random", "naive", "lattice"):
        for i in range(steps):
            sequence = generate_sequence(2**i, method)
            submaps1 = []
            submaps2 = []
            print(''.join([str(x) for x in sequence]))
            
            for j, submap in enumerate(feature_map):
                if sequence[j] == 0:
                    submaps1.append(submap[::2, ::2])
                    submaps2.append(submap[1::2, 1::2])
                else:
                    submaps1.append(submap[::2, 1::2])
                    submaps2.append(submap[1::2, ::2])
            feature_map = submaps1 + submaps2

        for k, submap in enumerate(feature_map):
            if steps < 4 and color:
                submap[:, :] = color_vals[color_names[k]]
            else:
                submap[:, :] = (0, 0, 0)

        im = Image.fromarray(np.uint8(im*255))
        if png_resize:
            im = im.resize([png_resize, png_resize])
        im.save("visualize_output/checkered_{}_{}_{}.png".format(method, im_size, steps), "PNG")

    elif method == "tri":
        # Multisampling by randomly choosing one of six 3x3 samplers.
        for i in range(steps):
                sequence = generate_sequence(3**i, "random", sampler_count=6)
                submaps1 = []
                submaps2 = []
                submaps3 = []
                print(sequence)
                
                for j, submap in enumerate(feature_map):
                    if sequence[j] == 0:
                        submaps1.append(submap[::3, ::3])
                        submaps2.append(submap[1::3, 2::3])
                        submaps3.append(submap[2::3, 1::3])
                    elif sequence[j] == 1:
                        submaps1.append(submap[::3, 1::3])
                        submaps2.append(submap[1::3, ::3])
                        submaps3.append(submap[2::3, 2::3])
                    elif sequence[j] == 2:
                        submaps1.append(submap[::3, 2::3])
                        submaps2.append(submap[1::3, 1::3])
                        submaps3.append(submap[2::3, ::3])
                    elif sequence[j] == 3:
                        submaps1.append(submap[::3, 2::3])
                        submaps2.append(submap[1::3, ::3])
                        submaps3.append(submap[2::3, 1::3])
                    elif sequence[j] == 4:
                        submaps1.append(submap[::3, 1::3])
                        submaps2.append(submap[1::3, 2::3])
                        submaps3.append(submap[2::3, ::3])     
                    elif sequence[j] == 5:
                        submaps1.append(submap[::3, ::3])
                        submaps2.append(submap[1::3, 1::3])
                        submaps3.append(submap[2::3, 2::3])      
                feature_map = submaps1 + submaps2 + submaps3

        for k, submap in enumerate(feature_map):
            submap[:, :] = (0, 0, 0)

        im = Image.fromarray(np.uint8(im*255))
        im.save("visualize_output/tri_random_{}_{}.png".format(im_size, steps), "PNG")

    else:
        exit("Method not recognized.")

if __name__ == "__main__":
    """
    Generates depictions of samples taken by traditional downsampling, checkered subsampling, and one instance
    of a random multisampling layer with 3x3 samplers.

    Args:
        --im_size (int) - size (height and width) of the input image we're subsampling. (default 32)
        --steps (int) - how many times to apply our subsampling method on the image. (default 3)
        --method (string) - what sort of subsampling method should we use from [naive, lattice, random, traditional, tri] (default lattice)
        --png_size (int) - used to upsample the final image so it's not tiny, if desired. (default None)
        --color (bool) - turns on coloring of first 3 checkered subsampling steps (default False)

    python visualize.py --im_size 32 --steps 3 --method lattice
    or
    python visualize.py 32 3 lattice
    """
    fire.Fire(run)