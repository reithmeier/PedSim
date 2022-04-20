"""
Pedestrian Simulation Example
"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from simulate.models.pedestrian import Actor, SocialForceModel

actors: List[Actor] = [
    Actor(identifier=0, position=np.array([0.0, 0.0]), path=[np.array([2, 2])]),
    Actor(identifier=1, position=np.array([1.0, 1.0]), path=[np.array([-1, -1])]),
]

model = SocialForceModel(lambda x: x, actors, [])
labels = model.labels()

# Define the metadata for the movie
metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
writer = animation.FFMpegWriter(fps=15, metadata=metadata)

# Initialize the movie
fig, ax = plt.subplots()

# plot the sine wave line
color_map = plt.cm.get_cmap("Set1", 2)
circles = []
i = 0
for _ in actors:
    circle = plt.Circle((0, 0), 0.2, color=color_map(i))
    ax.add_patch(circle)
    circles.append(circle)
    i += 1
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

plt.xlabel("x")
plt.ylabel("y")


def update(_):
    """update func"""
    result = model.simulate(0, 0.01)
    j = 0
    for actor in result[labels["actors"]]:
        circles[j].set_center(actor.position)
        j += 1
    return circles


ani = animation.FuncAnimation(fig, update, frames=100, blit=True)
plt.show()
