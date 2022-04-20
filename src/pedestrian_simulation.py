"""
Pedestrian Simulation Example
"""
from typing import List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from simulate.models.pedestrian import Actor, SocialForceModel

actors: List[Actor] = [
    Actor(identifier=0, position=np.array([0, 0]), path=[np.array([2, 2])]),
    Actor(identifier=1, position=np.array([1, 1]), path=[np.array([-1, -1])]),
]

model = SocialForceModel(lambda x: x, actors, [])
labels = model.labels()

# Define the metadata for the movie
metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
writer = animation.FFMpegWriter(fps=15, metadata=metadata)

# Initialize the movie
fig, ax = plt.subplots()

# plot the sine wave line
circles = []
for _ in actors:
    circle = plt.plot([], [], "ro", markersize=10)
    circles.append(circle[0])
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

plt.xlabel("x")
plt.ylabel("y")


def update(_):
    """update func"""
    result = model.simulate(0, 0.01)
    j = 0
    for actor in result[labels["actors"]]:
        x = actor.position[0]
        y = actor.position[1]
        circles[j].set_data(x, y)
        j += 1
    return circles


ani = animation.FuncAnimation(fig, update, frames=100, blit=True)
plt.show()
