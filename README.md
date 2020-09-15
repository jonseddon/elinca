Copyright (c) 2020, Jon Seddon

Please see LICENSE for further details.

Requires:

git@bitbucket.org:bvidmar/fieldanimation.git (https://bvidmar.bitbucket.io/fieldanimation/latest/ for documentation)

Contains:

`wind_arrows.py` generates a series of images with the wind represented with arrows that can be combined into a video using ffmpeg.

`save_fa.py` generates an animation of the wind using the Field Animation package. Video is written directly using CV2.

`make_positions.ipynb` generates the hourly positions from the raw GPX file.

`make_background.ipynb` generates a background image for the animation and will be moved into that code soon.
