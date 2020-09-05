"""
save_fa.py
from
https://medium.com/@shintaroshiba/saving-3d-rendering-images-without-displays-on-python-opengl-f534a4638a0d
"""
import copy
import os

import cv2
import iris
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from OpenGL.GL import (
    glClear,
    glClearColor,
    glReadPixels,
    GL_LINEAR,
    GL_BGR,
    GL_UNSIGNED_BYTE,
    GL_COLOR_BUFFER_BIT,
)
import glfw

from fieldanimation import FieldAnimation, field2RGB, modulus, Texture
from fieldanimation.examples.glfwBackend import glfwApp

OUTPUT_FILE = "image.avi"

# Various predefined PIL colours with full opacity
PIL_BLACK = (0, 0, 0, 255)
PIL_ORANGE = (235, 119, 52, 255)
PIL_GREEN = (90, 252, 3, 255)

# The path to a suitable monospaced true-type font. May vary from
# computer to computer; not sure how to automate this other than by
# including the font in this repository
FONT_PATH = '/usr/share/fonts/truetype/freefont/FreeMono.ttf'

# The top-level directory containing the reanalysis data
ERA5_DIR = "/data/jseddon/era5"

# The JSON Pandas output containing the interpolated hourly positions
HOURLY_FILE = '/home/jseddon/python/elinca/hourly_positions.json'

# The file containing the background image
BACKGROUND_FILE = "/home/jseddon/python/elinca/background.png"


class UpdateableAnimation(FieldAnimation):
    """
    A FieldAnimation class that allows the vector field to be updated as
    the animation is running and also allows a video frame to be extracted.
    """

    def __init__(self, width, height, field, compute_shader=False, image=None):
        """Initialise the class"""
        # Prevent linter warnings for attributes used in this class
        self._fieldAsRGB = None
        self.modulus = None
        self.fieldTexture = None
        # Initialise parent class
        super().__init__(
            width, height, field, computeSahder=compute_shader, image=image
        )

    def update_field(self, field):
        """Update the 2D vector field but leave the existing tracers"""
        field_as_rgb, u_min, u_max, v_min, v_max = field2RGB(field)
        self._fieldAsRGB = field_as_rgb
        self.modulus = modulus(field)
        self.fieldTexture = Texture(
            data=field_as_rgb,
            width=field_as_rgb.shape[1],
            height=field_as_rgb.shape[0],
            filt=GL_LINEAR,
        )

    def get_video_frame(self):
        """Get the current video frame as a BGR array for cv2"""
        image_buffer = glReadPixels(
            0, 0, self.w_width, self.w_height, GL_BGR, GL_UNSIGNED_BYTE
        )
        image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(
            self.w_width, self.w_height, 3
        )
        return np.flipud(image)


class VideoWriteGlfwApp(glfwApp):
    """
    An glfwApp that supports writing videos.
    """

    def __init__(self, videopath, fps=50, title="", width=800, height=600):
        super().__init__(title=title, width=width, height=height, resizable=False)
        # Setup the video writing
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._writer = cv2.VideoWriter(videopath, fourcc, fps, (width, height), True)
        self._fa = None

    def set_fa(self, fa):
        self._fa = fa

    def onResize(self, window, width, height):
        """Resizing's not allowed"""
        pass

    def close(self):
        """Additionally, close the video writer"""
        self._writer.release()
        super().close()

    def run_frame(self, skip_write=False):
        """
        Update a frame

        :param bool skip_write: if True then don't write this frame to the
            output video.
        """
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT)
        self._fa.draw()
        glfw.swap_buffers(self._window)
        if not skip_write:
            self._writer.write(self._fa.get_video_frame())


class BackgroundImage:
    """
    Load a background image and allow the addition of overlays
    """
    def __init__(self, filepath):
        """
        Load the image and calculate any necessary properties
        """
        self.orig_image = Image.open(filepath)
        self.width, self.height = self.orig_image.size

    def get_frame(self, datetime_string, lat, lon, vessel_colour,
                  lat_range, lon_range, font_size=30, vessel_radius=3,
                  text_width_proportion=0.47,
                  text_height_proportion=0.9375,
                  font_path=FONT_PATH
                  ):
        """
        Get a background image with the vessel and date overlaid

        :param str datetime_string: The date and time text to display
        :param float lat: the vessel's latitude
        :param float lon: the vessel's longitude
        :param tuple vessel_colour: The 4-component PIL colour to plot the
            vessel as
        :param tuple lat_range: the min and max latitudes in the background
            image
        :param tuple lon_range: the min and max longitudes in the background
            image
        :param int font_size: The size of the font in PIL units
        :param int vessel_radius: The radius of the vessel marker in pixels:
        :param float text_width_proportion: the proportion of the way across
            the screen to put the date time string
        :param float text_height_proportion: the proportion of the way down
            the screen to put the date time string
        :param str font_path: The path to the true-type font to use
        """
        frame_image = copy.copy(self.orig_image)
        draw = ImageDraw.Draw(frame_image)
        font = ImageFont.truetype(font_path, font_size)
        text_position = (
            int(self.width * text_width_proportion),
            int(self.height * text_height_proportion)
        )
        draw.text(text_position, datetime_string, font=font, fill=PIL_BLACK)
        lat_pixel = self.height - int((lat - lat_range[0]) /
                                      abs(lat_range[1] - lat_range[0]) *
                                      self.height)
        lon_pixel = int((lon - lon_range[0]) / abs(lon_range[1] - lon_range[0]) *
                        self.width)
        vessel_box = (
            lon_pixel - vessel_radius,
            lat_pixel - vessel_radius,
            lon_pixel + vessel_radius,
            lat_pixel + vessel_radius
        )
        draw.ellipse(vessel_box, fill=vessel_colour, outline=vessel_colour)
        return np.flipud(np.asarray(frame_image, np.uint8))


def load_era5_field(u_file, v_file, lats, longs):
    """
    Load ERA5 u and v wind files, trim to the specified latitude and longitude
    range and convert to a FieldAnimation uv field.

    :param str u_file: path to u wind component file
    :param str v_file: path to v wind component file
    :param tuple lats: tuple of minimum and maximum latitudes
    :param tuple longs: tuple of minimum and maximum longitudes
    :returns: the uv field
    :rtype: numpy.array
    """
    u_cube = iris.load_cube(u_file)
    u_atl = u_cube[0].intersection(latitude=lats, longitude=longs)
    u = u_atl.data[::-1]
    v_cube = iris.load_cube(v_file)
    v_atl = v_cube[0].intersection(latitude=lats, longitude=longs)
    v = v_atl.data[::-1]
    field_uv = np.flipud(np.dstack((u, -v)))
    return field_uv


def main():
    display_width = 800
    display_height = 800

    lat_range = (-28, 42)
    long_range = (-70, 10)

    # Load the positions
    with open(HOURLY_FILE) as fh:
        hourly = pd.read_json(fh, convert_dates=['time'])

    # Get just October for a reduced subset
    october = hourly[hourly.time.dt.strftime('%Y%m%d').between('20131001',
                                                               '20131106')]

    background_image = BackgroundImage(BACKGROUND_FILE)

    app = VideoWriteGlfwApp(
        OUTPUT_FILE,
        title="Elinca Animation",
        width=display_width,
        height=display_height,
    )

    for i, dp in enumerate(october.iterrows()):
        de = dp[1]
        print(f'{de.time.year}{de.time.month:02}{de.time.day:02}')
        # Load data
        day_dir = os.path.join(ERA5_DIR, f'{de.time.year}',
                               f'{de.time.month:02}', f'{de.time.day:02}')
        file_prefix = (f'ecmwf-era5_oper_an_sfc_{de.time.year}'
                       f'{de.time.month:02}{de.time.day:02}'
                       f'{de.time.hour:02}00')
        field_uv = load_era5_field(
            os.path.join(day_dir, file_prefix + '.10u.nc'),
            os.path.join(day_dir, file_prefix + '.10v.nc'),
            lat_range,
            long_range,
        )

        date_str = de.time.strftime('%d/%m/%Y %H:%M')
        boat_colour = PIL_GREEN if de.src == 'f' else PIL_ORANGE
        background = background_image.get_frame(date_str,
                                                de.lat, de.lon,
                                                boat_colour,
                                                lat_range, long_range)

        if i == 0:
            # If first field then create the animation
            fa = UpdateableAnimation(display_width, display_height, field_uv,
                                     True, background)
            fa.palette = False
            app.set_fa(fa)
        else:
            # On subsequent iterations then just update
            fa.update_field(field_uv)
            fa.imageTexture = Texture(data=background, dtype=GL_UNSIGNED_BYTE)

        # Allow animation to update num_updates_per_time for each frame
        # Advance the animation 8 times for each time period but only write
        # 4 of these out to the video
        num_updates_per_time = 8
        for n in range(num_updates_per_time):
            if (n + 1) % 2:
                app.run_frame(skip_write=True)
            else:
                app.run_frame(skip_write=False)

    app.close()


if __name__ == "__main__":
    main()
