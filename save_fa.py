"""
save_fa.py
Copyright (c) 2020, Jon Seddon

from
https://medium.com/@shintaroshiba/saving-3d-rendering-images-without-displays-on-python-opengl-f534a4638a0d
"""
import argparse
import logging
import os
import sys

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


config = {
    "global": {
        "font_path": "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "era5_dir": "/data/jseddon/era5",
        "hourly_file": "/home/jseddon/python/elinca/hourly_positions.json",
    },
    "videos": {
        "pre_delivery": {
            "background_file": "/home/jseddon/python/elinca/backgrounds/"
                               "leg_pre_delivery.png",
            "opening_image": "credits/start_leg_pre_delivery.png",
            "opening_seconds": 5,
            "end_image": "credits/end.png",
            "end_seconds": 10,
            "width": 800,
            "height": 800,
            "lat_range": (49.7, 58.7),
            "lon_range": (-11, -2),
            "start_date": "20130917",
            "end_date": "20130923",
            "filename": "elinca_pre_delivery.avi",
        },
        "leg_1": {
            "background_file": "/home/jseddon/python/elinca/backgrounds/"
                               "leg_01.png",
            "opening_image": "credits/start_leg_01.png",
            "opening_seconds": 5,
            "end_image": "credits/end.png",
            "end_seconds": 10,
            "width": 800,
            "height": 800,
            "lat_range": (37, 51),
            "lon_range": (-15, -1),
            "start_date": "20130924",
            "end_date": "20131001",
            "filename": "elinca_leg_01.avi",
        },
        "legs_2_3": {
            "background_file": "/home/jseddon/python/elinca/backgrounds/"
                               "leg_02_03.png",
            "opening_image": "credits/start_leg_02_03.png",
            "opening_seconds": 5,
            "end_image": "credits/end.png",
            "end_seconds": 10,
            "width": 800,
            "height": 800,
            "lat_range": (-33, 47),
            "lon_range": (-70, 10),
            "start_date": "20131001",
            "end_date": "20131106",
            "filename": "elinca_leg_02_03.avi",
        },
        "leg_4": {
            "background_file": "/home/jseddon/python/elinca/backgrounds/"
                               "leg_04.png",
            "opening_image": "credits/start_leg_04.png",
            "opening_seconds": 5,
            "end_image": "credits/end.png",
            "end_seconds": 10,
            "width": 800,
            "height": 800,
            "lat_range": (-57, -22),
            "lon_range": (-75, -40),
            "start_date": "20131110",
            "end_date": "20131203",
            "filename": "elinca_leg_04.avi",
        },
        "leg_5": {
            "background_file": "/home/jseddon/python/elinca/backgrounds/"
                               "leg_05.png",
            "opening_image": "credits/start_leg_05.png",
            "opening_seconds": 5,
            "end_image": "credits/end.png",
            "end_seconds": 10,
            "width": 800,
            "height": 800,
            "lat_range": (-68, -54),
            "lon_range": (-70, -56),
            "start_date": "20131209",
            "end_date": "20140104",
            "filename": "elinca_leg_05.avi",
        },
        "leg_6": {
            "background_file": "/home/jseddon/python/elinca/backgrounds/"
                               "leg_05.png",
            "opening_image": "credits/start_leg_06.png",
            "opening_seconds": 5,
            "end_image": "credits/end.png",
            "end_seconds": 10,
            "width": 800,
            "height": 800,
            "lat_range": (-68, -54),
            "lon_range": (-70, -56),
            "start_date": "20140105",
            "end_date": "20140129",
            "filename": "elinca_leg_06.avi",
        },
        "legs_7_8": {
            "background_file": "/home/jseddon/python/elinca/backgrounds/"
                               "leg_07.png",
            "opening_image": "credits/start_leg_07.png",
            "opening_seconds": 5,
            "end_image": "credits/end.png",
            "end_seconds": 10,
            "width": 800,
            "height": 800,
            "lat_range": (-67, -33),
            "lon_range": (-69, -35),
            "start_date": "20140201",
            "end_date": "20140308",
            "filename": "elinca_leg_07_08.avi",
        },
        "leg_9": {
            "background_file": "/home/jseddon/python/elinca/backgrounds/"
                               "leg_09.png",
            "opening_image": "credits/start_leg_09.png",
            "opening_seconds": 5,
            "end_image": "credits/end.png",
            "end_seconds": 10,
            "width": 800,
            "height": 800,
            "lat_range": (-37.5, -22),
            "lon_range": (-57.5, -42),
            "start_date": "20140312",
            "end_date": "20140329",
            "filename": "elinca_leg_09.avi",
        },
        "leg_10": {
            "background_file": "/home/jseddon/python/elinca/backgrounds/"
                               "leg_10.png",
            "opening_image": "credits/start_leg_10.png",
            "opening_seconds": 5,
            "end_image": "credits/end.png",
            "end_seconds": 10,
            "width": 800,
            "height": 800,
            "lat_range": (-24, 39),
            "lon_range": (-70, -7),
            "start_date": "20140331",
            "end_date": "20140428",
            "filename": "elinca_leg_10.avi",
        },
        "leg_11": {
            "background_file": "/home/jseddon/python/elinca/backgrounds/"
                               "leg_11.png",
            "opening_image": "credits/start_leg_11.png",
            "opening_seconds": 5,
            "end_image": "credits/end.png",
            "end_seconds": 10,
            "width": 800,
            "height": 800,
            "lat_range": (31.6, 49.6),
            "lon_range": (-26, -8),
            "start_date": "20140429",
            "end_date": "20140509",
            "filename": "elinca_leg_11.avi",
        },
        "leg_12": {
            "background_file": "/home/jseddon/python/elinca/backgrounds/"
                               "leg_12.png",
            "opening_image": "credits/start_leg_12.png",
            "opening_seconds": 5,
            "end_image": "credits/end.png",
            "end_seconds": 10,
            "width": 800,
            "height": 800,
            "lat_range": (43, 51),
            "lon_range": (-10.7, -2.7),
            "start_date": "20140510",
            "end_date": "20140517",
            "filename": "elinca_leg_12.avi",
        },
        "post_delivery": {
            "background_file": "/home/jseddon/python/elinca/backgrounds/"
                               "leg_pre_delivery.png",
            "opening_image": "credits/start_leg_post_delivery.png",
            "opening_seconds": 5,
            "end_image": "credits/end.png",
            "end_seconds": 10,
            "width": 800,
            "height": 800,
            "lat_range": (49.7, 58.7),
            "lon_range": (-11, -2),
            "start_date": "20140518",
            "end_date": "20140524",
            "filename": "elinca_post_delivery.avi",
        },

    },
}

# Various predefined PIL colours with full opacity
PIL_BLACK = (0, 0, 0, 255)
PIL_ORANGE = (235, 119, 52, 255)
PIL_GREEN = (90, 252, 3, 255)


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

    def __init__(self, videopath, font_path, fps=30, title="", width=800, height=800):
        super().__init__(title=title, width=width, height=height, resizable=False)
        # Setup the video writing
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._writer = cv2.VideoWriter(videopath, fourcc, fps, (width, height), True)
        self._fa = None
        self.width = width
        self.height = height
        self._fps = fps
        self._font_path = font_path

    def set_fa(self, fa):
        self._fa = fa

    def onResize(self, window, width, height):
        """Resizing's not allowed"""
        pass

    def close(self):
        """Additionally, close the video writer"""
        self._writer.release()
        super().close()

    def run_frame(
        self,
        datetime_string,
        lat,
        lon,
        vessel_colour,
        lat_range,
        lon_range,
        skip_write=False,
    ):
        """
        Update a frame

        :param str datetime_string: The string to display in the bottom right
            of each frame, typically the date and time
        :param float lat: The decimal latitude position of the vessel
        :param float lon: The decimal longitude position of the vessel
        :param tuple vessel_colour: The PIL colour to plot the vessel with
        :param tuple lat_range: the min and max latitudes in the background
            image
        :param tuple lon_range: the min and max longitudes in the background
            image
        :param bool skip_write: if True then don't write this frame to the
            output video.
        """
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT)
        self._fa.draw()
        glfw.swap_buffers(self._window)
        if not skip_write:
            self._writer.write(
                overlay_video_frame(
                    self._fa.get_video_frame(),
                    datetime_string,
                    lat,
                    lon,
                    vessel_colour,
                    lat_range,
                    lon_range,
                    self.width,
                    self.height,
                    self._font_path,
                )
            )

    def write_image(self, image_path, number_seconds):
        """
        Load the specified image and save it to the output video for the
        specified number of frames.

        :param str image_path: The path of the image
        :param int number_seconds: The number of seconds to show it for
        """
        image = cv2.imread(image_path)
        for index in range(number_seconds * self._fps):
            self._writer.write(image)


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

    def get_frame(self):
        """
        Get a background image with the vessel and date overlaid

        :returns: The backgrund image as a Numpy array orientated for the
            FieldAnimation package.
        """
        return np.flipud(np.asarray(self.orig_image, np.uint8))


def overlay_video_frame(
    frame_image,
    datetime_string,
    lat,
    lon,
    vessel_colour,
    lat_range,
    lon_range,
    width,
    height,
    font_path,
    font_size=30,
    vessel_radius=3,
    text_width_proportion=0.6,
    text_height_proportion=0.9375,
):
    """"
    Overlay the vessel position and the specified date and time string on

    :param numpy.ndarray frame_image: The video frame in CV2 format
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
    :returns: The frame with the overlaid information in CV2 format
    :rtype: numpy.ndarray
    """
    im_pil = Image.fromarray(frame_image)
    draw = ImageDraw.Draw(im_pil)
    font = ImageFont.truetype(font_path, font_size)
    text_position = (
        int(width * text_width_proportion),
        int(height * text_height_proportion),
    )
    draw.text(text_position, datetime_string, font=font, fill=PIL_BLACK)
    lat_pixel = height - int(
        (lat - lat_range[0]) / abs(lat_range[1] - lat_range[0]) * height
    )
    lon_pixel = int((lon - lon_range[0]) / abs(lon_range[1] - lon_range[0]) * width)
    vessel_box = (
        lon_pixel - vessel_radius,
        lat_pixel - vessel_radius,
        lon_pixel + vessel_radius,
        lat_pixel + vessel_radius,
    )
    draw.ellipse(vessel_box, fill=vessel_colour, outline=vessel_colour)
    return np.asarray(im_pil)


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


def produce_leg(global_config, leg_config):
    """
    Generate a window and animation for the leg with the configuration that
    has been passed in.

    :param dict global_config: global configuration parameters
    :param dict leg_config: the configuration for this leg
    """
    logging.info(f'Producing {leg_config["filename"]}')

    # Load the positions
    with open(global_config["hourly_file"]) as fh:
        hourly = pd.read_json(fh, convert_dates=["time"])

    # Get the date range for this leg
    leg_positions = hourly[
        hourly.time.dt.strftime("%Y%m%d").between(
            leg_config["start_date"], leg_config["end_date"]
        )
    ]

    background_image = BackgroundImage(leg_config["background_file"])
    background = background_image.get_frame()

    app = VideoWriteGlfwApp(
        leg_config["filename"],
        global_config["font_path"],
        title="Elinca Animation",
        width=leg_config["width"],
        height=leg_config["height"],
    )

    app.write_image(leg_config["opening_image"], leg_config["opening_seconds"])

    for i, dp in enumerate(leg_positions.iterrows()):
        de = dp[1]
        logging.debug(
            f"{de.time.year}{de.time.month:02}{de.time.day:02} {de.time.hour:02}:00"
        )
        # Load data
        day_dir = os.path.join(
            global_config["era5_dir"],
            f"{de.time.year}",
            f"{de.time.month:02}",
            f"{de.time.day:02}",
        )
        file_prefix = (
            f"ecmwf-era5_oper_an_sfc_{de.time.year}"
            f"{de.time.month:02}{de.time.day:02}"
            f"{de.time.hour:02}00"
        )
        field_uv = load_era5_field(
            os.path.join(day_dir, file_prefix + ".10u.nc"),
            os.path.join(day_dir, file_prefix + ".10v.nc"),
            leg_config["lat_range"],
            leg_config["lon_range"],
        )

        date_str = de.time.strftime("%d/%m/%Y %H:%M")
        boat_colour = PIL_GREEN if de.src == "f" else PIL_ORANGE

        if i == 0:
            # If first field then create the animation
            fa = UpdateableAnimation(
                leg_config["width"], leg_config["height"], field_uv, True, background
            )
            fa.palette = False
            app.set_fa(fa)
        else:
            # On subsequent iterations then just update
            fa.update_field(field_uv)

        # Allow animation to update num_updates_per_time for each frame
        # Advance the animation num_updates_per_time for each time period but
        # only write out to the video every output_frame_every_n_frames frames.
        num_updates_per_time = 9
        output_frame_every_n_frames = 3
        for n in range(num_updates_per_time):
            if (n + 1) % output_frame_every_n_frames:
                app.run_frame(
                    date_str,
                    de.lat,
                    de.lon,
                    boat_colour,
                    leg_config["lat_range"],
                    leg_config["lon_range"],
                    skip_write=True,
                )
            else:
                app.run_frame(
                    date_str,
                    de.lat,
                    de.lon,
                    boat_colour,
                    leg_config["lat_range"],
                    leg_config["lon_range"],
                    skip_write=False,
                )

    app.write_image(leg_config["end_image"], leg_config["end_seconds"])
    app.close()


def parse_args():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Produce Elinca wind particle " "animation videos"
    )
    all_or_leg = parser.add_mutually_exclusive_group(required=True)
    all_or_leg.add_argument("-a", "--all", help="Produce all legs", action="store_true")
    all_or_leg.add_argument(
        "-l",
        "--leg_name",
        help="The name of the leg from " "the config file to " "produce",
    )
    parser.add_argument(
        "-d", "--debug", help="dsiplay debug information", action="store_true"
    )
    return parser.parse_args()


def main(args):
    """Main entry"""
    if not args.all:
        if not args.leg_name in config["videos"]:
            logging.error(f"Leg name {args.leg_name} not found in the configuration.")
            sys.exit(1)
        produce_leg(config["global"], config["videos"][args.leg_name])
    else:
        for leg_name in config["videos"]:
            produce_leg(config["global"], config["videos"][leg_name])


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    main(args)
