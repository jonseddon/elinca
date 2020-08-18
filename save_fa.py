"""
save_fa.py
from
https://medium.com/@shintaroshiba/saving-3d-rendering-images-without-displays-on-python-opengl-f534a4638a0d
"""
import copy
import os

import cv2
import iris
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

PIL_BLACK = (0, 0, 0, 255)
PIL_ORANGE = (235, 119, 52, 255)

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

    def __init__(self, videopath, fps=20, title="", width=800, height=600):
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

    def run_frame(self):
        """Update a frame"""
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT)
        self._fa.draw()
        glfw.swap_buffers(self._window)
        self._writer.write(self._fa.get_video_frame())


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
    era5_dir = "/data/jseddon/era5/2013/10"
    # First time slice
    field_uv = load_era5_field(
        os.path.join(era5_dir, "01/ecmwf-era5_oper_an_sfc_201310010000.10u.nc"),
        os.path.join(era5_dir, "01/ecmwf-era5_oper_an_sfc_201310010000.10v.nc"),
        lat_range,
        long_range,
    )

    # Second time slice
    field_uv2 = load_era5_field(
        os.path.join(era5_dir, "05/ecmwf-era5_oper_an_sfc_201310050000.10u.nc"),
        os.path.join(era5_dir, "05/ecmwf-era5_oper_an_sfc_201310050000.10v.nc"),
        lat_range,
        long_range,
    )

    background_file = "/home/jseddon/python/elinca/background.png"
    # Load background image and convert from OpenCV BGR to Pillow RGB (+ alpha)
    # orig_image = cv2.imread(background_file, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0, 3]]
    orig_image = Image.open(background_file)
    background_width, background_height = orig_image.size
    frame_image = copy.copy(orig_image)
    draw = ImageDraw.Draw(frame_image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf',
                              30)
    draw.text((272, 540), '01/10/2013 12:00', font=font, fill=PIL_BLACK)
    draw.ellipse((0, 0, 5, 5), fill=PIL_ORANGE, outline=PIL_ORANGE)
    #  = cv2.putText(, '01/10/2013 12:00', )
    background = np.flipud(np.asarray(frame_image, np.uint8))
    frame_image = copy.copy(orig_image)
    draw = ImageDraw.Draw(frame_image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf',
                              30)
    draw.text((272, 540), '05/10/2013 12:00', font=font, fill=PIL_BLACK)
    draw.ellipse((background_width-5, background_height-5,
                    background_width, background_height), fill=PIL_ORANGE, outline=PIL_ORANGE)
    background2 = np.flipud(np.asarray(frame_image, np.uint8))

    app = VideoWriteGlfwApp(
        OUTPUT_FILE,
        title="Elinca Animation",
        width=display_width,
        height=display_height,
    )

    fa = UpdateableAnimation(display_width, display_height, field_uv, True, background)
    fa.palette = False
    app.set_fa(fa)

    n = 0
    while n < 400:
        n += 1
        if n == 200:
            print("Next frame")
            fa.update_field(field_uv2)
            fa.imageTexture = Texture(data=background2, dtype=GL_UNSIGNED_BYTE)
        app.run_frame()

    app.close()


if __name__ == "__main__":
    main()
