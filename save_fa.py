"""
save_fa.py
from
https://medium.com/@shintaroshiba/saving-3d-rendering-images-without-displays-on-python-opengl-f534a4638a0d
"""
import os

import cv2
import iris
import numpy as np
from OpenGL.GL import *
from PIL import Image
import glfw

from fieldanimation import FieldAnimation, field2RGB, modulus, Texture

OUTPUT_FILE = 'image.avi'


class UpdateableAnimation(FieldAnimation):
    """
    A FieldAnimation class that allows the vector field to be updated as
    the animation is running and also allows a video frame to be extracted.
    """
    def __init__(self, width, height, field, compute_shader=False,
                 image=None):
        """Initialise the class"""
        super().__init__(width, height, field, computeSahder=compute_shader,
                         image=image)

    def update_field(self, field):
        """Update the 2D vector field but leave the existing tracers"""
        field_as_rgb, u_min, u_max, v_min, v_max = field2RGB(field)
        self._fieldAsRGB = field_as_rgb
        self.modulus = modulus(field)
        self.fieldTexture = Texture(data=field_as_rgb,
                                    width=field_as_rgb.shape[1],
                                    height=field_as_rgb.shape[0],
                                    filt=OpenGL.GL.GL_LINEAR)

    def get_video_frame(self):
        """Get the current video frame as a BGR array for cv2"""
        image_buffer = glReadPixels(0, 0, self.w_width, self.w_height,
                                    OpenGL.GL.GL_BGR,
                                    OpenGL.GL.GL_UNSIGNED_BYTE)
        image = (np.frombuffer(image_buffer, dtype=np.uint8).
                 reshape(self.w_width, self.w_height, 3))
        return np.flipud(image)


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
    # Initialize the library
    if not glfw.init():
        print('Not initialised')
        return
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(display_width, display_height,
                                "Elinca Animation", None, None)
    if not window:
        glfw.terminate()
        print('Cannot create window')
        return
    # Make the window's context current
    glfw.make_context_current(window)

    # Setup the video writing
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 20
    writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps,
                             (display_width, display_height), True)

    lat_range = (-28, 42)
    long_range = (-70, 10)
    era5_dir = '/data/jseddon/era5/2013/10'
    # First time slice
    field_uv = load_era5_field(
        os.path.join(era5_dir, '01/ecmwf-era5_oper_an_sfc_201310010000.10u.nc'),
        os.path.join(era5_dir, '01/ecmwf-era5_oper_an_sfc_201310010000.10v.nc'),
        lat_range,
        long_range
    )

    # Second time slice
    field_uv2 = load_era5_field(
        os.path.join(era5_dir, '05/ecmwf-era5_oper_an_sfc_201310050000.10u.nc'),
        os.path.join(era5_dir, '05/ecmwf-era5_oper_an_sfc_201310050000.10v.nc'),
        lat_range,
        long_range
    )

    background_file = '/home/jseddon/python/elinca/background.png'
    background = np.flipud(np.asarray(Image.open(background_file), np.uint8))

    fa = UpdateableAnimation(display_width, display_height, field_uv, True,
                             background)

    n = 0
    while n < 400:
        n += 1
        if n == 200:
            print('Next frame')
            fa.update_field(field_uv2)
        glClear(GL_COLOR_BUFFER_BIT)
        glfw.poll_events()
        fa.draw()
        glfw.swap_buffers(window)

        writer.write(fa.get_video_frame())

    writer.release()
    glfw.destroy_window(window)
    glfw.terminate()

if __name__ == "__main__":
    main()
