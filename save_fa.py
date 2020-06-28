"""
save_image.py
from
https://medium.com/@shintaroshiba/saving-3d-rendering-images-without-displays-on-python-opengl-f534a4638a0d
"""
import time
import cv2
import numpy as np
from OpenGL.GL import *
import glfw

from fieldanimation import FieldAnimation

def main():
    DISPLAY_WIDTH = 800
    DISPLAY_HEIGHT = 800
    # Initialize the library
    if not glfw.init():
        print('Not initialised')
        return
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(DISPLAY_WIDTH, DISPLAY_HEIGHT, "hidden window",
                                None, None)
    if not window:
        glfw.terminate()
        print('Cannot create window')
        return
    # Make the window's context current
    glfw.make_context_current(window)

    # Set up the field animation
    field = np.load('/home/jseddon/python/fieldanimation/fieldanimation/'
                    'examples/wind_2016-11-20T00-00Z.npy')
    U = field[:, :, 0][::-1]
    V = - field[:, :, 1][::-1]
    field_uv = np.flipud(np.dstack((U, -V)))
    fa = FieldAnimation(DISPLAY_WIDTH, DISPLAY_HEIGHT, field_uv, True, None)

    glClear(GL_COLOR_BUFFER_BIT)
    fa.draw()
    glfw.swap_buffers(window)

    time.sleep(5)
    image_buffer = glReadPixels(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT,
                                OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
    image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(DISPLAY_WIDTH,
                                                                DISPLAY_HEIGHT,
                                                                3)
    cv2.imwrite("image.png", image)
    glfw.destroy_window(window)
    glfw.terminate()

if __name__ == "__main__":
    main()
