"""
save_fa.py
from
https://medium.com/@shintaroshiba/saving-3d-rendering-images-without-displays-on-python-opengl-f534a4638a0d
"""
import cv2
import iris
import numpy as np
from OpenGL.GL import *
import glfw

from fieldanimation import FieldAnimation, field2RGB, modulus, Texture

OUTPUT_FILE = 'image.avi'

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

    # Setup the video writing
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 20
    writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps,
                             (DISPLAY_WIDTH, DISPLAY_HEIGHT), True)

    # Set up the field animation
    # field = np.load('/home/jseddon/python/fieldanimation/fieldanimation/'
    #                 'examples/wind_2016-11-20T00-00Z.npy')
    # U = field[:, :, 0][::-1]
    # V = - field[:, :, 1][::-1]

    # First time slice
    u_cube = iris.load_cube('/data/jseddon/era5/2013/10/01/'
                            'ecmwf-era5_oper_an_sfc_201310010000.10u.nc')
    u_atl = u_cube[0].intersection(latitude=(-28, 42), longitude=(-70, 10))
    U = u_atl.data[::-1]
    v_cube = iris.load_cube('/data/jseddon/era5/2013/10/01/'
                            'ecmwf-era5_oper_an_sfc_201310010000.10v.nc')
    v_atl = v_cube[0].intersection(latitude=(-28, 42), longitude=(-70, 10))
    V = v_atl.data[::-1]
    field_uv = np.flipud(np.dstack((U, -V)))

    # Second time slice
    u_cube = iris.load_cube('/data/jseddon/era5/2013/10/05/'
                            'ecmwf-era5_oper_an_sfc_201310050000.10u.nc')
    u_atl = u_cube[0].intersection(latitude=(-28, 42), longitude=(-70, 10))
    U = u_atl.data[::-1]
    v_cube = iris.load_cube('/data/jseddon/era5/2013/10/05/'
                            'ecmwf-era5_oper_an_sfc_201310050000.10v.nc')
    v_atl = v_cube[0].intersection(latitude=(-28, 42), longitude=(-70, 10))
    V = v_atl.data[::-1]
    field_uv2 = np.flipud(np.dstack((U, -V)))


    fa = FieldAnimation(DISPLAY_WIDTH, DISPLAY_HEIGHT, field_uv, True, None)

    n = 0
    while n < 1200:
        # TODO set frame rate on the OpenGL FieldAnimation side
        n += 1

        if n == 200:
            print('Next frame')
            fieldAsRGB, uMin, uMax, vMin, vMax = field2RGB(field_uv2)
            fa._fieldAsRGB = fieldAsRGB
            fa.modulus = modulus(field_uv2)
            fa.fieldTexture = Texture(data=fieldAsRGB,
                                      width=fieldAsRGB.shape[1],
                                      height=fieldAsRGB.shape[0],
                                      filt=OpenGL.GL.GL_LINEAR)
            # fa.setField(field_uv2)
        glClear(GL_COLOR_BUFFER_BIT)
        fa.draw()
        glfw.swap_buffers(window)

        image_buffer = glReadPixels(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT,
                                    OpenGL.GL.GL_BGR, OpenGL.GL.GL_UNSIGNED_BYTE)
        image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(DISPLAY_WIDTH,
                                                                    DISPLAY_HEIGHT,
                                                                    3)
        writer.write(np.flipud(image))

    writer.release()
    glfw.destroy_window(window)
    glfw.terminate()

if __name__ == "__main__":
    main()
