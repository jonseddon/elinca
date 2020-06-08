"""
wind_arrows.py
(c) Jon Seddon 2020

Plot Elinca's position in October 2013.

Convert the images to a video with:
ffmpeg -framerate 5 -i plots/img%06d.png -c:v libx264 -r 30 
    -pix_fmt yuv420p wind_arrows.mp4
"""
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import gpxpy
import iris
import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import os
import pandas as pd

# Define a Cartopy 'ordinary' lat-lon coordinate reference system.
crs_latlon = ccrs.PlateCarree()

# The JSON Pandas output containing teh interpolated hourly positions
hourly_file = '/home/jseddon/python/elinca/hourly_positions.json'

# The top-level directory containing the reanalysis data
era5_dir = '/home/jseddon/python/elinca/era5'

# The directory to save the output images into
output_dir = '/home/jseddon/python/elinca/plots'

def main():
    """Main entry point"""
    # Load the positions   
    with open(hourly_file) as fh:
        hourly = pd.read_json(fh, convert_dates=['time'])

    # Get just October for a reduced subset
    october = hourly[hourly.time.dt.strftime('%Y%m%d').between('20131001', 
                                                               '20131107')]

    # chose every nth wind point in the netCDF
    nth = 8
    for i, dp in enumerate(october.iterrows()):
        de = dp[1]
        print(f'{de.time.year}{de.time.month:02}{de.time.day:02}')
        # Load data
        day_dir = os.path.join(era5_dir, f'{de.time.year}', f'{de.time.month:02}',
                               f'{de.time.day:02}')
        file_prefix = (f'ecmwf-era5_oper_an_sfc_{de.time.year}'
                       f'{de.time.month:02}{de.time.day:02}{de.time.hour:02}00')
        mslp = iris.load_cube(os.path.join(day_dir, file_prefix + '.msl.nc'))
        mslp_atl = mslp.intersection(longitude=(-90, 10))
        mslp_atl.convert_units('hPa')
        u = iris.load_cube(os.path.join(day_dir, file_prefix + '.10u.nc'))
        u_atl = u.intersection(longitude=(-90, 10))
        u_atl_2deg = u_atl[0, ::nth, ::nth]
        u_atl_2deg.convert_units('knot')
        v = iris.load_cube(os.path.join(day_dir, file_prefix + '.10v.nc'))
        v_atl = v.intersection(longitude=(-90, 10))
        v_atl_2deg = v_atl[0, ::nth, ::nth]
        v_atl_2deg.convert_units('knot')
        # Plot data
        plt.figure(figsize=(12, 12))
        iplt.quiver(u_atl_2deg, v_atl_2deg)
        iplt.contour(mslp_atl[0], 20,
                         linewidths=0.5, colors='black', linestyles='-')
        ax = plt.gca()
        # ax.set_extent((-70.0, 10.0, -90.0, 60.0), crs=crs_latlon)
        ax.set_extent((-70.0, 10.0, -28.0, 42.0), crs=crs_latlon)
        ax.stock_img()
        ax.coastlines(resolution='10m')
        ax.gridlines()
        ax.set_xticks([-60, -40, -20, 0], crs=crs_latlon)
        ax.set_yticks([-20, 0, 20, 40], crs=crs_latlon)
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        # Add the date
        date_str = de.time.strftime('%d/%m/%Y %H:%M')
        props = dict(boxstyle='round', facecolor='white')
        ax.text(0.78, 0.02, date_str, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props, fontfamily='monospace')
        # Add the start and end points
        for port in [['Lisbon', 38.725267, -9.150019], 
                     ['Rio de Janeiro', -22.908333, -43.196389]]:
            plt.plot(port[2], port[1], marker='.', markersize=5.0,
                     markerfacecolor='red', markeredgecolor='red',
                     transform=crs_latlon)
            plt.text(port[2], port[1] - 3, port[0], horizontalalignment='center',
                     color='black', transform=crs_latlon)
        # Add the vessel's position
        boat_colour = 'lime' if de.src == 'f' else 'black'
        plt.plot(de.lon, de.lat, marker='o', markersize=7.0, markeredgewidth=2.5,
                         markerfacecolor=boat_colour, markeredgecolor='white',
                         transform=crs_latlon)
        # Save plot
        output_file = f'img{i:06}.png'
        plt.savefig(os.path.join(output_dir, output_file))
        plt.close()


if __name__ == "__main__":
    main()
 
