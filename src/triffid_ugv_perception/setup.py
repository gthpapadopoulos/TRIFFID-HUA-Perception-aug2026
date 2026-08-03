from setuptools import setup
import os
from glob import glob

package_name = 'triffid_ugv_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        # Register package with ament index
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # Package manifest
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='triffid',
    maintainer_email='amichailidou@hua.gr',
    description='TRIFFID UGV perception pipeline',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'ugv_node = triffid_ugv_perception.ugv_node:main',
            'geojson_bridge = triffid_ugv_perception.geojson_bridge:main',
        ],
    },
)
