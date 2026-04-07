from glob import glob
from setuptools import find_packages, setup

package_name = 'fetch'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name, ['README.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ferdinand',
    maintainer_email='ferdinand@todo.todo',
    description='ROS 2 Humble package for Go2 cube tracking, policy rollout, and state-machine control.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cube_tracker_node = fetch.cube_tracker_node:main',
            'policy_node = fetch.policy_node:main',
            'state_machine_node = fetch.state_machine_node:main',
        ],
    },
)
