from setuptools import find_packages, setup

package_name = 'lidar_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'mmengine',
        'torch',
        'mmdet3d',
    ],
    zip_safe=True,
    maintainer='shantnav',
    maintainer_email='shantnav@todo.todo',
    description='TODO: Package description',
    license='No License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_detector = lidar_perception.lidar_detector:main'
        ],
    },
)
