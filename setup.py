from setuptools import setup

setup(
    name='aikapi',
    version='1.0.0',
    description='Bonn Activity Maps (BAM) API',
    url='https://github.com/bonn-activity-maps/aikapi',
    packages=['aikapi', 'aikapi.utils'],
    install_requires=['numpy==1.19.2',
                      'opencv-python==4.4.0.44',
                      'numba==0.51.2',
                      'scipy==1.5.4'
                      ],
)
