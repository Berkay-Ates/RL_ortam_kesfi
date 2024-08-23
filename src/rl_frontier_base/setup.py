from setuptools import find_packages, setup

package_name = "rl_frontier_base"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="berkay",
    maintainer_email="atesberkay2356@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rl_control_similation_node = rl_frontier_base.rl_control_similation_node:main",
            "rl_control_test_node = rl_frontier_base.rl_control_model_test_node:main",
        ],
    },
)
