from setuptools import find_packages, setup


setup(
    name="latte-attention",
    version="1.2.0",
    description="LATTE post-hoc linearization for DETR and RT-DETR decoder attention",
    packages=find_packages(),
    python_requires=">=3.10,<3.11",
    install_requires=[
        "torch==2.10.0",
        "torchvision==0.25.0",
        "transformers==5.2.0",
        "timm>=1.0,<2.0",
        "PyYAML==6.0.3",
        "numpy==2.2.6",
        "tqdm==4.67.3",
        "Pillow==12.1.1",
        "pycocotools==2.0.11",
    ],
    entry_points={
        "console_scripts": [
            "run_latte=latte.rtdetr_pipeline:main",
            "run_latte_rtdetr=latte.rtdetr_pipeline:main",
            "run_latte_detr=latte.detr_pipeline:main",
            "eval_latte_rtdetr=latte.rtdetr_evaluator:main",
            "eval_latte_detr=latte.detr_evaluator:main",
        ]
    },
)
