# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.18] - 2022-01-11
### Changed:
    - fix regions

## [3.0.17] - 2022-12-28
### Changed:
    - fix events

## [3.0.16] - 2022-12-26
### Changed:
    - fix classes

## [3.0.15] - 2022-12-25
### Changed:
    - models_yolov5_persons>=1.8.1.2022.12.25

## [3.0.14] - 2022-12-23
### Changed:
    - update re-id block

## [3.0.13] - 2022-12-05
### Changed:
    - bscreen_share>=3.52.0,<4
    - bscreen>=3.39.5,<4
    - info obejct event in reid block
    - fix readme

## [3.0.12] - 2022-11-22
### Changed:
    - openvino==2022.2.0

## [3.0.11] - 2022-11-22
### Changed:
    - move models in repos
    - models_yolov5_persons>=1.8.1.2022.11.22
    - models_reid_persons>=1.8.1.2022.11.22
    - move image_size from config to models info

## [3.0.10] - 2022-11-15
### Changed:
    - bscreen_share imports to bscreen_cv(as possible)

## [3.0.9] - 2022-11-14
### Changed:
    - bscreen_share>=3.49.0,<4
    - bscreen>=3.39.1,<4
    - fix object wrapper empty output

## [3.0.8] - 2022-11-03
### Changed:
    - gdown==4.5.3
    - reuired classes filtering
### Added:
    - README.md

## [3.0.7] - 2022-11-03
### Changed:
    - add reid block with strongsort object_reid.py
    - bscreen_fixtures>=1.10.0,<2.0.0

## [3.0.6] - 2022-11-03
### Changed:
    - torch>=1.8.1,<=1.12.1
    - torchvision>=0.9.1,<=0.13.1
    - yolov5==6.2.3
    - bscreen_fixtures>=1.4.0,<2.0.0

## [3.0.5] - 2022-11-03
### Changed:
    - bscreen_fixtures>=1.10.0,<2.0.0
    - add utils/yolov5
    - remove yolov5==6.2.3 (test docker image size)

## [3.0.4] - 2022-10-31
### Changed:
    - remove utils/yolov5
    - yolov5==6.2.3

## [3.0.3] - 2022-10-31
### Changed:
    - bscreen_share>=3.48.0,<4
    - zero events for regions
    - no_region counter event

## [3.0.2] - 2022-10-24
### Changed:
    - add region objects counters events

## [3.0.1] - 2022-10-19
### Changed:
    - unfix reqs versions
    - add new 640/1820 yolov5s persons models

## [3.0.0] - 2022-09-29
### Changed:
    - init
