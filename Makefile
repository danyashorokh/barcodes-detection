
DVC_REMOTE_NAME := storage

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: download_weights
download_weights:
	dvc pull -R weights -r $(DVC_REMOTE_NAME)

.PHONY: run_unit_tests
run_unit_tests:
	PYTHONPATH=. pytest tests/

.PHONY: lint
lint:
	flake8 ./

.PHONY: init_dvc
init_dvc:
	dvc init --no-scm
	dvc remote add --default $(DVC_REMOTE_NAME) ssh://91.206.15.25/home/$(USERNAME)/dvc_files
	dvc remote modify $(DVC_REMOTE_NAME) user $(USERNAME)
	dvc config cache.type hardlink,symlink

.PHONY: install_c_libs
install_c_libs:
	apt-get update && apt-get install -y --no-install-recommends gcc ffmpeg libsm6 libxext6

.PHONY: train_object_detection
train_object_detection:
	yolov5 train --img 1280 --batch 16 --epochs 100 --data data/barcode.yaml \
  	  --weights yolov5n.pt --device cpu --project barcode \
 	  --name experiement_1 > log.out

.PHONY: simple_predict
simple_predict:
	yolov5 detect --weights weights/best_openvino_model --source tests/fixtures/images/0b56af7e-386c-410a-8f46-74350f755d77--ru.4c7208d1-cba4-4539-967d-1c87ad0f6d2e.jpg
