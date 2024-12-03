# Modifications on the original ultralytics code

## New files

    module[ir], url： ultralytics/nn/modules/ir
        __init__.py
        ir_hm_fusion.py
        position_encoding.py
        transformer_fusion.py
    [yolov8-irhm.yaml](ultralytics/cfg/models/v8/yolov8-irhm.yaml)
    [irhm1128.yaml](irhm1128.yaml)      # dataset yaml
    [test.py](test.py)                  # train script 

## Modified files

    [tasks.py](ultralytics/nn/tasks.py)
    [dataset.py](ultralytics/data/dataset.py)
    [build.py](ultralytics/data/build.py)
    
## Prepare ir data in yolo format, run test.py
