### About this fork

 This repo was forked from the [Anatolix fork](https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation) of [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).

For changes to original [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) repository, see the [Michal Faber fork](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation) and the [Anatolix fork](https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation)

We have edited the Anatolix fork for transfer learning, starting with the trained CMU model weights.

## Changes

- Add config files to main folder
- Add video demo
- Remove segmentation mask from coco_masks_hdf5.py (replace with bounding box)
- load cmu model weights in train_pose.py

## How to run demo/training

- Download the data set (~25 GB) `cd dataset; sh get_dataset.sh`,
- Download [COCO official toolbox](https://github.com/pdollar/coco) in `dataset/coco/` . 
- `cd coco/PythonAPI; sudo python setup.py install` to install pycocotools.

- Download converted CMU keras model to model folder
`cd /model
sudo ./get_keras_model.sh`

Run demo on image:
cd ..
python3 demo_image.py --image sample_images/ski.jpg
Output saved in result.png in main folder

Run demo on video:
Add video to video_data folder
python3 demo_video.py
Output saved to video_data folder


Training steps

Download data to /dataset

coco:
sudo ./get_dataset.sh

Or add own data files to /dataset

Use coco-api to view data


Create .h5 data files
Edit /training/coco_masks_hdf5.py 
#!/usr/bin/env python # point to python env
point to correct .h5 data files
Set size of validation set

./coco_masks_hdf5.py to generate .h5 training files

Run training
Edit ./train_pose.py 
#!/usr/bin/env python # point to python env
Select gpu numbers
Change batch size if needed
Select model to train on

Run ./train_pose.py

Model files saved in /training/Canonical

    
## Related repository
- CVPR'16, [Convolutional Pose Machines](https://github.com/shihenw/convolutional-pose-machines-release).
- CVPR'17, [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).

## Citation
Please cite the paper in your publications if it helps your research:    

    @InProceedings{cao2017realtime,
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2017}
      }
	  
