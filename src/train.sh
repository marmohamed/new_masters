#!/bin/bash



data_path=../../Data/
epochs_img_head_city=5
epochs_img_all_city=10
epochs_img_head_kitti=15
epochs_img_all_kitti=0

epochs_bev=150
epochs_fusion=0
epochs_end=60


start_epoch_kitti=$((epochs_img_head_city+epochs_img_all_city))
start_epoch_bev=0
start_epoch_fusion=$((epochs_bev+start_epoch_bev))
start_epoch_end=$((epochs_fusion+epochs_bev+start_epoch_bev))
start_epoch_end=0


num_summary_images_segmentation=5
num_summary_images_detection=5

train_kitti=false
train_city=false
train_bev=true
train_fusion=false
train_end_to_end=false
train_bev_lr_find=false



if [ "$train_city" = true ]; then
    python Main.py --data_path $data_path \
                --train_images_seg \
                --epochs_img_head $epochs_img_head_city \
                --epochs_img_all $epochs_img_all_city \
                --batch_size 1 \
                --segmentation_cityscapes \
                --num_summary_images $num_summary_images_segmentation \
                --start_epoch 0 
fi

if [ "$train_kitti" = true ]; then
    python Main.py --data_path $data_path \
                --train_images_seg \
                --restore \
                --epochs_img_head $epochs_img_head_kitti \
                --epochs_img_all $epochs_img_all_kitti \
                --batch_size 1 \
                --segmentation_kitti \
                --segmentation_cityscapes  \
                --num_summary_images $num_summary_images_segmentation \
fi



if [ "$train_bev" = true ]; then

    python Main.py --data_path $data_path \
                --train_bev  \
                --epochs $epochs_bev \
                --start_epoch $start_epoch_bev \
                --num_summary_images $num_summary_images_detection \
                --batch_size 2 \

    # python write_prediction_in_files.py 
  
fi


if [ "$train_bev_lr_find" = true ]; then

    python Main.py --data_path $data_path \
                --train_bev_lr_find \
                --epochs $epochs_bev \
                --start_epoch $start_epoch_bev \
                --num_summary_images $num_summary_images_detection \
                --batch_size 1 \

    # python write_prediction_in_files.py 
  
fi

if [ "$train_fusion" = true ]; then
    python Main.py --data_path $data_path \
                --train_fusion \
                --epochs $epochs_fusion \
                --start_epoch $start_epoch_fusion \
                --num_summary_images $num_summary_images_detection \
                --batch_size 1 \

fi


if [ "$train_end_to_end" = true ]; then
    python Main.py --data_path $data_path \
                --train_end_to_end \
                --epochs $epochs_end \
                --start_epoch $start_epoch_end \
                --num_summary_images $num_summary_images_detection \
                --batch_size 1 \

fi
