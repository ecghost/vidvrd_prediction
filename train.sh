#!/bin/sh

CUDA_VISIBLE_DEVICES=2 python train_stage1.py

CUDA_VISIBLE_DEVICES=2 python train_stage2.py