%% This is a demo code to show how to generate training and testing samples from the HSI %%
clc
clear
close all

addpath('/home/shiyanshi/dyx/CEGATSR/mcodes/predataset');

%% Step 1: generate the training and testing images from the original HSI
load('/home/shiyanshi/dyx/datasets/Chikusei/HyperspecVNIR_Chikusei_20140729.mat');%% Please down the Chikusei dataset (mat format) from https://www.sal.t.u-tokyo.ac.jp/hyperdata/
% load('/home/shiyanshi/dyx/datasets/Pavia/Pavia.mat');
%% center crop this image to size 2304 x 2048
a = chikusei(107:2410,144:2191,50:80);
% a = pavia(:,:,35:65);
clear chikusei;
% clear pavia;
% normalization
a = a ./ max(max(max(a)));
a = single(a);
% save('/home/shiyanshi/dyx/datasets/Pavia/all/Pavia.mat', 'a');
%% select first row as test images
[H, W, C] = size(a);
test_img_size = 512;
% test_pic_num = floor(W / test_img_size);
mkdir ('/home/shiyanshi/dyx/datasets/Chikusei/test');
test = a(1:test_img_size,:,:);
% test = a(:,(test_img_size+1):end,:);
save('/home/shiyanshi/dyx/datasets/Chikusei/test/Chikusei_test.mat', 'test');


% for i = 1:test_pic_num
    % left = (i - 1) * test_img_size + 1;
    % right = left + test_img_size - 1;
    % test = a(1:test_img_size,left:right,:);
    % save(strcat('/home/shiyanshi/dyx/datasets/Chikusei/test/Chikusei_test_', int2str(i), '.mat'),'a');
% end

%% the rest left for training
mkdir ('/home/shiyanshi/dyx/datasets/Chikusei/train');
train = a((test_img_size+1):end,:,:);
% train = a(:,1:test_img_size,:);
save('/home/shiyanshi/dyx/datasets/Chikusei/train/Chikusei_train.mat', 'train');

%% Step 2: generate the training samples (patches) and test samples cropped from the training/test images
generate_train_test;

%% Step 3: Please manually remove 10% of the samples to the folder of evals