%% This file obtains the training dataset and test dataset.
clc;
clear; 
close all;
% addpath('include');
% Convert HS dataset to patches

% List all '.mat' file in folder
file_folder=fullfile('/home/shiyanshi/dyx/datasets/Cave/test/');   % change the train file or test file.
file_list=dir(fullfile(file_folder,'*.mat'));
file_names={file_list.name};

% store cropped images in folders
for i = 1:1:numel(file_names)
    name = file_names{i};
    name = name(1:end-4);
    load(strcat('/home/shiyanshi/dyx/datasets/Cave/test/',file_names{i}));
    % crop_image(train, 32, 16, 0.5, name);     % train
    crop_image(a, 128, 64, 0.5, name);          % test
    
    % crop_image(train, 64, 32, 0.25, name);    % train
    % crop_image(a, 128, 64, 0.25, name);       % test
    
    % crop_image(train, 128, 64, 0.125, name);  % train
    % crop_image(test, 128, 64, 0.125, name);   % test
end
