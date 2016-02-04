clc; clear all; close all;

%% 
addpath(genpath('.'));
gpuDevice(1);

%% Load the image
% img = imread('lena.png');
% img = rgb2gray(img);
% img = imresize(img, [128, 128]);

%% Load a 3D object
% load kiwi_128_uint8
% % S0 = rgb2gray(imread('lena.png'));
% % S0 = imresize(S0, [128, 128]);
% % S0 = double(S0);
% S0 = double(vol(:,:,end/2));
% S0 = S0 - mean2(S0);

S0 = imread('kiwi_128.png');
% S0 = img;
% S0 = single(S0);
S0 = im2single(S0);
% S0 = S0-mean(vec(S0));

%% Seed the randomness
rng(2016);

plan.elemSize = [128, 128,  1,   1];
plan.dataSize = [128, 128,  1, 512]; % For example
plan.atomSize = [ 11,  11,  1,   1];
plan.dictSize = [ 11,  11,  1, 64];
plan.blobSize = [128, 128,  1, 64];
plan.iterSize = [128, 128,  1, 64]; 


%% Initialize the plan
plan.alpha  = params; % See param.m
plan.gamma  = params; 
plan.delta  = params;
plan.theta  = params;
plan.omega  = params;
plan.lambda = params; 
plan.sigma  = params; 
plan.rho    = params; 

plan.lambda.Value	= 1;
plan.weight         = 1.0/255;
plan.sigma.Value	= 0.5;
plan.rho.Value		= 0.5;

%% Solver initialization
plan.Verbose = 1;
plan.MaxIter = 100;
plan.AbsStopTol = 1e-6;
plan.RelStopTol = 1e-6;

%% Initialize the dictionary
%D0 = zeros(plan.dictSize); % Turn on if using single precision
D0 = rand(plan.dictSize);

%% Run the CSC algorithm
isTrainingDictionary=1;
[resD] = ecsc_gpu(D0, S0, plan, isTrainingDictionary);

[resX] = ecsc_gpu(resD.G, S0, plan, 0);
Slicer(squeeze(resX.Y));
Slicer(squeeze(resX.GY));

figure; imagesc(squeeze(sum(resX.GY, 4))); axis equal off; colormap gray; drawnow;
